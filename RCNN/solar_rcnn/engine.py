"""Corre una combinación arquitectura × estrategia × fold, o las 18.

Importar este módulo no carga torch. torch y torchvision se importan recién
dentro de ``run_one``, cuando empieza un entrenamiento que no esté terminado.

Los folds, las verificaciones de datos, los marcadores de reanudación y los
CSV vienen de ``solar_yolo``. Las filas de ``fold_metrics.csv`` y
``summary.csv`` tienen las mismas columnas que las de YOLO, así las dos
familias se comparan en una sola tabla.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from solar_rcnn.architectures import (
    ARCHITECTURES,
    ArchitectureError,
    STRATEGIES,
    assert_freeze_state,
    audit_architectures,
    build_model,
    describe_architectures,
    freeze_plan,
)
from solar_rcnn.config import Config, ConfigError, estimated_iterations
from solar_rcnn.metrics import (
    best_threshold,
    classification_metrics,
    detection_metrics,
    image_scores,
)
from solar_yolo.data import FOLDS, describe_folds, fold_definitions, inventory_dataset
from solar_yolo.evaluation import (
    combined_metric_row,
    count_false_positives,
    kingston2_row,
    positive_source_row,
    ultralytics_fitness,
)
from solar_yolo.storage import (
    _read_rows,
    _write_rows,
    archive_incomplete_run,
    ensure_result_dirs,
    experiment_id,
    is_completed,
    now_iso,
    read_experiment_row,
    read_marker,
    rebuild_summary,
    replace_fold_metrics,
    run_dir,
    training_finished,
    upsert_row,
    weights_path,
    write_config_snapshot,
    write_marker,
)


LOGGER = logging.getLogger("solar_rcnn")

EXPERIMENT_FIELDS = [
    "experiment_id",
    "status",
    "architecture",
    "weights",
    "strategy",
    "fold",
    "train_sources",
    "val_sources",
    "trainable_backbone_layers",
    "frozen_modules",
    "trainable_modules",
    "params_trainable",
    "params_frozen",
    "epochs",
    "imgsz",
    "batch",
    "seed",
    "patience",
    "lr",
    "momentum",
    "weight_decay",
    "estimated_iterations",
    "best_epoch",
    "epochs_ran",
    "train_time_sec",
    "wall_time_sec",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "fitness",
    "best_weights",
    "run_dir",
    "predictions_dir",
    "error",
    "warning",
    "started_at",
    "finished_at",
    "config_hash",
    "torch_version",
    "torchvision_version",
    "device_name",
]

HISTORY_FIELDS = [
    "epoch",
    "lr",
    "loss",
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "fitness",
    "epoch_time_sec",
]

CLASSIFICATION_FIELDS = [
    "experiment_id",
    "architecture",
    "strategy",
    "fold",
    "threshold_kind",
    "threshold",
    "n_images",
    "n_positive_images",
    "tp",
    "fp",
    "fn",
    "tn",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "note",
]


def verify_setup(
    config: Config,
    architectures: list[str] | None = None,
    strategies: list[str] | None = None,
    folds: list[int] | None = None,
) -> str:
    """Revisa datos, la tabla de congelamiento y la grilla pedida. No carga modelos."""
    config = config.resolved()
    audit_architectures()
    if config.epochs < 1 or config.batch < 1 or config.patience < 1 or config.imgsz < 32:
        raise ConfigError("epochs, batch y patience tienen que ser enteros positivos, e imgsz al menos 32.")

    inventories = inventory_dataset(config.dataset_dir)
    jobs = select_jobs(architectures, strategies, folds)
    iteration_lines = []
    for architecture, strategy, fold in jobs:
        n_train = len(inventories[fold].of("train"))
        iterations = estimated_iterations(n_train, config.epochs, config.batch)
        iteration_lines.append(
            f"{experiment_id(architecture, strategy, fold)}: {n_train} imágenes de train, hasta {iterations} iteraciones"
        )

    return "\n\n".join(
        [
            describe_architectures(),
            describe_folds(inventories),
            "Experimentos seleccionados\n==========================\n" + "\n".join(iteration_lines),
            (
                f"SGD lr={config.lr} momentum={config.momentum} weight_decay={config.weight_decay}, "
                f"warmup de {config.warmup_iters} iteraciones y coseno. batch={config.batch}, "
                f"imgsz={config.imgsz} (min_size=max_size), epochs={config.epochs}, patience={config.patience}."
            ),
        ]
    )


def run_experiments(
    config: Config,
    architectures: list[str] | None = None,
    strategies: list[str] | None = None,
    folds: list[int] | None = None,
    stop_on_error: bool = False,
) -> dict:
    """Entrena la grilla. Un experimento fallido no cancela el resto, salvo ``stop_on_error``."""
    config = config.resolved()
    report = verify_setup(config, architectures, strategies, folds)
    _configure_logging(config.results_dir / "logs" / "session.log")
    LOGGER.info("\n%s", report)

    ensure_result_dirs(config.results_dir)
    jobs = select_jobs(architectures, strategies, folds)
    write_config_snapshot(
        config.results_dir,
        {
            "config": config.to_json(),
            "config_hash": config.config_hash(freeze_plan(), fold_definitions()),
            "freeze_plan": freeze_plan(),
            "folds": fold_definitions(),
            "jobs": [experiment_id(*job) for job in jobs],
            "score": "(mAP50-95 fold 1 + mAP50-95 fold 2 + mAP50-95 fold 3) / 3",
            "written_at": now_iso(),
        },
    )
    device = resolve_device(config)

    inventories = inventory_dataset(config.dataset_dir)
    failures = []
    for architecture, strategy, fold in jobs:
        exp_id = experiment_id(architecture, strategy, fold)
        try:
            if config.resume and is_completed(config.results_dir, architecture, strategy, fold):
                LOGGER.info("Ya está terminado, se saltea: %s", exp_id)
                continue
            run_one(config, inventories[fold], architecture, strategy, device)
        except KeyboardInterrupt:
            LOGGER.exception("Interrumpido en %s", exp_id)
            raise
        except Exception as exc:
            LOGGER.exception("Falló %s", exp_id)
            failures.append({"experiment_id": exp_id, "error": str(exc)})
            if stop_on_error:
                break
        finally:
            rebuild_summary(config.results_dir)

    return {"jobs": [experiment_id(*job) for job in jobs], "failures": failures}


def run_one(config: Config, inventory, architecture: str, strategy: str, device: str) -> None:
    import torch
    from torch.utils.data import DataLoader

    from solar_rcnn.dataset import PanelDataset, collate

    fold = inventory.spec.fold
    exp_id = experiment_id(architecture, strategy, fold)
    results_dir = config.results_dir
    directory = run_dir(results_dir, exp_id)
    stable_weights = weights_path(results_dir, architecture, strategy, fold)
    config_hash = config.config_hash(freeze_plan(), fold_definitions())
    n_train = len(inventory.of("train"))
    iterations = estimated_iterations(n_train, config.epochs, config.batch)

    _configure_logging(results_dir / "logs" / f"{exp_id}.log")
    LOGGER.info("Empieza %s. trainable_backbone_layers=%s", exp_id, ARCHITECTURES[architecture].trainable_layers[strategy])

    if training_finished(results_dir, exp_id) and stable_weights.is_file():
        LOGGER.info("El entrenamiento de %s ya había terminado. Se rehacen solo las métricas.", exp_id)
        _evaluate_finished_run(config, inventory, architecture, strategy, stable_weights, config_hash, iterations, device)
        return

    last_checkpoint = directory / "weights" / "last.pt"
    resume_run = config.resume and last_checkpoint.is_file()
    if directory.exists() and not resume_run:
        archived = archive_incomplete_run(directory)
        LOGGER.info("Se archivó una corrida incompleta de %s en %s", exp_id, archived)
    (directory / "weights").mkdir(parents=True, exist_ok=True)
    best_checkpoint = directory / "weights" / "best.pt"
    history_path = directory / "results.csv"

    row = _base_row(config, inventory, architecture, strategy, config_hash, iterations)
    row["status"] = "running"
    row["started_at"] = now_iso()
    upsert_row(results_dir / "experiments.csv", EXPERIMENT_FIELDS, exp_id, row)
    write_marker(results_dir, exp_id, {"status": "running", "training_finished": False, "experiment_id": exp_id})

    try:
        set_seed(config.seed)
        model = build_model(architecture, strategy, config.imgsz)
        counts = assert_freeze_state(model, architecture, strategy)
        LOGGER.info("Congelamiento verificado: %d parámetros entrenables, %d congelados.", counts["trainable"], counts["frozen"])
        model.roi_heads.score_thresh = config.eval_score_thresh
        model.to(device)

        val_records = inventory.of("val")
        ground_truths = _ground_truths(val_records)
        generator = torch.Generator()
        train_loader = DataLoader(
            PanelDataset(inventory.of("train"), train=True, hflip=config.hflip, vflip=config.vflip),
            batch_size=config.batch,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=collate,
            generator=generator,
            worker_init_fn=_seed_worker,
            pin_memory=device == "cuda",
        )
        val_loader = _eval_loader(val_records, config, device)

        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        total_iters = len(train_loader) * config.epochs
        warmup = min(config.warmup_iters, total_iters)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _lr_factor(step, warmup, total_iters))
        use_amp = config.amp and device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        state = {"epoch": 0, "best_fitness": -1.0, "best_epoch": None, "bad_epochs": 0, "train_time_sec": 0.0}
        if resume_run:
            checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            state = checkpoint["state"]
            LOGGER.info("Se reanuda %s desde la época %d.", exp_id, state["epoch"])
            row["warning"] = f"Reanudado desde la época {state['epoch']}."

        started = time.perf_counter()
        for epoch in range(state["epoch"] + 1, config.epochs + 1):
            if state["bad_epochs"] >= config.patience:
                # Una corrida reanudada que ya había cortado por early stopping.
                break
            # Semilla por época: una corrida reanudada baraja igual que una entera.
            torch.manual_seed(config.seed + epoch)
            generator.manual_seed(config.seed + epoch)
            epoch_start = time.perf_counter()
            epoch_lr = optimizer.param_groups[0]["lr"]
            losses = _train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp)
            predictions = predict(model, val_loader, device, use_amp)
            metrics = detection_metrics(predictions, ground_truths)
            fitness = ultralytics_fitness(metrics["map50"], metrics["map50_95"])
            epoch_time = time.perf_counter() - epoch_start
            state["train_time_sec"] += epoch_time
            _append_history(
                history_path,
                {
                    "epoch": epoch,
                    "lr": epoch_lr,
                    **losses,
                    "map50": metrics["map50"],
                    "map50_95": metrics["map50_95"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "fitness": fitness,
                    "epoch_time_sec": epoch_time,
                },
            )
            LOGGER.info(
                "%s época %d/%d: loss %.4f  mAP50 %.4f  mAP50-95 %.4f  (%.0f s)",
                exp_id, epoch, config.epochs, losses["loss"], metrics["map50"], metrics["map50_95"], epoch_time,
            )

            state["epoch"] = epoch
            if fitness > state["best_fitness"]:
                state.update(best_fitness=fitness, best_epoch=epoch, bad_epochs=0)
                _save_weights(best_checkpoint, model, architecture, strategy, fold, epoch, config)
            else:
                state["bad_epochs"] += 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "state": state,
                },
                last_checkpoint,
            )
            if state["bad_epochs"] >= config.patience:
                LOGGER.info("Early stopping en la época %d: %d épocas sin mejorar.", epoch, config.patience)
                break
        wall_time = time.perf_counter() - started

        if not best_checkpoint.is_file() or best_checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"No quedó best.pt después de entrenar {exp_id}.")
        stable_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_checkpoint, stable_weights)
        _plot_history(history_path, results_dir / "plots" / exp_id / "entrenamiento.png", exp_id)

        best_row = _history_row(history_path, state["best_epoch"])
        row.update(
            {
                "status": "trained",
                "best_epoch": state["best_epoch"],
                "epochs_ran": state["epoch"],
                "train_time_sec": state["train_time_sec"],
                "wall_time_sec": wall_time,
                "map50": float(best_row["map50"]),
                "map50_95": float(best_row["map50_95"]),
                "precision": float(best_row["precision"]),
                "recall": float(best_row["recall"]),
                "fitness": float(best_row["fitness"]),
                "best_weights": str(stable_weights),
                "run_dir": str(directory),
            }
        )
        upsert_row(results_dir / "experiments.csv", EXPERIMENT_FIELDS, exp_id, row)
        write_marker(
            results_dir,
            exp_id,
            {
                "status": "trained",
                "training_finished": True,
                "experiment_id": exp_id,
                "best_weights": str(stable_weights),
                "run_dir": str(directory),
                "metrics": {key: row[key] for key in ("map50", "map50_95", "precision", "recall", "fitness", "best_epoch")},
            },
        )
    except KeyboardInterrupt:
        row["status"] = "interrupted"
        row["error"] = "interrumpido"
        row["finished_at"] = now_iso()
        upsert_row(results_dir / "experiments.csv", EXPERIMENT_FIELDS, exp_id, row)
        raise
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = _short_error(exc)
        row["finished_at"] = now_iso()
        upsert_row(results_dir / "experiments.csv", EXPERIMENT_FIELDS, exp_id, row)
        write_marker(results_dir, exp_id, {"status": "failed", "training_finished": False, "error": _short_error(exc)})
        if isinstance(exc, ArchitectureError) and directory.exists():
            archive_incomplete_run(directory)
        (results_dir / "logs" / f"{exp_id}.traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

    _evaluate_finished_run(config, inventory, architecture, strategy, stable_weights, config_hash, iterations, device, existing_row=row)


def _evaluate_finished_run(
    config: Config,
    inventory,
    architecture: str,
    strategy: str,
    stable_weights: Path,
    config_hash: str,
    iterations: int,
    device: str,
    existing_row: dict | None = None,
) -> None:
    """Predice la validación con el best.pt y escribe todas las métricas. No entrena.

    Se predice una sola vez. Las métricas por fuente salen de las mismas
    predicciones, filtradas por imagen.
    """
    import torch

    exp_id = experiment_id(architecture, strategy, inventory.spec.fold)
    results_dir = config.results_dir
    row = existing_row or _row_from_marker(results_dir, exp_id) or _base_row(
        config, inventory, architecture, strategy, config_hash, iterations
    )
    row["best_weights"] = str(stable_weights)
    try:
        model = build_model(architecture, strategy, config.imgsz)
        checkpoint = torch.load(stable_weights, map_location="cpu", weights_only=False)
        if (checkpoint.get("architecture"), checkpoint.get("strategy")) != (architecture, strategy):
            raise ArchitectureError(
                f"{stable_weights} es de {checkpoint.get('architecture')} / {checkpoint.get('strategy')}, "
                f"no de {architecture} / {strategy}."
            )
        model.load_state_dict(checkpoint["model"])
        model.roi_heads.score_thresh = config.eval_score_thresh
        model.to(device)

        val_records = inventory.of("val")
        use_amp = config.amp and device == "cuda"
        predictions = predict(model, _eval_loader(val_records, config, device), device, use_amp)
        ground_truths = _ground_truths(val_records)

        combined = detection_metrics(predictions, ground_truths)
        combined["fitness"] = ultralytics_fitness(combined["map50"], combined["map50_95"])
        metric_rows = [combined_metric_row(exp_id, architecture, strategy, inventory, combined)]
        if len(inventory.spec.val_sources) == 1:
            source_row = positive_source_row(exp_id, architecture, strategy, inventory, inventory.spec.val_sources[0], combined)
            source_row["note"] = "La fuente ocupa toda la validación del fold. Es la misma métrica combinada."
            metric_rows.append(source_row)
        else:
            for source in inventory.spec.per_source_eval:
                if source == "kingston2":
                    continue
                index = [i for i, record in enumerate(val_records) if record.source == source]
                metrics = detection_metrics([predictions[i] for i in index], [ground_truths[i] for i in index])
                metrics["fitness"] = ultralytics_fitness(metrics["map50"], metrics["map50_95"])
                metric_rows.append(positive_source_row(exp_id, architecture, strategy, inventory, source, metrics))

        prediction_dir = _save_predictions(results_dir / "predictions" / exp_id, val_records, predictions, config)
        if "kingston2" in inventory.spec.val_sources:
            fp_images, fp_boxes = count_false_positives(prediction_dir / "labels", inventory.of("val", "kingston2"))
            metric_rows.append(
                kingston2_row(exp_id, architecture, strategy, inventory, fp_images, fp_boxes, config.prediction_conf)
            )
        replace_fold_metrics(results_dir / "fold_metrics.csv", exp_id, metric_rows)

        _write_classification_rows(results_dir, exp_id, architecture, strategy, inventory.spec.fold, val_records, predictions, config)
        _write_pr_curve(results_dir / "plots" / exp_id, combined, exp_id)

        row.update(
            {
                "map50": combined["map50"],
                "map50_95": combined["map50_95"],
                "precision": combined["precision"],
                "recall": combined["recall"],
                "fitness": combined["fitness"],
                "predictions_dir": str(prediction_dir),
                "status": "completed",
                "finished_at": now_iso(),
                "error": "",
            }
        )
        upsert_row(results_dir / "experiments.csv", EXPERIMENT_FIELDS, exp_id, row)
        write_marker(
            results_dir,
            exp_id,
            {
                "status": "completed",
                "training_finished": True,
                "experiment_id": exp_id,
                "best_weights": str(stable_weights),
                "predictions_dir": str(prediction_dir),
            },
        )
        LOGGER.info("Terminado %s. mAP50-95 %.4f. best.pt en %s", exp_id, combined["map50_95"], stable_weights)
    except Exception as exc:
        row["status"] = "failed_eval"
        row["error"] = _short_error(exc)
        row["finished_at"] = now_iso()
        upsert_row(results_dir / "experiments.csv", EXPERIMENT_FIELDS, exp_id, row)
        write_marker(
            results_dir,
            exp_id,
            {
                "status": "failed_eval",
                "training_finished": True,
                "experiment_id": exp_id,
                "best_weights": str(stable_weights),
                "error": _short_error(exc),
            },
        )
        (results_dir / "logs" / f"{exp_id}.traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def select_jobs(
    architectures: list[str] | None,
    strategies: list[str] | None,
    folds: list[int] | None,
) -> list[tuple[str, str, int]]:
    chosen_architectures = list(architectures or ARCHITECTURES)
    chosen_strategies = list(strategies or STRATEGIES)
    chosen_folds = list(folds or FOLDS)
    unknown_architectures = [name for name in chosen_architectures if name not in ARCHITECTURES]
    unknown_strategies = [name for name in chosen_strategies if name not in STRATEGIES]
    unknown_folds = [fold for fold in chosen_folds if fold not in FOLDS]
    if unknown_architectures or unknown_strategies or unknown_folds:
        raise ConfigError(
            "Selección desconocida. "
            f"Arquitecturas: {unknown_architectures or '-'}; "
            f"estrategias: {unknown_strategies or '-'}; "
            f"folds: {unknown_folds or '-'}. "
            f"Válidas: {list(ARCHITECTURES)}, {list(STRATEGIES)}, {list(FOLDS)}."
        )
    return [
        (architecture, strategy, fold)
        for architecture in chosen_architectures
        for strategy in chosen_strategies
        for fold in chosen_folds
    ]


def resolve_device(config: Config) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("No está instalado torch. Colab ya lo trae con torchvision.") from exc
    if torch.cuda.is_available():
        LOGGER.info("CUDA disponible: %s", torch.cuda.get_device_name(0))
        return "cuda"
    if config.allow_cpu:
        LOGGER.warning("CUDA no está disponible. Se sigue en CPU porque allow_cpu=True.")
        return "cpu"
    raise RuntimeError(
        "CUDA no está disponible. El experimento está preparado para una GPU NVIDIA de Colab. "
        "No se inició ningún entrenamiento. Para forzar CPU, pasá allow_cpu=True."
    )


def set_seed(seed: int) -> None:
    import os
    import random

    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # roi_align no tiene backward determinista en CUDA: no se pide
    # use_deterministic_algorithms. Con la misma semilla los resultados se
    # parecen, pero no son idénticos bit a bit.
    torch.backends.cudnn.benchmark = False


def predict(model, loader, device: str, use_amp: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(boxes, scores)`` por imagen, en el orden del loader, en píxeles ``xyxy``."""
    import torch

    model.eval()
    outputs = []
    with torch.inference_mode(), torch.autocast(device_type=device, enabled=use_amp):
        for images, _ in loader:
            for result in model([image.to(device, non_blocking=True) for image in images]):
                outputs.append((result["boxes"].float().cpu().numpy(), result["scores"].float().cpu().numpy()))
    return outputs


def _train_one_epoch(model, loader, optimizer, scheduler, scaler, device: str, use_amp: bool) -> dict:
    import torch

    model.train()
    totals: dict[str, float] = {}
    seen = 0
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{key: (value.to(device) if torch.is_tensor(value) else value) for key, value in target.items()} for target in targets]
        with torch.autocast(device_type=device, enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"La loss dio {loss.item()}. Componentes: { {k: v.item() for k, v in loss_dict.items()} }")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        size = len(images)
        seen += size
        totals["loss"] = totals.get("loss", 0.0) + loss.item() * size
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + value.item() * size
    return {key: value / max(seen, 1) for key, value in totals.items()}


def _lr_factor(step: int, warmup: int, total: int) -> float:
    """Warmup lineal desde 0.001 y después coseno hasta 0."""
    if warmup and step < warmup:
        alpha = step / warmup
        return 0.001 * (1 - alpha) + alpha
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _seed_worker(worker_id: int) -> None:
    import random

    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _eval_loader(records: list, config: Config, device: str):
    from torch.utils.data import DataLoader

    from solar_rcnn.dataset import PanelDataset, collate

    return DataLoader(
        PanelDataset(records),
        batch_size=config.batch,
        shuffle=False,
        num_workers=config.workers,
        collate_fn=collate,
        pin_memory=device == "cuda",
    )


def _ground_truths(records: list) -> list[np.ndarray]:
    from PIL import Image

    from solar_rcnn.dataset import read_boxes

    boxes = []
    for record in records:
        with Image.open(record.path) as image:
            width, height = image.size
        gt = read_boxes(record.label_path, width, height)
        if len(gt) != record.boxes:
            raise RuntimeError(f"{record.label_path}: el inventario contó {record.boxes} cajas y se leyeron {len(gt)}.")
        boxes.append(gt)
    return boxes


def _save_weights(path: Path, model, architecture: str, strategy: str, fold: int, epoch: int, config: Config) -> None:
    import torch
    import torchvision

    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": architecture,
            "strategy": strategy,
            "fold": fold,
            "epoch": epoch,
            "imgsz": config.imgsz,
            "torchvision_version": torchvision.__version__,
        },
        temporary,
    )
    temporary.replace(path)


def _save_predictions(destination: Path, records: list, predictions: list, config: Config) -> Path:
    """Guarda las predicciones de validación de tres formas.

    - ``labels/<imagen>.txt``: formato YOLO ``0 cx cy w h conf`` con
      confianza >= ``prediction_conf``. Sin archivo si no hay cajas, igual que
      Ultralytics, así ``count_false_positives`` sirve para las dos familias.
    - ``raw.npz``: todas las cajas desde ``eval_score_thresh``, para recalcular
      cualquier métrica sin volver a correr el modelo.
    - ``image_scores.csv``: la confianza máxima por imagen, que es el score
      del detector usado como clasificador.
    """
    from PIL import Image

    label_dir = destination / "labels"
    if label_dir.is_dir():
        for stale in label_dir.glob("*.txt"):
            stale.unlink()
    label_dir.mkdir(parents=True, exist_ok=True)

    names, image_index, all_boxes, all_scores, score_rows = [], [], [], [], []
    for i, (record, (boxes, scores)) in enumerate(zip(records, predictions)):
        with Image.open(record.path) as image:
            width, height = image.size
        keep = scores >= config.prediction_conf
        lines = []
        for (x1, y1, x2, y2), score in zip(boxes[keep], scores[keep]):
            lines.append(
                f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f} {score:.5f}"
            )
        if lines:
            (label_dir / f"{record.path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        names.append(record.path.stem)
        image_index.append(np.full(len(scores), i, dtype=np.int32))
        all_boxes.append(boxes.reshape(-1, 4))
        all_scores.append(scores)
        score_rows.append(
            {
                "image": record.path.stem,
                "source": record.source,
                "gt_boxes": record.boxes,
                "has_panels": int(record.boxes > 0),
                "max_score": float(scores.max()) if len(scores) else 0.0,
                "detections_at_conf": int(keep.sum()),
            }
        )

    np.savez_compressed(
        destination / "raw.npz",
        names=np.array(names),
        image_index=np.concatenate(image_index),
        boxes=np.concatenate(all_boxes),
        scores=np.concatenate(all_scores),
    )
    _write_rows(destination / "image_scores.csv", list(score_rows[0]), score_rows)
    return destination


def _write_classification_rows(
    results_dir: Path, exp_id: str, architecture: str, strategy: str, fold: int, records: list, predictions: list, config: Config
) -> None:
    scores = image_scores(predictions)
    truth = np.array([record.boxes > 0 for record in records])
    base = {"experiment_id": exp_id, "architecture": architecture, "strategy": strategy, "fold": fold}
    calibrated = best_threshold(scores, truth)
    rows = [
        {
            **base,
            "threshold_kind": "fabrica",
            **classification_metrics(scores, truth, config.prediction_conf),
            "note": "Tiene paneles si alguna detección llega a esta confianza.",
        },
        {
            **base,
            "threshold_kind": "calibrado",
            **classification_metrics(scores, truth, calibrated),
            "note": "Umbral elegido sobre esta misma validación para maximizar F1 macro. Diagnóstico, no estimación.",
        },
        {
            **base,
            "threshold_kind": "trivial_todo_positivo",
            **classification_metrics(np.ones_like(scores), truth, 0.5),
            "note": "Referencia: decir siempre que hay paneles. Hay que superarla.",
        },
    ]
    path = results_dir / "classification_metrics.csv"
    kept = [row for row in _read_rows(path) if row.get("experiment_id") != exp_id]
    kept.extend({field: _cell(row.get(field, "")) for field in CLASSIFICATION_FIELDS} for row in rows)
    kept.sort(key=lambda item: (item.get("experiment_id", ""), item.get("threshold_kind", "")))
    _write_rows(path, CLASSIFICATION_FIELDS, kept)


def _write_pr_curve(destination: Path, metrics: dict, exp_id: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    curve = metrics["curve"]
    rows = [
        {"confidence": f"{c:.6f}", "precision": f"{p:.6f}", "recall": f"{r:.6f}"}
        for c, p, r in zip(curve["confidence"], curve["precision"], curve["recall"])
    ]
    _write_rows(destination / "pr_curve_iou50.csv", ["confidence", "precision", "recall"], rows)
    plt = _pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(curve["recall"], curve["precision"], lw=1.4)
    ax.scatter([metrics["recall"]], [metrics["precision"]], s=18, zorder=3, label=f"F1 máx (conf {metrics['conf_at_best_f1']:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precisión")
    ax.set_title(f"{exp_id}\nIoU 0.5 · mAP50 {metrics['map50']:.3f} · mAP50-95 {metrics['map50_95']:.3f}", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(destination / "pr_curve.png", dpi=130)
    plt.close(fig)


def _plot_history(history_path: Path, destination: Path, exp_id: str) -> None:
    plt = _pyplot()
    if plt is None or not history_path.is_file():
        return
    rows = _read_rows(history_path)
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for key in ("loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
        if rows and key in rows[0]:
            axes[0].plot(epochs, [float(row[key]) for row in rows], label=key, lw=1.2)
    axes[0].set_xlabel("época")
    axes[0].set_title("loss de entrenamiento", fontsize=9)
    axes[0].legend(fontsize=6)
    for key in ("map50", "map50_95"):
        axes[1].plot(epochs, [float(row[key]) for row in rows], label=key, lw=1.2)
    axes[1].set_xlabel("época")
    axes[1].set_title("validación", fontsize=9)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(exp_id, fontsize=9)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("No está matplotlib: se guardan los CSV de las curvas, sin gráficos.")
        return None
    return plt


def _append_history(path: Path, row: dict) -> None:
    rows = [existing for existing in _read_rows(path) if int(existing["epoch"]) < int(row["epoch"])]
    rows.append({field: _cell(row.get(field, "")) for field in HISTORY_FIELDS})
    _write_rows(path, HISTORY_FIELDS, rows)


def _history_row(path: Path, epoch: int) -> dict:
    for row in _read_rows(path):
        if int(row["epoch"]) == int(epoch):
            return row
    raise RuntimeError(f"{path} no tiene la época {epoch}.")


def _base_row(config: Config, inventory, architecture: str, strategy: str, config_hash: str, iterations: int) -> dict:
    import importlib.metadata as metadata

    spec = ARCHITECTURES[architecture]
    exp_id = experiment_id(architecture, strategy, inventory.spec.fold)
    versions = {}
    for package in ("torch", "torchvision"):
        try:
            versions[f"{package}_version"] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[f"{package}_version"] = ""
    try:
        import torch

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        device_name = ""
    return {
        "experiment_id": exp_id,
        "architecture": architecture,
        "weights": f"{spec.weights_enum}.COCO_V1",
        "strategy": strategy,
        "fold": inventory.spec.fold,
        "train_sources": "+".join(inventory.spec.train_sources),
        "val_sources": "+".join(inventory.spec.val_sources),
        "trainable_backbone_layers": spec.trainable_layers[strategy],
        "frozen_modules": spec.describe_frozen(strategy),
        "trainable_modules": spec.describe_trainable(strategy),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "seed": config.seed,
        "patience": config.patience,
        "lr": config.lr,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "estimated_iterations": iterations,
        "config_hash": config_hash,
        "run_dir": str(run_dir(config.results_dir, exp_id)),
        "best_weights": str(weights_path(config.results_dir, architecture, strategy, inventory.spec.fold)),
        "device_name": device_name,
        **versions,
    }


def _row_from_marker(results_dir: Path, exp_id: str) -> dict | None:
    row = read_experiment_row(results_dir, exp_id)
    if row is not None:
        return row
    marker = read_marker(results_dir, exp_id)
    if not marker:
        return None
    return {"experiment_id": exp_id, **marker.get("metrics", {})}


def _configure_logging(log_path: Path) -> None:
    global SESSION_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if SESSION_LOG is None:
        SESSION_LOG = log_path
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    for path in dict.fromkeys((log_path, SESSION_LOG)):
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


SESSION_LOG: Path | None = None


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)
