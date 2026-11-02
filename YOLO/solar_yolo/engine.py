"""Corre una combinación arquitectura × estrategia × fold, o las 27.

Importar este módulo no carga Ultralytics. ``YOLO()`` se llama recién dentro
de ``run_one``, cuando empieza un entrenamiento que no esté ya terminado.
"""

from __future__ import annotations

import logging
import shutil
import time
import traceback
from pathlib import Path

from solar_yolo.architectures import (
    ARCHITECTURES,
    ArchitectureError,
    STRATEGIES,
    assert_freeze_state,
    audit_architectures,
    describe_architectures,
    verify_loaded_architecture,
)
from solar_yolo.config import (
    AUTO_OPTIMIZER_ITERATION_LIMIT,
    Config,
    ConfigError,
    estimated_iterations,
)
from solar_yolo.data import (
    FOLDS,
    describe_folds,
    fold_definitions,
    inventory_dataset,
    write_fold_yaml,
    write_source_yaml,
)
from solar_yolo.evaluation import (
    combined_metric_row,
    count_false_positives,
    kingston2_row,
    metrics_from_validator,
    positive_source_row,
    read_best_epoch_metrics,
)
from solar_yolo.storage import (
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
    EXPERIMENT_FIELDS,
)


LOGGER = logging.getLogger("solar_yolo")


def verify_setup(
    config: Config,
    architectures: list[str] | None = None,
    strategies: list[str] | None = None,
    folds: list[int] | None = None,
) -> str:
    """Revisa datos, tablas de capas y la grilla pedida. No carga modelos."""
    config = config.resolved()
    audit_architectures()
    if config.imgsz % 32 != 0:
        raise ConfigError(f"imgsz={config.imgsz} no es múltiplo de 32. YOLO necesita ese múltiplo del stride.")
    if config.epochs < 1 or config.batch < 1 or config.patience < 1:
        raise ConfigError("epochs, batch y patience tienen que ser enteros positivos.")

    inventories = inventory_dataset(config.dataset_dir)
    jobs = select_jobs(architectures, strategies, folds)
    iteration_lines = []
    for architecture, strategy, fold in jobs:
        n_train = len(inventories[fold].of("train"))
        iterations = estimated_iterations(n_train, config.epochs, config.batch)
        iteration_lines.append(
            f"{experiment_id(architecture, strategy, fold)}: {n_train} imágenes de train, cerca de {iterations} iteraciones"
        )
        if config.optimizer == "auto" and iterations > AUTO_OPTIMIZER_ITERATION_LIMIT:
            raise ConfigError(
                f"{experiment_id(architecture, strategy, fold)} supera las {AUTO_OPTIMIZER_ITERATION_LIMIT} "
                "iteraciones. Con optimizer=auto ese fold usaría MuSGD y los demás AdamW. "
                "Bajá epochs o fijá un optimizador único en solar_yolo/config.py antes de entrenar."
            )

    report = "\n\n".join(
        [
            describe_architectures(),
            describe_folds(inventories),
            "Experimentos seleccionados\n==========================\n" + "\n".join(iteration_lines),
            (
                "optimizer=auto queda en AdamW para todos estos folds "
                f"(umbral de Ultralytics: {AUTO_OPTIMIZER_ITERATION_LIMIT} iteraciones). "
                "lr0 lo calcula Ultralytics como 0.002 * 5 / (4 + nc); con una clase es 0.002."
            ),
        ]
    )
    return report


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
            "config_hash": config.config_hash(_freeze_plan_public(), fold_definitions()),
            "freeze_plan": _freeze_plan_public(),
            "folds": fold_definitions(),
            "jobs": [experiment_id(*job) for job in jobs],
            "score": "(mAP50-95 fold 1 + mAP50-95 fold 2 + mAP50-95 fold 3) / 3",
            "written_at": now_iso(),
        },
    )
    assert_cuda(config)

    inventories = inventory_dataset(config.dataset_dir)
    failures = []
    for architecture, strategy, fold in jobs:
        exp_id = experiment_id(architecture, strategy, fold)
        try:
            if config.resume and is_completed(config.results_dir, architecture, strategy, fold):
                LOGGER.info("Ya está terminado, se saltea: %s", exp_id)
                continue
            run_one(config, inventories[fold], architecture, strategy)
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


def run_one(config: Config, inventory, architecture: str, strategy: str) -> None:
    spec = ARCHITECTURES[architecture]
    fold = inventory.spec.fold
    exp_id = experiment_id(architecture, strategy, fold)
    results_dir = config.results_dir
    directory = run_dir(results_dir, exp_id)
    stable_weights = weights_path(results_dir, architecture, strategy, fold)
    frozen = spec.freeze_indices(strategy)
    config_hash = config.config_hash(_freeze_plan_public(), fold_definitions())
    n_train = len(inventory.of("train"))
    iterations = estimated_iterations(n_train, config.epochs, config.batch)

    _configure_logging(results_dir / "logs" / f"{exp_id}.log")
    LOGGER.info("Empieza %s. Freeze: %s", exp_id, list(frozen) or "(ninguna capa)")

    if training_finished(results_dir, exp_id) and stable_weights.is_file():
        LOGGER.info("El entrenamiento de %s ya había terminado. Se rehacen solo las métricas.", exp_id)
        _evaluate_finished_run(
            config=config,
            inventory=inventory,
            architecture=architecture,
            strategy=strategy,
            stable_weights=stable_weights,
            config_hash=config_hash,
            iterations=iterations,
        )
        return

    last_checkpoint = directory / "weights" / "last.pt"
    resume_run = config.resume and last_checkpoint.is_file()
    if directory.exists() and not resume_run:
        archived = archive_incomplete_run(directory)
        LOGGER.info("Se archivó una corrida incompleta de %s en %s", exp_id, archived)

    dataset_yaml = write_fold_yaml(inventory, results_dir / "datasets" / f"{inventory.spec.dataset_dirname}.yaml")
    row = _base_row(config, inventory, architecture, strategy, config_hash, iterations)
    row["status"] = "running"
    row["started_at"] = now_iso()
    upsert_row(results_dir / "experiments.csv", EXPERIMENT_FIELDS, exp_id, row)
    write_marker(results_dir, exp_id, {"status": "running", "training_finished": False, "experiment_id": exp_id})

    try:
        set_seed(config.seed)
        from ultralytics import YOLO

        if resume_run:
            LOGGER.info("Se reanuda desde %s", last_checkpoint)
            model = YOLO(str(last_checkpoint))
        else:
            # El nombre .pt oficial hace que Ultralytics use los pesos preentrenados.
            # No se acepta un YAML: eso entrenaría desde cero.
            if not spec.weights.endswith(".pt"):
                raise ArchitectureError(f"{spec.weights} no es un peso preentrenado.")
            model = YOLO(spec.weights)

        verify_loaded_architecture(model, architecture)
        state = {"checked": False}

        def on_train_start(trainer) -> None:
            named = [(name, parameter.requires_grad) for name, parameter in trainer.model.named_parameters()]
            assert_freeze_state(named, frozen, len(spec.layers))
            state["checked"] = True
            LOGGER.info("Congelamiento verificado para %s / %s.", architecture, strategy)

        def on_train_epoch_start(trainer) -> None:
            # Respaldo por si el arranque no dispara on_train_start. Corre una sola vez.
            if state["checked"]:
                return
            on_train_start(trainer)

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        train_args = config.train_args(list(frozen))
        started = time.perf_counter()
        if resume_run:
            model.train(resume=True)
        else:
            model.train(data=str(dataset_yaml), project=str(results_dir / "runs"), name=exp_id, **train_args)
        wall_time = time.perf_counter() - started
        if not state["checked"]:
            raise ArchitectureError("Terminó el arranque sin pasar por la verificación de congelamiento.")

        trainer = model.trainer
        produced = Path(getattr(trainer, "best", directory / "weights" / "best.pt"))
        if not produced.is_file():
            produced = directory / "weights" / "best.pt"
        if not produced.is_file() or produced.stat().st_size <= 0:
            raise RuntimeError(f"No quedó best.pt después de entrenar {exp_id}.")

        stable_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, stable_weights)
        save_dir = Path(trainer.save_dir)
        _copy_plots(save_dir, results_dir / "plots" / exp_id)
        args_yaml = save_dir / "args.yaml"
        if args_yaml.is_file():
            shutil.copy2(args_yaml, results_dir / "logs" / f"{exp_id}_args.yaml")

        best_epoch = getattr(getattr(trainer, "stopper", None), "best_epoch", None)
        best_fitness = getattr(trainer, "best_fitness", None)
        metrics = read_best_epoch_metrics(Path(trainer.csv), None if best_epoch is None else int(best_epoch), _float_or_none(best_fitness))
        batch_used = int(getattr(trainer, "batch_size", config.batch))
        warning = _warnings(config, batch_used, str(trainer.args.optimizer))
        row.update(
            {
                "status": "trained",
                "batch_used": batch_used,
                "optimizer": str(trainer.args.optimizer),
                "lr0": float(trainer.args.lr0),
                "best_epoch": metrics["epoch"],
                "epochs_ran": metrics["epochs_ran"],
                "train_time_sec": metrics["train_time_sec"],
                "wall_time_sec": wall_time,
                "map50": metrics["map50"],
                "map50_95": metrics["map50_95"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fitness": metrics["fitness"],
                "best_weights": str(stable_weights),
                "run_dir": str(save_dir),
                "warning": warning,
                "ultralytics_version": _ultralytics_version(),
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
                "run_dir": str(save_dir),
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
        if isinstance(exc, ArchitectureError) and directory.exists() and not training_finished(results_dir, exp_id):
            archive_incomplete_run(directory)
        (results_dir / "logs" / f"{exp_id}.traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

    _evaluate_finished_run(
        config=config,
        inventory=inventory,
        architecture=architecture,
        strategy=strategy,
        stable_weights=stable_weights,
        config_hash=config_hash,
        iterations=iterations,
        existing_row=row,
    )


def _evaluate_finished_run(
    config: Config,
    inventory,
    architecture: str,
    strategy: str,
    stable_weights: Path,
    config_hash: str,
    iterations: int,
    existing_row: dict | None = None,
) -> None:
    """Valida por fuente y guarda predicciones del best.pt. No vuelve a entrenar."""
    exp_id = experiment_id(architecture, strategy, inventory.spec.fold)
    results_dir = config.results_dir
    row = existing_row or _row_from_marker(results_dir, exp_id)
    if row is None:
        row = _base_row(config, inventory, architecture, strategy, config_hash, iterations)
    row["best_weights"] = str(stable_weights)
    try:
        from ultralytics import YOLO

        model = YOLO(str(stable_weights))
        verify_loaded_architecture(model, architecture)
        metric_rows = [
            combined_metric_row(
                exp_id,
                architecture,
                strategy,
                inventory,
                {
                    "map50": float(row["map50"]),
                    "map50_95": float(row["map50_95"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "fitness": float(row["fitness"]),
                },
            )
        ]
        if len(inventory.spec.val_sources) == 1:
            source = inventory.spec.val_sources[0]
            source_row = positive_source_row(exp_id, architecture, strategy, inventory, source, metric_rows[0])
            source_row["note"] = "La fuente ocupa toda la validación del fold. Es la misma métrica combinada."
            metric_rows.append(source_row)
        else:
            for source in inventory.spec.per_source_eval:
                if source == "kingston2":
                    continue
                yaml_path = write_source_yaml(inventory, source, results_dir / "datasets")
                LOGGER.info("Validando %s solo con %s", exp_id, source)
                raw_metrics = model.val(
                    data=str(yaml_path),
                    split="val",
                    imgsz=config.imgsz,
                    batch=int(row.get("batch_used") or config.batch),
                    device=config.device,
                    workers=config.workers,
                    plots=False,
                    verbose=False,
                    save_json=False,
                    rect=False,
                    conf=0.001,
                    iou=config.prediction_iou,
                    project=str(results_dir / "evals"),
                    name=f"{exp_id}__{source}",
                    exist_ok=True,
                )
                metric_rows.append(
                    positive_source_row(
                        exp_id,
                        architecture,
                        strategy,
                        inventory,
                        source,
                        metrics_from_validator(raw_metrics),
                    )
                )

        prediction_dir = _save_val_predictions(model, inventory, config, exp_id)
        row["predictions_dir"] = str(prediction_dir)
        if "kingston2" in inventory.spec.val_sources:
            label_dir = prediction_dir / "labels"
            fp_images, fp_boxes = count_false_positives(label_dir, inventory.of("val", "kingston2"))
            metric_rows.append(
                kingston2_row(
                    exp_id,
                    architecture,
                    strategy,
                    inventory,
                    fp_images,
                    fp_boxes,
                    config.prediction_conf,
                )
            )

        replace_fold_metrics(results_dir / "fold_metrics.csv", exp_id, metric_rows)
        row["status"] = "completed"
        row["finished_at"] = now_iso()
        row["error"] = ""
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
        LOGGER.info("Terminado %s. best.pt en %s", exp_id, stable_weights)
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


def assert_cuda(config: Config) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("No está instalado torch. En Colab instalá YOLO/requirements.txt.") from exc
    if torch.cuda.is_available():
        LOGGER.info("CUDA disponible: %s", torch.cuda.get_device_name(0))
        return
    if config.allow_cpu:
        LOGGER.warning("CUDA no está disponible. Se sigue en CPU porque allow_cpu=True.")
        return
    raise RuntimeError(
        "CUDA no está disponible. El experimento está preparado para una GPU NVIDIA de Colab. "
        "No se inició ningún entrenamiento. Para forzar CPU, pasá allow_cpu=True."
    )


def set_seed(seed: int) -> None:
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_val_predictions(model, inventory, config: Config, exp_id: str) -> Path:
    image_dir = inventory.root / "images" / "val"
    project = config.results_dir / "predictions" / exp_id
    save_dir = None
    predicted = model.predict(
        source=str(image_dir),
        imgsz=config.imgsz,
        conf=config.prediction_conf,
        iou=config.prediction_iou,
        device=config.device,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(project),
        name="val",
        exist_ok=True,
        stream=True,
        verbose=False,
    )
    for result in predicted:
        save_dir = Path(result.save_dir)
    if save_dir is None:
        raise RuntimeError(f"No se guardaron predicciones de {exp_id}.")
    return save_dir


def _copy_plots(save_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in save_dir.glob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            shutil.copy2(path, destination / path.name)


def _base_row(config: Config, inventory, architecture: str, strategy: str, config_hash: str, iterations: int) -> dict:
    spec = ARCHITECTURES[architecture]
    frozen = spec.freeze_indices(strategy)
    exp_id = experiment_id(architecture, strategy, inventory.spec.fold)
    return {
        "experiment_id": exp_id,
        "architecture": architecture,
        "weights": spec.weights,
        "strategy": strategy,
        "fold": inventory.spec.fold,
        "train_sources": "+".join(inventory.spec.train_sources),
        "val_sources": "+".join(inventory.spec.val_sources),
        "frozen_indices": ",".join(str(index) for index in frozen),
        "frozen_modules": spec.describe_indices(frozen),
        "trainable_modules": spec.describe_trainable(strategy),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch_config": config.batch,
        "seed": config.seed,
        "patience": config.patience,
        "estimated_iterations": iterations,
        "config_hash": config_hash,
        "run_dir": str(run_dir(config.results_dir, exp_id)),
        "best_weights": str(weights_path(config.results_dir, architecture, strategy, inventory.spec.fold)),
    }


def _row_from_marker(results_dir: Path, exp_id: str) -> dict | None:
    row = read_experiment_row(results_dir, exp_id)
    if row is not None:
        return row
    marker = read_marker(results_dir, exp_id)
    if not marker:
        return None
    return {"experiment_id": exp_id, **marker.get("metrics", {})}


def _warnings(config: Config, batch_used: int, optimizer: str) -> str:
    notes = []
    if batch_used != int(config.batch):
        notes.append(
            f"Ultralytics bajó el batch de {config.batch} a {batch_used} por memoria. "
            "Este experimento no es comparable con uno que haya usado el batch configurado."
        )
    if config.optimizer == "auto" and optimizer not in {"AdamW", "auto"}:
        notes.append(f"optimizer=auto resolvió a {optimizer}, no a AdamW.")
    return " ".join(notes)


def _freeze_plan_public() -> dict:
    from solar_yolo.architectures import freeze_plan

    return freeze_plan()


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
    text = f"{type(exc).__name__}: {exc}"
    return text[:500]


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _ultralytics_version() -> str:
    import ultralytics

    return ultralytics.__version__

