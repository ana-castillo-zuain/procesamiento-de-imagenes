"""Modelo final: se entrena con las 900 imágenes y se evalúa en el test.

La configuración es la mejor de la validación cruzada: Faster R-CNN ResNet50
``partial`` con las anclas "paneles", el aumento "fuerte" y 20 épocas fijas.
Empató con la base en mAP50-95 macro (0.332 contra 0.331), pero tuvo mejor
mAP50 y fue mejor como clasificador. Y, como no usa early stopping, se puede
entrenar con todo el dataset sin apartar una validación para elegir la
época: el modelo final es el de la última época.

El test (``data/test``) tiene la estructura que entregó la cátedra:
``etiquetas.csv`` con ``archivo,etiqueta`` (con/sin) y ``lote_N/test_XXX.txt``
en formato YOLO, con la imagen ``lote_N/test_XXX.png`` al lado.

Importar este módulo no carga torch.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solar_rcnn.architectures import assert_freeze_state, build_model
from solar_rcnn.config import SHORT_SCHEDULE, Config, ConfigError
from solar_rcnn.engine import (
    _append_history,
    _eval_loader,
    _ground_truths,
    _lr_factor,
    _plot_history,
    _save_predictions,
    _save_weights,
    _seed_worker,
    _train_one_epoch,
    _write_pr_curve,
    predict,
    resolve_device,
    set_seed,
)
from solar_rcnn.metrics import classification_metrics, detection_metrics, image_scores
from solar_yolo.data import inventory_dataset
from solar_yolo.storage import _write_rows

LOGGER = logging.getLogger("solar_rcnn")

FINAL_ARCHITECTURE = "fasterrcnn_resnet50_fpn"
FINAL_STRATEGY = "partial"
N_DATASET_IMAGES = 900


class TestSetError(RuntimeError):
    """El test no está completo o no coincide con ``etiquetas.csv``."""


@dataclass(frozen=True)
class TestRecord:
    """Misma forma que ``ImageRecord`` de solar_yolo, para reusar el motor."""

    path: Path
    label_path: Path
    boxes: int
    source: str
    split: str = "test"


def final_config(dataset_dir: Path, results_dir: Path, **overrides) -> Config:
    """La configuración del modelo final. ``overrides`` es solo para pruebas."""
    values = dict(
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        batch=8,
        anchors="paneles",
        augment="fuerte",
        epochs=SHORT_SCHEDULE["epochs"],
        patience=SHORT_SCHEDULE["patience"],
    )
    values.update(overrides)
    return Config(**values).resolved()


def all_training_records(dataset_dir: Path) -> list:
    """Las 900 imágenes del dataset, una sola vez cada una.

    Cada fold es una partición completa del dataset (train + val), así que el
    fold 1 alcanza. Se comprueba que no falte ni se repita ninguna.
    """
    records = inventory_dataset(dataset_dir)[1].records
    names = {record.path.name for record in records}
    if len(records) != N_DATASET_IMAGES or len(names) != N_DATASET_IMAGES:
        raise ConfigError(
            f"Se esperaban {N_DATASET_IMAGES} imágenes únicas y hay {len(records)} registros "
            f"con {len(names)} nombres distintos."
        )
    return records


def test_records(test_dir: Path, require_images: bool = True) -> list[TestRecord]:
    """Lee ``etiquetas.csv`` y comprueba que cada fila tenga su etiqueta y su imagen.

    Corta si una etiqueta "con" no tiene cajas o una "sin" tiene alguna, y si
    faltan imágenes dice cuántas y dónde se esperan.
    """
    test_dir = Path(test_dir)
    csv_path = test_dir / "etiquetas.csv"
    if not csv_path.is_file():
        raise TestSetError(f"No está {csv_path}.")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    records, missing, inconsistent = [], [], []
    for row in rows:
        image = test_dir / row["archivo"]
        label = image.with_suffix(".txt")
        if not label.is_file():
            raise TestSetError(f"Falta la etiqueta {label}.")
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(line.split()[0] != "0" for line in lines):
            raise TestSetError(f"{label} tiene una clase distinta de 0 (panel).")
        if (row["etiqueta"] == "con") != bool(lines):
            inconsistent.append(row["archivo"])
        if not image.is_file():
            missing.append(row["archivo"])
        records.append(TestRecord(image, label, len(lines), source=image.parent.name))

    if inconsistent:
        raise TestSetError(f"{len(inconsistent)} filas de etiquetas.csv no coinciden con su .txt: {inconsistent[:5]}.")
    if missing and require_images:
        raise TestSetError(
            f"Faltan {len(missing)} de {len(rows)} imágenes del test, por ejemplo {missing[0]}. "
            f"Cada PNG va al lado de su .txt: {test_dir}/lote_N/test_XXX.png."
        )
    return records


def attach_test_images(test_dir: Path, images_dir: Path) -> int:
    """Copia los PNG del test al lado de su etiqueta y devuelve cuántos copió.

    Sirve para imágenes subidas aparte (por ejemplo a Drive), con o sin las
    carpetas ``lote_N``: cada imagen se ubica por su nombre (``test_XXX``).
    Las que ya están en su lugar no se tocan.
    """
    import shutil

    test_dir, images_dir = Path(test_dir), Path(images_dir)
    available = {}
    for image in images_dir.rglob("*.png"):
        if image.stem in available:
            raise TestSetError(f"Hay dos imágenes {image.stem} en {images_dir}.")
        available[image.stem] = image
    copied = 0
    for record in test_records(test_dir, require_images=False):
        if record.path.is_file() or record.path.stem not in available:
            continue
        shutil.copy2(available[record.path.stem], record.path)
        copied += 1
    return copied


def describe_test(records: list[TestRecord]) -> str:
    positives = sum(record.boxes > 0 for record in records)
    lines = [
        f"Test: {len(records)} imágenes, {positives} con paneles, {len(records) - positives} sin, "
        f"{sum(record.boxes for record in records)} cajas."
    ]
    for lote in sorted({record.source for record in records}):
        group = [record for record in records if record.source == lote]
        lines.append(
            f"  {lote}: {len(group)} imágenes, {sum(r.boxes > 0 for r in group)} con paneles, "
            f"{sum(r.boxes for r in group)} cajas"
        )
    return "\n".join(lines)


def train_final(config: Config, device: str | None = None) -> Path:
    """Entrena con las 900 imágenes y devuelve la ruta de ``final.pt``.

    Guarda ``last.pt`` al final de cada época: si Colab se corta, la próxima
    llamada sigue desde ahí. Si ``final.pt`` ya existe con la misma
    configuración, no entrena de nuevo.
    """
    import torch
    from torch.utils.data import DataLoader

    from solar_rcnn.architectures import freeze_plan
    from solar_rcnn.dataset import PanelDataset, collate
    from solar_yolo.data import fold_definitions

    device = device or resolve_device(config)
    out = config.results_dir / "final"
    weights_dir = out / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    final_weights, last_checkpoint = weights_dir / "final.pt", weights_dir / "last.pt"
    history_path = out / "results.csv"
    config_hash = config.config_hash(freeze_plan(), fold_definitions())

    stored = out / "config.json"
    if stored.is_file():
        previous = json.loads(stored.read_text(encoding="utf-8")).get("config_hash")
        if previous != config_hash:
            raise ConfigError(
                f"{out} tiene un modelo final de otra configuración ({previous}, actual {config_hash}). "
                "Usá otra carpeta de resultados."
            )
    stored.write_text(json.dumps({"config_hash": config_hash, "config": config.to_json()}, indent=2), encoding="utf-8")

    if final_weights.is_file():
        LOGGER.info("El modelo final ya estaba entrenado: %s", final_weights)
        return final_weights

    records = all_training_records(config.dataset_dir)
    set_seed(config.seed)
    model = build_model(FINAL_ARCHITECTURE, FINAL_STRATEGY, config.imgsz, config.anchors)
    counts = assert_freeze_state(model, FINAL_ARCHITECTURE, FINAL_STRATEGY)
    LOGGER.info(
        "Modelo final %s %s, anclas %s, aumento %s, %d épocas con %d imágenes. %d parámetros entrenables.",
        FINAL_ARCHITECTURE, FINAL_STRATEGY, config.anchors, config.augment, config.epochs, len(records), counts["trainable"],
    )
    model.to(device)

    generator = torch.Generator()
    loader = DataLoader(
        PanelDataset(records, train=True, hflip=config.hflip, vflip=config.vflip, augment=config.augment),
        batch_size=config.batch,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate,
        generator=generator,
        worker_init_fn=_seed_worker,
        pin_memory=device == "cuda",
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    total_iters = len(loader) * config.epochs
    warmup = min(config.warmup_iters, total_iters)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _lr_factor(step, warmup, total_iters))
    use_amp = config.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        LOGGER.info("Se reanuda el modelo final desde la época %d.", start_epoch)

    for epoch in range(start_epoch + 1, config.epochs + 1):
        # Semilla por época: una corrida reanudada baraja igual que una entera.
        torch.manual_seed(config.seed + epoch)
        generator.manual_seed(config.seed + epoch)
        epoch_start = time.perf_counter()
        epoch_lr = optimizer.param_groups[0]["lr"]
        losses = _train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp)
        epoch_time = time.perf_counter() - epoch_start
        _append_history(history_path, {"epoch": epoch, "lr": epoch_lr, **losses, "epoch_time_sec": epoch_time})
        LOGGER.info("final época %d/%d: loss %.4f  (%.0f s)", epoch, config.epochs, losses["loss"], epoch_time)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
            },
            last_checkpoint,
        )

    # fold 0: entrenado con los tres folds juntos.
    _save_weights(final_weights, model, FINAL_ARCHITECTURE, FINAL_STRATEGY, 0, config.epochs, config)
    _plot_history(history_path, out / "plots" / "entrenamiento.png", "modelo final (900 imágenes)")
    LOGGER.info("Modelo final guardado en %s", final_weights)
    return final_weights


def load_final(config: Config, weights: Path, device: str):
    import torch

    model = build_model(FINAL_ARCHITECTURE, FINAL_STRATEGY, config.imgsz, config.anchors)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if (checkpoint.get("architecture"), checkpoint.get("strategy")) != (FINAL_ARCHITECTURE, FINAL_STRATEGY):
        raise ConfigError(f"{weights} no es un {FINAL_ARCHITECTURE} {FINAL_STRATEGY}.")
    model.load_state_dict(checkpoint["model"])
    model.roi_heads.score_thresh = config.eval_score_thresh
    return model.to(device)


def evaluate_test(config: Config, test_dir: Path, weights: Path | None = None, device: str | None = None) -> dict:
    """Predice el test una vez y escribe todas las métricas en ``final/test``.

    - Detección: mAP50, mAP50-95, precisión y recall en el punto de F1 máximo,
      en total y por lote.
    - Imágenes sin paneles: cuántas tienen alguna detección con confianza
      >= ``prediction_conf`` (0.25) y cuántas cajas.
    - Detector como clasificador con el umbral de fábrica (0.25), fijado de
      antemano, y la referencia de decir siempre que sí. No se calibra un
      umbral sobre el test.
    """
    device = device or resolve_device(config)
    records = test_records(test_dir)
    weights = weights or config.results_dir / "final" / "weights" / "final.pt"
    if not Path(weights).is_file():
        raise TestSetError(f"No está el modelo final en {weights}. Corré train_final antes.")

    model = load_final(config, weights, device)
    use_amp = config.amp and device == "cuda"
    predictions = predict(model, _eval_loader(records, config, device), device, use_amp)
    ground_truths = _ground_truths(records)

    out = config.results_dir / "final" / "test"
    out.mkdir(parents=True, exist_ok=True)
    detection_rows = []
    groups = [("todo", list(range(len(records))))] + [
        (lote, [i for i, record in enumerate(records) if record.source == lote])
        for lote in sorted({record.source for record in records})
    ]
    for name, index in groups:
        gts = [ground_truths[i] for i in index]
        negatives = [i for i in index if records[i].boxes == 0]
        fp_boxes = [int(np.sum(predictions[i][1] >= config.prediction_conf)) for i in negatives]
        row = {
            "scope": name,
            "n_images": len(index),
            "n_positive_images": len(index) - len(negatives),
            "n_negative_images": len(negatives),
            "n_boxes": sum(len(gt) for gt in gts),
            "fp_images_in_negatives": sum(count > 0 for count in fp_boxes),
            "fp_boxes_in_negatives": sum(fp_boxes),
        }
        if row["n_boxes"]:
            metrics = detection_metrics([predictions[i] for i in index], gts)
            row.update({key: metrics[key] for key in ("map50", "map50_95", "precision", "recall", "conf_at_best_f1")})
            if name == "todo":
                _write_pr_curve(out / "plots", metrics, "test")
        detection_rows.append(row)
    fields = list(dict.fromkeys(key for row in detection_rows for key in row))
    _write_rows(out / "detection_metrics.csv", fields, detection_rows)

    scores = image_scores(predictions)
    truth = np.array([record.boxes > 0 for record in records])
    classification_rows = [
        {"threshold_kind": "fabrica", **classification_metrics(scores, truth, config.prediction_conf)},
        {"threshold_kind": "trivial_todo_positivo", **classification_metrics(np.ones_like(scores), truth, 0.5)},
    ]
    _write_rows(out / "classification_metrics.csv", list(classification_rows[0]), classification_rows)

    # _save_predictions borra las etiquetas viejas de la carpeta antes de escribir.
    prediction_dir = out / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    _save_predictions(prediction_dir, records, predictions, config)

    summary = {"weights": str(weights), "detection": detection_rows, "classification": classification_rows}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    LOGGER.info("Métricas del test en %s", out)
    return summary
