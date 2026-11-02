"""Métricas de un fold y de cada fuente de validación.

El mAP del fold sale de la validación que hizo el entrenamiento sobre su propio
conjunto. No se juntan las predicciones de los tres folds para fabricar un
mAP global. El promedio macro se calcula después, en el CSV de resumen, con
un tercio para cada fold.

kingston2 no tiene objetos. No se le pide AP ni recall a Ultralytics. Cada
detección sobre esas imágenes es un falso positivo.
"""

from __future__ import annotations

import csv
from pathlib import Path

from solar_yolo.data import FoldInventory, ImageRecord


FITNESS_MAP50_WEIGHT = 0.1
FITNESS_MAP5095_WEIGHT = 0.9


def ultralytics_fitness(map50: float, map50_95: float) -> float:
    """Fitness de detección de Ultralytics: 0.1 * mAP50 + 0.9 * mAP50-95."""
    return FITNESS_MAP50_WEIGHT * map50 + FITNESS_MAP5095_WEIGHT * map50_95


def read_best_epoch_metrics(results_csv: Path, best_epoch: int | None, best_fitness: float | None) -> dict:
    """Lee la fila de la mejor época en ``results.csv``.

    La validación final de Ultralytics a veces agrega otra fila. Si hay
    empate de fitness, se queda la época más temprana, que es la del ``best.pt``.
    """
    if not results_csv.is_file():
        raise FileNotFoundError(f"No está el CSV de entrenamiento: {results_csv}")
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{results_csv} no tiene épocas.")

    parsed = []
    for row in rows:
        map50 = _first_float(row, "metrics/mAP50(B)", "metrics/mAP50")
        map50_95 = _first_float(row, "metrics/mAP50-95(B)", "metrics/mAP50-95")
        precision = _first_float(row, "metrics/precision(B)", "metrics/precision")
        recall = _first_float(row, "metrics/recall(B)", "metrics/recall")
        parsed.append(
            {
                "epoch": int(float(row["epoch"])),
                "time": float(row["time"]) if row.get("time") else None,
                "map50": map50,
                "map50_95": map50_95,
                "precision": precision,
                "recall": recall,
                "fitness": ultralytics_fitness(map50, map50_95),
            }
        )

    chosen = None
    if best_epoch is not None:
        matches = [row for row in parsed if row["epoch"] == int(best_epoch)]
        if matches:
            chosen = matches[0]
    if chosen is None and best_fitness is not None:
        chosen = min(parsed, key=lambda row: (abs(row["fitness"] - best_fitness), row["epoch"]))
    if chosen is None:
        chosen = max(parsed, key=lambda row: (row["fitness"], -row["epoch"]))

    chosen["epochs_ran"] = max(row["epoch"] for row in parsed)
    last = max(parsed, key=lambda row: row["epoch"])
    chosen["train_time_sec"] = last["time"]
    return chosen


def metrics_from_validator(metrics) -> dict:
    """Lee mAP, precisión, recall y fitness del objeto que devuelve ``model.val``."""
    box = getattr(metrics, "box", None)
    if box is None and isinstance(metrics, dict):
        return {
            "map50": float(metrics.get("metrics/mAP50(B)", metrics.get("metrics/mAP50"))),
            "map50_95": float(metrics.get("metrics/mAP50-95(B)", metrics.get("metrics/mAP50-95"))),
            "precision": float(metrics.get("metrics/precision(B)", metrics.get("metrics/precision"))),
            "recall": float(metrics.get("metrics/recall(B)", metrics.get("metrics/recall"))),
            "fitness": float(metrics["fitness"]) if "fitness" in metrics else ultralytics_fitness(
                float(metrics.get("metrics/mAP50(B)")),
                float(metrics.get("metrics/mAP50-95(B)")),
            ),
        }
    if box is None:
        raise RuntimeError("La validación no devolvió métricas de cajas.")
    map50 = float(box.map50)
    map50_95 = float(box.map)
    precision = float(box.mp)
    recall = float(box.mr)
    fitness = getattr(metrics, "fitness", None)
    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "fitness": float(fitness) if fitness is not None else ultralytics_fitness(map50, map50_95),
    }


def count_false_positives(prediction_label_dir: Path, images: list[ImageRecord]) -> tuple[int, int]:
    """Cuenta detecciones guardadas. Una imagen sin txt no tuvo cajas.

    Ultralytics no escribe un txt vacío cuando no hay detecciones. Sobre
    kingston2, que no tiene etiquetas positivas, toda detección es un falso
    positivo.
    """
    images_with_boxes = 0
    boxes = 0
    for image in images:
        label = prediction_label_dir / f"{image.path.stem}.txt"
        if not label.is_file():
            continue
        n_boxes = sum(1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip())
        if n_boxes:
            images_with_boxes += 1
            boxes += n_boxes
    return images_with_boxes, boxes


def source_counts(records: list[ImageRecord]) -> dict[str, int]:
    return {
        "n_images": len(records),
        "n_positive_images": sum(record.boxes > 0 for record in records),
        "n_negative_images": sum(record.boxes == 0 for record in records),
        "n_boxes": sum(record.boxes for record in records),
    }


def combined_metric_row(exp_id: str, architecture: str, strategy: str, inventory: FoldInventory, metrics: dict) -> dict:
    val_records = inventory.of("val")
    row = {
        "experiment_id": exp_id,
        "architecture": architecture,
        "strategy": strategy,
        "fold": inventory.spec.fold,
        "scope": "combined",
        "source": "+".join(inventory.spec.val_sources),
        **source_counts(val_records),
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "fitness": metrics["fitness"],
        "fp_images": "",
        "fp_boxes": "",
        "conf_threshold": "",
        "note": inventory.spec.caution,
    }
    return row


def positive_source_row(
    exp_id: str,
    architecture: str,
    strategy: str,
    inventory: FoldInventory,
    source: str,
    metrics: dict,
) -> dict:
    records = inventory.of("val", source)
    return {
        "experiment_id": exp_id,
        "architecture": architecture,
        "strategy": strategy,
        "fold": inventory.spec.fold,
        "scope": "source",
        "source": source,
        **source_counts(records),
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "fitness": metrics["fitness"],
        "fp_images": "",
        "fp_boxes": "",
        "conf_threshold": "",
        "note": "",
    }


def kingston2_row(
    exp_id: str,
    architecture: str,
    strategy: str,
    inventory: FoldInventory,
    fp_images: int,
    fp_boxes: int,
    conf_threshold: float,
) -> dict:
    records = inventory.of("val", "kingston2")
    if any(record.boxes > 0 for record in records):
        raise RuntimeError("kingston2 tiene cajas. No se puede tratar como un conjunto solo negativo.")
    return {
        "experiment_id": exp_id,
        "architecture": architecture,
        "strategy": strategy,
        "fold": inventory.spec.fold,
        "scope": "source",
        "source": "kingston2",
        **source_counts(records),
        "map50": "",
        "map50_95": "",
        "precision": "",
        "recall": "",
        "fitness": "",
        "fp_images": fp_images,
        "fp_boxes": fp_boxes,
        "conf_threshold": conf_threshold,
        "note": "Sin objetos positivos. No se calculó AP ni recall. fp_* son detecciones a esta confianza.",
    }


def _first_float(row: dict, *names: str) -> float:
    for name in names:
        if name in row and row[name] not in ("", None):
            return float(row[name])
    raise KeyError(f"No están las columnas {names} en results.csv. Columnas: {list(row)}")
