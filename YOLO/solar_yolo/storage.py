"""CSV, marcadores y rutas de resultados.

Un experimento terminado queda identificado por su ``experiment_id`` y por la
existencia de ``best.pt``. Volver a lanzar la corrida no pisa esa fila ni ese
peso. Una corrida interrumpida no tiene el marcador ``completed`` y se reanuda.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_FIELDS = [
    "experiment_id",
    "status",
    "architecture",
    "weights",
    "strategy",
    "fold",
    "train_sources",
    "val_sources",
    "frozen_indices",
    "frozen_modules",
    "trainable_modules",
    "epochs",
    "imgsz",
    "batch_config",
    "batch_used",
    "seed",
    "patience",
    "optimizer",
    "lr0",
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
    "ultralytics_version",
]

FOLD_METRIC_FIELDS = [
    "experiment_id",
    "architecture",
    "strategy",
    "fold",
    "scope",
    "source",
    "n_images",
    "n_positive_images",
    "n_negative_images",
    "n_boxes",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "fitness",
    "fp_images",
    "fp_boxes",
    "conf_threshold",
    "note",
]

SUMMARY_FIELDS = [
    "architecture",
    "strategy",
    "fold1_map50_95",
    "fold2_map50_95",
    "fold3_map50_95",
    "macro_map50_95",
    "fold1_map50",
    "fold2_map50",
    "fold3_map50",
    "fold1_precision",
    "fold2_precision",
    "fold3_precision",
    "fold1_recall",
    "fold2_recall",
    "fold3_recall",
    "fold1_fitness",
    "fold2_fitness",
    "fold3_fitness",
    "fold1_best_epoch",
    "fold2_best_epoch",
    "fold3_best_epoch",
    "fold1_train_time_sec",
    "fold2_train_time_sec",
    "fold3_train_time_sec",
    "fold1_weights",
    "fold2_weights",
    "fold3_weights",
    "n_folds_completed",
    "score_status",
    "note",
]


def experiment_id(architecture: str, strategy: str, fold: int) -> str:
    return f"{architecture}__{strategy}__fold{fold}"


def ensure_result_dirs(results_dir: Path) -> None:
    for relative in (
        "logs",
        "weights",
        "predictions",
        "plots",
        "runs",
        "datasets",
        "evals",
    ):
        (results_dir / relative).mkdir(parents=True, exist_ok=True)


def weights_path(results_dir: Path, architecture: str, strategy: str, fold: int) -> Path:
    return results_dir / "weights" / architecture / strategy / f"fold{fold}" / "best.pt"


def run_dir(results_dir: Path, exp_id: str) -> Path:
    return results_dir / "runs" / exp_id


def marker_path(results_dir: Path, exp_id: str) -> Path:
    return results_dir / "logs" / f"{exp_id}.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_experiment_row(results_dir: Path, exp_id: str) -> dict | None:
    for row in _read_rows(results_dir / "experiments.csv"):
        if row.get("experiment_id") == exp_id:
            return row
    return None


def read_marker(results_dir: Path, exp_id: str) -> dict | None:
    path = marker_path(results_dir, exp_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_marker(results_dir: Path, exp_id: str, payload: dict) -> None:
    path = marker_path(results_dir, exp_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def is_completed(results_dir: Path, architecture: str, strategy: str, fold: int) -> bool:
    exp_id = experiment_id(architecture, strategy, fold)
    marker = read_marker(results_dir, exp_id)
    if not marker or marker.get("status") != "completed":
        return False
    weights = weights_path(results_dir, architecture, strategy, fold)
    recorded = Path(str(marker.get("best_weights", "")))
    return weights.is_file() and weights.stat().st_size > 0 and recorded.is_file() and recorded.stat().st_size > 0


def training_finished(results_dir: Path, exp_id: str) -> bool:
    marker = read_marker(results_dir, exp_id)
    if not marker or not marker.get("training_finished"):
        return False
    best = run_dir(results_dir, exp_id) / "weights" / "best.pt"
    return best.is_file() and best.stat().st_size > 0


def upsert_row(path: Path, fields: list[str], key: str, row: dict) -> None:
    rows = _read_rows(path)
    rows = [existing for existing in rows if existing.get("experiment_id") != key]
    rows.append({field: _cell(row.get(field, "")) for field in fields})
    rows.sort(key=lambda item: item.get("experiment_id", ""))
    _write_rows(path, fields, rows)


def replace_fold_metrics(path: Path, exp_id: str, metric_rows: list[dict]) -> None:
    rows = [row for row in _read_rows(path) if row.get("experiment_id") != exp_id]
    for metric in metric_rows:
        rows.append({field: _cell(metric.get(field, "")) for field in FOLD_METRIC_FIELDS})
    rows.sort(key=lambda item: (item.get("experiment_id", ""), item.get("scope", ""), item.get("source", "")))
    _write_rows(path, FOLD_METRIC_FIELDS, rows)


def rebuild_summary(results_dir: Path) -> Path:
    """Una fila por arquitectura y estrategia. El macro solo existe con los tres folds."""
    experiments = _read_rows(results_dir / "experiments.csv")
    completed = [row for row in experiments if row.get("status") == "completed"]
    grouped: dict[tuple[str, str], dict[int, dict]] = {}
    for row in completed:
        try:
            fold = int(row["fold"])
        except (KeyError, ValueError):
            continue
        grouped.setdefault((row["architecture"], row["strategy"]), {})[fold] = row

    summary_rows = []
    for architecture, strategy in sorted(grouped):
        folds = grouped[(architecture, strategy)]
        summary = {
            "architecture": architecture,
            "strategy": strategy,
            "n_folds_completed": len(folds),
            "note": "fold3 tiene pocas cajas; no interpretar su mAP como una estimación precisa.",
        }
        for fold in (1, 2, 3):
            row = folds.get(fold, {})
            summary[f"fold{fold}_map50_95"] = row.get("map50_95", "")
            summary[f"fold{fold}_map50"] = row.get("map50", "")
            summary[f"fold{fold}_precision"] = row.get("precision", "")
            summary[f"fold{fold}_recall"] = row.get("recall", "")
            summary[f"fold{fold}_fitness"] = row.get("fitness", "")
            summary[f"fold{fold}_best_epoch"] = row.get("best_epoch", "")
            summary[f"fold{fold}_train_time_sec"] = row.get("train_time_sec", "")
            summary[f"fold{fold}_weights"] = row.get("best_weights", "")
        if all(fold in folds and folds[fold].get("map50_95") not in ("", None) for fold in (1, 2, 3)):
            values = [float(folds[fold]["map50_95"]) for fold in (1, 2, 3)]
            # Cada fold pesa 1/3, aunque tenga distinta cantidad de imágenes o cajas.
            summary["macro_map50_95"] = f"{sum(values) / 3:.8f}"
            summary["score_status"] = "complete"
        else:
            summary["macro_map50_95"] = ""
            summary["score_status"] = "incomplete"
        summary_rows.append({field: _cell(summary.get(field, "")) for field in SUMMARY_FIELDS})

    path = results_dir / "summary.csv"
    _write_rows(path, SUMMARY_FIELDS, summary_rows)
    return path


def write_config_snapshot(results_dir: Path, payload: dict) -> None:
    path = results_dir / "config.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != text:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        archive = results_dir / "logs" / f"config_{stamp}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    _atomic_text(path, text)


def archive_incomplete_run(directory: Path) -> Path | None:
    """Corre un directorio de corrida fallida a un lado para no mezclar CSV."""
    if not directory.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    archive = directory.parent / f"{directory.name}__archived_{stamp}"
    directory.rename(archive)
    return archive


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)
