from __future__ import annotations

import csv
import json
from pathlib import Path


def save_config(config, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text(
        json.dumps(config.to_json(), indent=2), encoding="utf-8")


def append_metrics(results_dir: Path, row: dict):
    path = results_dir / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(row)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
