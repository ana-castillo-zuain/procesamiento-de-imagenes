"""Parámetros compartidos por los 18 entrenamientos de Faster R-CNN.

Todo lo que debe permanecer constante entre arquitecturas, estrategias y folds
está en este módulo. Cambiar un valor acá lo cambia para todos los experimentos.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass
class Config:
    """Rutas y hiperparámetros de una corrida.

    ``dataset_dir`` es el mismo ``data/yolo`` que usa YOLO: las carpetas
    ``PV01_val``, ``solarpanels_val`` y ``geo_kingston_val``. Faster R-CNN lee
    las etiquetas YOLO de ahí y las pasa a píxeles; no se escribe otro dataset.
    """

    dataset_dir: Path
    results_dir: Path

    epochs: int = 50
    # La consigna pide min_size=max_size=256. Si no, torchvision lleva las
    # imágenes a 800 px y el entrenamiento se vuelve mucho más lento.
    imgsz: int = 256
    batch: int = 8
    seed: int = 42
    # Épocas sin mejorar el fitness antes de cortar. Igual que YOLO, el best.pt
    # es la época con mejor fitness sobre la validación del fold.
    patience: int = 10

    # SGD como en las referencias de torchvision (lr 0.02 para batch 16), con
    # escalado lineal: 0.01 para batch 8. Warmup lineal y después coseno.
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = 200

    # Las imágenes son aéreas: no hay un "arriba" natural, así que los dos
    # volteos son aumentos válidos. Es el único aumento.
    hflip: bool = True
    vflip: bool = True

    workers: int = 2
    device: str = "cuda"
    amp: bool = True

    # Umbral de score para medir mAP. Es el mismo que usa Ultralytics al
    # validar (0.001): con 0.05, el default de torchvision, se recorta la
    # cola de la curva precisión-recall y el mAP sale más bajo.
    eval_score_thresh: float = 0.001
    # Confianza de las predicciones guardadas, de los falsos positivos de
    # kingston2 y del "detector como clasificador". Igual que YOLO.
    prediction_conf: float = 0.25

    allow_cpu: bool = False
    resume: bool = True

    def resolved(self) -> Config:
        return replace(
            self,
            dataset_dir=Path(self.dataset_dir).expanduser().resolve(),
            results_dir=Path(self.results_dir).expanduser().resolve(),
        )

    def config_hash(self, freeze_plan: dict, folds: dict) -> str:
        payload = {key: value for key, value in self.to_json().items() if key not in ("dataset_dir", "results_dir", "device", "workers", "resume", "allow_cpu")}
        payload["freeze_plan"] = freeze_plan
        payload["folds"] = folds
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_json(self) -> dict:
        data = asdict(self)
        data["dataset_dir"] = str(self.dataset_dir)
        data["results_dir"] = str(self.results_dir)
        return data


def default_dataset_dir() -> Path:
    """``data/yolo`` del repositorio, relativo a este paquete."""
    return Path(__file__).resolve().parents[2] / "data" / "yolo"


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def estimated_iterations(n_train_images: int, epochs: int, batch: int) -> int:
    """Pasos del optimizador si no corta el early stopping."""
    return -(-int(n_train_images) // int(batch)) * int(epochs)


class ConfigError(ValueError):
    """La configuración haría que los experimentos dejen de ser comparables."""
