"""Parámetros compartidos por los 27 entrenamientos.

Todo lo que debe permanecer constante entre arquitecturas, estrategias y folds
está en este módulo. Cambiar un valor acá lo cambia para todos los experimentos.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


# Ultralytics elige el optimizador con optimizer=auto según la cantidad de
# iteraciones: AdamW si son 10000 o menos, MuSGD si son más. Con estos tamaños
# de fold, batch y epochs, los tres folds quedan en AdamW. El motor vuelve a
# comprobar esa cuenta antes de entrenar y aborta si un fold cruzaría el umbral,
# porque si no los experimentos dejarían de ser comparables.
AUTO_OPTIMIZER_ITERATION_LIMIT = 10_000
NOMINAL_BATCH_SIZE = 64

# Fórmula que usa Ultralytics dentro de la rama AdamW de optimizer=auto.
# lr0 = round(0.002 * 5 / (4 + nc), 6). Con nc=1 da 0.002.
# No fijamos lr0 nosotros: auto lo pisa y, además, pone warmup_bias_lr=0.


@dataclass
class Config:
    """Rutas y hiperparámetros de una corrida.

    ``dataset_dir`` apunta al directorio que contiene ``PV01_val``,
    ``solarpanels_val`` y ``geo_kingston_val``. No se asume una ruta de Drive.
    ``results_dir`` es la carpeta ``results/`` donde se escribe todo lo que
    tiene que sobrevivir al cierre de la sesión.
    """

    dataset_dir: Path
    results_dir: Path

    epochs: int = 100
    imgsz: int = 256
    batch: int = 16
    seed: int = 42
    patience: int = 15

    # workers=8 satura Colab. 2 es el valor común para una GPU de Colab.
    workers: int = 2
    device: str | int = 0
    cache: bool | str = False
    amp: bool = True
    deterministic: bool = True

    # optimizer=auto se resuelve igual en los tres folds con la configuración
    # inicial. Ver AUTO_OPTIMIZER_ITERATION_LIMIT.
    optimizer: str = "auto"

    # Confianza de las predicciones guardadas y de los falsos positivos de
    # kingston2. El mAP de Ultralytics sigue usando su umbral de validación
    # (0.001). No son la misma cuenta.
    prediction_conf: float = 0.25
    prediction_iou: float = 0.7

    allow_cpu: bool = False
    resume: bool = True

    # Claves que se pasan tal cual a model.train y que no dependen del fold.
    extra_train_args: dict = field(default_factory=dict)

    def resolved(self) -> Config:
        return Config(
            dataset_dir=Path(self.dataset_dir).expanduser().resolve(),
            results_dir=Path(self.results_dir).expanduser().resolve(),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            seed=self.seed,
            patience=self.patience,
            workers=self.workers,
            device=self.device,
            cache=self.cache,
            amp=self.amp,
            deterministic=self.deterministic,
            optimizer=self.optimizer,
            prediction_conf=self.prediction_conf,
            prediction_iou=self.prediction_iou,
            allow_cpu=self.allow_cpu,
            resume=self.resume,
            extra_train_args=dict(self.extra_train_args),
        )

    def train_args(self, freeze: list[int] | None) -> dict:
        """Argumentos comunes de model.train, más el freeze de esta estrategia."""
        args = {
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "seed": self.seed,
            "patience": self.patience,
            "workers": self.workers,
            "device": self.device,
            "cache": self.cache,
            "amp": self.amp,
            "deterministic": self.deterministic,
            "optimizer": self.optimizer,
            "pretrained": True,
            "plots": True,
            "save": True,
            "val": True,
            "exist_ok": True,
            "verbose": True,
            "fraction": 1.0,
            "close_mosaic": 10,
            "cos_lr": False,
            "single_cls": False,
            "rect": False,
            "multi_scale": 0.0,
            "compile": False,
            "profile": False,
            "iou": self.prediction_iou,
        }
        args.update(self.extra_train_args)
        if freeze:
            args["freeze"] = list(freeze)
        return args

    def config_hash(self, freeze_plan: dict, folds: dict) -> str:
        payload = {
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "seed": self.seed,
            "patience": self.patience,
            "workers": self.workers,
            "cache": self.cache,
            "amp": self.amp,
            "deterministic": self.deterministic,
            "optimizer": self.optimizer,
            "prediction_conf": self.prediction_conf,
            "prediction_iou": self.prediction_iou,
            "extra_train_args": self.extra_train_args,
            "freeze_plan": freeze_plan,
            "folds": folds,
        }
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


def estimated_iterations(n_train_images: int, epochs: int, batch: int, nbs: int = NOMINAL_BATCH_SIZE) -> int:
    """Misma cuenta que BaseTrainer: ceil(n / max(batch, nbs)) * epochs."""
    step = max(int(batch), int(nbs))
    return -(-int(n_train_images) // step) * int(epochs)


class ConfigError(ValueError):
    """La configuración haría que los experimentos dejen de ser comparables."""
