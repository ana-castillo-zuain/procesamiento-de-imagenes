from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ARCHITECTURES = ("efficientnetb0", "mobilenetv2")
STRATEGIES = ("random", "head", "fine_tune")
SPLITS = ("solarpanels_val", "geo_kingston_val", "PV01_val")


@dataclass
class Config:
    dataset_dir: Path
    results_dir: Path
    image_size: int = 224
    batch_size: int = 32
    head_epochs: int = 10
    fine_tune_epochs: int = 15
    seed: int = 42
    workers: int = 2
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    dropout: float = 0.2
    patience: int = 5

    def resolved(self) -> "Config":
        return Config(Path(self.dataset_dir).expanduser().resolve(),
                      Path(self.results_dir).expanduser().resolve(),
                      self.image_size, self.batch_size, self.head_epochs,
                      self.fine_tune_epochs, self.seed, self.workers,
                      self.learning_rate, self.fine_tune_learning_rate,
                      self.dropout, self.patience)

    def to_json(self) -> dict:
        value = asdict(self)
        value["dataset_dir"] = str(self.dataset_dir)
        value["results_dir"] = str(self.results_dir)
        return value


def default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "yolo"


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results"
