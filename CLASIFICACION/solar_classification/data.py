from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def label_from_annotation(path: Path) -> int:
    """Clase 0 válida significa imagen positiva; un archivo vacío es negativa."""
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and int(float(fields[0])) == 0:
            return 1
    return 0


def samples_for_split(dataset_dir: Path, split: str, part: str) -> list[Sample]:
    image_dir = dataset_dir / split / "images" / part
    label_dir = dataset_dir / split / "labels" / part
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Falta el split {split}/{part}: {image_dir}")
    samples = []
    for image in sorted(image_dir.glob("*.png")):
        annotation = label_dir / f"{image.stem}.txt"
        if not annotation.is_file():
            raise FileNotFoundError(f"Falta etiqueta para {image.name}")
        samples.append(Sample(image, label_from_annotation(annotation)))
    if not samples:
        raise ValueError(f"No hay imágenes en {image_dir}")
    return samples


def validate_dataset(dataset_dir: Path, splits: tuple[str, ...]) -> dict:
    report = {}
    for split in splits:
        train = samples_for_split(dataset_dir, split, "train")
        val = samples_for_split(dataset_dir, split, "val")
        train_paths = {s.path.name for s in train}
        val_paths = {s.path.name for s in val}
        if train_paths & val_paths:
            raise ValueError(f"Hay imágenes repetidas entre train y val en {split}")
        report[split] = {
            "train": len(train), "val": len(val),
            "train_positive": sum(s.label for s in train),
            "val_positive": sum(s.label for s in val),
        }
    return report


def make_dataset(samples: list[Sample], image_size: int, batch_size: int,
                 training: bool, seed: int):
    import tensorflow as tf

    paths = tf.constant([str(s.path) for s in samples])
    labels = tf.constant([s.label for s in samples], dtype=tf.float32)

    def load(path, label):
        image = tf.io.decode_png(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        return tf.cast(image, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
