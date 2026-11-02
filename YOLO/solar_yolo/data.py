"""Folds por fuente a partir de los directorios YOLO que ya existen.

``data/yolo`` no está partido por el nombre de la fuente. Cada carpeta es el
conjunto de validación de un fold, y el train es el resto:

- ``PV01_val``: validación PV01. Train = solarpanels_hd + geo_unknown + kingston + kingston2.
- ``solarpanels_val``: validación solarpanels_hd. Train = PV01 + geo_unknown + kingston + kingston2.
- ``geo_kingston_val``: validación geo_unknown + kingston + kingston2. Train = PV01 + solarpanels_hd.

Esas carpetas ya tienen ``images/{train,val}``, ``labels/{train,val}`` y
``data.yaml``. Las imágenes están copiadas ahí por el preprocesamiento. Este
módulo solo las lee y escribe YAML nuevos dentro de ``results/``, con rutas
absolutas, para no depender del directorio de trabajo de Colab ni modificar
los datos originales.

La fuente se lee del nombre de archivo. No hay una carpeta por fuente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Conteos medidos sobre data/yolo. Si el disco no coincide, se aborta en vez de
# entrenar con otra partición. Vienen del inventario de este repositorio.
EXPECTED_SOURCES = {
    "PV01": {"images": 478, "positive": 477, "negative": 1, "boxes": 3556},
    "solarpanels_hd": {"images": 316, "positive": 284, "negative": 32, "boxes": 500},
    "geo_unknown": {"images": 44, "positive": 10, "negative": 34, "boxes": 13},
    "kingston": {"images": 34, "positive": 34, "negative": 0, "boxes": 66},
    "kingston2": {"images": 28, "positive": 0, "negative": 28, "boxes": 0},
}


class DatasetError(RuntimeError):
    """Los datos no alcanzan para construir los folds sin inventar asignaciones."""


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    dataset_dirname: str
    train_sources: tuple[str, ...]
    val_sources: tuple[str, ...]
    # Fuentes de validación para las que se guardan métricas aparte.
    # kingston2 está incluida, pero no se le calcula AP.
    per_source_eval: tuple[str, ...]
    caution: str = ""


FOLDS: dict[int, FoldSpec] = {
    1: FoldSpec(
        fold=1,
        dataset_dirname="PV01_val",
        train_sources=("solarpanels_hd", "geo_unknown", "kingston", "kingston2"),
        val_sources=("PV01",),
        per_source_eval=("PV01",),
    ),
    2: FoldSpec(
        fold=2,
        dataset_dirname="solarpanels_val",
        train_sources=("PV01", "geo_unknown", "kingston", "kingston2"),
        val_sources=("solarpanels_hd",),
        per_source_eval=("solarpanels_hd",),
    ),
    3: FoldSpec(
        fold=3,
        dataset_dirname="geo_kingston_val",
        train_sources=("PV01", "solarpanels_hd"),
        val_sources=("geo_unknown", "kingston", "kingston2"),
        per_source_eval=("geo_unknown", "kingston", "kingston2"),
        caution=(
            "El fold 3 tiene 79 cajas en total y kingston2 no tiene ninguna. "
            "Su mAP no es una estimación precisa."
        ),
    ),
}


@dataclass
class ImageRecord:
    path: Path
    label_path: Path
    source: str
    split: str
    boxes: int


@dataclass
class FoldInventory:
    spec: FoldSpec
    root: Path
    records: list[ImageRecord] = field(default_factory=list)

    def of(self, split: str, source: str | None = None) -> list[ImageRecord]:
        return [
            record
            for record in self.records
            if record.split == split and (source is None or record.source == source)
        ]


def source_of(filename: str) -> str:
    """Identifica la fuente por el nombre, no por la carpeta.

    ``kingston2`` se prueba antes que ``kingston_``. Si el nombre no entra en
    ninguna regla, se lanza un error: geo_unknown no es el cajón de resto.
    """
    name = Path(filename).name
    if name.startswith("PV01"):
        return "PV01"
    if name.startswith("solarpanels_hd"):
        return "solarpanels_hd"
    if name.startswith("kingston2"):
        return "kingston2"
    if name.startswith("kingston_"):
        return "kingston"
    if "_unknown_" in name:
        return "geo_unknown"
    raise DatasetError(
        f"No se pudo identificar la fuente de '{name}'. "
        "Las reglas conocidas son el prefijo PV01, solarpanels_hd, kingston2, kingston_ "
        "y el texto _unknown_ para geo_unknown. No se asignó esta imagen a ninguna fuente."
    )


def count_boxes(label_path: Path) -> int:
    """Cuenta cajas YOLO de clase 0. Un archivo vacío es una imagen negativa."""
    text = label_path.read_text(encoding="utf-8")
    boxes = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise DatasetError(f"{label_path} línea {line_number}: se esperaban 5 columnas y hay {len(parts)}.")
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise DatasetError(f"{label_path} línea {line_number}: no se pudo leer como número.") from exc
        class_id = int(values[0])
        if class_id != 0 or values[0] != 0.0:
            raise DatasetError(
                f"{label_path} línea {line_number}: la clase {parts[0]} no es 0. "
                "Las etiquetas de data/yolo ya deberían haber descartado las clases que no son paneles."
            )
        if any(value < -1e-4 or value > 1.0 + 1e-4 for value in values[1:]):
            raise DatasetError(f"{label_path} línea {line_number}: coordenada fuera de [0, 1]: {values[1:]}.")
        boxes += 1
    return boxes


def inventory_dataset(dataset_dir: Path) -> dict[int, FoldInventory]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise DatasetError(f"No existe el directorio de datos: {dataset_dir}")

    missing = [spec.dataset_dirname for spec in FOLDS.values() if not (dataset_dir / spec.dataset_dirname).is_dir()]
    if missing:
        raise DatasetError(
            "Faltan carpetas de folds en "
            f"{dataset_dir}: {', '.join(missing)}. "
            "Se esperaban PV01_val, solarpanels_val y geo_kingston_val. No se inventaron otras rutas."
        )

    inventories = {fold_id: _inventory_fold(dataset_dir, spec) for fold_id, spec in FOLDS.items()}
    _check_same_universe(inventories)
    _check_source_totals(inventories)
    return inventories


def _inventory_fold(dataset_dir: Path, spec: FoldSpec) -> FoldInventory:
    root = dataset_dir / spec.dataset_dirname
    data_yaml = root / "data.yaml"
    if data_yaml.is_file():
        yaml_text = data_yaml.read_text(encoding="utf-8")
        if "nc: 1" not in yaml_text or "panel" not in yaml_text:
            raise DatasetError(
                f"{data_yaml} no declara nc: 1 y la clase panel. No se reescribió el YAML original."
            )
    inventory = FoldInventory(spec=spec, root=root)
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise DatasetError(f"{spec.dataset_dirname} no tiene images/{split} y labels/{split}.")
        images = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            raise DatasetError(f"{image_dir} no tiene imágenes.")
        for image_path in sorted(images, key=lambda path: path.name):
            if image_path.stat().st_size <= 0:
                raise DatasetError(f"Imagen vacía: {image_path}")
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise DatasetError(f"Falta la etiqueta de {image_path.name} en {label_dir}.")
            inventory.records.append(
                ImageRecord(
                    path=image_path,
                    label_path=label_path,
                    source=source_of(image_path.name),
                    split=split,
                    boxes=count_boxes(label_path),
                )
            )
        extra_labels = sorted(path.name for path in label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        orphan_labels = [name for name in extra_labels if Path(name).stem not in image_stems]
        if orphan_labels:
            sample = ", ".join(orphan_labels[:5])
            raise DatasetError(f"Hay etiquetas sin imagen en {label_dir}: {sample}.")

    _check_fold_partition(inventory)
    return inventory


def _check_fold_partition(inventory: FoldInventory) -> None:
    spec = inventory.spec
    train = inventory.of("train")
    val = inventory.of("val")
    train_names = {record.path.name for record in train}
    val_names = {record.path.name for record in val}
    shared = sorted(train_names & val_names)
    if shared:
        raise DatasetError(
            f"Fold {spec.fold} ({spec.dataset_dirname}) comparte {len(shared)} imágenes "
            f"entre train y val. Ejemplo: {shared[0]}."
        )

    train_sources = {record.source for record in train}
    val_sources = {record.source for record in val}
    if train_sources != set(spec.train_sources) or val_sources != set(spec.val_sources):
        raise DatasetError(
            f"Fold {spec.fold} ({spec.dataset_dirname}) no tiene las fuentes esperadas. "
            f"Train encontrado: {sorted(train_sources)}; esperado: {list(spec.train_sources)}. "
            f"Val encontrado: {sorted(val_sources)}; esperado: {list(spec.val_sources)}."
        )
    if train_sources & val_sources:
        raise DatasetError(f"Fold {spec.fold} tiene fuentes en train y en val a la vez: {sorted(train_sources & val_sources)}.")

    for source in sorted(train_sources | val_sources):
        splits = {record.split for record in inventory.records if record.source == source}
        if len(splits) != 1:
            raise DatasetError(f"La fuente {source} quedó partida entre train y val en el fold {spec.fold}.")


def _check_same_universe(inventories: dict[int, FoldInventory]) -> None:
    universes = {fold_id: {record.path.name for record in inventory.records} for fold_id, inventory in inventories.items()}
    reference = universes[1]
    for fold_id, names in universes.items():
        if names != reference:
            missing = sorted(reference - names)[:5]
            extra = sorted(names - reference)[:5]
            raise DatasetError(
                f"El fold {fold_id} no contiene las mismas imágenes que el fold 1. "
                f"Faltan {missing} y sobran {extra}."
            )


def _check_source_totals(inventories: dict[int, FoldInventory]) -> None:
    # Las tres carpetas son copias del mismo universo. Se controla una.
    records = inventories[1].records
    for source, expected in EXPECTED_SOURCES.items():
        group = [record for record in records if record.source == source]
        observed = {
            "images": len(group),
            "positive": sum(record.boxes > 0 for record in group),
            "negative": sum(record.boxes == 0 for record in group),
            "boxes": sum(record.boxes for record in group),
        }
        if observed != expected:
            raise DatasetError(
                f"La fuente {source} no coincide con el inventario conocido. "
                f"Observado: {observed}. Esperado: {expected}. "
                "No se rearmó el fold con estos números distintos."
            )


def source_summary(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    summary = {}
    for source in EXPECTED_SOURCES:
        group = [record for record in records if record.source == source]
        if not group:
            continue
        summary[source] = {
            "images": len(group),
            "positive": sum(record.boxes > 0 for record in group),
            "negative": sum(record.boxes == 0 for record in group),
            "boxes": sum(record.boxes for record in group),
        }
    return summary


def write_fold_yaml(inventory: FoldInventory, destination: Path) -> Path:
    """YAML de entrenamiento que apunta a las carpetas existentes, sin copiarlas."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_dataset_yaml(
        destination,
        dataset_root=inventory.root,
        train="images/train",
        val="images/val",
    )
    return destination


def write_source_yaml(inventory: FoldInventory, source: str, destination_dir: Path) -> Path:
    """YAML solo para validar una fuente del fold. No se usa para entrenar.

    El train del YAML sigue siendo el train del fold, así un uso accidental no
    entrena con la fuente que se está evaluando. La validación es una lista de
    rutas a las imágenes que ya están en ``images/val``.
    """
    if source not in inventory.spec.val_sources:
        raise DatasetError(f"{source} no está en la validación del fold {inventory.spec.fold}.")
    records = inventory.of("val", source)
    if not records:
        raise DatasetError(f"El fold {inventory.spec.fold} no tiene imágenes de {source} en validación.")

    destination_dir.mkdir(parents=True, exist_ok=True)
    list_path = destination_dir / f"fold{inventory.spec.fold}_{source}_val.txt"
    # Separador nativo: Ultralytics busca ``{sep}images{sep}`` para hallar la etiqueta.
    list_path.write_text("\n".join(str(record.path.resolve()) for record in records) + "\n", encoding="utf-8")
    yaml_path = destination_dir / f"fold{inventory.spec.fold}_{source}_val.yaml"
    _write_dataset_yaml(
        yaml_path,
        dataset_root=inventory.root,
        train="images/train",
        val=str(list_path.resolve()),
    )
    return yaml_path


def _write_dataset_yaml(path: Path, dataset_root: Path, train: str, val: str) -> None:
    # Comillas simples: una ruta de Windows tiene ``:`` y barras invertidas, y YAML
    # las interpretaría mal si quedaran sueltas.
    content = (
        f"path: {_yaml_quote(str(dataset_root.resolve()))}\n"
        f"train: {_yaml_quote(train)}\n"
        f"val: {_yaml_quote(val)}\n"
        "nc: 1\n"
        "names:\n"
        "  0: panel\n"
    )
    path.write_text(content, encoding="utf-8")


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def describe_folds(inventories: dict[int, FoldInventory]) -> str:
    lines = [
        "Folds por fuente",
        "================",
        "",
        "Cada carpeta de data/yolo ya es un fold. El nombre de la carpeta es la validación,",
        "no una fuente suelta. Las imágenes de una fuente quedan todas del mismo lado.",
        "",
    ]
    for fold_id, inventory in inventories.items():
        spec = inventory.spec
        train = inventory.of("train")
        val = inventory.of("val")
        lines.append(f"Fold {fold_id} — {spec.dataset_dirname}")
        lines.append(f"  train: {', '.join(spec.train_sources)} ({len(train)} imágenes, {sum(r.boxes for r in train)} cajas)")
        lines.append(f"  val:   {', '.join(spec.val_sources)} ({len(val)} imágenes, {sum(r.boxes for r in val)} cajas)")
        overlap = {record.path.name for record in train} & {record.path.name for record in val}
        lines.append(f"  imágenes compartidas train/val: {len(overlap)}")
        for source, stats in source_summary(val).items():
            lines.append(
                f"    val {source}: {stats['images']} imágenes, "
                f"{stats['positive']} positivas, {stats['negative']} negativas, {stats['boxes']} cajas"
            )
        if spec.caution:
            lines.append(f"  cuidado: {spec.caution}")
        lines.append("")
    lines.append(
        "kingston2 solo entra en la validación del fold 3 y no tiene cajas. "
        "No se le calcula AP ni recall. Se cuentan los falsos positivos."
    )
    return "\n".join(lines)


def fold_definitions() -> dict:
    return {
        str(spec.fold): {
            "dataset_dirname": spec.dataset_dirname,
            "train_sources": list(spec.train_sources),
            "val_sources": list(spec.val_sources),
            "per_source_eval": list(spec.per_source_eval),
            "caution": spec.caution,
        }
        for spec in FOLDS.values()
    }
