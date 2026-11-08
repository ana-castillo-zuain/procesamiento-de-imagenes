"""Dataset de torch que lee las etiquetas YOLO de ``data/yolo``.

YOLO guarda ``clase cx cy w h`` normalizados. torchvision espera
``x1 y1 x2 y2`` en píxeles y etiquetas que empiezan en 1, porque la 0 es el
fondo. Una imagen sin cajas es una negativa: se pasa con ``boxes`` vacío, que
torchvision acepta desde la 0.6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import pil_to_tensor


PANEL_LABEL = 1

# Cambios de color del aumento "fuerte". Son chicos: alcanzan para cubrir las
# diferencias de cámara e iluminación entre fuentes sin que un panel deje de
# parecerse a un panel.
COLOR_JITTER = {"brightness": 0.25, "contrast": 0.25, "saturation": 0.25, "hue": 0.02}


def read_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    """Cajas en píxeles ``(N, 4)``, recortadas al marco.

    El inventario de ``solar_yolo.data`` ya comprobó que todas las filas sean
    clase 0 con coordenadas en [0, 1]. Acá solo se convierten.
    """
    rows = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = (float(value) for value in line.split())
        rows.append(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(rows, dtype=np.float32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]


def rotate90(image: torch.Tensor, boxes: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rota ``k`` veces 90 grados en sentido antihorario, con sus cajas.

    Un giro lleva el punto ``(x, y)`` de una imagen de ancho ``W`` a
    ``(y, W - x)``; la imagen nueva tiene ancho ``H`` y alto ``W``.
    """
    for _ in range(k % 4):
        width = image.shape[-1]
        image = torch.rot90(image, 1, dims=(-2, -1))
        boxes = torch.stack([boxes[:, 1], width - boxes[:, 2], boxes[:, 3], width - boxes[:, 0]], dim=1)
    return image, boxes


def load_image(path: Path) -> torch.Tensor:
    """RGB en [0, 1]. Hay 32 PNG en modo paleta: ``convert`` los pasa a RGB.

    No se normaliza: ``GeneralizedRCNNTransform`` resta la media y divide por
    el desvío de ImageNet adentro del modelo.
    """
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float().div_(255.0)


class PanelDataset(Dataset):
    def __init__(
        self,
        records: list,
        train: bool = False,
        hflip: bool = False,
        vflip: bool = False,
        augment: str = "volteos",
    ) -> None:
        if augment not in ("volteos", "fuerte"):
            raise ValueError(f"Aumento desconocido: {augment}. Válidos: volteos, fuerte.")
        self.records = list(records)
        self.train = train
        self.hflip = hflip
        self.vflip = vflip
        self.strong = augment == "fuerte"
        self.color = ColorJitter(**COLOR_JITTER) if self.strong else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_image(record.path)
        height, width = image.shape[-2:]
        boxes = torch.from_numpy(read_boxes(record.label_path, width, height))
        if self.train:
            if self.hflip and torch.rand(1).item() < 0.5:
                image = image.flip(-1)
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            if self.vflip and torch.rand(1).item() < 0.5:
                image = image.flip(-2)
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
            if self.strong:
                # Con los dos volteos, la rotación completa las 8 orientaciones
                # posibles de una imagen cuadrada.
                image, boxes = rotate90(image, boxes, int(torch.randint(4, (1,)).item()))
                image = self.color(image).clamp_(0.0, 1.0)
        target = {
            "boxes": boxes,
            "labels": torch.full((len(boxes),), PANEL_LABEL, dtype=torch.int64),
            "image_id": index,
        }
        return image, target


def collate(batch):
    """Las imágenes de detección no se apilan: el modelo recibe listas."""
    return tuple(zip(*batch))
