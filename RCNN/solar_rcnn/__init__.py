"""Entrenamiento de detección de paneles solares con Faster R-CNN de torchvision.

Usa los mismos folds, verificaciones y CSV que ``YOLO/solar_yolo``. Este paquete
importa ``solar_yolo.data``, ``solar_yolo.evaluation`` y ``solar_yolo.storage``
en vez de copiarlos: así las dos familias no pueden partir distinto el dataset.
Importarlo no carga torch. El entrenamiento vive en ``engine``.
"""

import sys
from pathlib import Path


_YOLO_DIR = Path(__file__).resolve().parents[2] / "YOLO"
if not (_YOLO_DIR / "solar_yolo").is_dir():
    raise ImportError(
        f"No está YOLO/solar_yolo en {_YOLO_DIR}. solar_rcnn reutiliza sus folds y sus CSV; "
        "cloná el repositorio completo."
    )
if str(_YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(_YOLO_DIR))

__version__ = "1.0.0"
