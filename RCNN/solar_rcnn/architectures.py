"""Las dos Faster R-CNN de torchvision y qué congela cada estrategia.

torchvision congela con ``trainable_backbone_layers``: cuenta bloques del
extractor desde el final, y 0 congela todo el extractor. El mismo entero no
corta en el mismo lugar de las dos redes (ResNet50 tiene 5 bloques y
MobileNetV3 tiene 6 estadios), así que cada estrategia guarda el valor de cada
arquitectura y la lista de unidades de ``backbone.body`` que quedan congeladas.

Las tres estrategias son las mismas que en YOLO:

- ``full``: no se congela nada del extractor.
- ``backbone``: se congela todo el extractor. Se entrenan el FPN, la RPN y la
  cabeza ROI.
- ``partial``: se congelan el stem y el estadio de stride 4 (P2). Se entrena
  desde la bajada a stride 8 (P3) en adelante.

En las dos redes, las capas de batch norm del extractor son
``FrozenBatchNorm2d``: torchvision las congela siempre que parte de pesos
preentrenados, también con ``full``. Ni sus estadísticas ni sus parámetros
afines se entrenan.

Este módulo no importa torch al cargarse. ``build_model`` lo importa adentro.
"""

from __future__ import annotations

from dataclasses import dataclass

from solar_rcnn.config import ANCHOR_PRESETS


# torchvision reserva la clase 0 para el fondo. Panel es la 1.
NUM_CLASSES = 2

STRATEGIES = ("full", "backbone", "partial")


class ArchitectureError(RuntimeError):
    """El modelo cargado no coincide con la tabla de congelamiento."""


@dataclass(frozen=True)
class Unit:
    """Un bloque de ``backbone.body``: nombre del submódulo y qué es."""

    name: str
    description: str


@dataclass(frozen=True)
class RCNNSpec:
    name: str
    builder: str
    weights_enum: str
    max_trainable: int
    units: tuple[Unit, ...]
    # Posición en ``units`` donde empieza cada estadio que torchvision cuenta
    # para ``trainable_backbone_layers``, del primero al último.
    stage_starts: tuple[int, ...]
    trainable_layers: dict
    frozen_units: dict
    notes: tuple[str, ...] = ()

    def frozen_names(self, strategy: str) -> tuple[str, ...]:
        return tuple(unit.name for unit in self.units[: self.frozen_units[strategy]])

    def describe_frozen(self, strategy: str) -> str:
        frozen = self.units[: self.frozen_units[strategy]]
        return "; ".join(f"{unit.name} {unit.description}" for unit in frozen) or "(ninguno)"

    def describe_trainable(self, strategy: str) -> str:
        trainable = self.units[self.frozen_units[strategy] :]
        backbone = "; ".join(f"{unit.name} {unit.description}" for unit in trainable)
        return (backbone + "; " if backbone else "") + "FPN, RPN y cabeza ROI"


_RESNET_UNITS = (
    Unit("conv1", "stem conv 7x7 stride 2 (P1/2) y maxpool (P2/4)"),
    Unit("layer1", "3 Bottleneck, stride 4 (P2/4)"),
    Unit("layer2", "4 Bottleneck, stride 8 (P3/8)"),
    Unit("layer3", "6 Bottleneck, stride 16 (P4/16)"),
    Unit("layer4", "3 Bottleneck, stride 32 (P5/32)"),
)

# features[0..16] de MobileNetV3-Large. Los strides están medidos con una
# entrada de 256 px.
_MOBILENET_UNITS = (
    Unit("0", "stem conv stride 2 (P1/2)"),
    Unit("1", "bneck stride 2"),
    Unit("2", "bneck, baja a stride 4 (P2/4)"),
    Unit("3", "bneck stride 4"),
    Unit("4", "bneck, baja a stride 8 (P3/8)"),
    Unit("5", "bneck stride 8"),
    Unit("6", "bneck stride 8"),
    Unit("7", "bneck, baja a stride 16 (P4/16)"),
    Unit("8", "bneck stride 16"),
    Unit("9", "bneck stride 16"),
    Unit("10", "bneck stride 16"),
    Unit("11", "bneck stride 16"),
    Unit("12", "bneck stride 16"),
    Unit("13", "bneck, baja a stride 32 (P5/32), primera salida al FPN"),
    Unit("14", "bneck stride 32"),
    Unit("15", "bneck stride 32"),
    Unit("16", "conv 1x1 a 960 canales, stride 32, segunda salida al FPN"),
)


ARCHITECTURES: dict[str, RCNNSpec] = {
    "fasterrcnn_mobilenet_v3_large_fpn": RCNNSpec(
        name="fasterrcnn_mobilenet_v3_large_fpn",
        builder="fasterrcnn_mobilenet_v3_large_fpn",
        weights_enum="FasterRCNN_MobileNet_V3_Large_FPN_Weights",
        max_trainable=6,
        units=_MOBILENET_UNITS,
        # Igual que torchvision: el primer bloque, los bloques con stride y el último.
        stage_starts=(0, 2, 4, 7, 13, 16),
        trainable_layers={"full": 6, "backbone": 0, "partial": 4},
        frozen_units={"full": 0, "backbone": 17, "partial": 4},
        notes=(
            "El FPN de esta red toma solo las salidas de features[13] y features[16], las dos de stride 32, "
            "más un max pool (stride 64). Con 256 px eso es una grilla de 8x8 y otra de 4x4.",
            "Las anclas van de 32 a 512 px. El 15 % de las cajas tiene un lado de menos de 16 px.",
            "trainable_backbone_layers=3, el default de torchvision, congelaría hasta features[6] (stride 8). "
            "Para partial se usa 4, que corta en el mismo lugar que YOLO.",
        ),
    ),
    "fasterrcnn_resnet50_fpn": RCNNSpec(
        name="fasterrcnn_resnet50_fpn",
        builder="fasterrcnn_resnet50_fpn",
        weights_enum="FasterRCNN_ResNet50_FPN_Weights",
        max_trainable=5,
        units=_RESNET_UNITS,
        stage_starts=(0, 1, 2, 3, 4),
        trainable_layers={"full": 5, "backbone": 0, "partial": 3},
        frozen_units={"full": 0, "backbone": 5, "partial": 2},
        notes=(
            "El FPN toma layer1 a layer4 (strides 4, 8, 16 y 32) más un max pool (stride 64).",
            "trainable_backbone_layers=3 es el default de torchvision y coincide con partial.",
        ),
    ),
}


def _torchvision_frozen_units(spec: RCNNSpec, trainable: int) -> int:
    """Cuántas unidades congela torchvision para ese ``trainable_backbone_layers``.

    Replica ``_resnet_fpn_extractor`` y ``_mobilenet_extractor`` sin importar
    torchvision, para comprobar la tabla antes de cargar pesos.
    """
    if not 0 <= trainable <= spec.max_trainable:
        raise ArchitectureError(f"{spec.name}: trainable_backbone_layers={trainable} fuera de [0, {spec.max_trainable}].")
    if trainable == 0:
        return len(spec.units)
    return spec.stage_starts[len(spec.stage_starts) - trainable]


def audit_architectures() -> None:
    """Comprueba que la tabla declarada sea la que produce la regla de torchvision."""
    for spec in ARCHITECTURES.values():
        if set(spec.trainable_layers) != set(STRATEGIES) or set(spec.frozen_units) != set(STRATEGIES):
            raise ArchitectureError(f"{spec.name} no declara las tres estrategias.")
        if len(spec.stage_starts) != spec.max_trainable:
            raise ArchitectureError(f"{spec.name}: {len(spec.stage_starts)} estadios y max_trainable={spec.max_trainable}.")
        for strategy in STRATEGIES:
            expected = _torchvision_frozen_units(spec, spec.trainable_layers[strategy])
            if expected != spec.frozen_units[strategy]:
                raise ArchitectureError(
                    f"{spec.name} / {strategy}: trainable_backbone_layers={spec.trainable_layers[strategy]} "
                    f"congela {expected} unidades y la tabla dice {spec.frozen_units[strategy]}."
                )


def freeze_plan() -> dict:
    return {
        name: {
            strategy: {
                "trainable_backbone_layers": spec.trainable_layers[strategy],
                "frozen_units": list(spec.frozen_names(strategy)),
            }
            for strategy in STRATEGIES
        }
        for name, spec in ARCHITECTURES.items()
    }


def build_model(architecture: str, strategy: str, imgsz: int, anchors: str = "coco"):
    """Faster R-CNN con pesos de COCO y la cabeza cambiada a fondo + panel.

    Con ``anchors`` distinto de "coco" se reemplazan las anclas de la RPN y,
    como cambia la cantidad de anclas por posición, también las dos capas de
    salida de la cabeza de la RPN. Su convolución compartida conserva los pesos
    de COCO; ``cls_logits`` y ``bbox_pred`` arrancan de cero.
    """
    from torchvision.models import detection
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.rpn import RPNHead

    if anchors not in ANCHOR_PRESETS:
        raise ArchitectureError(f"Anclas desconocidas: {anchors}. Válidas: {list(ANCHOR_PRESETS)}.")
    spec = ARCHITECTURES[architecture]
    builder = getattr(detection, spec.builder)
    weights = getattr(detection, spec.weights_enum).COCO_V1
    model = builder(
        weights=weights,
        trainable_backbone_layers=spec.trainable_layers[strategy],
        min_size=imgsz,
        max_size=imgsz,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    preset = ANCHOR_PRESETS[anchors]
    if preset is not None:
        n_levels = len(model.rpn.anchor_generator.sizes)
        sizes = preset["sizes"]
        # La ResNet50 tiene un nivel del FPN por tamaño. La MobileNet tiene 3
        # niveles y torchvision pone todos los tamaños en cada uno; se respeta.
        if n_levels == len(sizes):
            level_sizes = tuple((size,) for size in sizes)
        else:
            level_sizes = (tuple(sizes),) * n_levels
        generator = AnchorGenerator(level_sizes, (tuple(preset["aspect_ratios"]),) * n_levels)
        old_head = model.rpn.head
        new_head = RPNHead(model.backbone.out_channels, generator.num_anchors_per_location()[0])
        new_head.conv.load_state_dict(old_head.conv.state_dict())
        model.rpn.anchor_generator = generator
        model.rpn.head = new_head
    return model


def assert_freeze_state(model, architecture: str, strategy: str) -> dict:
    """Compara ``requires_grad`` de cada parámetro contra la tabla.

    Dentro de ``backbone.body`` se congelan exactamente las unidades de la
    tabla. Todo lo demás (FPN, RPN, cabeza ROI) tiene que ser entrenable.
    Devuelve la cantidad de parámetros de cada lado.
    """
    spec = ARCHITECTURES[architecture]
    frozen = set(spec.frozen_names(strategy))
    known = {unit.name for unit in spec.units}
    wrong = []
    counts = {"trainable": 0, "frozen": 0}
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        if parts[:2] == ["backbone", "body"]:
            if parts[2] not in known:
                raise ArchitectureError(f"{architecture}: backbone.body.{parts[2]} no está en la tabla de unidades.")
            should_train = parts[2] not in frozen
        else:
            should_train = True
        if parameter.requires_grad != should_train:
            wrong.append(name)
        counts["trainable" if parameter.requires_grad else "frozen"] += parameter.numel()
    if wrong:
        raise ArchitectureError(
            f"{architecture} / {strategy}: {len(wrong)} parámetros no tienen el requires_grad esperado. "
            f"Ejemplo: {wrong[:3]}."
        )
    if strategy == "backbone" and any(
        parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("backbone.body.")
    ):
        raise ArchitectureError(f"{architecture}: la estrategia backbone dejó parámetros del extractor entrenables.")
    return counts


def describe_architectures() -> str:
    lines = ["Arquitecturas y congelamiento", "=============================", ""]
    for spec in ARCHITECTURES.values():
        lines.append(f"{spec.name}  (pesos {spec.weights_enum}.COCO_V1, cabeza cambiada a {NUM_CLASSES} clases)")
        for strategy in STRATEGIES:
            lines.append(
                f"  {strategy:<8} trainable_backbone_layers={spec.trainable_layers[strategy]}  "
                f"congela: {spec.describe_frozen(strategy)}"
            )
        for note in spec.notes:
            lines.append(f"  nota: {note}")
        lines.append("")
    lines.append("En las dos redes el batch norm del extractor es FrozenBatchNorm2d en todas las estrategias.")
    return "\n".join(lines)
