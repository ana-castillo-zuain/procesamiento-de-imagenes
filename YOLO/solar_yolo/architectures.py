"""Capas y congelamiento de yolov8n, yolo11n y yolo26n.

Los índices son los de la lista top-level del YAML de Ultralytics
(``model.model``), que es lo que interpreta el argumento ``freeze``.
No son el conteo de subcapas que imprime ``model.info()``.

Un entero ``freeze=N`` congela los índices ``0 .. N-1``. Por eso el mismo
entero no significa lo mismo en las tres redes:

- YOLOv8n termina el backbone en la capa 9 (SPPF). ``freeze=10`` lo cubre.
  ``freeze=11`` también congelaría el primer Upsample del cuello.
- YOLO11n y YOLO26n agregan C2PSA como capa 10. El backbone completo es
  ``freeze=11``. ``freeze=10`` dejaría C2PSA entrenable.

YOLO11n y YOLO26n comparten esa numeración, pero no la misma red. YOLO26
cambia los argumentos del SPPF, los C3k2 del cuello, la atención de la capa 22
y la cabeza (end-to-end, sin DFL). Esas diferencias se comprueban al cargar
el peso, antes de entrenar.
"""

from __future__ import annotations

from dataclasses import dataclass


class ArchitectureError(RuntimeError):
    """El peso cargado no coincide con la arquitectura que se va a congelar."""


@dataclass(frozen=True)
class LayerSpec:
    index: int
    module: str
    section: str  # backbone | head
    stage: str
    note: str


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    weights: str
    layers: tuple[LayerSpec, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        indexes = [layer.index for layer in self.layers]
        if indexes != list(range(len(self.layers))):
            raise ArchitectureError(f"{self.name}: los índices de capa no son 0..{len(self.layers) - 1}.")

    def indices(self, *, section: str | None = None, stages: set[str] | None = None) -> tuple[int, ...]:
        chosen = []
        for layer in self.layers:
            if section is not None and layer.section != section:
                continue
            if stages is not None and layer.stage not in stages:
                continue
            chosen.append(layer.index)
        return tuple(chosen)

    @property
    def backbone_indices(self) -> tuple[int, ...]:
        return self.indices(section="backbone")

    @property
    def partial_indices(self) -> tuple[int, ...]:
        # Stem (dos convoluciones stride 2) y el primer bloque CSP, el estadio P2.
        # El downsample a P3/8 queda afuera y se entrena.
        return self.indices(stages={"stem", "p2"})

    def freeze_indices(self, strategy: str) -> tuple[int, ...]:
        if strategy == "full":
            return ()
        if strategy == "backbone":
            return self.backbone_indices
        if strategy == "partial":
            return self.partial_indices
        raise KeyError(strategy)

    def layer_map(self) -> dict[int, LayerSpec]:
        return {layer.index: layer for layer in self.layers}

    def describe_indices(self, indices: tuple[int, ...]) -> str:
        if not indices:
            return "(ninguna)"
        layers = self.layer_map()
        return "; ".join(f"{index}:{layers[index].module} {layers[index].note}" for index in indices)

    def describe_trainable(self, strategy: str) -> str:
        frozen = set(self.freeze_indices(strategy))
        trainable = [layer for layer in self.layers if layer.index not in frozen]
        if not trainable:
            return "(ninguna)"
        first, last = trainable[0], trainable[-1]
        return (
            f"{first.index}:{first.module} ({first.note}) "
            f"hasta {last.index}:{last.module} ({last.note})"
        )


def _layers_v8() -> tuple[LayerSpec, ...]:
    return (
        LayerSpec(0, "Conv", "backbone", "stem", "P1/2"),
        LayerSpec(1, "Conv", "backbone", "stem", "P2/4"),
        LayerSpec(2, "C2f", "backbone", "p2", "estadio P2"),
        LayerSpec(3, "Conv", "backbone", "p3", "downsample P3/8"),
        LayerSpec(4, "C2f", "backbone", "p3", "P3 que lee el cuello"),
        LayerSpec(5, "Conv", "backbone", "p4", "downsample P4/16"),
        LayerSpec(6, "C2f", "backbone", "p4", "P4 que lee el cuello"),
        LayerSpec(7, "Conv", "backbone", "p5", "downsample P5/32"),
        LayerSpec(8, "C2f", "backbone", "p5", "estadio P5"),
        LayerSpec(9, "SPPF", "backbone", "spp", "fin del backbone; no hay C2PSA"),
        LayerSpec(10, "Upsample", "head", "neck", "cuello"),
        LayerSpec(11, "Concat", "head", "neck", "concat P4"),
        LayerSpec(12, "C2f", "head", "neck", "cuello"),
        LayerSpec(13, "Upsample", "head", "neck", "cuello"),
        LayerSpec(14, "Concat", "head", "neck", "concat P3"),
        LayerSpec(15, "C2f", "head", "neck", "salida P3"),
        LayerSpec(16, "Conv", "head", "neck", "cuello"),
        LayerSpec(17, "Concat", "head", "neck", "cuello"),
        LayerSpec(18, "C2f", "head", "neck", "salida P4"),
        LayerSpec(19, "Conv", "head", "neck", "cuello"),
        LayerSpec(20, "Concat", "head", "neck", "concat P5"),
        LayerSpec(21, "C2f", "head", "neck", "salida P5"),
        LayerSpec(22, "Detect", "head", "detect", "cabeza"),
    )


def _layers_11_and_26() -> tuple[LayerSpec, ...]:
    """Misma numeración top-level en YOLO11 y YOLO26. Los módulos no lo son."""
    return (
        LayerSpec(0, "Conv", "backbone", "stem", "P1/2"),
        LayerSpec(1, "Conv", "backbone", "stem", "P2/4"),
        LayerSpec(2, "C3k2", "backbone", "p2", "estadio P2"),
        LayerSpec(3, "Conv", "backbone", "p3", "downsample P3/8"),
        LayerSpec(4, "C3k2", "backbone", "p3", "P3 que lee el cuello"),
        LayerSpec(5, "Conv", "backbone", "p4", "downsample P4/16"),
        LayerSpec(6, "C3k2", "backbone", "p4", "P4 que lee el cuello"),
        LayerSpec(7, "Conv", "backbone", "p5", "downsample P5/32"),
        LayerSpec(8, "C3k2", "backbone", "p5", "estadio P5"),
        LayerSpec(9, "SPPF", "backbone", "spp", "pirámide espacial"),
        LayerSpec(10, "C2PSA", "backbone", "psa", "atención; el cuello lee esta capa, no la 9"),
        LayerSpec(11, "Upsample", "head", "neck", "cuello"),
        LayerSpec(12, "Concat", "head", "neck", "concat P4"),
        LayerSpec(13, "C3k2", "head", "neck", "cuello"),
        LayerSpec(14, "Upsample", "head", "neck", "cuello"),
        LayerSpec(15, "Concat", "head", "neck", "concat P3"),
        LayerSpec(16, "C3k2", "head", "neck", "salida P3"),
        LayerSpec(17, "Conv", "head", "neck", "cuello"),
        LayerSpec(18, "Concat", "head", "neck", "cuello"),
        LayerSpec(19, "C3k2", "head", "neck", "salida P4"),
        LayerSpec(20, "Conv", "head", "neck", "cuello"),
        LayerSpec(21, "Concat", "head", "neck", "concat P5, desde la capa 10"),
        LayerSpec(22, "C3k2", "head", "neck", "salida P5"),
        LayerSpec(23, "Detect", "head", "detect", "cabeza"),
    )


ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "yolov8n": ArchitectureSpec(
        name="yolov8n",
        weights="yolov8n.pt",
        layers=_layers_v8(),
        notes=(
            "Backbone de 10 módulos (0-9). El último es SPPF. No existe C2PSA.",
            "La capa 2 del estadio P2 es C2f, no C3k2.",
            "La cabeza es Detect con DFL (reg_max=16). Ultralytics deja .dfl congelado siempre.",
        ),
    ),
    "yolo11n": ArchitectureSpec(
        name="yolo11n",
        weights="yolo11n.pt",
        layers=_layers_11_and_26(),
        notes=(
            "Backbone de 11 módulos (0-10). El último es C2PSA.",
            "SPPF (capa 9) no tiene atajo residual.",
            "Los C3k2 del cuello en las capas 13, 16 y 19 usan Bottleneck (c3k=False).",
            "La capa 22 es C3k2 con C3k y sin PSABlock.",
            "Detect no tiene cabeza one-to-one. reg_max=16 y el DFL queda congelado.",
        ),
    ),
    "yolo26n": ArchitectureSpec(
        name="yolo26n",
        weights="yolo26n.pt",
        layers=_layers_11_and_26(),
        notes=(
            "La numeración del backbone coincide con YOLO11 (0-10) y no con YOLOv8.",
            "SPPF (capa 9) suma un atajo residual: el YAML pasa shortcut=True.",
            "Los C3k2 del cuello usan C3k. La capa 22 además mete PSABlock.",
            "Detect entrena cabeza one-to-many y one-to-one (end2end). reg_max=1: el DFL es Identity y no tiene parámetros.",
            "Congelar la cabeza no forma parte de ninguna estrategia: la capa 23 se entrena en las tres.",
        ),
    ),
}

STRATEGIES: dict[str, str] = {
    "full": (
        "Full fine-tuning. Se entrenan todas las capas. "
        "Ultralytics igual congela los parámetros cuyo nombre contiene .dfl cuando existen."
    ),
    "backbone": (
        "Backbone congelado. Se congelan todos los módulos del backbone y se entrenan el cuello y la cabeza. "
        "El corte es el último módulo declarado bajo backbone en el YAML de esa arquitectura."
    ),
    "partial": (
        "Fine-tuning parcial. Se congelan solo el stem y el estadio P2 "
        "(dos convoluciones stride 2 y el primer bloque CSP). "
        "Se entrena desde el downsample a P3/8 inclusive, junto con el resto del backbone, el cuello y la cabeza."
    ),
}


def architecture_names() -> list[str]:
    return list(ARCHITECTURES)


def strategy_names() -> list[str]:
    return list(STRATEGIES)


def freeze_plan() -> dict:
    plan = {}
    for name, spec in ARCHITECTURES.items():
        plan[name] = {}
        for strategy in STRATEGIES:
            indexes = spec.freeze_indices(strategy)
            plan[name][strategy] = {
                "weights": spec.weights,
                "frozen_indices": list(indexes),
                "frozen_modules": spec.describe_indices(indexes),
                "trainable_modules": spec.describe_trainable(strategy),
                "notes": list(spec.notes),
            }
    return plan


def describe_architectures() -> str:
    lines = [
        "Estrategias de congelamiento",
        "============================",
        "",
        "El argumento freeze de Ultralytics usa el índice top-level del YAML.",
        "freeze=[0, 1, 2] congela esas capas. freeze=10 congela 0..9, no la capa 10.",
        "",
    ]
    for strategy, text in STRATEGIES.items():
        lines.append(f"{strategy}: {text}")
    lines.append("")
    for spec in ARCHITECTURES.values():
        lines.append(f"## {spec.name}  ({spec.weights})")
        for note in spec.notes:
            lines.append(f"- {note}")
        lines.append("Capas:")
        for layer in spec.layers:
            lines.append(
                f"  {layer.index:2d}  {layer.module:<10} {layer.section:<10} {layer.stage:<8} {layer.note}"
            )
        for strategy in STRATEGIES:
            indexes = spec.freeze_indices(strategy)
            shown = ", ".join(str(index) for index in indexes) if indexes else "(ninguna)"
            lines.append(f"Congela {strategy}: {shown}")
            lines.append(f"  entrenable: {spec.describe_trainable(strategy)}")
        lines.append("")
    lines.append(
        "YOLOv8n congela 10 capas de backbone; YOLO11n y YOLO26n congelan 11. "
        "El fine-tuning parcial congela los índices 0, 1 y 2 en las tres, pero la capa 2 "
        "es C2f en YOLOv8n y C3k2 en YOLO11n y YOLO26n. El mismo índice no es el mismo bloque."
    )
    return "\n".join(lines)


def audit_architectures() -> None:
    """Coherencia interna de las tablas. No carga modelos."""
    v8 = ARCHITECTURES["yolov8n"]
    y11 = ARCHITECTURES["yolo11n"]
    y26 = ARCHITECTURES["yolo26n"]

    if v8.backbone_indices != tuple(range(10)):
        raise ArchitectureError("YOLOv8n: el backbone dejó de ser el rango 0-9.")
    if y11.backbone_indices != tuple(range(11)) or y26.backbone_indices != tuple(range(11)):
        raise ArchitectureError("YOLO11n/YOLO26n: el backbone dejó de ser el rango 0-10.")
    if v8.partial_indices != (0, 1, 2) or y11.partial_indices != (0, 1, 2) or y26.partial_indices != (0, 1, 2):
        raise ArchitectureError("El corte parcial dejó de ser stem + estadio P2.")
    for spec in (v8, y11, y26):
        if not set(spec.partial_indices) < set(spec.backbone_indices):
            raise ArchitectureError(f"El corte parcial de {spec.name} no es un subconjunto estricto del backbone.")

    if v8.layers[2].module != "C2f" or y11.layers[2].module != "C3k2" or y26.layers[2].module != "C3k2":
        raise ArchitectureError("La capa 2 ya no distingue el estadio P2 de cada arquitectura.")
    if v8.layers[9].module != "SPPF" or any(layer.module == "C2PSA" for layer in v8.layers):
        raise ArchitectureError("YOLOv8n no debe tener C2PSA; su backbone termina en SPPF.")
    if y11.layers[10].module != "C2PSA" or y26.layers[10].module != "C2PSA":
        raise ArchitectureError("YOLO11n y YOLO26n deben terminar el backbone en C2PSA, capa 10.")
    if v8.layers[-1].module != "Detect" or y11.layers[-1].module != "Detect" or y26.layers[-1].module != "Detect":
        raise ArchitectureError("La última capa top-level debe ser Detect.")

    modules_v8 = [layer.module for layer in v8.layers[:3]]
    modules_11 = [layer.module for layer in y11.layers[:3]]
    if modules_v8 == modules_11:
        raise ArchitectureError("El stem+P2 de YOLOv8n y YOLO11n no debería tener los mismos módulos.")
    if [layer.module for layer in y11.layers] != [layer.module for layer in y26.layers]:
        raise ArchitectureError(
            "YOLO11n y YOLO26n dejaron de compartir la secuencia top-level. "
            "Hay que rehacer el corte en vez de reutilizar la tabla."
        )
    if v8.weights != "yolov8n.pt" or y11.weights != "yolo11n.pt" or y26.weights != "yolo26n.pt":
        raise ArchitectureError("Los experimentos tienen que arrancar de los pesos .pt oficiales.")

    _audit_prefix_matching()
    _audit_freeze_states()


def _audit_prefix_matching() -> None:
    """El prefijo model.{i}. de Ultralytics no puede confundir la capa 1 con la 10."""
    samples = {
        "model.0.conv.weight": 0,
        "model.1.conv.weight": 1,
        "model.2.m.0.cv1.conv.weight": 2,
        "model.10.cv2.conv.weight": 10,
        "model.11.conv.weight": 11,
        "model.23.one2one_cv2.0.weight": 23,
        "model.22.dfl.conv.weight": 22,
        "module.model.9.cv1.conv.weight": 9,
    }
    for name, expected in samples.items():
        index, _is_dfl = classify_parameter(name)
        if index != expected:
            raise ArchitectureError(f"{name} se clasificó como capa {index}, se esperaba {expected}.")
    if classify_parameter("model.22.dfl.conv.weight") != (22, True):
        raise ArchitectureError("El parámetro DFL tiene que detectarse aparte de la cabeza.")
    if classify_parameter("model.22.cv2.0.weight") != (22, False):
        raise ArchitectureError("Un peso de la cabeza no es DFL.")

    partial = freeze_prefixes((0, 1, 2))
    backbone_v8 = freeze_prefixes(tuple(range(10)))
    backbone_later = freeze_prefixes(tuple(range(11)))
    if _matches("model.10.cv.weight", partial) or _matches("model.3.conv.weight", partial):
        raise ArchitectureError("El corte parcial está alcanzando capas que tienen que entrenarse.")
    if not _matches("model.9.cv1.weight", backbone_v8):
        raise ArchitectureError("El backbone de YOLOv8n no cubre la capa 9.")
    if _matches("model.10.cv.weight", backbone_v8):
        raise ArchitectureError("El backbone de YOLOv8n no debe congelar la capa 10 (Upsample del cuello).")
    if not _matches("model.10.m.0.weight", backbone_later):
        raise ArchitectureError("El backbone de YOLO11/26 debe congelar C2PSA, la capa 10.")
    if _matches("model.11.conv.weight", backbone_later):
        raise ArchitectureError("El backbone de YOLO11/26 no debe congelar el primer Upsample, capa 11.")


def _audit_freeze_states() -> None:
    """Simula requires_grad. No instancia un modelo."""
    for spec in ARCHITECTURES.values():
        detect_index = spec.layers[-1].index
        for strategy in STRATEGIES:
            frozen = spec.freeze_indices(strategy)
            assert_freeze_state(_fake_parameters(len(spec.layers), set(frozen), detect_index), frozen, len(spec.layers))

        partial = set(spec.partial_indices)
        leaked = _fake_parameters(len(spec.layers), partial | {3}, detect_index)
        try:
            assert_freeze_state(leaked, spec.partial_indices, len(spec.layers))
        except ArchitectureError:
            continue
        raise ArchitectureError(f"{spec.name}: no detectó una capa 3 congelada de más.")


def _fake_parameters(n_layers: int, frozen: set[int], detect_index: int) -> list[tuple[str, bool]]:
    parameters = [(f"model.{index}.conv.weight", index not in frozen) for index in range(n_layers)]
    parameters.append((f"model.{detect_index}.dfl.conv.weight", False))
    return parameters


def freeze_prefixes(indices: tuple[int, ...]) -> list[str]:
    return [f"model.{index}." for index in indices]


def _matches(parameter_name: str, prefixes: list[str]) -> bool:
    return any(prefix in parameter_name for prefix in prefixes)


def classify_parameter(name: str) -> tuple[int | None, bool]:
    """Devuelve (índice top-level, es_dfl). Índice None si el nombre no es de una capa."""
    parts = name.split(".")
    index = None
    for position, part in enumerate(parts[:-1]):
        if part == "model" and parts[position + 1].isdigit():
            index = int(parts[position + 1])
            break
    is_dfl = ".dfl." in f".{name}." or name.endswith(".dfl")
    return index, is_dfl


def top_level_modules(yolo_model) -> list:
    """Lista top-level de un objeto YOLO ya cargado. No descarga nada por sí misma."""
    inner = getattr(yolo_model, "model", None)
    sequential = getattr(inner, "model", None)
    if sequential is None:
        raise ArchitectureError(
            "No se encontró model.model, la lista top-level que Ultralytics numera para freeze."
        )
    return list(sequential)


def verify_loaded_architecture(yolo_model, architecture: str) -> list:
    """Comprueba el grafo cargado contra la tabla de esa arquitectura."""
    spec = ARCHITECTURES[architecture]
    modules = top_level_modules(yolo_model)
    if len(modules) != len(spec.layers):
        raise ArchitectureError(
            f"{architecture}: se esperaban {len(spec.layers)} capas top-level y el peso tiene {len(modules)}: "
            + ", ".join(type(module).__name__ for module in modules)
        )
    mismatches = []
    for layer, module in zip(spec.layers, modules):
        actual = type(module).__name__
        if actual != layer.module:
            mismatches.append(f"capa {layer.index}: se esperaba {layer.module} y hay {actual}")
    if mismatches:
        raise ArchitectureError(f"{architecture} no coincide con el peso cargado. " + "; ".join(mismatches))
    _verify_variant_signature(architecture, modules)
    return modules


def _verify_variant_signature(architecture: str, modules: list) -> None:
    detect = modules[-1]
    if architecture == "yolov8n":
        _require(not any(type(module).__name__ == "C2PSA" for module in modules), "YOLOv8n no tiene C2PSA.")
        _require(not hasattr(detect, "one2one_cv2"), "YOLOv8n no debe traer cabeza one-to-one.")
        _require(getattr(detect, "reg_max", None) == 16, "YOLOv8n debe usar reg_max=16.")
        _require(type(getattr(detect, "dfl", None)).__name__ == "DFL", "YOLOv8n debe tener DFL.")
        _require(getattr(modules[9], "add", False) is False, "El SPPF de YOLOv8n no lleva atajo residual.")
        return

    sppf = modules[9]
    neck = modules[13]
    deep_neck = modules[22]
    if architecture == "yolo11n":
        _require(getattr(sppf, "add", False) is False, "El SPPF de YOLO11n no lleva atajo residual.")
        _require("C3k" not in _inner_types(neck), "La capa 13 de YOLO11n es C3k2 con Bottleneck, no con C3k.")
        _require("PSABlock" not in _inner_types(deep_neck), "La capa 22 de YOLO11n no lleva PSABlock.")
        _require("C3k" in _inner_types(deep_neck), "La capa 22 de YOLO11n debe usar C3k.")
        _require(not hasattr(detect, "one2one_cv2"), "YOLO11n no debe traer cabeza one-to-one.")
        _require(getattr(detect, "reg_max", None) == 16, "YOLO11n debe usar reg_max=16.")
        _require(type(getattr(detect, "dfl", None)).__name__ == "DFL", "YOLO11n debe tener DFL.")
        return

    if architecture == "yolo26n":
        _require(getattr(sppf, "add", None) is True, "El SPPF de YOLO26n debe tener el atajo residual (add=True).")
        _require("C3k" in _inner_types(neck), "La capa 13 de YOLO26n debe usar C3k.")
        _require("PSABlock" in _inner_types(deep_neck), "La capa 22 de YOLO26n debe incluir PSABlock.")
        _require(hasattr(detect, "one2one_cv2"), "YOLO26n debe tener la cabeza one-to-one (one2one_cv2).")
        _require(getattr(detect, "reg_max", None) == 1, "YOLO26n debe usar reg_max=1.")
        _require(type(getattr(detect, "dfl", None)).__name__ == "Identity", "El DFL de YOLO26n debe ser Identity.")
        return

    raise ArchitectureError(f"No hay firma de verificación para {architecture}.")


def _inner_types(module) -> set[str]:
    return {type(child).__name__ for child in module.modules()}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchitectureError(message)


def assert_freeze_state(parameters, frozen_indices, n_layers: int) -> None:
    """Comprueba requires_grad después de que el trainer aplicó freeze.

    ``parameters`` es un iterable de ``(nombre, requires_grad)``. Los parámetros
    DFL pueden quedar congelados dentro de una cabeza entrenable: Ultralytics
    los congela siempre. El resto de una capa congelada tiene que estar
    congelado, y el resto de una capa entrenable tiene que poder entrenarse.
    """
    frozen = set(frozen_indices)
    seen = {index: {"frozen": 0, "trainable": 0, "dfl_frozen": 0, "dfl_trainable": 0} for index in range(n_layers)}
    unmatched = []
    for name, requires_grad in parameters:
        index, is_dfl = classify_parameter(name)
        if index is None or index not in seen:
            unmatched.append(f"{name} requires_grad={requires_grad}")
            continue
        if is_dfl:
            seen[index]["dfl_trainable" if requires_grad else "dfl_frozen"] += 1
        else:
            seen[index]["trainable" if requires_grad else "frozen"] += 1

    problems = []
    if unmatched:
        problems.append("parámetros fuera de la tabla: " + ", ".join(unmatched))
    for index, counts in seen.items():
        if index in frozen:
            if counts["trainable"] or counts["dfl_trainable"]:
                problems.append(f"la capa {index} debía congelarse y tiene parámetros entrenables")
        else:
            if counts["frozen"]:
                problems.append(f"la capa {index} debía entrenarse y tiene parámetros congelados")
            if counts["dfl_trainable"]:
                problems.append(f"el DFL de la capa {index} quedó entrenable")
    if problems:
        raise ArchitectureError("El congelamiento no coincidió con la estrategia. " + "; ".join(problems))
