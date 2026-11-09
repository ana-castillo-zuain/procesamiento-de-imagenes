# Comparativa de detección: Faster R-CNN vs YOLO

Promedio **macro** (no ponderado) de los tres folds por fuente: fold 1 PV01, fold 2 solarpanels_hd, fold 3 geo_unknown + kingston. Misma partición, imágenes de 256 px, una semilla. Fuentes: [RCNN/RESULTADOS.md](RCNN/RESULTADOS.md) y `YOLO/results/summary.csv`. El F1 imagen de YOLO se calculó sobre las predicciones en `YOLO/results/predictions/` (score = confianza máxima por imagen).

## Mejores modelos

| Modelo | mAP50-95 macro | mAP50 macro | Precisión | Recall | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ResNet50 `partial`** | **0.331** | **0.566** | **0.759** | 0.534 | **0.436** | **0.351** | **0.204** |
| YOLO26n `backbone` | 0.271 | 0.510 | 0.558 | 0.504 | 0.331 | 0.315 | 0.166 |

ResNet50 FPN con fine-tuning parcial gana **0.060** de mAP50-95 macro y **0.056** de mAP50. La ventaja aparece en los tres folds. YOLO26n es el mejor de su familia, pero ninguna receta YOLO alcanza a las dos mejores ResNet50 (`partial` 0.331, `full` 0.325).

Precisión y recall de R-CNN: punto de F1 máximo, IoU 0.5. YOLO: métricas Ultralytics del `best.pt`. No son el mismo umbral; el mAP es la comparación principal.

## mAP50-95 por fold

| Familia | Configuración | Fold 1 (PV01) | Fold 2 (solarpanels_hd) | Fold 3 (geo + kingston) | **Macro** |
|---|---|---:|---:|---:|---:|
| R-CNN | MobileNetV3 `backbone` | 0.031 | 0.170 | 0.117 | 0.106 |
| R-CNN | MobileNetV3 `full` | 0.124 | 0.266 | 0.123 | 0.171 |
| R-CNN | MobileNetV3 `partial` | 0.106 | 0.245 | 0.149 | 0.167 |
| R-CNN | ResNet50 `backbone` | 0.344 | 0.328 | 0.163 | 0.279 |
| R-CNN | ResNet50 `full` | **0.466** | 0.323 | 0.187 | 0.325 |
| R-CNN | ResNet50 `partial` | 0.436 | **0.351** | **0.204** | **0.331** |
| YOLO | yolov8n `backbone` | 0.263 | 0.256 | 0.102 | 0.207 |
| YOLO | yolov8n `full` | 0.275 | 0.258 | 0.048 | 0.194 |
| YOLO | yolov8n `partial` | 0.343 | 0.255 | 0.076 | 0.225 |
| YOLO | yolo11n `backbone` | 0.243 | 0.283 | 0.081 | 0.202 |
| YOLO | yolo11n `full` | 0.278 | 0.265 | 0.051 | 0.198 |
| YOLO | yolo11n `partial` | 0.305 | 0.244 | 0.074 | 0.208 |
| YOLO | yolo26n `backbone` | 0.331 | 0.315 | 0.166 | **0.271** |
| YOLO | yolo26n `full` | 0.364 | 0.297 | 0.093 | 0.251 |
| YOLO | yolo26n `partial` | 0.324 | 0.298 | 0.137 | 0.253 |

## Métricas macro (todas las recetas)

| Familia | Configuración | mAP50-95 | mAP50 | Precisión | Recall |
|---|---|---:|---:|---:|---:|
| R-CNN | MobileNetV3 `backbone` | 0.106 | 0.255 | 0.508 | 0.270 |
| R-CNN | MobileNetV3 `full` | 0.171 | 0.383 | 0.671 | 0.379 |
| R-CNN | MobileNetV3 `partial` | 0.167 | 0.360 | 0.556 | 0.387 |
| R-CNN | ResNet50 `backbone` | 0.279 | 0.560 | 0.727 | 0.525 |
| R-CNN | ResNet50 `full` | 0.325 | 0.568 | 0.719 | 0.529 |
| R-CNN | ResNet50 `partial` | **0.331** | 0.566 | **0.759** | **0.534** |
| YOLO | yolov8n `backbone` | 0.207 | 0.416 | 0.477 | 0.428 |
| YOLO | yolov8n `full` | 0.194 | 0.378 | 0.455 | 0.378 |
| YOLO | yolov8n `partial` | 0.225 | 0.418 | 0.519 | 0.394 |
| YOLO | yolo11n `backbone` | 0.202 | 0.410 | 0.511 | 0.418 |
| YOLO | yolo11n `full` | 0.198 | 0.375 | 0.483 | 0.376 |
| YOLO | yolo11n `partial` | 0.208 | 0.399 | 0.536 | 0.376 |
| YOLO | yolo26n `backbone` | 0.271 | **0.510** | 0.558 | 0.504 |
| YOLO | yolo26n `full` | 0.251 | 0.445 | 0.597 | 0.413 |
| YOLO | yolo26n `partial` | 0.253 | 0.470 | 0.565 | 0.459 |

Las macros YOLO de mAP50, P y R son el promedio de las tres columnas de `summary.csv` (igual que el mAP50-95).

## Qué se sostiene con el macro

- **La arquitectura pesa más que el congelamiento.** ResNet50 duplica a MobileNet. YOLO26n supera a v8n y 11n por ~0.05–0.08 de mAP50-95.
- **El congelamiento óptimo no coincide.** En R-CNN, `backbone` es el peor (0.279 vs 0.331). En YOLO26n, `backbone` es el mejor (0.271 vs 0.251–0.253).
- **El fold 3 es el techo compartido:** 0.204 R-CNN vs 0.166 YOLO. Ninguna familia generaliza bien a geo+kingston (79 cajas; el mAP de ese fold es inestable).
- **La variante R-CNN con aumento** (fuera de la grilla de 18) queda en 0.332 de mAP50-95 macro: empate con la base. No cambia el ranking frente a YOLO.

## Detector como clasificador (F1 macro imagen)

Cada imagen recibe como score su confianza máxima de detección. Clases: tiene paneles / no tiene. *Fábrica* es umbral 0.25. *Calibrado* es el umbral que maximiza el F1 macro sobre la misma validación: es un diagnóstico, no una estimación.

Referencia trivial (siempre positivo): fold 1 0.499, fold 2 0.473, fold 3 0.293.

### Mejores detectores

| Modelo | Fold 1 fábrica | Fold 2 fábrica | Fold 3 fábrica | Fold 2 calibrado | Fold 3 calibrado |
|---|---:|---:|---:|---:|---:|
| ResNet50 `partial` | **0.749** | 0.557 | 0.465 | 0.698 | 0.757 |
| YOLO26n `backbone` | 0.515 | **0.647** | **0.679** | 0.647 | 0.726 |

Con umbral de fábrica, YOLO26n `backbone` gana en los folds 2 y 3 (0.647 y 0.679 contra 0.557 y 0.465). ResNet50 `partial` dispara demasiado a 0.25; calibrando el umbral el fold 3 sube a 0.757. El fold 1 no sirve para medir clasificación (PV01: 477 positivas y una negativa).

YOLO26n `backbone`, umbral 0.25 (TP / FP / TN / FN): fold 1 447 / 0 / 1 / 30; fold 2 216 / 4 / 28 / 68; fold 3 34 / 24 / 38 / 10. En el fold 3 el umbral calibrado fue 0.41.

### Faster R-CNN (todas las recetas)

| Arquitectura | Estrategia | Fold 1 fábrica | Fold 2 fábrica | Fold 3 fábrica | Fold 2 calibrado | Fold 3 calibrado |
|---|---|---:|---:|---:|---:|---:|
| MobileNetV3 | `backbone` | 0.420 | 0.598 | **0.789** | 0.657 | **0.824** |
| MobileNetV3 | `full` | 0.487 | 0.650 | 0.774 | **0.753** | 0.802 |
| MobileNetV3 | `partial` | 0.532 | 0.633 | 0.719 | 0.750 | 0.774 |
| ResNet50 | `backbone` | 0.498 | **0.711** | 0.539 | 0.717 | 0.776 |
| ResNet50 | `full` | 0.586 | 0.675 | 0.600 | 0.723 | 0.735 |
| ResNet50 | `partial` | **0.749** | 0.557 | 0.465 | 0.698 | 0.757 |

El mejor detector no es el mejor clasificador: en el fold 3 con umbral 0.25, MobileNet `backbone` (0.789) supera a ResNet50 `partial` (0.465) porque inventa menos en imágenes vacías.

### YOLO (todas las recetas)

| Arquitectura | Estrategia | Fold 1 fábrica | Fold 2 fábrica | Fold 3 fábrica | Fold 2 calibrado | Fold 3 calibrado |
|---|---|---:|---:|---:|---:|---:|
| yolov8n | `full` | 0.503 | 0.598 | 0.656 | 0.598 | 0.662 |
| yolov8n | `backbone` | 0.530 | **0.707** | 0.682 | 0.710 | 0.682 |
| yolov8n | `partial` | 0.528 | 0.549 | 0.575 | 0.549 | 0.575 |
| yolo11n | `full` | 0.483 | 0.634 | 0.538 | 0.634 | 0.545 |
| yolo11n | `backbone` | 0.494 | 0.694 | 0.511 | **0.732** | 0.722 |
| yolo11n | `partial` | 0.490 | 0.635 | 0.665 | 0.636 | 0.687 |
| yolo26n | `full` | 0.505 | 0.624 | 0.620 | 0.624 | 0.620 |
| yolo26n | `backbone` | 0.515 | 0.647 | **0.679** | 0.647 | **0.726** |
| yolo26n | `partial` | 0.489 | 0.569 | 0.644 | 0.569 | 0.644 |

En varios YOLO el umbral calibrado queda en ~0.01: a 0.25 ya casi no hay detecciones débiles que recortar. yolo11n `backbone` en el fold 3 es la excepción (fábrica 0.511 → calibrado 0.722 con umbral 0.53): a 0.25 marca demasiadas vacías (FP 46).

## Límites de la comparación

- Recetas distintas: R-CNN usa SGD, 50 épocas, patience 10, batch 8; YOLO usa AdamW, 100 épocas, patience 15, batch 16 y más aumento.
- El mAP de torchvision-COCO y el de Ultralytics no emparejan igual. Diferencias de ~0.01 no son concluyentes.
- Una semilla. `best.pt` se elige sobre la misma validación en las dos familias.
- Precisión/recall y F1 calibrado no son el mismo protocolo entre familias.

## Conclusiones

1. En detección, gana **ResNet50 `partial`** (0.331 de mAP50-95 macro) sobre **YOLO26n `backbone`** (0.271).
2. Dentro de YOLO, **yolo26n** es claramente mejor que v8n y 11n; congelar el backbone es la mejor estrategia de esa arquitectura.
3. En clasificación imagen con umbral 0.25, **YOLO26n `backbone` supera a ResNet50 `partial`** en los folds 2 y 3. R-CNN necesita un umbral más alto para no inventar paneles.
4. Ninguna familia generaliza bien al fold 3 (~0.20 de mAP50-95). El cuello de botella sigue siendo la brecha de dominio entre fuentes.
