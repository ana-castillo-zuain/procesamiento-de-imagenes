# Resultados de Faster R-CNN

Son las 18 corridas de `entrenamiento_colab.ipynb`: 2 arquitecturas, 3 estrategias de congelamiento y 3 folds. Se entrenó en Colab con una Tesla T4 (torch 2.11, torchvision 0.26) y no falló ninguna. Tardaron 2 h 19 min en total. La receta está en el [README](README.md). Los números salen de `summary.csv`, `fold_metrics.csv`, `classification_metrics.csv` y `experiments.csv`, más el log por época del notebook.

## Detección

mAP50-95 por fold con el `best.pt` de cada corrida. La métrica principal es el promedio macro de los tres folds.

| Arquitectura | Estrategia | Fold 1 (PV01) | Fold 2 (solarpanels_hd) | Fold 3 (geo_unknown + kingston) | **Macro** |
|---|---|---|---|---|---|
| MobileNetV3 | `backbone` | 0.031 | 0.170 | 0.117 | 0.106 |
| MobileNetV3 | `full` | 0.124 | 0.266 | 0.123 | 0.171 |
| MobileNetV3 | `partial` | 0.106 | 0.245 | 0.149 | 0.167 |
| ResNet50 | `backbone` | 0.344 | 0.328 | 0.163 | 0.279 |
| ResNet50 | `full` | **0.466** | 0.323 | 0.187 | 0.325 |
| ResNet50 | `partial` | 0.436 | **0.351** | **0.204** | **0.331** |

Promedio macro de las otras métricas (P y R en el punto de F1 máximo, IoU 0.5):

| Arquitectura | Estrategia | mAP50 | Precisión | Recall | mAP50-95 |
|---|---|---|---|---|---|
| MobileNetV3 | `backbone` | 0.255 | 0.508 | 0.270 | 0.106 |
| MobileNetV3 | `full` | 0.383 | 0.671 | 0.379 | 0.171 |
| MobileNetV3 | `partial` | 0.360 | 0.556 | 0.387 | 0.167 |
| ResNet50 | `backbone` | 0.560 | 0.727 | 0.525 | 0.279 |
| ResNet50 | `full` | 0.568 | 0.719 | 0.529 | 0.325 |
| ResNet50 | `partial` | 0.566 | 0.759 | 0.534 | 0.331 |

### Qué se ve

- **La arquitectura pesa más que el congelamiento.** Con la misma estrategia, la ResNet50 duplica el mAP50-95 de la MobileNet. La diferencia es más grande en PV01 (0.47 contra 0.12), la fuente con más cajas por imagen y más cajas chicas. Coincide con lo previsto en el README: a 256 px, el FPN de la MobileNet de torchvision solo ve una grilla de 8 × 8 y sus anclas más chicas son de 32 px.
- **Congelar todo el extractor (`backbone`) es lo peor en las dos redes.** Las features de COCO sirven, pero hay que adaptarlas a la vista aérea. En la ResNet50 cuesta 0.05 de mAP50-95 macro; en la MobileNet, 0.06.
- **`full` y `partial` empatan.** En la ResNet50 la diferencia es 0.006 y en la MobileNet 0.004. Las dos son mucho menores que el ruido entre épocas (ver *Dinámica del entrenamiento*). No se puede elegir una sobre la otra con estos datos. En la MobileNet era esperable, porque `partial` congela solo 8 072 parámetros.
- **El fold 3 es el más difícil en los seis casos.** No pasa de 0.20, y además tiene solo 79 cajas, así que su mAP es poco preciso.
- **Hacia mAP50-95 se pierde más que hacia mAP50.** La ResNet50 llega a 0.57 de mAP50 macro, pero cae a 0.33 en mAP50-95: encuentra los paneles, pero las cajas no ajustan bien al borde. En imágenes de 256 px con paneles de pocos píxeles, un píxel de error ya baja mucho el IoU.

### Fold 3 por fuente

| Arquitectura | Estrategia | geo_unknown mAP50-95 (13 cajas) | kingston mAP50-95 (66 cajas) | kingston2: imágenes con FP (de 28) | kingston2: cajas FP |
|---|---|---|---|---|---|
| MobileNetV3 | `backbone` | 0.196 | 0.109 | 5 | 5 |
| MobileNetV3 | `full` | 0.182 | 0.116 | 4 | 4 |
| MobileNetV3 | `partial` | 0.301 | 0.131 | 9 | 9 |
| ResNet50 | `backbone` | 0.178 | 0.189 | 19 | 31 |
| ResNet50 | `full` | 0.331 | 0.182 | 15 | 25 |
| ResNet50 | `partial` | 0.384 | 0.182 | 23 | 56 |

kingston2 no tiene paneles, así que cualquier detección con confianza ≥ 0.25 es un falso positivo. **La ResNet50 detecta más pero también inventa más**: dispara en 15 a 23 de las 28 imágenes vacías, mientras que la MobileNet lo hace en 4 a 9. Lo de geo_unknown sale de solo 13 cajas, así que sus diferencias entre estrategias no son confiables.

## Detector como clasificador

Cada imagen recibe como score su confianza máxima de detección. F1 macro de "tiene paneles / no tiene":

| Arquitectura | Estrategia | Fold 1 fábrica | Fold 2 fábrica | Fold 3 fábrica | Fold 2 calibrado | Fold 3 calibrado |
|---|---|---|---|---|---|---|
| Referencia trivial (todo positivo) | | 0.499 | 0.473 | 0.293 | 0.473 | 0.293 |
| MobileNetV3 | `backbone` | 0.420 | 0.598 | **0.789** | 0.657 | **0.824** |
| MobileNetV3 | `full` | 0.487 | 0.650 | 0.774 | **0.753** | 0.802 |
| MobileNetV3 | `partial` | 0.532 | 0.633 | 0.719 | 0.750 | 0.774 |
| ResNet50 | `backbone` | 0.498 | **0.711** | 0.539 | 0.717 | 0.776 |
| ResNet50 | `full` | 0.586 | 0.675 | 0.600 | 0.723 | 0.735 |
| ResNet50 | `partial` | **0.749** | 0.557 | 0.465 | 0.698 | 0.757 |

*Fábrica* es umbral 0.25. *Calibrado* es el umbral que maximiza el F1 macro sobre la misma validación: es un diagnóstico, no una estimación.

- **El fold 1 no sirve para medir clasificación.** PV01 tiene 477 imágenes con paneles y una sola sin. El F1 macro depende de si esa única imagen se acierta. Decir siempre "sí" ya da 99.8 % de accuracy.
- **En los folds 2 y 3 todos los modelos superan la referencia trivial**, también con el umbral de fábrica.
- **El mejor detector no es el mejor clasificador.** En el fold 3, con umbral 0.25, la ResNet50 `partial` acierta 40 de 44 imágenes con paneles, pero marca 49 de las 62 vacías (TP 40, FP 49, FN 4, TN 13). La MobileNet `backbone` se equivoca menos con las vacías (TP 35, FP 13, FN 9, TN 49). Es el mismo efecto que en kingston2.
- **El problema de la ResNet50 es el umbral, no el ranking.** Con el umbral calibrado, su F1 macro en el fold 3 sube de 0.47–0.60 a 0.74–0.78: los scores sí separan las clases. Sus confianzas son más altas que las de la MobileNet, y 0.25 le queda bajo. Para usarla como clasificador habría que elegir el umbral en un conjunto aparte, no en esta validación.

## Dinámica del entrenamiento

| Arquitectura | Estrategia | Épocas corridas (f1 / f2 / f3) | Mejor época (f1 / f2 / f3) | Minutos (3 folds) |
|---|---|---|---|---|
| MobileNetV3 | `backbone` | 23 / 43 / 32 | 13 / 33 / 22 | 23 |
| MobileNetV3 | `full` | 21 / 19 / 17 | 11 / 9 / 7 | 15 |
| MobileNetV3 | `partial` | 20 / 23 / 26 | 10 / 13 / 16 | 17 |
| ResNet50 | `backbone` | 26 / 38 / 16 | 16 / 28 / 6 | 32 |
| ResNet50 | `full` | 16 / 18 / 16 | 6 / 8 / 6 | 25 |
| ResNet50 | `partial` | 22 / 15 / 16 | 12 / 5 / 6 | 26 |

- **Todas pararon por early stopping antes de la época 50.** El tope de épocas no limitó a ninguna. La más larga fue la MobileNet `backbone` del fold 2, con 43.
- **Las estrategias con más parámetros entrenables llegan antes a su mejor época.** Con `backbone`, que solo entrena FPN, RPN y cabeza, tardan más en llegar a su techo.
- **El mAP de validación oscila mucho entre épocas.** En la ResNet50 `full` del fold 1 da 0.466 en la época 6, 0.287 en la 9 y 0.465 en la 12. La variación típica de una época a la siguiente es de 0.02 a 0.05 en la ResNet50 y de 0.01 a 0.03 en la MobileNet.
- **El learning rate casi nunca llegó a bajar.** El coseno está pensado para 50 épocas y las corridas pararon entre la 15 y la 26. En 16 de las 18, el learning rate todavía estaba entre el 30 % y el 80 % del máximo. Ningún modelo pasó por la fase de learning rate bajo, que es la que suele estabilizar el mAP.

## Limitaciones

- **`best.pt` es optimista.** Se elige la mejor época sobre la misma validación que después se informa, y con esa oscilación el máximo incluye algo de suerte. YOLO elige `best.pt` de la misma forma, así que la comparación entre familias es pareja en esto. Aun así, las diferencias chicas no se pueden leer como reales.
- **Hay una sola semilla por combinación.** No hay intervalos de confianza. Diferencias menores a unos 0.03 de mAP50-95 no son concluyentes.
- **Las recetas de entrenamiento son distintas.** Esta usa la receta estándar de torchvision (SGD, 50 épocas, patience 10, batch 8, volteos). YOLO usa la de Ultralytics (AdamW, 100 épocas, patience 15, más aumento de datos). Se compara cada familia con su receta habitual, no con una receta idéntica.
- **El mAP no es exactamente el mismo.** El de Ultralytics empareja distinto que el de COCO, así que una diferencia de un punto con YOLO no es concluyente.

## Conclusiones para el informe

1. Dentro de la familia R-CNN, la mejor configuración es **ResNet50 FPN con `partial` o `full`**, alrededor de 0.33 de mAP50-95 macro. Entre las dos no hay ganador claro.
2. La **MobileNetV3 FPN de torchvision no sirve a 256 px**: su FPN no tiene resolución para paneles chicos.
3. **Congelar todo el extractor empeora** en las dos redes. Ajustar las features de COCO vale la pena.
4. **Generalizar a fuentes nuevas sigue siendo difícil**: el fold 3 no pasa de 0.20. Y la ResNet50 genera muchos falsos positivos en imágenes sin paneles, salvo que se le suba el umbral.
