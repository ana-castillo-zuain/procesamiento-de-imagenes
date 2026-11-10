# Detección de paneles solares con YOLO

Comparación de tres pesos preentrenados y tres formas de congelar capas, medida con tres folds que dejan fuentes enteras afuera del entrenamiento. Son 27 entrenamientos. La métrica principal es el promedio macro de mAP50-95: cada fold pesa un tercio, sin importar cuántas imágenes o cajas tenga.

Este directorio no entrena nada al importarse. `python train.py` sin `--train` solo revisa archivos y tablas. Los pesos se descargan recién cuando un entrenamiento arranca en Colab.

## Cómo ejecutarlo en Google Colab

1. Subí el repositorio a Google Drive, o cloná el repo dentro de la sesión y apuntá `RESULTS_DIR` a una carpeta de Drive.
2. Abrí `YOLO/entrenamiento_colab.ipynb`.
3. Editá las tres rutas del notebook: código, dataset y resultados. No hay una ruta de Drive fija en el código.
4. Corré las celdas en orden. La celda de verificación no carga modelos. La última sí entrena.

El notebook llama a `run_experiments`. No hace falta editar `train.py`.

Para un solo experimento, en la celda de selección:

```python
SELECTED_ARCHITECTURES = ["yolo11n"]   # yolov8n, yolo11n, yolo26n
SELECTED_STRATEGIES = ["partial"]      # full, backbone, partial
SELECTED_FOLDS = [1]                   # 1, 2, 3
```

`None` en cualquiera de las tres usa todas las opciones.

Desde una terminal, con el dataset ya montado:

```bash
python train.py --check
# equivalente: python train.py   (sin --train tampoco carga modelos)
python train.py --train --architectures yolo11n --strategies partial --folds 1 --batch 16
```

`--results` tiene que ser una carpeta de Drive si la sesión de Colab se puede cerrar. `--batch` es el único parámetro pensado para tocarse según la memoria de la GPU. El resto de la configuración está en `solar_yolo/config.py` y es común a los 27.

## Reanudar

Un experimento se considera terminado solo si `results/logs/<id>.json` dice `completed` y existe su `best.pt`. Esos no se vuelven a entrenar.

Si el entrenamiento llegó a guardar `best.pt` pero falló la evaluación por fuente, la siguiente corrida no reentrena: solo rehace métricas y predicciones.

Si quedó un `last.pt` de una corrida cortada, Ultralytics reanuda ese run. Una corrida fallida sin `last.pt` se archiva a `runs/<id>__archived_<fecha>` para no mezclar su `results.csv` con el intento nuevo.

`--rerun` ignora los experimentos ya terminados. Antes de pisar un run incompleto, el directorio anterior se archiva. El `best.pt` de un experimento completado no se borra solo: para repetirlo hay que usar `--rerun` o borrar ese peso y su marcador.

## Resultados

```text
results/
├── experiments.csv      # una fila por entrenamiento
├── summary.csv          # una fila por arquitectura × estrategia
├── fold_metrics.csv     # fold completo y, aparte, cada fuente
├── config.json
├── logs/
├── weights/
│   ├── yolov8n/{full,backbone,partial}/fold{1,2,3}/best.pt
│   ├── yolo11n/...
│   └── yolo26n/...
├── predictions/
└── plots/
```

`summary.csv` deja vacío `macro_map50_95` hasta que los tres folds de esa combinación terminaron bien. No promedia uno o dos folds y lo presenta como el score. El score no mezcla las predicciones de distintos folds.

`experiments.csv` guarda arquitectura, estrategia, fold, índices congelados, hiperparámetros, mAP50, mAP50-95, precision, recall, fitness, mejor época, tiempo y la ruta del `best.pt`.

## Congelamiento

Ultralytics numera la lista top-level del YAML, no las subcapas que imprime `model.info()`. `freeze=10` congela los índices 0 a 9. El mismo entero no corta en el mismo lugar de las tres redes, así que cada estrategia guarda la lista de índices de esa arquitectura. Al cargar el `.pt`, antes de entrenar, se compara el grafo real contra esta tabla. Si no coincide, ese experimento se aborta y los demás siguen.

Los pesos son siempre `yolov8n.pt`, `yolo11n.pt` y `yolo26n.pt`.

### Full fine-tuning

No se congela ninguna capa. Ultralytics deja congelados los parámetros `.dfl` cuando existen (YOLOv8n y YOLO11n, `reg_max=16`). YOLO26n usa `reg_max=1`: su DFL es `Identity` y no tiene parámetros.

### Backbone congelado

Se congela todo el bloque `backbone` del YAML y se entrenan el cuello y la cabeza.

| Arquitectura | Índices congelados | Última capa congelada | Qué queda entrenable |
|---|---|---|---|
| yolov8n | 0–9 (10 capas) | 9 SPPF | desde el Upsample 10 hasta Detect 22 |
| yolo11n | 0–10 (11 capas) | 10 C2PSA | desde el Upsample 11 hasta Detect 23 |
| yolo26n | 0–10 (11 capas) | 10 C2PSA | desde el Upsample 11 hasta Detect 23 |

YOLOv8n no tiene C2PSA. Usar 11 capas en YOLOv8n congelaría el primer Upsample del cuello. Usar 10 en YOLO11n o YOLO26n dejaría C2PSA entrenable, y el cuello de esas dos redes lee justamente la capa 10, no el SPPF de la 9.

YOLO11n y YOLO26n comparten esa numeración y no la misma red:

- El SPPF de YOLO26n (capa 9) tiene atajo residual. El de YOLO11n no.
- Los C3k2 del cuello de YOLO26n usan C3k. En YOLO11n las capas 13, 16 y 19 usan Bottleneck.
- La capa 22 de YOLO26n agrega `PSABlock`. La de YOLO11n es C3k sin esa atención.
- La cabeza de YOLO26n es dual: one-to-many y one-to-one (`one2one_cv2`). YOLO11n y YOLOv8n no la tienen.

Esas firmas se comprueban sobre el módulo cargado. La cabeza, incluida la one-to-one de YOLO26n, se entrena en las tres estrategias.

### Fine-tuning parcial

Se congelan solo el stem y el estadio P2: las dos convoluciones stride 2 y el primer bloque CSP. Se entrena desde el downsample a P3/8 (capa 3) en adelante, incluido el resto del backbone, el cuello y la cabeza.

| Arquitectura | Índices | Módulos congelados |
|---|---|---|
| yolov8n | 0, 1, 2 | Conv P1/2, Conv P2/4, C2f |
| yolo11n | 0, 1, 2 | Conv P1/2, Conv P2/4, C3k2 |
| yolo26n | 0, 1, 2 | Conv P1/2, Conv P2/4, C3k2 |

El índice 2 coincide, el bloque no: C2f en YOLOv8n y C3k2 en las otras dos. La verificación del peso rechaza el entrenamiento si esa clase no aparece donde dice la tabla.

## Folds

La fuente no está en el nombre de la carpeta. Está en el nombre de cada archivo:

- `PV01...` → PV01
- `solarpanels_hd...` → solarpanels_hd
- `kingston2...` → kingston2 (se prueba antes que kingston)
- `kingston_...` → kingston
- `..._unknown_...` → geo_unknown

Cualquier otro nombre aborta la corrida. geo_unknown no se usa como cajón de resto.

`data/yolo` ya tiene los tres folds, armados en el preprocesamiento. Cada carpeta es la validación, y el train es el resto de las fuentes. Las tres tienen el mismo layout (`images/{train,val}`, `labels/{train,val}`, `data.yaml`) y las mismas 900 imágenes, repartidas distinto. No se copian de nuevo. El código escribe un YAML con rutas absolutas dentro de `results/datasets/` y deja los originales intactos.

| Fold | Carpeta | Train | Validación |
|---|---|---|---|
| 1 | `PV01_val` | solarpanels_hd, geo_unknown, kingston, kingston2 (422) | PV01 (478, 3556 cajas) |
| 2 | `solarpanels_val` | PV01, geo_unknown, kingston, kingston2 (584) | solarpanels_hd (316, 500 cajas) |
| 3 | `geo_kingston_val` | PV01, solarpanels_hd (794) | geo_unknown + kingston + kingston2 (106, 79 cajas) |

Antes de entrenar se comprueba que train y val no compartan imágenes ni fuentes, que cada fuente quede entera de un solo lado, que cada imagen tenga su `.txt`, y que las cajas sean clase 0 con coordenadas en `[0, 1]`. Un archivo vacío es una imagen negativa. También se exigen estos totales, medidos sobre este dataset:

| Fuente | Imágenes | Positivas | Negativas | Cajas |
|---|---:|---:|---:|---:|
| PV01 | 478 | 477 | 1 | 3556 |
| solarpanels_hd | 316 | 284 | 32 | 500 |
| geo_unknown | 44 | 10 | 34 | 13 |
| kingston | 34 | 34 | 0 | 66 |
| kingston2 | 28 | 0 | 28 | 0 |

Si algún número no coincide, no se reescribe el fold: se informa el desvío y no se entrena.

El fold 3 guarda la métrica del conjunto combinado y, además, una validación aparte de geo_unknown y otra de kingston. Esas validaciones leen las imágenes que ya están en `images/val`; no duplican archivos. kingston2 no entra en ese cálculo de AP.

## Métricas

Por cada entrenamiento se registra mAP50, mAP50-95, precision, recall, fitness (`0.1 * mAP50 + 0.9 * mAP50-95`), la mejor época y el tiempo. Esas cifras salen de la fila de `results.csv` que corresponde al `best.pt`, no de la última época si hubo early stopping (`patience=15`).

El fold 3 tiene 79 cajas. Su mAP entra al promedio con el mismo peso que los otros dos, y no hay que leerlo como una estimación precisa. No se arma un mAP único juntando las predicciones de los tres folds.

kingston2 solo está en la validación del fold 3 y no tiene objetos. No se calcula AP ni recall para esa fuente. Después de predecir el fold, se cuentan imágenes con al menos una detección y la cantidad de cajas, con confianza 0.25. El mAP de Ultralytics sigue usando su umbral de validación (0.001). Las dos cuentas están en `fold_metrics.csv` y no se mezclan.

## Configuración común

En `solar_yolo/config.py`:

- pesos preentrenados `.pt`
- `epochs=100`
- `imgsz=256`
- `batch=16`
- `seed=42` y `deterministic=True`
- `patience=15`
- `optimizer=auto`

Con estos tamaños, `auto` elige AdamW: Ultralytics pasa a MuSGD recién arriba de 10000 iteraciones, y la cuenta es `ceil(n_train / max(batch, 64)) * epochs`. Los tres folds quedan cerca de 700, 1000 y 1300 iteraciones. Si un cambio de epochs o batch cruza el umbral, la corrida se aborta antes de entrenar, porque si no un fold usaría otro optimizador. En la rama AdamW, Ultralytics pisa `lr0` con `0.002 * 5 / (4 + nc)`. Con una clase, eso es 0.002, y pone `warmup_bias_lr=0`.

`requirements.txt` pide `ultralytics>=8.4.198` porque `yolo26n.pt` no está en versiones anteriores. El resto de los argumentos que no están en `config.py` quedan en el default de la versión instalada. Esa versión se guarda en cada fila. No hay búsqueda de hiperparámetros.

Si la GPU no entra en memoria, Ultralytics puede bajar el batch durante la primera época. La fila lo anota en `batch_used` y en `warning`. Conviene fijar un batch que entre y repetir el experimento, porque uno con el batch reducido no es comparable.

## Visualización

Notebook local (no Colab): con `results/` en `YOLO/results/`, abrí `visualizacion.ipynb`. Lee CSV, predicciones y plots (sin cargar modelos): detección (mAP, PR de cajas), clasificación imagen tiene/no tiene panel (TP/FP/TN/FN, F1), curvas loss y matrices de confusión. Necesita también `data/yolo` para las etiquetas GT.

## Archivos

- `train.py`: entrada de línea de comandos.
- `entrenamiento_colab.ipynb`: monta Drive, instala, revisa CUDA y llama al motor.
- `visualizacion.ipynb`: métricas y plots a partir de `YOLO/results/`.
- `solar_yolo/config.py`: parámetros comunes.
- `solar_yolo/architectures.py`: capas y congelamiento.
- `solar_yolo/data.py`: fuentes, folds y YAML.
- `solar_yolo/engine.py`: la grilla de 27, reanudación y errores aislados.
- `solar_yolo/evaluation.py`: métricas, fitness y falsos positivos.
- `solar_yolo/storage.py`: CSV y `best.pt`.
