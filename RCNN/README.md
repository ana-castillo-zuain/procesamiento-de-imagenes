# Detección de paneles solares con Faster R-CNN

Es la parte opcional *Familia R-CNN* de la consigna. Compara dos Faster R-CNN de torchvision y tres formas de congelar el extractor, con los mismos tres folds por fuente que usa YOLO. Son 18 entrenamientos. La métrica principal es la misma que en YOLO: el promedio macro de mAP50-95, donde cada fold pesa un tercio.

Este directorio no entrena nada al importarse. `python train.py` sin `--train` solo revisa archivos y tablas. Los pesos de COCO se descargan recién cuando arranca un entrenamiento.

## Qué comparte con YOLO

`solar_rcnn` importa de `YOLO/solar_yolo` el inventario de folds y sus verificaciones (`data.py`), las filas de métricas (`evaluation.py`) y los CSV y marcadores (`storage.py`). No los copia. Así las dos familias no pueden partir distinto el dataset, y `fold_metrics.csv` y `summary.csv` tienen las mismas columnas en las dos carpetas de resultados. Por eso hace falta el repositorio completo, no solo `RCNN/`.

El dataset es el mismo `data/yolo`. Las etiquetas YOLO se pasan a píxeles `x1 y1 x2 y2` al leerlas; no se escribe otro dataset.

## Cómo ejecutarlo en Google Colab

1. En Colab: *Archivo → Abrir cuaderno → GitHub*, buscá `ana-castillo-zuain/procesamiento-de-imagenes` y abrí `RCNN/entrenamiento_colab.ipynb`.
2. *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU*.
3. Corré las celdas en orden. La segunda clona el repo en `/content` (o hace `git pull` si ya estaba) y guarda los resultados en `MyDrive/paneles_resultados/RCNN`, que sobrevive si la sesión se corta. No hay nada para instalar: Colab ya trae torch y torchvision con CUDA. La celda de verificación no carga modelos. La de entrenamiento sí.
4. Al terminar, *Archivo → Descargar → .ipynb* guarda el notebook con las salidas.

Para un solo experimento:

```python
SELECTED_ARCHITECTURES = ["fasterrcnn_mobilenet_v3_large_fpn"]
SELECTED_STRATEGIES = ["partial"]   # full, backbone, partial
SELECTED_FOLDS = [1]                # 1, 2, 3
```

Desde una terminal:

```bash
python train.py
python train.py --train --architectures fasterrcnn_mobilenet_v3_large_fpn --strategies partial --folds 1
```

## Reanudar

Funciona igual que en YOLO. Un experimento se considera terminado solo si `results/logs/<id>.json` dice `completed` y existe su `best.pt`. Esos no se vuelven a entrenar.

Al final de cada época se guarda `runs/<id>/weights/last.pt`, con el modelo, el optimizador, el scheduler y el estado del early stopping. Si Colab se corta, la próxima corrida sigue desde la última época guardada. Si el entrenamiento terminó pero falló la evaluación, la siguiente corrida solo rehace las métricas.

`--rerun` ignora los experimentos ya terminados. Antes de pisar un run incompleto, el directorio anterior se archiva.

## Arquitecturas

Las dos parten de los pesos de COCO (`COCO_V1`). Se reemplaza `FastRCNNPredictor` por uno de 2 clases: fondo y panel, porque en torchvision la clase 0 es el fondo. Las dos reciben `min_size=256, max_size=256`, como pide la consigna. Si no, torchvision agranda las imágenes a 800 px.

| | MobileNetV3-Large FPN | ResNet50 FPN |
|---|---|---|
| Parámetros | 18.9 M | 41.3 M |
| Niveles que ve el FPN | features[13] y features[16], los dos con stride 32, más un max pool | layer1 a layer4 (strides 4, 8, 16 y 32), más un max pool |
| Grilla más fina con 256 px | 8 × 8 | 64 × 64 |

La MobileNet de torchvision está pensada para imágenes de 800 px y solo arma el FPN sobre las salidas de stride 32. Con 256 px trabaja sobre una grilla de 8 × 8, y las anclas más chicas son de 32 px. Mientras tanto, el 15 % de las cajas tiene algún lado de menos de 16 px. Hay que esperar que le cueste con los paneles chicos, y es algo para discutir en el informe.

## Congelamiento

torchvision congela con `trainable_backbone_layers`, que cuenta bloques del extractor desde el final; 0 congela todo el extractor. El mismo entero no corta en el mismo lugar de las dos redes. Por eso cada estrategia tiene su valor por arquitectura, elegido para que corte donde corta YOLO:

| Estrategia | Qué se congela | MobileNetV3 | ResNet50 |
|---|---|---|---|
| `full` | nada | `6` | `5` |
| `backbone` | todo el extractor. Se entrenan el FPN, la RPN y la cabeza ROI | `0` | `0` |
| `partial` | el stem y el estadio de stride 4 (P2). Se entrena desde la bajada a stride 8 | `4` → features[0..3] | `3` → conv1 y layer1 |

Antes de cargar pesos, `audit_architectures` comprueba que cada valor congele las unidades de la tabla, con la misma regla que usa torchvision. Al cargar el modelo, `assert_freeze_state` revisa el `requires_grad` de cada parámetro. Si algo no coincide, ese experimento se aborta y los demás siguen.

Tres detalles para el informe:

- El default de torchvision, 3, coincide con `partial` en la ResNet50, pero en la MobileNet congelaría hasta features[6], que está en stride 8.
- En las dos redes el batch norm del extractor es `FrozenBatchNorm2d`: torchvision lo congela siempre que parte de pesos preentrenados, también con `full`.
- En la MobileNet, `partial` congela solo 8 mil parámetros: los primeros bloques son muy chicos. La diferencia con `full` va a ser mínima.

## Entrenamiento

En `solar_rcnn/config.py`:

- `epochs=50`, `patience=10` sobre el fitness de validación
- `batch=8`, `imgsz=256`, `seed=42`
- SGD con `lr=0.01`, `momentum=0.9`, `weight_decay=1e-4`: es la receta de referencia de torchvision (0.02 para batch 16), escalada a batch 8
- warmup lineal de 200 iteraciones y después coseno hasta 0
- aumento: volteo horizontal y vertical. Son imágenes aéreas, así que no hay un "arriba" natural
- AMP en GPU

El `best.pt` es la época con mejor fitness (`0.1 * mAP50 + 0.9 * mAP50-95`), igual que en YOLO. Una época de validación es una predicción completa sobre el fold.

No es determinista bit a bit: `roi_align` no tiene backward determinista en CUDA.

## Métricas

torchvision no trae mAP, así que `solar_rcnn/metrics.py` implementa el de COCO para una clase: emparejamiento codicioso por score, AP interpolado en 101 puntos de recall, mAP50 y mAP50-95 sobre IoU 0.50 a 0.95. Se verificó contra `pycocotools` en datos sintéticos y da el mismo número (diferencias del orden de 1e-17). Solo usa numpy, así que se puede recalcular desde las predicciones guardadas sin GPU.

- Para medir mAP se predice con `score_thresh=0.001`, igual que la validación de Ultralytics. El default de torchvision, 0.05, recorta la cola de la curva precisión-recall.
- Precisión y recall se informan en el punto de F1 máximo, con IoU 0.5, como hace Ultralytics.
- Las métricas por fuente del fold 3 salen de las mismas predicciones, filtradas por imagen. kingston2 se mide con falsos positivos a confianza 0.25, igual que en YOLO.

El mAP de Ultralytics es parecido pero no idéntico al de COCO: empareja de otra forma. Al poner YOLO y Faster R-CNN en la misma tabla, diferencias de un punto o menos no son concluyentes.

### Detector como clasificador

La consigna pide medir el detector como clasificador. Cada imagen recibe como score su confianza máxima de detección (0 si no detecta nada). `classification_metrics.csv` tiene tres filas por experimento:

- `fabrica`: tiene paneles si alguna detección llega a 0.25.
- `calibrado`: el umbral que maximiza el F1 macro sobre esa misma validación. Es un diagnóstico, no una estimación de generalización.
- `trivial_todo_positivo`: la referencia de decir siempre que sí. Hay que superarla.

Cada fila trae matriz de confusión, accuracy, precision, recall, F1 y F1 macro. `predictions/<id>/image_scores.csv` guarda el score de cada imagen para calcular cualquier otra cosa.

## Resultados

```text
results/
├── experiments.csv            # una fila por entrenamiento
├── summary.csv                # una fila por arquitectura × estrategia
├── fold_metrics.csv           # fold completo y, aparte, cada fuente
├── classification_metrics.csv # detector como clasificador
├── config.json
├── logs/
├── weights/<arquitectura>/<estrategia>/fold{1,2,3}/best.pt
├── runs/<id>/results.csv      # una fila por época: losses, lr, mAP
├── predictions/<id>/
│   ├── labels/                # formato YOLO con confianza, desde 0.25
│   ├── raw.npz                # todas las cajas desde 0.001
│   └── image_scores.csv
└── plots/<id>/                # curva precisión-recall y curvas de entrenamiento
```

Los `best.pt` son un `state_dict` con metadatos. Para cargarlos: `build_model(arquitectura, estrategia, 256)` y después `model.load_state_dict(torch.load(ruta)["model"])`.

## Archivos

- `RESULTADOS.md`: tablas y análisis de las 18 corridas.
- `train.py`: entrada de línea de comandos.
- `entrenamiento_colab.ipynb`: monta Drive, revisa CUDA y llama al motor. Está guardado con las salidas de la corrida completa.
- `solar_rcnn/config.py`: parámetros comunes.
- `solar_rcnn/architectures.py`: modelos, cabeza y congelamiento.
- `solar_rcnn/dataset.py`: lee las etiquetas YOLO para torchvision.
- `solar_rcnn/metrics.py`: mAP de COCO, curva PR y detector como clasificador.
- `solar_rcnn/engine.py`: la grilla de 18, el bucle de entrenamiento, la reanudación y la evaluación.
