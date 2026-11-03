# Clasificación de paneles solares

Implementa la parte de clasificación de la consigna con **EfficientNetB0** y
**MobileNetV2**, usando los tres splits por fuente de `data/yolo/`.

El flujo recomendado es el notebook único `entrenamiento_colab.ipynb`: ejecuta el
chequeo, permite seleccionar la grilla, entrena y termina con tablas y gráficos
comparativos por arquitectura, estrategia y fuente de validación.

Desde la raíz, el chequeo local no carga modelos:

```powershell
python CLASIFICACION\train.py
```

En Colab/GPU, seleccionar una grilla o ejecutar todo:

```powershell
python CLASIFICACION\train.py --train --architectures efficientnetb0 mobilenetv2
```

Las estrategias son `random` (pesos aleatorios), `head` (ImageNet con extractor
congelado) y `fine_tune` (ImageNet con extractor entrenable). Cada experimento
guarda `best.keras`, `metrics.json`, `history.json` y un registro en
`results/metrics.csv`.

## Estructura

- `entrenamiento_colab.ipynb`: flujo completo recomendado: verifica datos, ejecuta
  experimentos y genera comparaciones.
- `train.py`: entrada de terminal para verificar o lanzar la misma grilla.
- `requirements.txt`: dependencias de TensorFlow, scikit-learn, pandas y gráficos.
- `solar_classification/config.py`: arquitecturas, estrategias, splits e hiperparámetros.
- `solar_classification/data.py`: lectura de imágenes/etiquetas y construcción de
  datasets de TensorFlow.
- `solar_classification/architectures.py`: modelos EfficientNetB0/MobileNetV2 y
  compilación.
- `solar_classification/engine.py`: orquestación de los entrenamientos.
- `solar_classification/evaluation.py`: accuracy, precision, recall, F1 y matriz de
  confusión.
- `solar_classification/storage.py`: guardado de configuración y métricas.
