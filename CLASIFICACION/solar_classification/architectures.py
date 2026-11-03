from __future__ import annotations


def build_model(name: str, strategy: str, image_size: int, dropout: float = 0.2):
    import tensorflow as tf
    from tensorflow import keras

    constructors = {
        "efficientnetb0": (keras.applications.EfficientNetB0,
                           keras.applications.efficientnet.preprocess_input),
        "mobilenetv2": (keras.applications.MobileNetV2,
                        keras.applications.mobilenet_v2.preprocess_input),
    }
    if name not in constructors:
        raise ValueError(f"Arquitectura desconocida: {name}")
    if strategy not in {"random", "head", "fine_tune"}:
        raise ValueError(f"Estrategia desconocida: {strategy}")
    constructor, preprocess = constructors[name]
    weights = None if strategy == "random" else "imagenet"
    base = constructor(include_top=False, weights=weights,
                       input_shape=(image_size, image_size, 3))
    base.trainable = strategy == "fine_tune"
    inputs = keras.Input((image_size, image_size, 3))
    x = keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = keras.layers.Lambda(preprocess, name="preprocess")(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name=f"{name}_{strategy}")


def compile_model(model, learning_rate: float):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
