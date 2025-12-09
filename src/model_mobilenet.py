# Improved MobileNetV2 model for personal emotion dataset
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
NUM_CLASSES = 5

def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # MobileNetV2 preprocess (important for different camera qualities)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )(x)

    # Unfreeze last 50 layers
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    for layer in base.layers[:-50]:
        layer.trainable = False
    for layer in base.layers[-50:]:
        layer.trainable = True

    # Replace base_model with the actual functional API output
    x = base_model

    # Custom head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
