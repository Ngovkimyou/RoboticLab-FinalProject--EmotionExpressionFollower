import os
import tensorflow as tf
from data_pipeline import get_data_generators
from model_mobilenet import build_model

# Mixed precision improves training on GPU (optional, safe)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Paths
train_path = os.path.join("data", "train")
test_path  = os.path.join("data", "test")

# Generators
train_gen, test_gen, train_count, test_count = get_data_generators(
    train_path, test_path, batch_size=32
)

# Build model (already compiled inside build_model)
model = build_model()

# Training settings
EPOCHS = 40
STEPS_PER_EPOCH = train_count // 32
VALIDATION_STEPS = None   # evaluate full test set

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/mobilenet_best_{val_accuracy:.3f}.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

print("🚀 Starting fine-tuning...")

history = model.fit(
    train_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=test_gen,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# Save final model
model.save("models/mobilenet_picam.keras")
print("\n🎉 Fine-tuning complete: models/mobilenet_picam.keras")
