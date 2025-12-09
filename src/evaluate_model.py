import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from data_pipeline import get_data_generators

# ===============================
# CONFIG
# ===============================
MODEL_PATH = r"models/mobilenet_boo.h5"   # <-- change if needed
TEST_DIR   = r"data/test"

# Emotion labels (must match your folder order)
EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise"]


# ===============================
# LOAD MODEL
# ===============================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)


# ===============================
# LOAD TEST SET
# ===============================
print("Loading test set...")
_, test_gen, _, test_count = get_data_generators("data/train", TEST_DIR, batch_size=32)

# We need arrays
y_true = []
y_pred = []

print("Running inference on test set...")

for i in range(test_count // 32):
    batch_x, batch_y = next(test_gen)

    pred = model.predict(batch_x)
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(batch_y, axis=1)

    y_pred.extend(pred_labels)
    y_true.extend(true_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ===============================
# CONFUSION MATRIX + REPORT
# ===============================
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=EMOTIONS))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))
