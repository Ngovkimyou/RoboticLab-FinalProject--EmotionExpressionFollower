import tensorflow as tf

# Load your fine-tuned model
model = tf.keras.models.load_model(
    r"C:\Users\ASUS\OneDrive\Desktop\robo-project\models\mobilenet_boo.h5"
)

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert model
tflite_model = converter.convert()

# Save TFLite file
output_path = r"C:\Users\ASUS\OneDrive\Desktop\robo-project\models\mobilenet_boo.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

print("🎉 TFLite model saved:", output_path)
