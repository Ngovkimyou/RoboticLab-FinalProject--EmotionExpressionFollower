# Improved data pipeline for RGB images + Pi-camera simulation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import random
import io
from PIL import Image, ImageFilter, ImageEnhance

IMG_SIZE = 224

# ===============================
# PI CAMERA SIMULATION FUNCTIONS
# ===============================

def apply_pi_simulation(img):
    pil = Image.fromarray(img.astype('uint8'))
    pil = pil.resize((640, 480), Image.BILINEAR)

    pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.6, 1.2))
    pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.7, 1.3))
    pil = ImageEnhance.Color(pil).enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.6:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.8)))

    buffer = io.BytesIO()
    pil.save(buffer, format='JPEG', quality=random.randint(40, 80))
    buffer.seek(0)
    pil = Image.open(buffer)

    arr = np.array(pil).astype(np.float32)

    noise = np.random.normal(0, random.uniform(5, 20), arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))  # keep float32

    return arr

def custom_preprocess(img):
    """Wrapper needed for ImageDataGenerator"""
    return apply_pi_simulation(img)


# =====================================
# DATA GENERATORS
# =====================================

def get_data_generators(train_dir, test_dir, batch_size=32):

    # ============================
    # TRAIN AUGMENTATION (WITH PI SIM)
    # ============================
    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocess,
        rescale=1.0/255,       # after custom_preprocess
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # ============================
    # TEST SET (NO PI SIM)
    # ============================
    test_datagen = ImageDataGenerator(
        rescale=1.0/255
    )

    # Training data
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True
    )

    # Testing data
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, test_gen, train_gen.samples, test_gen.samples
