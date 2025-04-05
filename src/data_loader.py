# src/data_loader.py

import tensorflow as tf
from .config import IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR

def load_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='binary',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='binary',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False
    )

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), test_ds.prefetch(AUTOTUNE)
