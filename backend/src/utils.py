# src/utils.py

import tensorflow as tf
from .config import MODEL_PATH

def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]
