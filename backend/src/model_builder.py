# src/model_builder.py

from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

def build_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    return model, base_model
