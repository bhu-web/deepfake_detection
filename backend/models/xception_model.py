import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from load_dataset import train_data, test_data  # Import dataset

IMG_SIZE = (128, 128, 3)

# Load Xception model with smaller input size
base_model = Xception(weights=None, include_top=False, input_shape=IMG_SIZE)

# Add classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
output_layer = Dense(1, activation="sigmoid")(x)  # Binary classification

# Create model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile model with reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.00001), loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Callbacks: Reduce LR, Early Stopping, and Save Best Model
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),  
    ModelCheckpoint("best_xception_model.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)  # Stop early if no improvement
]

# Train model
model.fit(train_data, epochs=10, validation_data=test_data, callbacks=callbacks)

# Save model
model.save("xception_model.keras")

print("✅ Model training complete and saved as 'xception_model.keras'")
