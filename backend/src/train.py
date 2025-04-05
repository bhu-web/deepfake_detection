# src/train.py

from .data_loader import load_datasets
from .model_builder import build_model
from .utils import get_callbacks
from .config import EPOCHS, MODEL_PATH

def main():
    train_ds, test_ds = load_datasets()
    model, base_model = build_model()

    print("[INFO] Starting initial training...")
    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=get_callbacks())

    print("[INFO] Fine-tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    model.fit(train_ds, validation_data=test_ds, epochs=5, callbacks=get_callbacks())

    model.save(MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
