from tensorflow.keras.models import load_model
from load_dataset import test_data  # Ensure test_data is imported

model = load_model("xception_model.keras")
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.4f} - Test Loss: {test_loss:.4f}")