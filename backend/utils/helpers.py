from sklearn.model_selection import train_test_split
from backend.models.preprocess import prepare_data

# Define dataset paths
real_data_path = r"C:\Users\archi\deepfakedetectionaudio\processed_wav\PA"
fake_data_path = r"C:\Users\archi\deepfakedetectionaudio\processed_wav\LA"

# Prepare dataset
X, y = prepare_data(real_data_path, fake_data_path)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training Set Size:", X_train.shape)
print("Validation Set Size:", X_val.shape)
print("Testing Set Size:", X_test.shape)

# Save prepared datasets (optional)
import pickle
with open("X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)