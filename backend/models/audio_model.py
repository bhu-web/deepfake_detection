from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# Load preprocessed data
with open("X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(13,)),  # 13 MFCC features
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save(r"C:\Users\archi\deepfakedetectionaudio\saved_models\audio_model.h5")
print("Model saved!")