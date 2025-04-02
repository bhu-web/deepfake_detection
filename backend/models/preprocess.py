import os
import librosa
import numpy as np

# Extract MFCC features
def extract_mfcc(file_path, n_mfcc=13):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Process a folder of audio files
def process_folder(folder_path, label):
    """Extract MFCC features from all audio files in the folder."""
    features = []
    labels = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    mfcc = extract_mfcc(file_path)
                    features.append(mfcc)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return features, labels

# Prepare the dataset
def prepare_data(real_base_path, fake_base_path):
    """Prepare dataset by processing real and fake audio files."""
    real_features, real_labels = [], []
    fake_features, fake_labels = [], []

    for subset in ['train', 'test', 'dev']:
        real_subset_path = os.path.join(real_base_path, subset)
        fake_subset_path = os.path.join(fake_base_path, subset)

        print(f"Processing {subset} - Real...")
        real_f, real_l = process_folder(real_subset_path, 0)  # Label 0 for real
        real_features.extend(real_f)
        real_labels.extend(real_l)

        print(f"Processing {subset} - Fake...")
        fake_f, fake_l = process_folder(fake_subset_path, 1)  # Label 1 for fake
        fake_features.extend(fake_f)
        fake_labels.extend(fake_l)

    # Combine real and fake data
    X = np.array(real_features + fake_features)
    y = np.array(real_labels + fake_labels)
    return X, y