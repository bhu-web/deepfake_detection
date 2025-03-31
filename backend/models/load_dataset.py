import os
import tensorflow as tf

# Define dataset paths
TRAIN_REAL_PATH = r"C:\Users\bhoom\datasets\deepfake_detection\train\real"
TRAIN_FAKE_PATH = r"C:\Users\bhoom\datasets\deepfake_detection\train\fake"
TEST_REAL_PATH = r"C:\Users\bhoom\datasets\deepfake_detection\test\real"
TEST_FAKE_PATH = r"C:\Users\bhoom\datasets\deepfake_detection\test\fake"

IMG_SIZE = (128, 128)  
BATCH_SIZE = 8        

# ✅ Define Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  
    tf.keras.layers.RandomRotation(0.1),  
    tf.keras.layers.RandomZoom(0.1),
])

# Function to load and preprocess images
def load_and_preprocess(image_path, label, augment=False):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  
    image = tf.image.resize(image, IMG_SIZE)        
    image = image / 255.0  # Normalize to [0,1]

    # Apply augmentation only to training data
    if augment:
        image = data_augmentation(image, training=True)  # ✅ Fix: Pass training=True

    return image, label

# Function to create dataset
def create_dataset(real_path, fake_path, num_samples, augment=False):
    real_images = [os.path.join(real_path, img) for img in os.listdir(real_path)[:num_samples]]
    fake_images = [os.path.join(fake_path, img) for img in os.listdir(fake_path)[:num_samples]]
    
    image_paths = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)  

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(len(image_paths))
    
    # ✅ Explicitly pass augment=augment
    dataset = dataset.map(lambda x, y: load_and_preprocess(x, y, augment=augment), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # ✅ Optimize loading
    return dataset

# ✅ Enable augmentation for training data
train_data = create_dataset(TRAIN_REAL_PATH, TRAIN_FAKE_PATH, 5000, augment=True)  
test_data = create_dataset(TEST_REAL_PATH, TEST_FAKE_PATH, 2000, augment=False)

print("✅ Dataset loaded and preprocessed successfully!")