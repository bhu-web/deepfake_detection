import os, random, shutil

# Paths
train_real = r"C:\Users\bhoom\datasets\deepfake_detection\train\real"
train_fake = r"C:\Users\bhoom\datasets\deepfake_detection\train\fake"
test_real = r"C:\Users\bhoom\datasets\deepfake_detection\test\real"
test_fake = r"C:\Users\bhoom\datasets\deepfake_detection\test\fake"

output = "dataset_reduced"
os.makedirs(output, exist_ok=True)

# Target sizes
train_size, test_size = 1000, 500  # Per class

# Function to copy sampled images
def sample_and_copy(src, dst, num):
    os.makedirs(dst, exist_ok=True)
    for img in random.sample(os.listdir(src), num):
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))

# Reduce dataset
sample_and_copy(train_real, f"{output}/train/real", train_size)
sample_and_copy(train_fake, f"{output}/train/fake", train_size)
sample_and_copy(test_real, f"{output}/test/real", test_size)
sample_and_copy(test_fake, f"{output}/test/fake", test_size)

print("Dataset reduced successfully.")