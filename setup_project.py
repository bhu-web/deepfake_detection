import os
import shutil

# Define the new directory structure
new_structure = {
    "backend": ["routes", "models", "utils", "tests"],
    "data": ["raw", "processed"],
    "models": [],
    "frontend": ["components", "pages", "styles"]
}

# Root directory of your current project
root = os.getcwd()

def create_new_structure(base_path, structure):
    for folder, subfolders in structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        for sub in subfolders:
            os.makedirs(os.path.join(folder_path, sub), exist_ok=True)

def move_files():
    # Define specific rules for moving files (customize as needed)
    files_to_move = {
        "audio_text_model.py": "backend/models",
        "helpers.py": "backend/utils",
        "test_audio_model.py": "backend/tests",
    }

    for file, new_folder in files_to_move.items():
        src = os.path.join(root, file)
        dest_folder = os.path.join(root, new_folder)
        if os.path.exists(src) and os.path.exists(dest_folder):
            shutil.move(src, dest_folder)

# Create the new structure
print("Creating new directory structure...")
create_new_structure(root, new_structure)

# Move specific files
print("Organizing files into new structure...")
move_files()

print("Reorganization complete!")