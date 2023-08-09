import os
import shutil

# Define the source directories and the target directory
source_dirs = ['./data/Overhead Angle Data/test', './data/Overhead Angle Data/train', './data/Overhead Angle Data/valid']
target_dir = './data/Overhead Angle Data/'

# Iterate through each source directory
for source_dir in source_dirs:
    # Check if the source directory exists
    if os.path.exists(source_dir):
        # Iterate through each class folder in the source directory
        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)
            # Ensure that it's a directory
            if os.path.isdir(class_path):
                # If the class directory doesn't exist in the target directory, create it
                if not os.path.exists(os.path.join(target_dir, class_name)):
                    os.makedirs(os.path.join(target_dir, class_name))
                # Move each image from the source class directory to the target class directory
                for filename in os.listdir(class_path):
                    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more conditions if you have other file types
                        shutil.move(os.path.join(class_path, filename), os.path.join(target_dir, class_name, filename))