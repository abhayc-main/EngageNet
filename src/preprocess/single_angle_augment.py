"""
Code to Augment the data into different angles for the CNN to learn from

Assumes all the images are facing 0 (down)
"""

import cv2
import os
import numpy as np

# Function to rotate image
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def generate_images(directory_path, target_directory_path, rotation_angle):
    # Get all the image paths in the directory
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    if not os.path.exists(target_directory_path):
        os.makedirs(target_directory_path)

    # For each image path
    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)
        
        # Check if the image is empty
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        # Rotate the image and save it
        rotated = rotate_image(image, rotation_angle)
        cv2.imwrite(os.path.join(target_directory_path, os.path.basename(image_path)), rotated)

source_directory = './data/angle_data/0'
target_directory = './data/angle_data/180'
rotation_angle = 180
generate_images(source_directory, target_directory, rotation_angle)
