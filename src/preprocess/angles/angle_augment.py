import cv2
import os
import numpy as np

# Function to rotate image
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def generate_images(directory_path):
    # Get all the image paths in the directory
    image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Define the list of angles
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    # For each image path
    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)
        
        # Check if the image is empty
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        # For each angle, rotate the image and save it
        for angle in angles:
            # Rotate the image and save it
            rotated = rotate_image(image, angle)
            if not os.path.exists(f'./data/angle_data/{angle}'):
                os.makedirs(f'./data/angle_data/{angle}')
            cv2.imwrite(f'./data/angle_data/{angle}/{os.path.basename(image_path)}', rotated)

generate_images('./data/angle_data/0')
