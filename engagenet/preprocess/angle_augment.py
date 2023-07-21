"""
Code to Augment the data into different angles for the CNN to learn from

Assumes all the images are facing 0 (down)
"""

import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# SSL certificate issues fix
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import Image
import numpy as np
from roboflow import Roboflow

def resize_image(image_path):
    img = Image.open(image_path)

    # Specify the maximum dimensions for resizing
    max_width = 800
    max_height = 800

    # Calculate the aspect ratio
    width_ratio = max_width / img.width
    height_ratio = max_height / img.height

    # Choose the smaller ratio to ensure the image fits within the desired dimensions
    resize_ratio = min(width_ratio, height_ratio)

    # Calculate the new width and height
    new_width = int(img.width * resize_ratio)
    new_height = int(img.height * resize_ratio)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Save the resized image
    resized_img.save(image_path)

def detect_head_centers(image_path):
    # Initialize Roboflow
    rf = Roboflow(api_key="K9A8vdmXXNpdi3lmHVBI")
    project = rf.workspace().project("people_counterv0")
    model = project.version(1).model

    #resize_image(image_path)

    # Run the model prediction
    prediction = model.predict(image_path, confidence=40, overlap=30).json()

    print(prediction)

    # Get the image width and height from the prediction dictionary
    image_width = int(prediction['image']['width'])
    image_height = int(prediction['image']['height'])

    # Initialize an empty list to hold the bounding boxes
    boxes = []

    # Iterate over each detection in the predictions
    for obj in prediction['predictions']:
        # Get the bounding box coordinates
        x = obj['x']
        y = obj['y']
        width = obj['width']
        height = obj['height']

        # Append the bounding box coordinates to the list
        boxes.append((x, y, width, height))

    person_boxes = len(boxes)

    # Convert the list to a NumPy array for easier manipulation
    boxes = np.array(boxes)

    return boxes, image_width, image_height, person_boxes


import cv2
import os
import numpy as np
from PIL import Image

# Function to rotate image
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def generate_images(directory_path, current_angle):
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
            if angle == current_angle:
                # If the angle is the current angle, save the original image
                if not os.path.exists(f'./data/angle_data/{angle}'):
                    os.makedirs(f'./data/angle_data/{angle}')
                cv2.imwrite(f'./data/angle_data/{angle}/{os.path.basename(image_path)}', image)
            else:
                # If the angle is not the current angle, rotate the image and save it
                rotated = rotate_image(image, angle - current_angle)
                if not os.path.exists(f'./data/angle_data/{angle}'):
                    os.makedirs(f'./data/angle_data/{angle}')
                cv2.imwrite(f'./data/angle_data/{angle}/{os.path.basename(image_path)}', rotated)

generate_images('./data/main/', 180)


