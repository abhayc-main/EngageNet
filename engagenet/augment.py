"""
Code to Augment the data into different angles for the CNN to learn from
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


# Function to rotate image
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# Function to calculate brightness of an image
def calculate_brightness(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.average(grayscale)

# Function to process image
## Function to process image
def process_image(image_path, brightness_threshold=50):
    image = cv2.imread(image_path)

    brightness = calculate_brightness(image)
    if brightness < brightness_threshold:
        return

    # Resize the image before sending it to the API
    resize_image(image_path)

    boxes, image_width, image_height, person_boxes = detect_head_centers(image_path)

    for i, box in enumerate(boxes):
        x, y, width, height = box
        center_x = x
        center_y = y

        new_width = width * 1.5  # Increase the bounding box size
        new_height = height * 1.5  # Increase the bounding box size

        x_start = max(0, int(center_x - new_width / 2))
        x_end = min(image_width, int(center_x + new_width / 2))
        y_start = max(0, int(center_y - new_height / 2))
        y_end = min(image_height, int(center_y + new_height / 2))

        cropped = image[int(y_start):int(y_end), int(x_start):int(x_end)]
        if cropped.size != 0:  # Check if the cropped region is not empty
            for angle in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
                rotated = rotate_image(cropped, angle)
                if not os.path.exists(f'./angle_data/{angle}'):
                    os.makedirs(f'./angle_data/{angle}')
                cv2.imwrite(f'./angle_data/{angle}/image_{i}.jpg', rotated)




# Directory containing images
image_directory = './data/test/'

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]

# Process each image file
for image_file in tqdm(image_files):
    process_image(os.path.join(image_directory, image_file))
