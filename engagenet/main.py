import cv2
import torch
import numpy as np
import math

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

    resize_image(image_path)

    # Run the model prediction
    prediction = model.predict(image_path, confidence=40, overlap=30).json()

    # Get the image width and height from the prediction dictionary
    image_width = int(prediction['image']['width'])
    image_height = int(prediction['image']['height'])

    # Initialize an empty list to hold the center coordinates
    centers = []

    # Iterate over each detection in the predictions
    for obj in prediction['predictions']:
        # Calculate the center of the bounding box
        x_center = (obj['x'] + obj['width']) / 2
        y_center = (obj['y'] + obj['height']) / 2

        # Append the center coordinates to the list
        centers.append((x_center, y_center))

    person_boxes = len(centers)

    # Convert the list to a NumPy array for easier manipulation
    centers = np.array(centers)

    return centers, image_width, image_height, person_boxes

def calculate_median_proximity(head_centers, image_width, image_height, head_count):
    # Check if no head centers were detected
    if len(head_centers) == 0:
        print("No head centers found.")
        return 0

    # Initialize a list to store distances
    distances = []

    # Calculate pairwise Euclidean distance for all head centers
    for i in range(len(head_centers)):
        for j in range(i+1, len(head_centers)):
            # Calculate Euclidean distance
            distance = np.sqrt((head_centers[i][0] - head_centers[j][0])**2 + (head_centers[i][1] - head_centers[j][1])**2)
            
            # Add this distance to the distances list
            distances.append(distance)

    # Calculate median distance
    if len(distances) > 0:
        median_distance = np.median(distances)
    else:
        print("Only one person detected, cannot calculate proximity.")
        return 0

    # Normalize the median distance using the diagonal length of the image (max possible distance)
    max_possible_distance = np.sqrt(image_width**2 + image_height**2)
    normalized_distance = median_distance / max_possible_distance

    # Higher score indicates people are closer
    score = 1 - normalized_distance

    print(score)

    normalized_count = min(head_count / 100, 1)

    engagement_score = 0.1 * normalized_count + 0.9 * score
    print("Engagement Score: ", engagement_score)

    return engagement_score   

# Calculate the proximity score using the new function

# Calculate the proximity score using the updated functions
head_positions, image_width, image_height, numppl= detect_head_centers("./data/slightangle.jpeg")

median_proximity = calculate_median_proximity(head_positions, image_width, image_height, numppl)

