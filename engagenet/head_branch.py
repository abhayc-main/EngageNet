import cv2
import torch
import numpy as np
import math

# SSL certificate issues fix
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from roboflow import Roboflow
import numpy as np

def detect_head_centers(image_path):
    # Initialize Roboflow
    rf = Roboflow(api_key="K9A8vdmXXNpdi3lmHVBI")
    project = rf.workspace().project("people_counterv0")
    model = project.version(1).model

    # Run the model prediction
    prediction = model.predict(image_path, confidence=40, overlap=30).json()

    # Initialize an empty list to hold the center coordinates
    centers = []

    # Iterate over each detection in the predictions
    for obj in prediction['objects']:
        # Calculate the center of the bounding box
        x_center = (obj['x'] + obj['width']) / 2
        y_center = (obj['y'] + obj['height']) / 2

        # Append the center coordinates to the list
        centers.append((x_center, y_center))

    # Convert the list to a NumPy array for easier manipulation
    centers = np.array(centers)

    return centers

# Test the function
image_path = "" 
head_centers = detect_head_centers(image_path)
print(head_centers)


# Define the function to calculate average proximity using the median of pairwise distances
def calculate_median_proximity(head_centers, image_width, image_height):
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
    return score

# Function to normalize head count
def normalize_head_count(head_count, max_heads=100):
    normalized_count = min(head_count / max_heads, 1)
    return normalized_count

# Calculate the proximity score and engagement score
head_positions, image_height, image_width, head_count = detect_head_centers("./data/group.jpg")
median_proximity = calculate_median_proximity(head_positions, image_width, image_height)
normalized_head_count = normalize_head_count(head_count)

engagement_score = 0.1 * normalized_head_count + 0.9 * median_proximity
print("Engagement Score: ", engagement_score)
