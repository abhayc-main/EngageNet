import cv2
import torch
import numpy as np
import math

# SSL certificate issues fix
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def detect_head_centers(image_path):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Read image
    image = cv2.imread(image_path)

    # Get image height and width
    image_height, image_width = image.shape[:2]

    # Perform head detection
    results = model(image)

    # Retrieve bounding boxes and class labels
    boxes = results.xyxy[0].cpu().numpy()
    class_labels = results.names[0]

    if 'person' not in class_labels:
        print("No person class detected in the image.")
        return [], image_height, image_width

    # Filter detections to keep only "person" class
    person_boxes = boxes[results.pred[0][:, -1] == class_labels.index('person')]

    if len(person_boxes) == 0:
        print("No person found in the image.")
        return [], image_height, image_width

    head_centers = []
    # Draw bounding boxes on the image and get head centers
    for box in person_boxes:
        x1, y1, x2, y2, _, __ = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Calculate center coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        head_centers.append((center_x, center_y))

    cv2.imshow('Detection', image)
    cv2.waitKey(0)

    print("Head Centers:", head_centers)
    return head_centers, image_height, image_width


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

# Calculate the proximity score using the new function

head_positions, image_height, image_width = detect_head_centers("./data/unengaged.jpg")
median_proximity = calculate_median_proximity(head_positions, image_width, image_height)
print(median_proximity)








