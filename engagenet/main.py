import cv2
import threading
import numpy as np
import math
import time
import torch
from ultralytics import YOLO
import os

# SSL certificate issues fix
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import Image
import numpy as np

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import Input

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = YOLO("./models/yolotrainedruns/detect/train/weights/best.pt")
model.conf=0.40

# Export the model to ONNX format
model.export(format='onnx')
print("Successfully exported the model into .onnx format")

import onnxruntime as ort

def load_onnx_model(onnx_path):
    session = ort.InferenceSession(onnx_path)
    print("Loaded ONNX model")
    return session

onnx_path = './best.onnx'
ort_session = load_onnx_model(onnx_path)

# Load the head angle model
angle_model = tf.keras.models.load_model('./models/angle_algorithm_model_end')

# Initialize a lock object
lock = threading.Lock()

# Initialize global variables for the results
engagement_score = 0
n_clusters = 0
n_noise = 0
boxes = []

def detect_head_centers(frame, ort_session):
    # Run the ONNX model prediction on the frame
    inputs = {ort_session.get_inputs()[0].name: frame[None]}
    results = ort_session.run(None, inputs)

    # Extract the detections from the ONNX model results
    detections = results[0]

    image_height, image_width = frame.shape[:2]

    # Initialize empty lists to hold the center coordinates and boxes
    centers = []
    boxes = []

    # Iterate over the detections and calculate the centers and boxes
    for detection in detections:
        # Extract the bounding box coordinates
        x1, y1, x2, y2 = detection[:4]

        # Convert the box coordinates to the original image scale
        box = [x1 * image_width, y1 * image_height, (x2 - x1) * image_width, (y2 - y1) * image_height]

        # Calculate the center of the bounding box
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2

        # Append the center coordinates to the list
        centers.append((x_center, y_center))

        # Append the box coordinates to the list
        boxes.append(box)

    person_boxes = len(centers)

    # Convert the list to a NumPy array for easier manipulation
    centers = np.array(centers)

    print(centers)
    return centers, person_boxes, boxes, image_height, image_width

def preprocess_image(image):
    # Resize the image to the input size expected by the model
    image_resized = cv2.resize(image, (180, 180))

    # Normalize the pixel values to the range [0, 1]
    image_normalized = image_resized / 255.0

    # Convert the image to a format suitable for the model
    image = img_to_array(image_normalized)

    return image

def get_head_angle(image_path):
    # Load the model

    image = cv2.imread(image_path)

    # Preprocess the image for the model
    image = preprocess_image(image)

    # Convert the image to a format suitable for the model
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    angle = angle_model.predict(image)

    # Squeeze the angle array to remove unnecessary dimensions
    angle = angle.squeeze()

    return angle

# Function to calculate proximity score based on head positions
def calculate_proximity_score(head_centers, image_width, image_height):
    # Initialize a list to store distances
    distances = []
    
    # Loop over each pair of heads
    for i in range(len(head_centers)):
        for j in range(i+1, len(head_centers)):
            # Calculate Euclidean distance between the pair
            distance = np.sqrt((head_centers[i][0] - head_centers[j][0])**2 + (head_centers[i][1] - head_centers[j][1])**2)
            distances.append(distance)
            
    # Calculate median of the distances
    if len(distances) > 0:
        median_distance = np.median(distances)
    else:
        median_distance = 0
        
    # Calculate maximum possible distance (diagonal of the image)
    max_possible_distance = np.sqrt(image_width**2 + image_height**2)
    
    # Normalize the median distance to be between 0 and 1
    normalized_distance = median_distance / max_possible_distance if max_possible_distance > 0 else 0
    
    # Calculate proximity score as 1 - normalized_distance
    proximity_score = 1 - normalized_distance
    
    return proximity_score

def calculate_cluster_engagement(head_centers, head_angles):

    if len(head_centers) == 0:
        return 0, 0, 0

    # Standardize the head centers for the clustering algorithm
    scaler = StandardScaler()
    X = scaler.fit_transform(head_centers)
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=1.5, min_samples=2).fit(X)
    labels = clustering.labels_
    
    # Calculate the number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    engaged_groups = 0
    total_size = 0
    
    # Iterate over each cluster
    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) < 2:
            continue

        # Extract the angles and centers for the current cluster
        group_angles = head_angles[indices]
        group_centers = head_centers[indices]
        
        # Calculate the centroid of the current cluster
        centroid = np.mean(group_centers, axis=0)
        
        # Calculate the angles from each point to the centroid
        centroid_angles = np.array([np.degrees(np.arctan2(centroid[1] - center[1], centroid[0] - center[0])) for center in group_centers])
        centroid_angles = (centroid_angles + 360) % 360
        group_angles = (group_angles + 360) % 360

        centroid_angles = centroid_angles.reshape(-1,1)
        # Check if the group is engaged based on the orientation towards the centroid
        diff_angles = np.abs(centroid_angles - group_angles)
        engaged_centroid = np.sum((diff_angles < 45) | (diff_angles > 315)) >= len(indices) / 2

        # Check if the group is engaged based on pairwise orientations
        pairwise_angles = pairwise_distances(group_angles.reshape(-1, 1), metric='manhattan') % 180
        engaged_pairs = np.sum((pairwise_angles < 45) | (pairwise_angles > 315)) >= len(indices) * (len(indices) - 1) / 2

        # Consider the group as engaged if either condition is met
        if engaged_centroid or engaged_pairs:
            engaged_groups += 1
        
        total_size += len(indices)

    # Calculate the engagement score
    cluster_engagement = engaged_groups / n_clusters if n_clusters > 0 else 0
    
    # Boost the engagement score based on the average cluster size
    avg_cluster_size = total_size / n_clusters if n_clusters > 0 else 0
    size_boost = min(avg_cluster_size / 10, 1)  # Normalize the average size to [0, 1] range by assuming maximum size to be 10
    boosted_cluster_engagement = cluster_engagement * (1 + size_boost)
    
    if boosted_cluster_engagement > 1:
        boosted_cluster_engagement = 1.3

    return boosted_cluster_engagement, n_clusters, n_noise


# Function to normalize head count to be between 0 and 1
def normalize_head_count(head_count):
    normalized_count = min(head_count / 100, 1)
    return normalized_count


def calculate_engagement(head_centers, head_angles, head_count, image_height, image_width, proximity_weight=0.4, cluster_weight=0.5, headcount_weight=0.1):
    # Calculate the proximity score and the cluster engagement
    proximity_score = calculate_proximity_score(head_centers, image_width, image_height)
    cluster_engagement, n_clusters, n_noise = calculate_cluster_engagement(head_centers, head_angles)
    
    # Normalize the head count
    normalized_count = normalize_head_count(head_count)
    
    # Calculate the noise penalty as the ratio of noise points to total points
    noise_penalty = n_noise / len(head_centers) if len(head_centers) > 0 else 0
    
    # Subtract the noise penalty from the cluster engagement score
    cluster_engagement = cluster_engagement * (1 - noise_penalty)

    if head_count == 0:
        proximity_weight = 0
        cluster_weight = 0
        headcount_weight = 0
    
    # If no clusters detected, adjust the weights
    if n_clusters == 0:
        proximity_weight = 0.7
        cluster_weight = 0.2
        headcount_weight = 0.1

    
    # Calculate the overall engagement score
    engagement_score = proximity_weight * proximity_score + cluster_weight * cluster_engagement + headcount_weight * normalized_count
    
    return engagement_score, n_clusters, n_noise


import threading

# Initialize a lock object
lock = threading.Lock()

# Initialize global variables for the results
engagement_score = 0
n_clusters = 0
n_noise = 0
boxes = []

def detect_and_calculate(frame):
    global engagement_score, n_clusters, n_noise, boxes
    # Detect the head centers and get image dimensions
    head_centers, person_boxes, boxes, image_height, image_width = detect_head_centers(frame, ort_session)

    # Initialize an empty list to hold the head angles
    head_angles = []

    for i, box in enumerate(boxes):

        box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]

        if i < len(boxes):
            # Crop the head from the frame
            head_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            
            # Check if the cropped image is not empty
            if head_img.size > 0 and head_img.shape[0] > 0 and head_img.shape[1] > 0:
                # Only calculate the angle if the head was also detected when calculating head_centers

                box = boxes[i]                    
                # Save the cropped head to the head_images directory
                cv2.imwrite(f'./head_images/head_{i}.jpg', head_img)

                # Predict the head angle
                head_angle = get_head_angle(f'./head_images/head_{i}.jpg')

                # Append the head angle to the list
                head_angles.append(head_angle)


    # Calculate the engagement
    engagement_score, n_clusters, n_noise = calculate_engagement(head_centers, head_angles, person_boxes, image_height, image_width)

# Open the video capture
cap = cv2.VideoCapture(1)

frame_count = 0

def draw_rounded_box(img, box, color, thickness, radius=10, confidence=None):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    if confidence is not None:
        label = f"{confidence * 100:.2f}%"
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = x1 + radius
        text_y = y1 + text_size[1] + 5
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, [255, 255, 255], font_thickness)


while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # If the frame was not read correctly, break the loop
    if not ret:
        break

    # Increment the frame count
    frame_count += 1

    # Perform detection and calculation every 30 frames
    if frame_count % 30 == 0:
        # Start a new thread for detection and calculation
        threading.Thread(target=detect_and_calculate, args=(frame, ort_session)).start()

        # Create a blank image to display the scores
        score_image = np.zeros((500, 500, 3), dtype="uint8")

        # Draw the scores on the image
        cv2.putText(score_image, f"Engagement Score: {engagement_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(score_image, f"Number of Clusters: {n_clusters}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(score_image, f"Noise: {n_noise}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the scores in a separate window
        cv2.imshow('Scores', score_image)

    # Perform YOLO detection on the frame
    results = model.predict(frame)
    #Draw the bounding boxes on the frame
    for result in results:
        for box in result.boxes:
           coordinates = box.xyxy[0].tolist()  # get box coordinates in (top, left, bottom, right) format
           confidence = box.conf[0].item()  # get confidence value
           cv2.rectangle(frame, (int(coordinates[0]), int(coordinates[1])), (int(coordinates[2]), int(coordinates[3])), (199, 58, 58), 2)
           cv2.putText(frame, f"{confidence:.2f}", (int(coordinates[0]), int(coordinates[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (199, 58, 58), 2)    # Display the frame with bounding boxes
    cv2.imshow('YOLO Detections', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
