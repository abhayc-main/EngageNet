import cv2
import threading
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict


import torch
from ultralytics import YOLO


import cv2

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
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.metrics.pairwise import pairwise_distances

import socketio
import time

sio = socketio.Client()
sio.connect('http://localhost:3001')

device = torch.device("cuda")

def preprocess_image(image):
    # Resize the image to the input size expected by the model
    image_resized = cv2.resize(image, (180, 180))

    # Normalize the pixel values to the range [0, 1]
    image_normalized = image_resized / 255.0

    # Convert the image to a format suitable for the model
    image = img_to_array(image_normalized)

    return image

def get_head_angle(cropped_img):
    # Convert numpy array to a format suitable for YOLO
    cropped_img_path = "temp_cropped.jpg"
    cv2.imwrite(cropped_img_path, cropped_img)

    # Load the model
    model = YOLO("./models/angle_best.pt")

    # Get predictions
    results = model(cropped_img_path)  # results list

    # Extract the class name for the top detected class (angle)
    top_detected_class_name = None
    for r in results:
        top_detected_class_index = r.probs.top1  # Extracting the index of the top detected class (angle)
        top_detected_class_name = r.names[top_detected_class_index]  # Getting the class name using the index
    print(f"Angle Returned {top_detected_class_name}")
    return top_detected_class_name

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

def select_algorithm(number_of_people, density_variation):
    threshold = 0.5  # Define the threshold value
    if number_of_people < 5:
        print("1st one")
        return DBSCAN(eps=1.75, min_samples=2)
    elif density_variation > threshold:
        print("2nd")
        return DBSCAN(eps = 0.25,min_samples=2)
    else:
        return DBSCAN(eps=0.5, min_samples=4)


def calculate_cluster_engagement(head_centers, head_angles, previous_clusters):
    if not head_centers:  # Check if head_centers is empty
        return 0, 0, 0
    
    if previous_clusters is None:
        previous_clusters = []

    # Variables for temporal threshold clustering
    truly_engaged_groups = 0
    threshold=3

    # Standardize the head centers for the clustering algorithm
    scaler = StandardScaler()
    X = scaler.fit_transform(head_centers)

    # Calculate density variation as the standard deviation of pairwise distances
    pairwise_distances_matrix = pairwise_distances(X)
    density_variation = np.std(pairwise_distances_matrix)


    # Apply DBSCAN clustering to calculate noise level
    # Determine the clustering algorithm based on the select_algorithm function
    clustering_algorithm = select_algorithm(len(head_centers), density_variation)

    # Apply the selected clustering algorithm
    clustering = clustering_algorithm.fit(X)
    labels = clustering.labels_

    # Calculate the number of noise points
    n_noise = list(labels).count(-1)
    print(n_noise)

    # Calculate the noise level
    noise_level = n_noise / len(head_centers) if len(head_centers) > 0 else 0
    print(noise_level)

    # Select the clustering algorithm
    
    print(f"Labels {labels}")
    # Create a set of all data point indices
    all_indices = set(range(len(head_centers)))
    

    # Calculate the number of clusters excluding noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    engaged_groups = 0
    total_size = 0
    print(f"Noise: {n_noise}")

    # Iterate over each cluster
    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        
        # Remove the indices of data points that belong to a cluster from the all_indices set
        all_indices -= set(indices.tolist())
        
        if len(indices) < 2:
            continue

        # Remove the indices of data points that belong to a cluster from the all_indices set
        all_indices -= set(indices.tolist())
        
        group_angles = np.array([head_angles[i] for i in indices])  # Convert to NumPy array
        group_centers = np.array([head_centers[i] for i in indices])

        # Calculate the centroid of the current cluster
        centroid = np.mean(group_centers, axis=0)

        # Now calculate centroid angles
        centroid_angles = np.array([np.degrees(np.arctan2(centroid[1] - center[1], centroid[0] - center[0])) for center in group_centers])
        centroid_angles = (centroid_angles + 360) % 360

        group_angles = np.array(group_angles, dtype=float)

        group_angles = (group_angles + 360) % 360
        
        # Calculate the centroid of the current cluster
        centroid = np.mean(group_centers, axis=0)

        # Calculate pairwise angles between individuals
        pairwise_angles_matrix = np.abs(np.subtract.outer(centroid_angles, centroid_angles))
        pairwise_angles = np.where(pairwise_angles_matrix > 180, 360 - pairwise_angles_matrix, pairwise_angles_matrix)

        # Dynamic Angle Threshold
        distances_to_centroid = np.linalg.norm(group_centers - centroid, axis=1)
        max_distance = np.max(distances_to_centroid)
        angle_thresholds = 45 - 30 * (distances_to_centroid / max_distance)  # Adjusting threshold based on distance

        # Check if the group is engaged based on the orientation towards the centroid
        diff_angles = np.abs(centroid_angles - group_angles)
        engaged_centroid = np.sum((diff_angles < angle_thresholds) | (diff_angles > (360 - angle_thresholds))) >= len(indices) / 2

        # Secondary Metrics: Check if individuals are facing each other within a certain distance
        pairwise_distances_to_each_other = pairwise_distances(group_centers)
        close_pairs = pairwise_distances_to_each_other < max_distance * 0.5  # Adjust the distance threshold as needed
        close_pair_angles = pairwise_angles[close_pairs]
        engaged_pairs = np.sum((close_pair_angles < 45) | (close_pair_angles > 315)) >= len(indices) * (len(indices) - 1) / 2

        # Consider the group as engaged if either condition is met
        if engaged_centroid or engaged_pairs:
            engaged_groups += 1
        
        total_size += len(indices)
        print(f"Noise in loop: {n_noise}")


    # By the end of the cluster iteration, any indices remaining in the set correspond to noise points
    #n_noise = len(all_indices)
    print(f"Noise: {n_noise}")

    # Calculate the engagement score based on truly engaged groups
    cluster_engagement = truly_engaged_groups / n_clusters if n_clusters > 0 else 0

    # Calculate the engagement score
    cluster_engagement = engaged_groups / n_clusters if n_clusters > 0 else 0
    
    # Boost the engagement score based on the average cluster size
    avg_cluster_size = total_size / n_clusters if n_clusters > 0 else 0
    size_boost = min(avg_cluster_size / 10, 1)  # Normalize the average size to [0, 1] range by assuming maximum size to be 10
    boosted_cluster_engagement = cluster_engagement * (1 + size_boost)
    
    if boosted_cluster_engagement > 1.2:
        boosted_cluster_engagement = 1.2
    elif boosted_cluster_engagement > 1:
        boosted_cluster_engagement = 1.1

    return boosted_cluster_engagement, n_clusters, n_noise

# Function to normalize head count to be between 0 and 1
def normalize_head_count(head_count):
    normalized_count = min(head_count / 100, 1)
    return normalized_count

# To prevent severe score drops - exponenital smoothing algorithm implementation
def exponential_smoothing(scores, alpha=0.1):
    smoothed_scores = [scores[0]]
    for score in scores[1:]:
        smoothed_scores.append(alpha * score + (1 - alpha) * smoothed_scores[-1])
    return smoothed_scores

def calculate_engagement(head_centers, head_angles, head_count, image_height, image_width, previous_clusters, previous_engagement_score=0, no_cluster_frames=0, initial_frames=0):

    # Calculate the proximity score and the cluster engagement
    proximity_score = calculate_proximity_score(head_centers, image_width, image_height)
    cluster_engagement, n_clusters, n_noise = calculate_cluster_engagement(head_centers, head_angles, previous_clusters)
    
    # Normalize the head count
    normalized_count = normalize_head_count(head_count)
    
    # Calculate the noise penalty as the ratio of noise points to total points
    noise_ratio = (n_noise / len(head_centers) if len(head_centers) > 0 else 0)
    noise_penalty = noise_ratio**2
    print(f"noisepenalty: {noise_penalty}")
    
    print(cluster_engagement)
    # Subtract the noise penalty from the cluster engagement score
    cluster_engagement = cluster_engagement * (1 - noise_penalty) 

    # Subtract the noise penalty from the cluster engagement score

    INITIAL_THRESHOLD = 10  # Number of initial frames to use weighted average
    THRESHOLD = 30  # Number of frames to carry over previous score
    DECAY_FACTOR = 0.95  # Decay factor to gradually reduce the engagement score

    if n_clusters == 0:
        if initial_frames < INITIAL_THRESHOLD:
            engagement_score = 0.7 * proximity_score + 0.3 * normalized_count
            initial_frames += 1
        elif no_cluster_frames < THRESHOLD:
            engagement_score = 0.7 * previous_engagement_score
        else:
            engagement_score = previous_engagement_score * DECAY_FACTOR
        no_cluster_frames += 1
    else:
        no_cluster_frames = 0
        initial_frames = 0
        engagement_score = 0.4 * proximity_score + 0.5 * cluster_engagement + 0.1 * normalized_count
    print(f"Clusters: {n_clusters}")
    if engagement_score > 1:
        engagement_score = 1

    return engagement_score, n_clusters, n_noise, head_count

import socketio
import threading

sio = socketio.Client()
sio.connect('http://localhost:3001')

model = YOLO("./models/best.pt")
rect_color = (235, 64, 52)
engagement_scores = []
previous_clusters = None
previous_engagement_score = 0
no_cluster_frames = 0
initial_frames = 0
frame_counter = 0

# For tracking CDF measures
interaction_counts = []
interaction_durations = []


def process_frame(result, engagement_scores, previous_clusters, previous_engagement_score, no_cluster_frames, initial_frames):
    frame = result.orig_img
    detections = result.boxes.xyxy
    boxes = result.boxes.data
    class_indices = boxes[:, 4].tolist()
    boxes = [box[:4] for box in detections]
    head_centers = [(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)) for box in boxes]

    if result.boxes is not None and hasattr(result.boxes, 'id') and result.boxes.id is not None:
        ids = result.boxes.id.tolist()
    else:
        ids = []

    head_angles = [get_head_angle(frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]) for box in result.boxes.xyxy]
    interactions_this_frame = defaultdict(int)
    durations_this_frame = []
    for i, (id1, center1) in enumerate(zip(ids, head_centers)):
        for j, (id2, center2) in enumerate(zip(ids, head_centers)):
            if id1 != id2:  # Exclude self-interaction
                distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                # Define a threshold for what you consider as interaction. Adjust this value accordingly.
                if distance < 50:
                    interactions_this_frame[id1] += 1
                    # Note: This is a simplistic way to record interaction duration. For more accuracy, track start and end times.
                    durations_this_frame.append(distance)

    interaction_counts.append(np.mean(list(interactions_this_frame.values())))
    interaction_durations.extend(durations_this_frame)

    engagement_score, n_clusters, n_noise, head_count = calculate_engagement(
        head_centers, head_angles, len(boxes), frame.shape[0], frame.shape[1], previous_clusters, previous_engagement_score, no_cluster_frames, initial_frames
    )

    previous_engagement_score = engagement_score
    engagement_scores.append(engagement_score)
    smoothed_scores = exponential_smoothing(engagement_scores)
    smoothed_engagement_score = smoothed_scores[-1]

    data = {
        'engagement_score': smoothed_engagement_score,
        'n_total': head_count,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'interaction_counts_raw': interactions_this_frame,
        'interaction_durations_raw': durations_this_frame
    }
    
    sio.emit('dataUpdate', data)
    print(data)
    print("sent")


for result in model.track(source=0, show=True, stream=True, agnostic_nms=True, conf=0.25, iou=0.10):
    frame_counter += 1
    if frame_counter % 30 != 0:
        continue
    threading.Thread(target=process_frame, args=(result, engagement_scores, previous_clusters, previous_engagement_score, no_cluster_frames, initial_frames)).start()


