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
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import Input

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.metrics.pairwise import pairwise_distances

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def preprocess_image(image):
    # Resize the image to the input size expected by the model
    image_resized = cv2.resize(image, (180, 180))

    # Normalize the pixel values to the range [0, 1]
    image_normalized = image_resized / 255.0

    # Convert the image to a format suitable for the model
    image = img_to_array(image_normalized)

    return image

def get_head_angle(image_region):
    # Preprocess the image for the model
    image = preprocess_image(image_region)

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


def select_algorithm(number_of_people, noise_level):
    threshold_noise = 0.5  # This can be adjusted based on the data
    desired_clusters = 4  # This can also be adjusted if necessary

    print(f"Number of people: {number_of_people}")

    # For smaller groups
    if number_of_people < 5:
        print("1st condition used")
        return DBSCAN(eps=0.75, min_samples=4)

    # If noise level is below the threshold
    elif noise_level < threshold_noise:
        print("2nd condition used")
        return KMeans(n_clusters=desired_clusters)

    # Default: If none of the above conditions are met
    else:
        print("Default condition used")
        return DBSCAN(eps=0.95, min_samples=4)
    
def calculate_cluster_engagement(head_centers, head_angles, previous_clusters):
    if previous_clusters is None:
        previous_clusters = []

    truly_engaged_groups = 0
    threshold = 3

    # Standardize the head centers for the clustering algorithm
    scaler = StandardScaler()
    X = scaler.fit_transform(head_centers)

    # Use DBSCAN clustering to calculate noise level
    db_clustering = DBSCAN().fit(X)
    db_labels = db_clustering.labels_
    n_noise = list(db_labels).count(-1)
    noise_level = n_noise / len(head_centers) if len(head_centers) > 0 else 0

    # Select the clustering algorithm
    clustering_algorithm = select_algorithm(len(head_centers), noise_level)

    # Apply the selected clustering algorithm
    clustering = clustering_algorithm.fit(X)
    labels = clustering.labels_

    engaged_groups = 0
    valid_clusters = 0
    total_size = 0
    
    for cluster_id in set(labels):
        if cluster_id == -1:  # Ignore the noise cluster
            continue
        
        indices = np.where(labels == cluster_id)[0]
        if len(indices) < 2:
            continue
        
        indices_list = indices.tolist()

        group_angles = np.array([head_angles[i] for i in indices_list])
        group_centers = np.array([head_centers[i] for i in indices_list])

        centroid = np.mean(group_centers, axis=0)
        centroid_angles = np.array([np.degrees(np.arctan2(centroid[1] - center[1], centroid[0] - center[0])) for center in group_centers])

        diff_angles = np.abs(centroid_angles - group_angles)
        engaged_centroid = np.sum((diff_angles < 45) | (diff_angles > 315)) >= len(indices) / 2

        if engaged_centroid:
            valid_clusters += 1

        same_direction = np.std(group_angles) < 30  # Threshold can be adjusted

        if engaged_centroid and same_direction:
            engaged_groups += 1

            if len(previous_clusters) >= threshold and all(prev_cluster[cluster_id] for prev_cluster in previous_clusters[-threshold:]):
                truly_engaged_groups += 1

        total_size += len(indices)

    previous_clusters.append([engaged_centroid for cluster_id in set(labels) if cluster_id != -1])

    cluster_engagement = truly_engaged_groups / valid_clusters if valid_clusters > 0 else 0

    print(f"Valid Clusters: {valid_clusters}")

    avg_cluster_size = total_size / valid_clusters if valid_clusters > 0 else 0
    size_boost = min(avg_cluster_size / 10, 1)
    boosted_cluster_engagement = cluster_engagement * (1 + size_boost)

    if boosted_cluster_engagement > 1:
        boosted_cluster_engagement = 1.1

    return boosted_cluster_engagement, valid_clusters, n_noise

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
    
    # Calculate the noise penalty as 0.01 for each noise point
    #noise_penalty = 0.01 * n_noise
    
    # Subtract the noise penalty from the cluster engagement score
    #cluster_engagement = cluster_engagement * (1 - noise_penalty) 

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
    print(cluster_engagement)
    print(proximity_score)
    
    return engagement_score, n_clusters, n_noise

# Assuming an image dimension for context in proximity calculations (can be adjusted)
image_width, image_height = 40, 40

def generate_very_tight_natural_circular_crowds(num_clusters, cluster_size, cluster_radius):
    all_points = []
    
    # Determine the radius of the circle on which the clusters are located
    clusters_circle_radius = 12  # This determines how spread out the clusters are
    
    for k in range(num_clusters):
        # Calculate cluster center
        angle_for_cluster = 2 * np.pi * k / num_clusters
        cluster_center_x = image_width/2 + clusters_circle_radius * np.cos(angle_for_cluster)
        cluster_center_y = image_height/2 + clusters_circle_radius * np.sin(angle_for_cluster)
        
        for i in range(cluster_size):
            angle_variation = np.random.uniform(-0.2, 0.2)  # slight variation in angle for more natural appearance
            r_variation = cluster_radius * np.random.uniform(-0.7, 0.3)  # increased variation in distance for individuals to be very close together
            angle = 2 * np.pi * i / cluster_size + angle_variation
            r = cluster_radius + r_variation
            x = cluster_center_x + r * np.cos(angle)
            y = cluster_center_y + r * np.sin(angle)
            all_points.append((x, y))
    return all_points

def generate_angles_towards_centroid(points, num_clusters, cluster_size, towards=True):
    angles = []
    for k in range(num_clusters):
        cluster_points = points[k*cluster_size:(k+1)*cluster_size]
        centroid = np.mean(cluster_points, axis=0)
        
        for point in cluster_points:
            angle = np.degrees(np.arctan2(centroid[1] - point[1], centroid[0] - point[0]))
            if not towards:
                angle += 180  # Make them face away from the centroid
            angle %= 360  # Ensure angle is between 0 and 360
            angles.append(angle)
    return angles


# Parameters for data generation
num_clusters = 5
cluster_size = 10
cluster_radius = 1.5

# Generate the points for very tight natural circular crowds
all_points_very_tight_natural_crowd_circular = generate_very_tight_natural_circular_crowds(num_clusters, cluster_size, cluster_radius)

head_count = len(all_points_very_tight_natural_crowd_circular)

# Highly Engaged without Noise
all_points_engaged_without_noise = generate_very_tight_natural_circular_crowds(num_clusters, cluster_size, cluster_radius) 
head_angles_engaged_without_noise = generate_angles_towards_centroid(all_points_engaged_without_noise, num_clusters, cluster_size, towards=True)

# Opposite (Highly Unengaged without Noise)
all_points_unengaged_without_noise = generate_very_tight_natural_circular_crowds(num_clusters, cluster_size, cluster_radius) 
head_angles_unengaged_without_noise = generate_angles_towards_centroid(all_points_unengaged_without_noise, num_clusters, cluster_size, towards=False)

# Engagement Calculation for Highly Engaged without Noise scenario
engagement_score_engaged, n_clusters_engaged, n_noise_engaged = calculate_engagement(
    all_points_engaged_without_noise, 
    head_angles_engaged_without_noise, 
    head_count,  
    image_height=image_height, 
    image_width=image_width,
    previous_clusters=None
)

# Engagement Calculation for Highly Unengaged without Noise scenario
engagement_score_unengaged, n_clusters_unengaged, n_noise_unengaged = calculate_engagement(
    all_points_unengaged_without_noise, 
    head_angles_unengaged_without_noise, 
    head_count,  
    image_height=image_height, 
    image_width=image_width,
    previous_clusters=None
)

# Print results
print(f"Engagement Score (Engaged without Noise Scenario): {engagement_score_engaged}")
print(f"Engagement Score (Unengaged without Noise Scenario): {engagement_score_unengaged}")

print(" ")

print(f"Clusters:  {n_clusters}")


# Adjust arrow plotting for better visualization
def plot_points_with_angles(points, angles, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    
    for point, angle in zip(points, angles):
        x, y = point
        dx = 0.8 * np.cos(np.radians(angle))
        dy = 0.8 * np.sin(np.radians(angle))
        ax.arrow(x, y, dx, dy, head_width=0.5, head_length=1, fc='red', ec='red')
        ax.plot(x, y, 'bo')
    
    ax.set_title(title)
    plt.show()


print(" ")


print(" ")

print(f"Number of Clusters (Engaged without Noise Scenario): {n_clusters_engaged}")
print(f"Number of Noise Points (Engaged without Noise Scenario): {n_noise_engaged}")
print(f"Number of Clusters (Unengaged without Noise Scenario): {n_clusters_unengaged}")
print(f"Number of Noise Points (Unengaged without Noise Scenario): {n_noise_unengaged}")

plot_points_with_angles(all_points_engaged_without_noise, head_angles_engaged_without_noise, "Engaged without Noise Scenario")

# Plot points for Highly Unengaged without Noise scenario
plot_points_with_angles(all_points_unengaged_without_noise, head_angles_unengaged_without_noise, "Unengaged without Noise Scenario")