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

def select_algorithm(number_of_people, density_variation):
    threshold = 0.5  # Define the threshold value
    if number_of_people < 5:
        print("1st one")
        return DBSCAN(eps=1, min_samples=2)
    elif density_variation > threshold:
        print("2nd")
        return DBSCAN(eps = 0.25,min_samples=2)
    else:
        return DBSCAN(eps=0.5, min_samples=4)

def calculate_cluster_engagement(head_centers, head_angles, previous_clusters):
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

    return engagement_score, n_clusters, n_noise

image_width, image_height = 40, 40

def generate_randomly_scattered_points(num_points):
    """Generate randomly scattered points across the image."""
    x = np.random.uniform(0, image_width, num_points)
    y = np.random.uniform(0, image_height, num_points)
    return list(zip(x, y))

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
            if towards:
                angle = np.degrees(np.arctan2(centroid[1] - point[1], centroid[0] - point[0]))
            else:
                angle = np.degrees(np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))
            angles.append(angle)
    return angles

# Parameters for data generation
num_clusters = 2
cluster_size = 5
cluster_radius = 1.5

# Generate the points for very tight natural circular crowds
all_points2 = generate_very_tight_natural_circular_crowds(num_clusters, cluster_size, cluster_radius)

# Generate head angles for engaged scenario (facing towards the centroid of their clusters)
head_angles2 = generate_angles_towards_centroid(all_points2, num_clusters, cluster_size, towards=True)


# Number of scattered points and cluster size
num_scattered_points = 10

# Generate scattered points
scattered_points = generate_randomly_scattered_points(num_scattered_points)


# Combine scattered points and cluster points
all_points = scattered_points + all_points2

# Generate random orientations for scattered points and oriented angles for the cluster
head_angles = list(np.random.uniform(0, 360, num_scattered_points)) + head_angles2

head_count = (len(all_points))

# Engagement Calculation for unengaged scenario
engagement_score_unengaged, n_clusters_unengaged, n_noise_unengaged = calculate_engagement(
    all_points, 
    head_angles, 
    head_count,
    image_height=image_height, 
    image_width=image_width,
    previous_clusters=None
)

# Print results
print(f"Engagement Score Slight Scenario): {engagement_score_unengaged}")

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

# Plot points for engaged scenario
plot_points_with_angles(all_points, head_angles, "Engaged Scenario")


