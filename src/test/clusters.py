# Numpy test - Engaged, Unengaged, Somewhat engaged, Somwhat unenaged tests

import cv2
import numpy as np
import math
import time
import matplotlib
import matplotlib.pyplot as plt

# SSL certificate issues fix
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import Image
from roboflow import Roboflow

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import Input

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

def normalize_head_count(head_count):
    normalized_count = min(head_count / 100, 1)
    return normalized_count

def calculate_proximity_score(head_centers, image_width, image_height):
    distances = []

    for i in range(len(head_centers)):
        for j in range(i+1, len(head_centers)):
            distance = np.sqrt((head_centers[i][0] - head_centers[j][0])**2 + (head_centers[i][1] - head_centers[j][1])**2)
            distances.append(distance)

    if len(distances) > 0:
        median_distance = np.median(distances)
    else:
        median_distance = 0

    max_possible_distance = np.sqrt(image_width**2 + image_height**2)
    normalized_distance = median_distance / max_possible_distance if max_possible_distance > 0 else 0
    proximity_score = 1 - normalized_distance

    return proximity_score

def calculate_cluster_engagement(head_centers, head_angles):
    scaler = StandardScaler()
    X = scaler.fit_transform(head_centers)
    clustering = DBSCAN(eps=1.5, min_samples=2).fit(X)
    labels = clustering.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    engaged_groups = 0
    total_size = 0

    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) < 2:
            continue

        group_angles = head_angles[indices]
        group_centers = head_centers[indices]
        centroid = np.mean(group_centers, axis=0)
        centroid_angles = np.array([np.degrees(np.arctan2(centroid[1] - center[1], centroid[0] - center[0])) for center in group_centers])
        centroid_angles = (centroid_angles + 360) % 360
        group_angles = (group_angles + 360) % 360
        centroid_angles = centroid_angles.reshape(-1,1)
        diff_angles = np.abs(centroid_angles - group_angles)
        engaged_centroid = np.sum((diff_angles < 45) | (diff_angles > 315)) >= len(indices) / 2
        pairwise_angles = pairwise_distances(group_angles.reshape(-1, 1), metric='manhattan') % 180
        engaged_pairs = np.sum((pairwise_angles < 45) | (pairwise_angles > 315)) >= len(indices) * (len(indices) - 1) / 2

        if engaged_centroid or engaged_pairs:
            engaged_groups += 1

        total_size += len(indices)

    cluster_engagement = engaged_groups / n_clusters if n_clusters > 0 else 0
    avg_cluster_size = total_size / n_clusters if n_clusters > 0 else 0
    size_boost = min(avg_cluster_size / 10, 1)
    boosted_cluster_engagement = cluster_engagement * (1 + size_boost)

    if boosted_cluster_engagement > 1:
        boosted_cluster_engagement = 1.3

    return boosted_cluster_engagement, n_clusters, n_noise

def calculate_engagement(head_centers, head_angles, image_width, image_height, head_count, proximity_weight=0.3, cluster_weight=0.6, headcount_weight=0.1):
    proximity_score = calculate_proximity_score(head_centers, image_width, image_height)
    cluster_engagement, n_clusters, n_noise = calculate_cluster_engagement(head_centers, head_angles)
    normalized_count = normalize_head_count(head_count)
    noise_penalty = n_noise / len(head_centers) if len(head_centers) > 0 else 0
    cluster_engagement = cluster_engagement * (1 - noise_penalty)
    print(f"Cluster Angles: {head_angles}")
    print(f"Cluster Centers: {head_centers}")
    print(f"Cluster engagement: {cluster_engagement}")
    if n_clusters == 0:
        proximity_weight = 0.7
        cluster_weight = 0.2
        headcount_weight = 0.1
    engagement_score = proximity_weight * proximity_score + cluster_weight * cluster_engagement + headcount_weight * normalized_count

    
    
    return engagement_score, n_clusters, n_noise

# Set the seed for reproducibility
np.random.seed(0)

# Generate cluster centers
cluster_centers = np.array([[250, 250], [750, 250], [250, 750], [750, 750]])

# Generate points around the cluster centers
points_cluster1 = np.random.normal(loc=cluster_centers[0], scale=50, size=(6, 2))
points_cluster2 = np.random.normal(loc=cluster_centers[1], scale=50, size=(6, 2))
points_cluster3 = np.random.normal(loc=cluster_centers[2], scale=50, size=(6, 2))
points_cluster4 = np.random.normal(loc=cluster_centers[3], scale=50, size=(6, 2))

# Generate the angles for each cluster
angles_cluster1 = (360 - np.degrees(np.arctan2(cluster_centers[0][1] - points_cluster1[:, 1], cluster_centers[0][0] - points_cluster1[:, 0])) + 90) % 360
angles_cluster2 = (360 - np.degrees(np.arctan2(cluster_centers[1][1] - points_cluster2[:, 1], cluster_centers[1][0] - points_cluster2[:, 0])) + 90) % 360
angles_cluster3 = (360 - np.degrees(np.arctan2(cluster_centers[2][1] - points_cluster3[:, 1], cluster_centers[2][0] - points_cluster3[:, 0])) + 90) % 360
angles_cluster4 = (360 - np.degrees(np.arctan2(cluster_centers[3][1] - points_cluster4[:, 1], cluster_centers[3][0] - points_cluster4[:, 0])) + 90) % 360

# Combine all points and angles
points = np.concatenate([points_cluster1, points_cluster2, points_cluster3, points_cluster4])
angles = np.concatenate([angles_cluster1, angles_cluster2, angles_cluster3, angles_cluster4])
head_count = len(points)


# Calculate engagement score
engagement_score, n_clusters, n_noise = calculate_engagement(points, angles, 1000, 1000, head_count)
print(engagement_score, n_clusters, n_noise)

