import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

# Function to calculate cluster engagement - on head positions and angles
def calculate_cluster_engagement_combined(head_centers, head_angles):
    scaler = StandardScaler()
    X = scaler.fit_transform(head_centers)
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
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
        centroid_angles = np.degrees(np.arctan2(centroid[1] - group_centers[:, 1], centroid[0] - group_centers[:, 0]))
        centroid_angles = (centroid_angles + 360) % 360
        group_angles = (group_angles + 360) % 360
        diff_angles = np.abs(centroid_angles - group_angles)
        engaged = np.sum((diff_angles < 45) | (diff_angles > 315))
        if engaged >= len(indices) / 2:
            engaged_groups += 1
        else:
            pairwise_angles = pairwise_distances(group_angles.reshape(-1, 1), metric='manhattan') % 180
            engaged_pairs = np.sum((pairwise_angles < 45) | (pairwise_angles > 135))
            total_pairs = len(indices) * (len(indices) - 1) / 2
            if engaged_pairs >= total_pairs / 2:
                engaged_groups += 1
        total_size += len(indices)
    cluster_engagement = engaged_groups / n_clusters if n_clusters > 0 else 0
    # Boost the engagement score based on the average cluster size
    avg_cluster_size = total_size / n_clusters if n_clusters > 0 else 0
    size_boost = min(avg_cluster_size / 10, 1)  # Normalize the average size to [0, 1] range by assuming maximum size to be 10
    boosted_cluster_engagement = cluster_engagement * (1 + size_boost)
    return boosted_cluster_engagement, n_clusters, n_noise

# 3 pairs of people close to each other but facing away
pair_centers = [(0, 1), (0, 1.1), (2, 1), (2, 1.1), (4, 1), (4, 1.1)]
pair_angles = [0, 180, 0, 180, 0, 180]

# Combine the data
head_centers = pair_centers
head_angles = pair_angles



# Test the function with the data
engagement_score, n_clusters, n_noise = calculate_cluster_engagement_combined(np.array(head_centers), np.array(head_angles))

print(engagement_score, n_clusters, n_noise)
