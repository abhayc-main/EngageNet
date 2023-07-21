import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

# Function to calculate cluster engagement - on head positions and angles
# THIS CODE WAS TESTED - Seems to work better where people are facing one common focal point
def calculate_cluster_engagement_v1(head_centers, head_angles):
    scaler = StandardScaler()
    X = scaler.fit_transform(head_centers)
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    engaged_groups = 0
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
    cluster_engagement = engaged_groups / n_clusters if n_clusters > 0 else 0
    return cluster_engagement, n_clusters, n_noise

# Function to calculate cluster - New Version
# Works better when situations are more complex
def calculate_cluster_engagement_v2(head_centers, head_angles):
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
        pairwise_angles = pairwise_distances(group_angles.reshape(-1, 1), metric='manhattan') % 180
        engaged_pairs = np.sum((pairwise_angles < 45) | (pairwise_angles > 135))
        total_pairs = len(indices) * (len(indices) - 1) / 2
        if engaged_pairs >= total_pairs / 2:
            engaged_groups += 1
    cluster_engagement = engaged_groups / n_clusters if n_clusters > 0 else 0
    # Boost the engagement score based on the average cluster size
    avg_cluster_size = total_size / n_clusters if n_clusters > 0 else 0
    size_boost = min(avg_cluster_size / 10, 1)  # Normalize the average size to [0, 1] range by assuming maximum size to be 10
    boosted_cluster_engagement = cluster_engagement * (1 + size_boost)
    return cluster_engagement, n_clusters, n_noise

# Test data
np.random.seed(0)

# Create two clusters of 6 people each facing inwards
cluster1_centers = np.random.normal(loc=[250, 250], scale=10, size=(6, 2))
cluster1_angles = np.degrees(np.arctan2(250 - cluster1_centers[:, 1], 250 - cluster1_centers[:, 0])) % 360
cluster2_centers = np.random.normal(loc=[750, 750], scale=10, size=(6, 2))
cluster2_angles = np.degrees(np.arctan2(750 - cluster2_centers[:, 1], 750 - cluster2_centers[:, 0])) % 360

# Create 3-4 random people scattered here and there
random_centers = np.random.rand(4, 2) * 1000
random_angles = np.random.rand(4) * 360

# Combine the data
head_centers = np.concatenate([cluster1_centers, cluster2_centers, random_centers], axis=0)
head_angles = np.concatenate([cluster1_angles, cluster2_angles, random_angles], axis=0)

# Calculate engagement
engagement_v1, n_clusters_v1, n_noise_v1 = calculate_cluster_engagement_v1(head_centers, head_angles)
engagement_v2, n_clusters_v2, n_noise_v2 = calculate_cluster_engagement_v2(head_centers, head_angles)

print(engagement_v1, n_clusters_v1, n_noise_v1, engagement_v2, n_clusters_v2, n_noise_v2)
