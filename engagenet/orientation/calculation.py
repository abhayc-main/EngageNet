"""
VERY IMPORTANT - REMEMBER TO ALSO HAVE COORDINATES
YOU CANT JUST USE angles also PROXIMITY AND COORDINATES from the algorithm

This implementation mixes clustering, distance matrices (proximity) with head angle iteration threshold

Mixes proxmity with facing inwards head angle scores

"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

def calculate_proximity_score(angles, coordinates):
    # Calculate distance matrix between head positions
    distance_matrix = pairwise_distances(coordinates)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=3)  # Adjust the parameters based on your needs
    labels = dbscan.fit_predict(distance_matrix)

    # Count the number of heads in each cluster
    cluster_sizes = [np.sum(labels == label) for label in set(labels)]

    # Assign proximity scores based on cluster sizes
    proximity_scores = [size for size in cluster_sizes]  # Customize the calculation based on your preference

    # Normalize proximity scores between 0 and 1
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(np.array(proximity_scores).reshape(-1, 1))

    # Calculate facing inward score based on angles
    facing_inward_scores = np.zeros(len(angles))
    for i in range(len(angles)):
        for j in range(len(angles)):
            if i != j:
                angle_diff = abs(angles[i] - angles[j])
                if angle_diff < threshold:  # Adjust the threshold based on your specific scenario
                    facing_inward_scores[i] += 1

    # Normalize facing inward scores between 0 and 1
    facing_inward_scores = facing_inward_scores / np.max(facing_inward_scores)

    # Combine proximity scores with facing inward scores
    combined_scores = np.multiply(normalized_scores.flatten(), facing_inward_scores)

    return combined_scores

