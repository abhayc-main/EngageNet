

# annotation functions

scaling_factor = 10  # Adjust this value based on the desired scoring range

def calculate_head_orientation_score(facing_inwards_heads, total_heads):
    if total_heads == 0:
        return 0.0
    score = (facing_inwards_heads / total_heads) * scaling_factor
    return score



total_heads = 10
facing_inwards_heads = 7
score = calculate_head_orientation_score(facing_inwards_heads, total_heads)
print("Head Orientation Score:", score)


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def calculate_proximity_score(head_positions):
    # Calculate distance matrix between head positions
    distance_matrix = calculate_distance_matrix(head_positions)
    
    # Perform clustering on the distance matrix
    clustering = KMeans(n_clusters=5)  # Adjust the number of clusters based on your specific scenario
    labels = clustering.fit_predict(distance_matrix)
    
    # Count the number of heads in each cluster
    cluster_sizes = [np.sum(labels == label) for label in set(labels)]
    
    # Assign proximity scores based on cluster sizes
    proximity_scores = [size for size in cluster_sizes]  # Customize the calculation based on your preference
    
    # Normalize proximity scores between 0 and 1
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(np.array(proximity_scores).reshape(-1, 1))
    
    return normalized_scores.flatten()

# Example usage
head_positions = [position1, position2, position3, ...]  # List of head positions (x, y coordinates)
proximity_scores = calculate_proximity_score(head_positions)

