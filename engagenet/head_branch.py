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

# Function to calculate cluster engagement based on head positions and angles
def calculate_cluster_engagement(head_centers, head_angles):
    # Standardize head positions using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(head_centers)
    
    # Apply DBSCAN clustering on the standardized positions
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(X)
    
    # Extract cluster labels
    labels = clustering.labels_
    
    # Calculate number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Initialize count of engaged groups
    engaged_groups = 0
    
    # Loop over each cluster
    for cluster_id in range(n_clusters):
        # Get indices of heads belonging to the current cluster
        indices = np.where(labels == cluster_id)[0]
        
        # Ignore clusters with less than 2 heads
        if len(indices) < 2:
            continue
        
        # Get angles and positions of heads in the current cluster
        group_angles = head_angles[indices]
        group_centers = head_centers[indices]
        
        # Calculate centroid of the current cluster
        centroid = np.mean(group_centers, axis=0)
        
        # Calculate angles from each head in the cluster to the centroid
        centroid_angles = np.degrees(np.arctan2(centroid[1] - group_centers[:, 1], centroid[0] - group_centers[:, 0]))
        centroid_angles = (centroid_angles + 360) % 360
        
        # Make sure group_angles are between 0 and 360
        group_angles = (group_angles + 360) % 360
        
        # Calculate absolute differences between group_angles and centroid_angles
        diff_angles = np.abs(centroid_angles - group_angles)
        
        # Count number of heads facing towards the centroid (within 45 degrees)
        engaged = np.sum((diff_angles < 45) | (diff_angles > 315))
        
        # If more than half of the heads in the cluster are engaged, count the cluster as an engaged group
        if engaged >= len(indices) / 2:
            engaged_groups += 1
    
    # Calculate cluster engagement as the fraction of engaged groups
    cluster_engagement = engaged_groups / n_clusters if n_clusters > 0 else 0
    
    return cluster_engagement, n_clusters, n_noise

# Function to normalize head count to be between 0 and 1
def normalize_head_count(head_count):
    normalized_count = min(head_count / 100, 1)
    return normalized_count

# Function to calculate overall engagement score
def calculate_engagement(head_centers, head_angles, image_width, image_height, head_count, proximity_weight=0.4, cluster_weight=0.5, headcount_weight=0.1):
    # Calculate each component of the engagement score
    proximity_score = calculate_proximity_score(head_centers, image_width, image_height)
    cluster_engagement, n_clusters, n_noise = calculate_cluster_engagement(head_centers, head_angles)
    normalized_count = normalize_head_count(head_count)
    
    # Combine components into an overall engagement score
    engagement_score = proximity_weight * proximity_score + cluster_weight * cluster_engagement + headcount_weight * normalized_count
    
    return engagement_score, n_clusters, n_noise

# Generate random data for testing
np.random.seed(0)  
head_centers = np.random.rand(20, 2) * 1000
head_angles = np.random.rand(20) * 360
head_count = 20

# Calculate engagement
engagement_score, n_clusters, n_noise = calculate_engagement(head_centers, head_angles, 1000, 1000, head_count)

engagement_score, n_clusters, n_noise
