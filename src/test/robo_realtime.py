import cv2
import numpy as np
import math
import time

# SSL certificate issues fix
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from PIL import Image
import numpy as np
from roboflow import Roboflow


from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import Input

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances


def resize_image(image_path):
    img = Image.open(image_path)

    # Specify the maximum dimensions for resizing
    max_width = 800
    max_height = 800

    # Calculate the aspect ratio
    width_ratio = max_width / img.width
    height_ratio = max_height / img.height

    # Choose the smaller ratio to ensure the image fits within the desired dimensions
    resize_ratio = min(width_ratio, height_ratio)

    # Calculate the new width and height
    new_width = int(img.width * resize_ratio)
    new_height = int(img.height * resize_ratio)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

    # Save the resized image
    resized_img.save(image_path)

def detect_head_centers(frame):
    
    # Convert the frame to an image and save it
    img = Image.fromarray(frame)
    image_path = "current_frame.jpg"
    img.save(image_path)

    # Initialize Roboflow
    rf = Roboflow(api_key="K9A8vdmXXNpdi3lmHVBI")
    project = rf.workspace().project("people_counterv0")
    model = project.version(1).model

    resize_image(image_path)

    # Run the model prediction
    prediction = model.predict(image_path, confidence=30, overlap=30).json()

    # Get the image width and height from the prediction dictionary
    image_width = int(prediction['image']['width'])
    image_height = int(prediction['image']['height'])

    # Initialize an empty list to hold the center coordinates
    centers = []
    boxes = []

    # Iterate over each detection in the predictions
    for obj in prediction['predictions']:
        # Calculate the center of the bounding box
        x_center = (obj['x'] + obj['width']) / 2
        y_center = (obj['y'] + obj['height']) / 2

        # Append the center coordinates to the list
        centers.append((x_center, y_center))

        boxes.append([obj['x'], obj['y'], obj['width'], obj['height']])


    person_boxes = len(centers)
    boxes.append([obj['x'], obj['y'], obj['width'], obj['height']])


    # Convert the list to a NumPy array for easier manipulation
    centers = np.array(centers)

    print(centers, image_width, image_height)
    return centers, image_width, image_height, person_boxes, boxes

def extract_roi(image, center, size=180):
    # Padding the image
    padded_image = cv2.copyMakeBorder(image, top=112, bottom=112, left=112, right=112, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

    # Adjust the center for the padding
    center = (center[0] + 112, center[1] + 112)

    # Calculate the top left and bottom right coordinates of the ROI
    top_left = (max(0, int(center[0] - size / 2)), max(0, int(center[1] - size / 2)))
    bottom_right = (min(padded_image.shape[1], int(center[0] + size / 2)), min(padded_image.shape[0], int(center[1] + size / 2)))

    print(top_left, bottom_right)
    
    # Extract the ROI from the padded image
    roi = padded_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return roi



def preprocess_roi(roi):
    # Resize the ROI to the input size expected by the model
    roi_resized = cv2.resize(roi, (180, 180))

    # Normalize the pixel values to the range [0, 1]
    roi_normalized = roi_resized / 255.0

    # Expand the dimensions of the image array if necessary
    # Some models expect a 4D array as input (batch_size, height, width, channels)
    # Convert the ROI to a format suitable for the model
    roi = img_to_array(roi_normalized)

    return roi

def get_head_angles(frame, centers):
    # Use the frame directly instead of reading an image
    image = frame
    # Define the input shape
    input_shape = (180, 180, 3)  # For color images
    # input_shape = (180, 180, 1)  # For grayscale images

    # Load the model with the specified input shape
    model = load_model('my_model.h5', custom_objects={'input': Input(shape=input_shape)})
    
    print("Model input shape:", model.input_shape)
    
    image = cv2.imread(image_path)

    # Initialize an empty list to hold the angles
    angles = []

    # Iterate over each center
    for center in centers:
        # Extract the region of interest around the center
        roi = extract_roi(image, center)

        # Preprocess the region of interest for the model
        roi = preprocess_roi(roi)

            # Print the roi and its shape
        print("ROI shape:", roi.shape)

        # Print the dimensions to resize to
        print("Resize dimensions:", (model.input_shape[1], model.input_shape[2]))


        # Resize the ROI to the input size expected by the model
        #roi = cv2.resize(roi, (model.input_shape[1], model.input_shape[2]))

        # Convert the ROI to a format suitable for the model
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        angle = model.predict(roi)
        print(f"angle shape: {angle.shape}")  # Add this line


        # Append the angle to the list
        angles.append(angle)

    # Convert the list to a NumPy array for easier manipulation
    angles = np.array(angles)

    angles = angles.squeeze()

    return angles


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

def calculate_engagement(head_centers, head_angles, image_width, image_height, head_count, proximity_weight=0.4, cluster_weight=0.5, headcount_weight=0.1):
    # Calculate the proximity score and the cluster engagement
    proximity_score = calculate_proximity_score(head_centers, image_width, image_height)
    cluster_engagement, n_clusters, n_noise = calculate_cluster_engagement(head_centers, head_angles)
    
    # Normalize the head count
    normalized_count = normalize_head_count(head_count)
    
    # Calculate the noise penalty as the ratio of noise points to total points
    noise_penalty = n_noise / len(head_centers) if len(head_centers) > 0 else 0
    
    # Subtract the noise penalty from the cluster engagement score
    cluster_engagement = cluster_engagement * (1 - noise_penalty)
    
    # If no clusters detected, adjust the weights
    if n_clusters == 0:
        proximity_weight = 0.7
        cluster_weight = 0.2
        headcount_weight = 0.1
    
    # Calculate the overall engagement score
    engagement_score = proximity_weight * proximity_score + cluster_weight * cluster_engagement + headcount_weight * normalized_count
    
    return engagement_score, n_clusters, n_noise


# Initialize video capture (0 for webcam, or a filename for a video file)
cap = cv2.VideoCapture(0)

# Initialize the counter
counter = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Resize and preprocess the frame as needed
        # frame = preprocess_frame(frame)

        # Every 5 frames, calculate engagement
        if counter % 5 == 0:
            # Detect the head centers and get frame dimensions
            head_centers, frame_width, frame_height, head_count, boxes = detect_head_centers(frame)

            # Predict the head angles
            head_angles = get_head_angles(frame, head_centers)

            # Calculate the engagement
            engagement_score, n_clusters, n_noise = calculate_engagement(head_centers, head_angles, frame_width, frame_height, head_count)

            print(f"Engagement score: {engagement_score}, Clusters: {n_clusters}, Noise: {n_noise}")

        # Draw bounding boxes and arrows on the frame
        # frame = draw_boxes_and_arrows(frame, boxes, angles)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # If 'q' is pressed on the keyboard, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        counter += 1

# After the loop release the cap object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()
