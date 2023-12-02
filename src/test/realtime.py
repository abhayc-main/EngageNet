import cv2

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

def detect_head_centers(frame, model):
    # Convert the frame to a torch tensor and normalize pixel values
    image_tensor = F.to_tensor(frame).unsqueeze(0)

    # Run the model prediction
    with torch.no_grad():
        prediction = model(image_tensor)

    # Extract bounding boxes and class IDs from the model prediction
    boxes = prediction[..., :4]
    class_ids = prediction[..., -1]

    # Filter out predictions for non-head classes
    head_boxes = boxes[class_ids == 0]  # Assuming head class ID is 0

    # Calculate the center of each bounding box
    centers = (head_boxes[:, :2] + head_boxes[:, 2:]) / 2

    # Convert the tensor to a numpy array
    centers = centers.numpy()

    # Get the image width and height from the frame
    image_width, image_height = frame.shape[1], frame.shape[0]

    # Get the number of people
    person_boxes = len(centers)

    return centers, image_width, image_height, person_boxes, head_boxes

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


def get_head_angles(image_path, centers):
    # Load the model
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
