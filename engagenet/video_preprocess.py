import cv2
import os
import numpy as np
from PIL import Image
from roboflow import Roboflow
from tqdm import tqdm

def detect_head_centers(image_path):
    # Initialize Roboflow
    rf = Roboflow(api_key="K9A8vdmXXNpdi3lmHVBI")
    project = rf.workspace().project("people_counterv0")
    model = project.version(1).model

    #resize_image(image_path)

    # Run the model prediction
    prediction = model.predict(image_path, confidence=40, overlap=30).json()

    print(prediction)

    # Get the image width and height from the prediction dictionary
    image_width = int(prediction['image']['width'])
    image_height = int(prediction['image']['height'])

    # Initialize an empty list to hold the bounding boxes
    boxes = []

    # Iterate over each detection in the predictions
    for obj in prediction['predictions']:
        # Get the bounding box coordinates
        x = obj['x']
        y = obj['y']
        width = obj['width']
        height = obj['height']

        # Append the bounding box coordinates to the list
        boxes.append((x, y, width, height))

    person_boxes = len(boxes)

    # Convert the list to a NumPy array for easier manipulation
    boxes = np.array(boxes)

    return boxes, image_width, image_height, person_boxes

# Function to calculate brightness of an image
def calculate_brightness(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.average(grayscale)

def process_video_and_save_snapshots(video_path, snapshot_dir='./data/test', frame_skip=10, brightness_threshold=50):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    too_dark = False

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_count % frame_skip == 0:  # Only process every nth frame
                brightness = calculate_brightness(frame)
                if brightness < brightness_threshold:
                    too_dark = True
                    break

                cv2.imwrite('frame.jpg', frame)
                boxes, image_width, image_height, person_boxes = detect_head_centers('frame.jpg')

                if person_boxes > 0:
                    # Save the whole frame as a snapshot
                    if not os.path.exists(snapshot_dir):
                        os.makedirs(snapshot_dir)
                    cv2.imwrite(f'{snapshot_dir}/frame_{frame_count}.jpg', frame)

            frame_count += 1
        else:
            break

    cap.release()

    # If the video is too dark, delete it
    if too_dark:
        os.remove(video_path)

# Directory containing .avi files - store it in a drive
avi_directory = '/Volumes/ABHAYDRIVE/videos'

# Get a list of all .avi files in the directory
avi_files = [f for f in os.listdir(avi_directory) if f.endswith('.avi') and not f.startswith('._')]

# Process each .avi file
for avi_file in tqdm(avi_files):
    process_video_and_save_snapshots(os.path.join(avi_directory, avi_file))
