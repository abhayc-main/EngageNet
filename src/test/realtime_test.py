import torch
from ultralytics import YOLO

device = torch.device("cuda")

# Load your custom YOLO model
model = YOLO("./models/newbest.pt")

video_path = "./videos/istockphoto-1180234043-640_adpp_is.mp4"


# Perform object detection on a video stream from the webcam
results = model.track(source=video_path, show=True, stream=True, conf=0.20, iou=0.20)

# Process each frame
for result in results:
    # Here, you can process the `result` for each frame
    print(result)