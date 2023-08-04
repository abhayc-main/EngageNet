import torch
from ultralytics import YOLO
print(torch.cuda.is_available())

# Load your custom YOLO model
model = YOLO("./models/yolotrainedruns/detect/train/weights/best.pt")
model.conf = 0.40


# Perform object detection on a video stream from the webcam
results = model.predict("./main (2).jpg", show=True)

# Print the results
print(results)
