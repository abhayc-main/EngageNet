from ultralytics import YOLO

# Load your custom YOLO model
model = YOLO("./models/yolotrainedruns/detect/train/weights/best.pt")
model.conf=0.40

# Perform object detection on a video stream from the webcam
results = model.predict(source="0", show=True)

# Print the results
print(results)