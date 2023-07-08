import cv2
import torch

# SSL certificate issues fix
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


image = cv2.imread('./test.png')  


results = model(image)

boxes = results.xyxy[0].cpu().numpy()
class_labels = results.names[0]

# Filter detections to keep only "person" class
person_boxes = boxes[boxes[:, -1] == 0]

# Draw bounding boxes on the image
for box in person_boxes:
    print(box)
    x1, y1, x2, y2, _ , __ = box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


cv2.imshow('Detection', image)
cv2.waitKey(0)
