import torch
from ultralytics import YOLO
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = YOLO("./models/yolotrainedruns/detect/train/weights/best.pt")
model.conf=0.40

# Export the model to ONNX format
model.export(format='onnx')
print("Successfully exported the model into .onnx format")

model.export(format='engine')
print("Successfully exported the model into .onnx format")
