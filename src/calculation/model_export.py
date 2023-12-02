from ultralytics import YOLO

# Load a model
model = YOLO('./models/best.pt')  # load a custom trained

# Export the model
model.export(format='onnx')

model.export(format='engine')