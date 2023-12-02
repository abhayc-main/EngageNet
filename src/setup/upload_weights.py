from roboflow import Roboflow

rf = Roboflow(api_key="A7C4W0rEXBbkSlEN8qT4")
project = rf.workspace().project("overhead-new")
project.version("1").deploy(model_type="yolov8", model_path="./models")
