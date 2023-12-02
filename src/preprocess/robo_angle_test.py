from roboflow import Roboflow

rf = Roboflow(api_key="R5i9d6qtGJCDn0LiaEhe")
project = rf.workspace().project("overhead-angle-detection-6lmpn")
model = project.version(1).model

# infer on a local image
print(model.predict("./data/test2.png").json())