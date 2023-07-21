import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image_class(img_path):
    # Load the trained model
    model = tf.keras.models.load_model("my_model.h5")

    # Load the image for prediction
    img = image.load_img(img_path, target_size=(180, 180))

    # Convert the image to a numpy array and scale it
    img_array = image.img_to_array(img) / 255.

    # Expand the dimensions of the image
    img_array = np.expand_dims(img_array, axis=0)

    # Use the model to predict the class of the image
    predictions = model.predict(img_array)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)

    # Define the class labels in the same order as they were during training
    class_labels = ["0", "45", "90", "135", "180", "225", "270", "315"]

    # Get the class label from the class labels list
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

# Call the function with the path to your model and test image
predicted_class = predict_image_class('./data/validate/L.png')

print(f"The predicted angle for the image is: {predicted_class}")

