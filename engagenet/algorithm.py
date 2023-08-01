import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import BatchNormalization


def test_model(img_path):

    # Load the model
    loaded_model = tf.keras.models.load_model('my_model.h5')

    # Load a single image for prediction
    img = image.load_img(img_path, target_size=(img_height, img_width))

    # Convert the image to a numpy array and scale it
    img_array = image.img_to_array(img) / 255.

    # Expand the dimensions of the image
    img_array = np.expand_dims(img_array, axis=0)

    # Use the model to predict the class of the image
    prediction = loaded_model.predict(img_array)

    # Get the index of the class with the highest probability
    prediction = np.argmax(prediction)

    # Print the prediction
    print(prediction)



# Define the path to the data directory
data_dir = './data/angle_dataV1'

# Define the image size
img_height = 180 
img_width = 180

# Create an image data generator object
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load images from the data directory
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = datagen.flow_from_directory(
    data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='validation') # set as validation data

# Print the class labels
print(train_generator.class_indices)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')  # 8 classes for 8 angles
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=30)

# Load the model
loaded_model = tf.keras.models.load_model('angle_algorithm_model_main')

# Continue training
history = loaded_model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // 32,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // 32,
    epochs = 50,  # Add the additional epochs
    callbacks=[reduce_lr, early_stopping])

model.save('algorithm.h5') 


