# CNN for focial angle orientation determination

import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

# Remove dodgy images
import cv2
import imghdr

data_dir = 'data' 
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

# Load and preprocess data
data = tf.keras.utils.image_dataset_from_directory('data', labels='inferred', label_mode='categorical',
                                                   validation_split=0.2, subset='training', seed=42,
                                                   image_size=(256, 256), batch_size=32)

# Split data into training and validation sets
train_data = data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = tf.keras.utils.image_dataset_from_directory('data', labels='inferred', label_mode='categorical',
                                                       validation_split=0.2, subset='validation', seed=42,
                                                       image_size=(256, 256), batch_size=32)

# Build the CNN model
model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(90, activation='softmax'))  # Assuming 90 classes for angle prediction

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(train_data, epochs=20, validation_data=val_data)

# Plot performance
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()

# Save the model
model.save('head_angle_model')

# Test the model
import cv2
import numpy as np

image_path = 'test_image.jpg'
img = cv2.imread(image_path)
resized_img = cv2.resize(img, (256, 256))
normalized_img = resized_img / 255.0
input_img = np.expand_dims(normalized_img, axis=0)

prediction = model.predict(input_img)
predicted_angle = np.argmax(prediction)
print("Predicted Angle:", predicted_angle)
