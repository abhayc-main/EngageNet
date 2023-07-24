import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import rgb_to_grayscale
import tensorflow as tf

def to_gray(image):
    return rgb_to_grayscale(image)

datagen = ImageDataGenerator(
    rescale=1./255,  # rescale pixel values to [0,1]
    zoom_range=0.2,  # randomly zooming inside pictures
    horizontal_flip=True,  # randomly flip images
    preprocessing_function=to_gray # convert images to grayscale
)

dir_path = './data/angle_data/'

for subdir, dirs, files in os.walk(dir_path):
    for file in files:
        image_path = os.path.join(subdir, file)
        image = tf.keras.preprocessing.image.load_img(image_path)
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = x.reshape((1,) + x.shape)

        # Determine the file format for saving based on the original file extension
        file_format = 'jpeg' if file.lower().endswith('.jpg') else 'png'

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=subdir, save_prefix='aug', save_format=file_format):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely
