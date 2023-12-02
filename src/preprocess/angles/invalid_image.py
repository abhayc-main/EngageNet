import os
from PIL import Image

data_dir = './data/angle_data'

# Iterate over all files in data_dir
for subdir, dirs, files in os.walk(data_dir):
    for file in files:
        filepath = subdir + os.sep + file

        # Try to open the file with PIL
        try:
            img = Image.open(filepath)
        except Exception as e:
            print(f"Can't open image file {filepath}")
            os.remove(filepath)
