import cv2
import os
import numpy as np
from PIL import Image
import cv2
import os
import numpy as np
from PIL import Image
import re
from datetime import datetime

# Function to rotate image
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]

def screenshot_sort_key(s):
    date_time_str = s.replace('Screenshot ', '').rsplit('.', 1)[0].replace(' at ', ' ')
    try:
        date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %I.%M.%S %p')
        return (date_time_obj, s.lower())
    except ValueError:
        return (datetime.max, s.lower())


def generate_images(directory_path, angles):
    # Define the list of angles
    all_angles = [0, 45, 90, 135, 180, 225, 270, 315]

    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    screenshot_files = []
    other_files = []

    for file in files:
        if file.startswith('Screenshot '):
            screenshot_files.append(file)
        else:
            other_files.append(file)

    screenshot_files.sort(key=screenshot_sort_key)
    other_files.sort(key=natural_sort_key)

    files = other_files + screenshot_files

    # Open a new markdown file in write mode
    with open('image_angles.md', 'w') as f:
        # Write the table headers
        f.write('| Filename | Angle |\n')
        f.write('|----------|-------|\n')

        # Iterate over all files in the directory
        for i, filename in enumerate(files):
            # Check if the file is an image
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                # Get the full path of the image
                image_path = os.path.join(directory_path, filename)

                # Load the image
                image = cv2.imread(image_path)

                # Get the current angle from the list of angles
                if i < len(angles):
                    current_angle = angles[i]
                else:
                    current_angle = angles[-1]  # Use the last angle in the list as a default

                # Write the filename and angle to the markdown file
                f.write(f'| {filename} | {current_angle} |\n')

                # For each angle, rotate the image and save it
                for angle in all_angles:
                    if angle == current_angle:
                        # If the angle is the current angle, save the original image
                        if not os.path.exists(f'./data/angle_data/{angle}'):
                            os.makedirs(f'./data/angle_data/{angle}')
                        cv2.imwrite(f'./data/angle_data/{angle}/{filename}', image)
                    else:
                        # If the angle is not the current angle, rotate the image and save it
                        rotated = rotate_image(image, angle - current_angle)
                        if not os.path.exists(f'./data/angle_data/{angle}'):
                            os.makedirs(f'./data/angle_data/{angle}')
                        cv2.imwrite(f'./data/angle_data/{angle}/{filename}', rotated)

# Call the function with the path to your directory and the list of angles
main = [180, 225, 270, 315, 180, 45, 45, 135, 45, 90, 135,  270, 315, 135, 45, 270, 270, 90,90, 315, 0, 90, 270, 0, 135,90,270, 0, 315, 315, 270, 270, 225, 180, 135, 90, 90, 45, 45, 45, 45, 270, 0, 270, 270, 180, 180, 135, 90, 45, 45, 0, 315, 270, 0, 0, 315, 270,270,225, 225, 225, 135, 90, 135, 90, 90, 0, 270, 270, 270, 270, 270, 270, 270, 270, 270, 270, 0, 315, 315, 270, 225, 225, 180, 135, 135, 90, 45, 45, 315, 315, 0, 0, 315, 315, 315, 270, 225, 180, 135, 135, 135, 90, 45, 45, 45, 0, 45, 0, 45, 315, 225, 90, 90, 90, 90, 135, 135, 0, 180, 270, 315, 315, 135, 0, 0, 0, 0, 0, 0, 135, 270, 270, 0, 45, 315, 45, 45, 0, 45, 315, 0, 0, 90, 135, 225, 270, 90, 0, 180, 315, 180, 0, 315, 315, 315, 315, 270, 270, 225, 225, 180, 180, 135, 135, 90, 90, 45, 45, 180, 90, 135, 180, 180, 225, 270, 90, 135, 270, 225, 45, 90, 135, 180, 180, 315, 315, 315, 0,0,0, 315, 315, 315, 270, 225, 225, 180, 135, 135, 90, 90, 45, 45, 225, 135, 0, 0, 0, 0, 315, 270, 270, 270, 225,  180, 225, 180, 180, 135, 135, 90, 45, 0, 0, 0, 90, 270, 225, 270, 135, 270, 90, 270, 90, 45, 180, 45, 0, 270, 180, 90, 90, 90, 90, 270, 270, 270, 90, 0, 0, 135, 180, 180, 0, 315, 90, 180, 270, 90, 45, 315, 315, 180, 0, 90, 135, 45, 225, 225, 270, 90, 90, 225, 315, 45, 90, 90, 90, 270, 135, 90, 45, 225, 135, 0, 0, 180, 180, 180, 180, 180, 180, 180, 180, 180, 45, 180, 225, 180, 180, 135, 225, 180, 180, 180, 180, 225, 135, 180, 180, 180, 180, 225, 225, 135]
print(len(main))
generate_images('./data/test/', main)
