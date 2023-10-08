import os
import cv2

def crop_objects_from_yolo_format(base_dir, output_dir):
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Folders like 'train', 'test', 'valid'
    for folder in ['train', 'valid', 'test']:
        image_folder = os.path.join(base_dir, folder, 'images')
        label_folder = os.path.join(base_dir, folder, 'labels')

        for filename in os.listdir(image_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(image_folder, filename)
                label_path = os.path.join(label_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

                # Read image
                img = cv2.imread(img_path)

                # Check if image is null or corrupted
                if img is None:
                    print(f"Skipping null or corrupted image: {img_path}")
                    continue

                # Check if label file exists
                if os.path.exists(label_path):
                    with open(label_path, 'r') as file:
                        lines = file.readlines()

                        for index, line in enumerate(lines):
                            parts = line.strip().split()
                            class_id, x_center, y_center, width, height = map(float, parts)
                            img_height, img_width = img.shape[:2]

                            # Convert YOLO format to pixel values
                            x_center, y_center, width, height = x_center * img_width, y_center * img_height, width * img_width, height * img_height
                            x1, y1, x2, y2 = int(x_center - width/2), int(y_center - height/2), int(x_center + width/2), int(y_center + height/2)

                            # Crop the object from the image
                            cropped_img = img[y1:y2, x1:x2]

                            # Check if cropped image is empty
                            if cropped_img.size == 0:
                                print(f"Skipping empty crop for image: {img_path}, with coordinates: {x1, y1, x2, y2}")
                                continue

                            # Save the cropped image
                            save_path = os.path.join(output_dir, f"{filename.split('.')[0]}_{index}.jpg")
                            cv2.imwrite(save_path, cropped_img)

if __name__ == "__main__":
    base_directory = './data/NEW/'  # Assuming your dataset is in the current directory with 'train', 'test', 'valid' subdirectories
    output_directory = './new'
    crop_objects_from_yolo_format(base_directory, output_directory)
