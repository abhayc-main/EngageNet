import os

# Define the base directory
base_dir = './data/Overhead Angle Data/'

# Check if the base directory exists
if not os.path.exists(base_dir):
    print(f"The directory {base_dir} does not exist.")
    exit()

# Initialize an empty dictionary to store class names as keys and counts as values
class_counts = {}

# Iterate through each class folder in the base directory
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    # Ensure that it's a directory
    if os.path.isdir(class_path):
        # Count the number of files in the class directory
        class_counts[class_name] = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])

# Print out the counts
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} files")

