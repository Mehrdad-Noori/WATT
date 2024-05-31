import os
from PIL import Image, UnidentifiedImageError
# import logging
import argparse

# Configure logging
# logging.basicConfig(filename='corrupted_images.log', level=logging.INFO, filemode='w')

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, help="Root path of the data directory")
args = parser.parse_args()

dataroot = args.dataroot

def verify_images(root_path):
    corrupted_files = []
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for common image file extensions
                filepath = os.path.join(subdir, file)
                try:
                    with Image.open(filepath) as img:
                        img.convert('RGB')  # Attempt to convert the image to RGB
                except (IOError, UnidentifiedImageError, SyntaxError) as e:
                    print(f"Corrupted file found: {filepath} - {e}")
                    # logging.info(f"Corrupted file: {filepath} - {e}")
                    corrupted_files.append(filepath)

    return corrupted_files

def clean_corrupted_files(file_list, delete=False):
    if delete:
        for file in file_list:
            os.remove(file)
            print(f"Deleted corrupted file: {file}")
            # logging.info(f"Deleted corrupted file: {file}")
    else:
        # Move corrupted files to a new directory for manual review
        review_folder = os.path.join(dataroot, "review_corrupted")
        os.makedirs(review_folder, exist_ok=True)
        for file in file_list:
            dest = os.path.join(review_folder, os.path.basename(file))
            os.rename(file, dest)
            print(f"Moved corrupted file for review: {dest}")
            # logging.info(f"Moved corrupted file for review: {dest}")

corrupted_files = verify_images(dataroot)
if corrupted_files:
    print(f"Found {len(corrupted_files)} corrupted files. Handling...")
    # Set delete=True if you want to delete the corrupted files.
    clean_corrupted_files(corrupted_files, delete=True)
else:
    print("No corrupted files found.")