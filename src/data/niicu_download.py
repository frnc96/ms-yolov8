from torchvision.transforms.functional import to_pil_image
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import subprocess
import requests
import zipfile
import torch
import os


# URL of the .zip file
zip_file_url = 'https://nii-cu-mapd-dataset.s3.ap-northeast-1.amazonaws.com/NII_CU_MAPD_dataset.zip'

# The local path where you want to save the downloaded ZIP file
local_file_path = '../zips/niicu.zip'

# The directory to which the contents of the zip file will be extracted
extract_to_dir = '../NII-CU'


def download_zip_file(url, save_path):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the local file in binary write mode
        with open(save_path, 'wb') as file:
            # Write the content of the response to the file
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
        print(f"ZIP file downloaded and saved to '{save_path}'")
    else:
        print(f"Failed to download the ZIP file. Status code: {response.status_code}")

def extract_zip(zip_path, extract_to):
    # Check if the ZIP file exists
    if not os.path.isfile(zip_path):
        print(f"The file {zip_path} does not exist.")
        return

    # Create the directory where the contents will be extracted if it doesn't exist
    if not os.path.isdir(extract_to):
        os.makedirs(extract_to)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted ZIP file to '{extract_to}'")
    directories = [
        '../niicu', '../niicu/train', '../niicu/train/images', '../niicu/train/labels',
        '../niicu/val', '../niicu/val/images', '../niicu/val/labels',
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def convert_labels_to_yolo():
    path = Path('../niicu')

    img_width, img_height = 2706, 1980

    # Convert labels to supported format
    for t in path.rglob('*.txt'):
        save = True
        new_lines = []

        with open(t, 'r') as file:
            for line in file:
                bb_data = line.strip()

                # Check if it has already been converted
                if len(bb_data.split()) == 4:
                    save = False
                    continue

                x_min, y_min, x_max, y_max, _, _, _ = map(float, bb_data.split())

                # Convert to YOLO format: class_id, x_center, y_center, width, height
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height

                new_lines.append(
                    " ".join(map(str, [0, x_center, y_center, width, height]))
                )

        # Overwrite the file with new format
        if save:
            with open(t, 'w') as file:
                for line in new_lines:
                    file.write(f"{line}\n")

def merge_images_into_tiff():
    source_dir = '../NII-CU/4-channel/images/{version}/{split}'
    destination_dir = '../niicu/{split}/images'

    for s in ('train', 'val'):
        rgb_dir = Path(source_dir.format(version='rgb', split=s))
        th_dir = Path(source_dir.format(version='thermal', split=s))
        dest_dir = Path(destination_dir.format(split=s))

        for rgb_img_path in tqdm(rgb_dir.rglob('*.jpg'), desc=s):
            # Skip if .tiff exists
            tiff_path = dest_dir.joinpath(
                rgb_img_path.name.replace('.jpg', '.tiff')
            )
            if os.path.exists(tiff_path):
                continue

            rgb_image = Image.open(rgb_img_path)
            thermal_image = Image.open(th_dir.joinpath(rgb_img_path.name))\
                                .resize(rgb_image.size)\
                                .convert('L')

            # Convert images to NumPy arrays
            rgb_array = np.array(rgb_image)
            thermal_array = np.array(thermal_image)

            # Ensure the thermal image is expanded to have the same dimensions as the RGB image
            thermal_expanded = np.expand_dims(thermal_array, axis=2)

            # Stack along the third dimension to get an array of shape [H, W, 4]
            combined_array = np.concatenate((rgb_array, thermal_expanded), axis=2)

            # Convert the NumPy array to a PyTorch tensor of shape [H, W, 4]
            combined_tensor = torch.tensor(combined_array, dtype=torch.uint8)
            combined_tensor = combined_tensor.permute(2, 0, 1)

            # Convert tensor to PIL image
            image = to_pil_image(combined_tensor)

            # Save image
            image.save(tiff_path, compression='tiff_lzw')


if __name__ == "__main__":
    # Download
    if not os.path.exists("../zips/niicu.zip"):
        print("Downloading niicu.zip")
        download_zip_file(zip_file_url, local_file_path)
    else:
        print("ZIP file already there")

    # Extract
    if not os.path.exists("../NII-CU"):
        print("Extracting niicu.zip")
        extract_zip(local_file_path, extract_to_dir)
    else:
        print("ZIP data already extracted")

    # Structure
    if not os.path.exists("../niicu"):
        print("Constructing folder structure")
        subprocess.run(['./niicu_struct.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    else:
        print("Folder structure already created")

    # Label formatting
    print("Converting labels to YOLO format")
    convert_labels_to_yolo()

    # TIFF image files
    print("Merging RGB & T images into .tiff format")
    merge_images_into_tiff()
