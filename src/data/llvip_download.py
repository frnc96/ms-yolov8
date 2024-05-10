from torchvision.transforms.functional import to_pil_image
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import requests
import zipfile
import torch
import glob
import os


# URL of the .zip file
zip_file_url = 'https://drive.usercontent.google.com/download?id=1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv&export=download&authuser=0'

# The local path where you want to save the downloaded ZIP file
local_file_path = '../zips/llvip.zip'

# The directory to which the contents of the zip file will be extracted
extract_to_dir = '../LLVIP'


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

def convert_llvip_labels_to_yolo(xml_path, txt_path):
    # Read XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # Open a file to write the YOLO formatted labels
    with open(txt_path, 'w') as file:
        for obj in root.iter('object'):
            # Get the class name and convert it to YOLO format (e.g., person -> 0)
            class_name = obj.find('name').text

            if class_name == 'person':
                class_id = 0
            else:
                continue
            
            # Extract bounding box and convert to YOLO format
            bndbox = obj.find('bndbox')

            x_min = int(bndbox.find('xmin').text)
            y_min = int(bndbox.find('ymin').text)
            x_max = int(bndbox.find('xmax').text)
            y_max = int(bndbox.find('ymax').text)
            
            # Calculate the width and height from x_min, x_max, y_min, and y_max
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # Calculate the center coordinates from x_min, x_max, y_min, and y_max
            x_center = (x_min + (x_max - x_min) / 2) / img_width
            y_center = (y_min + (y_max - y_min) / 2) / img_height
            
            # Write to file
            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def convert_llvip_images_to_tiff(rgb_path, t_path, tiff_path):
    rgb_image = Image.open(rgb_path)
    thermal_image = Image.open(t_path).convert('L')

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
    # # Download
    # if not os.path.exists("../zips/llvip.zip"):
    #     print("Downloading llvip.zip")
    #     download_zip_file(zip_file_url, local_file_path)
    # else:
    #     print("ZIP file already there")

    # # Extract
    # if not os.path.exists("../LLVIP"):
    #     print("Extracting llvip.zip")
    #     extract_zip(local_file_path, extract_to_dir)
    # else:
    #     print("llvip.zip data already extracted")

    # Convert llvip labels to yolo
    for split in ('train', 'test'):
        """
            version: infrared/visible
            split: train/test
            type: images/labels
        """
        source_dir = "/home/frencis/D1/datasets/LLVIP/LLVIP/{version}/{split}"
        destination_dir = "/home/frencis/D1/datasets/llvip/{split}/{type}"

        rgb_dir = Path(source_dir.format(version='visible', split=split))
        t_dir = Path(source_dir.format(version='infrared', split=split))
        dest_dir = Path(destination_dir.format(split=split, type='images'))

        xml_dir = Path("/home/frencis/D1/datasets/LLVIP/LLVIP/Annotations")
        txt_dir = Path(destination_dir.format(split=split, type='labels'))

        for rgb_img_path in tqdm(rgb_dir.rglob('*.jpg'), desc=split):
            tiff_path = dest_dir.joinpath(
                rgb_img_path.name.replace('.jpg', '.tiff')
            )
            xml_path = xml_dir.joinpath(
                rgb_img_path.name.replace('.jpg', '.xml')
            )
            txt_path = txt_dir.joinpath(
                rgb_img_path.name.replace('.jpg', '.txt')
            )

            # Merge images into .tiff file
            if not os.path.exists(tiff_path):
                convert_llvip_images_to_tiff(
                    rgb_path=rgb_dir.joinpath(rgb_img_path.name),
                    t_path=t_dir.joinpath(rgb_img_path.name),
                    tiff_path=tiff_path,
                )

            # Convert labels to yolo format
            if not os.path.exists(txt_path):
                convert_llvip_labels_to_yolo(
                    xml_path=xml_path,
                    txt_path=txt_path,
                )