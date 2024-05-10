from torchvision.transforms.functional import to_pil_image
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import requests
import tarfile
import torch
import glob
import os


# URL of the .zip file
zip_file_url = 'https://onedrive.live.com/download?cid=1570430EADF56512&resid=1570430EADF56512%21109419&authkey=AJcMP-7Yp86PWoE'

# The local path where you want to save the downloaded ZIP file
local_file_path = '../zips/kaist.tar.gz'

# The directory to which the contents of the zip file will be extracted
extract_to_dir = '../KAIST'


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

def extract_tar_gz(tar_gz_path, extract_to):
    # Check if the TAR.GZ file exists
    if not os.path.isfile(tar_gz_path):
        print(f"The file {tar_gz_path} does not exist.")
        return

    # Create the directory where the contents will be extracted if it doesn't exist
    if not os.path.isdir(extract_to):
        os.makedirs(extract_to)

    # Extract the TAR.GZ file
    with tarfile.open(tar_gz_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)
        print(f"Extracted TAR.GZ file to '{extract_to}'")

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

def convert_kaist_labels_to_yolo(split, entity_path):
    """
        split: train or test
        entity_path: set06/V000/I00000
    """
    # RGB_PATH = Path("/home/frencis/D1/datasets/KAIST/kaist-cvpr15/images")
    # T_PATH = Path("/home/frencis/D1/datasets/KAIST/kaist-cvpr15/images")

    XML_LABEL_PATH = Path(f"/home/frencis/D1/datasets/KAIST/kaist-cvpr15/annotations-xml-new-sanitized/{entity_path}.xml")
    TXT_LABEL_PATH = Path(f"/home/frencis/D1/datasets/kaist/{split}/labels/{entity_path.replace('/', '_')}.txt")

    # Read XML file
    tree = ET.parse(XML_LABEL_PATH)
    root = tree.getroot()

    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    # Open a file to write the YOLO formatted labels
    with open(TXT_LABEL_PATH, 'w') as file:
        for obj in root.iter('object'):
            # Get the class name and convert it to YOLO format (e.g., person -> 0)
            class_name = obj.find('name').text

            if class_name == 'person':
                class_id = 0
            elif class_name == 'people':
                class_id = 1
            elif class_name == 'cyclist':
                class_id = 2
            else:
                continue
            
            # Extract bounding box and convert to YOLO format
            bndbox = obj.find('bndbox')

            x = int(bndbox.find('x').text)
            y = int(bndbox.find('y').text)
            w = int(bndbox.find('w').text)
            h = int(bndbox.find('h').text)
            
            # Calculate YOLO coordinates
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            
            # Write to file
            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def convert_kaist_images_to_tiff(split, entity_path):
    """
        split: train or test
        entity_path: set06/V000/I00000
        rgb_path: set06/V000/visible/I00000
        t_path: set06/V000/lwir/I00000 
    """
    filename = entity_path.split('/')[-1]
    split_path = '/'.join(
        entity_path.split('/')[:-1]
    )

    RGB_PATH = Path(f"/home/frencis/D1/datasets/KAIST/kaist-cvpr15/images/{split_path}/visible/{filename}.jpg")
    T_PATH = Path(f"/home/frencis/D1/datasets/KAIST/kaist-cvpr15/images/{split_path}/lwir/{filename}.jpg")
    DEST_PATH = Path(f"/home/frencis/D1/datasets/kaist/{split}/images/{entity_path.replace('/', '_')}.tiff")

    if os.path.exists(DEST_PATH):
        return

    rgb_image = Image.open(RGB_PATH)
    thermal_image = Image.open(T_PATH).convert('L')

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
    image.save(DEST_PATH, compression='tiff_lzw')


if __name__ == "__main__":
    # Download
    # if not os.path.exists("../zips/kaist.tar.gz"):
    #     print("Downloading kaist.tar.gz")
    #     download_zip_file(zip_file_url, local_file_path)
    # else:
        # print("ZIP file already there")

    # # Extract
    # if not os.path.exists("../KAIST"):
    #     print("Extracting kaist.tar.gz")
    #     extract_tar_gz(local_file_path, extract_to_dir)
    # else:
    #     print("TAR.GZ data already extracted")

    # Convert kaist labels to yolo
    for split in ('train', 'test'):
        SETS_PATTERN = f"/home/frencis/D1/datasets/KAIST/kaist-cvpr15/imageSets/{split}-all-*.txt"

        for file_path in tqdm(glob.glob(SETS_PATTERN), desc=split):
            # Dont process the seq file
            if file_path.split('/')[-1] == 'test-all-01-Seq.txt':
                continue

            with open(file_path, 'r') as file:
                for line in tqdm(file, desc='Lines'):
                    # convert_kaist_labels_to_yolo(
                    #     split=split,
                    #     entity_path=line.strip(),
                    # )
                    convert_kaist_images_to_tiff(
                        split=split,
                        entity_path=line.strip(),
                    )