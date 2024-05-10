from torchvision.transforms.functional import to_pil_image
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import zipfile
import torch
import glob
import shutil
import os


# The local path where you want to save the downloaded ZIP file
local_file_path = '../zips/M3FD_Detection.zip'

# The directory to which the contents of the zip file will be extracted
extract_to_dir = '../M3FD'


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

def convert_labels_to_yolo(xml_path, txt_path):
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

            # People, Car, Bus, Motorcycle, Lamp, Truck
            if class_name == 'People':
                class_id = 0
            elif class_name == 'Car':
                class_id = 1
            elif class_name == 'Bus':
                class_id = 2
            elif class_name == 'Motorcycle':
                class_id = 3
            elif class_name == 'Lamp':
                class_id = 4
            elif class_name == 'Truck':
                class_id = 5
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

def convert_images_to_tiff(rgb_path, t_path, tiff_path):
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
    # # Extract
    # if not os.path.exists("../M3FD"):
    #     print("Extracting M3FD_Detection.zip")
    #     extract_zip(local_file_path, extract_to_dir)
    # else:
    #     print("M3FD_Detection.zip data already extracted")



    # # Split data into train/val on a 10:1 ratio
    # vis_path = Path("/home/frencis/D1/datasets/M3FD/Vis")
    # ir_path = Path("/home/frencis/D1/datasets/M3FD/Ir")

    # vis_dest_path = "/home/frencis/D1/datasets/M3FD/{split}/visible/{name}"
    # ir_dest_path = "/home/frencis/D1/datasets/M3FD/{split}/ir/{name}"

    # # Split 10% of the dataset into validation
    # vis_images = [image for image in os.listdir(vis_path)]
    # ten_percent = random.sample(vis_images, k=int(len(vis_images) * 0.1))

    # # Move the selected 10% to the first target directory
    # for filename in ten_percent:
    #     # Move visible image file
    #     shutil.move(
    #         os.path.join(vis_path, filename),
    #         vis_dest_path.format(split="val", name=filename),
    #     )
    #     # Move ir image file
    #     shutil.move(
    #         os.path.join(ir_path, filename),
    #         ir_dest_path.format(split="val", name=filename),
    #     )

    # # Move the rest of the files
    # for filename in [file for file in os.listdir(vis_path)]:
    #     # Move visible image file
    #     shutil.move(
    #         os.path.join(vis_path, filename),
    #         vis_dest_path.format(split="train", name=filename),
    #     )
    #     # Move ir image file
    #     shutil.move(
    #         os.path.join(ir_path, filename),
    #         ir_dest_path.format(split="train", name=filename),
    #     )



    # Convert llvip labels to yolo
    for split in ('train', 'val'):
        """
            version: ir/visible
            split: train/val
            type: images/labels
        """
        source_dir = "/home/frencis/D1/datasets/M3FD/{split}/{version}"
        destination_dir = "/home/frencis/D1/datasets/m3fd/{split}/{type}"

        rgb_dir = Path(source_dir.format(version='visible', split=split))
        t_dir = Path(source_dir.format(version='ir', split=split))
        dest_dir = Path(destination_dir.format(split=split, type='images'))

        xml_dir = Path("/home/frencis/D1/datasets/M3FD/Annotation")
        txt_dir = Path(destination_dir.format(split=split, type='labels'))

        for rgb_img_path in tqdm(rgb_dir.rglob('*.png'), desc=split):
            tiff_path = dest_dir.joinpath(
                rgb_img_path.name.replace('.png', '.tiff')
            )
            xml_path = xml_dir.joinpath(
                rgb_img_path.name.replace('.png', '.xml')
            )
            txt_path = txt_dir.joinpath(
                rgb_img_path.name.replace('.png', '.txt')
            )

            # Merge images into .tiff file
            if not os.path.exists(tiff_path):
                convert_images_to_tiff(
                    rgb_path=rgb_dir.joinpath(rgb_img_path.name),
                    t_path=t_dir.joinpath(rgb_img_path.name),
                    tiff_path=tiff_path,
                )

            # Convert labels to yolo format
            if not os.path.exists(txt_path):
                convert_labels_to_yolo(
                    xml_path=xml_path,
                    txt_path=txt_path,
                )