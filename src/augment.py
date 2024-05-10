import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


def random_hsv_augmentation(image_path, hue_shift_limit=(-180, 180), saturation_scale_limit=(0.5, 1.5), brightness_scale_limit=(0.5, 1.5)):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV

    # Randomly adjust hue, saturation, brightness
    hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
    saturation_scale = np.random.uniform(saturation_scale_limit[0], saturation_scale_limit[1])
    brightness_scale = np.random.uniform(brightness_scale_limit[0], brightness_scale_limit[1])

    # Hue
    image[:, :, 0] = (image[:, :, 0].astype(np.float32) + hue_shift) % 180

    # Saturation
    image[:, :, 1] = np.clip(image[:, :, 1].astype(np.float32) * saturation_scale, 0, 255)

    # Brightness
    image[:, :, 2] = np.clip(image[:, :, 2].astype(np.float32) * brightness_scale, 0, 255)

    # Convert back to BGR and return
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def apply_random_perspective_augmentation(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("The image file was not found.")

    # Convert the image from BGR (OpenCV format) to RGB (PIL format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    
    # Define the transformation: since perspective is 0.0, focusing on translate and scale
    transform = T.RandomAffine(
        degrees=0.0,          # No rotation
        translate=(0.1, 0.1), # 10% translation
        scale=(0.5, 1),     # Scale between 50% and 150%
        shear=0.0             # No shear
    )

    # Apply the transformation
    transformed_img = transform(pil_img)
    
    # Convert back to OpenCV format if needed
    transformed_img_cv = cv2.cvtColor(np.array(transformed_img), cv2.COLOR_RGB2BGR)
    
    return transformed_img_cv


# Usage
image_path = '/home/frencis/D1/datasets/niicu/val/images/flight3_frame20001.tiff'
augmented_image = apply_random_perspective_augmentation(image_path)

# Create a PIL Image from the NumPy array
pil_image = Image.fromarray(augmented_image)
tiff_path = '/home/frencis/D1/sbatch/images/flight3_frame20001_rpersp.tiff'
pil_image.save(tiff_path, format='TIFF')
