from PIL import Image
import numpy as np
import torch


def merge(rgb_image_path, t_image_path):
    # Load your images using PIL (replace 'path/to/your/image' with your image paths)
    rgb_image = Image.open(rgb_image_path)
    thermal_image = Image.open(t_image_path).resize(rgb_image.size).convert('L')

    # Convert images to NumPy arrays
    rgb_array = np.array(rgb_image)
    thermal_array = np.array(thermal_image)

    # Ensure the thermal image is expanded to have the same dimensions as the RGB image
    thermal_expanded = np.expand_dims(thermal_array, axis=2)

    # Stack along the third dimension to get an array of shape [H, W, 4]
    combined_array = np.concatenate((rgb_array, thermal_expanded), axis=2)

    # Convert the NumPy array to a PyTorch tensor
    combined_tensor = torch.tensor(combined_array, dtype=torch.float32)

    return combined_tensor
