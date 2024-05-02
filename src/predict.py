from ultralytics import YOLO
# from src.rgbt import merge
from pathlib import Path


# Load a model
model = YOLO(
    model=Path("/home/frencis/D1/sbatch/runs/detect/4ch-yolov8x-niicu-640 (bicubic)/weights/best.pt"),
    task="detect",
    verbose=True,
)

# # Compose the WxHx4 Tensor from two images
# rgbt_tensor = merge(
#     rgb_image_path='ultralytics/cfg/datasets/niicu/val/rgb-images/flight3_frame12721.jpg',
#     t_image_path='ultralytics/cfg/datasets/niicu/val/t-images/flight3_frame12721.jpg',
# )

results = model.predict(
    # source=rgbt_tensor,
    source='/home/frencis/D1/datasets/niicu/val/images/flight3_frame12721.tiff',
    save=True,
)
