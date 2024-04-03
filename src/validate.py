from ultralytics import YOLO
from pathlib import Path


# Config
MODEL_YAML_PATH = Path("/home/frencis/D1/sbatch/runs/detect/4ch-yolov8x-coco-llvip_niicu-640/weights/best.pt")
# DATA_YAML_PATH = Path("/home/frencis/D1/datasets/llvip/data.yaml")


# Load a model
model = YOLO(
    model=MODEL_YAML_PATH,
    verbose=True,
)

results = model.val(
    # data=DATA_YAML_PATH,
)
