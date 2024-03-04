from ultralytics import YOLO
from pathlib import Path


# Config
MODEL_YAML_PATH = Path("/home/frencis/D1/sbatch/runs/detect/base-niicu-tl-4ch-640/weights/best.pt")


# Load a model
model = YOLO(
    model=MODEL_YAML_PATH,
    verbose=True,
)

results = model.val()
