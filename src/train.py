from ultralytics import YOLO
from pathlib import Path


# Config
MODEL_YAML_PATH = Path("/home/frencis/D1/ms-yolov8/ultralytics/cfg/models/v8/ms-yolov8n.yaml")
DATA_YAML_PATH = Path("/home/frencis/D1/datasets/niicu/niicu.yaml")


# Load a model
model = YOLO(
    model=MODEL_YAML_PATH,
    verbose=True,
)

results = model.train(
    data=DATA_YAML_PATH,
    single_cls=True,
    plots=True,
    epochs=100,
    imgsz=640,
)
