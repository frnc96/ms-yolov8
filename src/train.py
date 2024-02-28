from ultralytics import YOLO
from pathlib import Path


# Load a model
model = YOLO(
    model=Path("ultralytics/cfg/models/v8/ms-yolov8n.yaml"),
    verbose=True,
)

results = model.train(
    data='ultralytics/cfg/datasets/niicu.yaml',
    single_cls=True,
    plots=True,
    epochs=100,
    imgsz=640,
)
