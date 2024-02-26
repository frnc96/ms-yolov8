from ultralytics import YOLO
from pathlib import Path


# Load a model
# model = YOLO("ms-yolov8n.yaml")  # build a new model from scratch
model = YOLO(
    model=Path("ultralytics/cfg/models/v8/ms-yolov8.yaml"),
    # model=Path("checkpoints/yolov8n.pt"),
    task="detect",
    verbose=True,
)

results = model.predict(
    source=Path("runs/people.jpeg"),
    save=True,
)
