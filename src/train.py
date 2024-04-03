from ultralytics import YOLO
from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description='MS-YOLOv8 training script.')

parser.add_argument('--modelsz', type=str, default='x', help='n, s, m, l or x')
parser.add_argument('--imgsz', type=int, default=640, help='640 or 1280')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')

args = parser.parse_args()


# Config
MODEL_YAML_PATH = Path(f"yolov8{args.modelsz}")
DATA_YAML_PATH = Path(
    "/home/frencis/D1/datasets/niicu_rgb/data.yaml",
    # "/home/frencis/D1/datasets/llvip_niicu_rgb/data.yaml",
)
EXP_NAME = f"yolov8{args.modelsz}-coco-niicu-{args.imgsz}"


# Load a model
model = YOLO(
    model=MODEL_YAML_PATH,
    verbose=True,
).load(
    f'yolov8{args.modelsz}.pt',
)


results = model.train(
    data=DATA_YAML_PATH,
    single_cls=True,
    plots=True,
    epochs=args.epochs,
    imgsz=args.imgsz,
    name=EXP_NAME,
)
