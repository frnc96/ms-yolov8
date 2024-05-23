from ultralytics import YOLO
import numpy as np
import cv2


# Load a pretrained YOLOv8n model
model = YOLO('/home/frencis/D1/sbatch/runs/detect/4ch-yolov8x-niicu-640 (bicubic)/weights/best.pt')

# Define path to video file
source = f'/home/frencis/D1/datasets/niicu/train/images'

# Run inference on the source
results = model(source, stream=True)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(f'/home/frencis/D1/sbatch/videos/4ch_training_set.avi', fourcc, 30, (2706, 1980))

# Process results list
for idx, result in enumerate(results):
    frame = result.plot()  # TRGB-order numpy array

    r = frame[:, :, 0]
    g = frame[:, :, 1]
    b = frame[:, :, 2]

    frame = np.dstack((r, g, b))

    # Write the frame into the video file
    out.write(frame)

    # if idx == 180:
    #     break

# Release the video writer
out.release()
