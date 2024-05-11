from ultralytics import YOLO
import cv2


VIDEO_FILE_NAME = 'flight3_rgb_part2'


# Load a pretrained YOLOv8n model
model = YOLO('/home/frencis/D1/sbatch/runs/detect/yolov8x-coco-niicu-640/weights/best.pt')

# Define path to video file
# source = f'/home/frencis/D1/datasets/zips/raw_data/{VIDEO_FILE_NAME}.mov'
source = f'/home/frencis/D1/datasets/niicu_rgb/val/images'

# Run inference on the source
results = model(source, stream=True)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter(f'/home/frencis/D1/sbatch/videos/{VIDEO_FILE_NAME}.avi', fourcc, 30, (3840, 2160))
out = cv2.VideoWriter(f'/home/frencis/D1/sbatch/videos/validation_set_60.avi', fourcc, 60, (2706, 1980))

# Process results list
for result in results:
    frame = result.plot()  # BGR-order numpy array

    # Write the frame into the video file
    out.write(frame)

# Release the video writer
out.release()
