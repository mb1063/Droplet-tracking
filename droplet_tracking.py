# -*- coding: utf-8 -*-

import ultralytics
from ultralytics import YOLO
import torch
import argparse
import os 

# ===========================
# Check for GPU
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ===========================
# Parse command-line arguments
# ===========================
parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation Model")
parser.add_argument("--path", type=str, required=True, help="path to dataset folder")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
parser.add_argument("--imgsz", type=int, default=640, help="Input image size")

args = parser.parse_args()

# Construct the dataset.yaml path
dataset_path = os.path.join(args.path, "dataset.yaml")

# Check if dataset.yaml exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Error: {dataset_path} not found. Please check the dataset path.")

# ===========================
# Load Segmentation Model
# ===========================
yolo_model = os.path.join(args.path, "yolov8n-seg.pt")
model = YOLO(yolo_model)

# ===========================
# Train the model
# ===========================
model.train(data = dataset_path, epochs = args.epochs, imgsz = args.imgsz, batch = args.batch, device = device, plots = True)

# ===========================
# Load Trained Model for Testing
# ===========================
# trained_model_path = os.path.join(args.path, "runs/segment/train/weights/best.pt")  # Update this path if needed
# model = YOLO(trained_model_path)


# ===========================
# Test Model on a Video File
# ===========================
video_path = os.path.join(args.path,"test/ewod_from_ppt_mix2.mp4") # make sure the video is in the test folder
project_path = os.path.join(args.path,"runs/segment")
save_path = os.path.join(project_path,"predict")
result = model.predict(source = video_path, save=True, conf=0.15, stream = True, project=project_path, name='predict')

# Visualize the results
for i, r in enumerate(result):

    filename = os.path.join(save_path,"f'results{i}.jpg")
    # Save results to disk
    r.save(filename = filename)