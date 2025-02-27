# YOLOv8 Segmentation Training & Inference on a custim dataset

This repository provides scripts for training and evaluating a **YOLOv8 segmentation model** on a custom toy dataset.  
It utilizes [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for training and inference. The code remains the same for any custom dataset. 

## Citation
If you use this repository, please cite Ultralytics YOLOv8 as:
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}

## Use labelme to annotate and create your dataset ad convert json to YOLO format

## Install necessary packages
pip install -r requirements.txt

##dataset.yaml
train: your_folder_path/train
val: your_folder_path/val
nc: 1
names: ['drop']

Create this yaml file and save it in your folder

##Dataset Structure
dataset/
│── train
│   ├── images/   # Training images
│   ├── labels/   # Training labels in YOLO format
│── validation/
│   ├── images/   # Val images
│   ├── labels/   # Val labels in YOLO format
│── dataset.yaml

##Training
python your_folder_path/train.py --path your_folder_path --epochs 100 --batch 32 --imgsz 640

##View Training Results
1.  Navigate to 
    cd runs/segment/trainX/
	Open results.png to view training loss/metrics.

2.  Alternatively, visyalize using Matplotlib
    ```	
	import matplotlib.pyplot as plt
	import cv2

	img = cv2.imread("runs/segment/trainX/results.png")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	plt.figure(figsize=(10, 6))
	plt.imshow(img)
	plt.axis("off")
	plt.title("YOLOv8 Training Results")
	plt.show()
	
	```