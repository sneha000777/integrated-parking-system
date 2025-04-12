import os
import re
from glob import glob

# Paths to images and labels
image_dir = "abukidataset/train/images"
label_dir = "abukidataset/train/labels"

# Rename images
for file_path in glob(f"{image_dir}/*.jpg"):
    filename = os.path.basename(file_path)
    new_filename = re.sub(r"_jpg\.rf\.[a-z0-9]+", "", filename)  # Remove .rf.xxxxxx
    os.rename(file_path, os.path.join(image_dir, new_filename))

# Rename labels
for file_path in glob(f"{label_dir}/*.txt"):
    filename = os.path.basename(file_path)
    new_filename = re.sub(r"_jpg\.rf\.[a-z0-9]+", "", filename)  # Remove .rf.xxxxxx
    os.rename(file_path, os.path.join(label_dir, new_filename))

print("✅ Filenames cleaned successfully!")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load pretrained YOLOv8n model
model.train(data="/content/abukidataset/data.yaml", epochs=50, imgsz=640, patience=20)