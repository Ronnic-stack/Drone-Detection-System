from roboflow import Roboflow
import os
import shutil

print("Connecting to Roboflow API...")

rf = Roboflow(api_key="5ERQdGLh5DiFSSP9LTTB")
project = rf.workspace("ronnicekka4-gmail-com").project("drone-detection-hwfsv-rlapn")
version = project.version(1)
dataset = version.download("yolov8")

print(f"\nDataset downloaded successfully to: {dataset.location}")

dest_dir = "./training_set"
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

shutil.move(dataset.location, dest_dir)
print(f"Data moved to {dest_dir}. Ready for training!")