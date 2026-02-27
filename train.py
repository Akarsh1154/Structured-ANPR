import os
from ultralytics import YOLO

# Get the absolute path to your project folder
base_path = r"C:\Users\ayush\OneDrive\Desktop\PROJECTS\MINOR PERSONAL PROJECT\Structured-ANPR"
data_yaml_path = os.path.join(base_path, "datasets", "data.yaml")

# Load the base model
model = YOLO("yolo11n.pt")

# Start training
# This will create the 'runs' folder once it starts
model.train(
    data=data_yaml_path, 
    epochs=50, 
    imgsz=640,
    device='cpu' # Using CPU as per your system specs
)