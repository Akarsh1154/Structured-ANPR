import os
import cv2
from ultralytics import YOLO

# Updated to point exactly to your successful train3 results
custom_path = r"C:\Users\ayush\runs\detect\train3\weights\best.pt"

if os.path.exists(custom_path):
    print(f"Success: Loading Custom Plate Detector from {custom_path}")
    model = YOLO(custom_path)
else:
    # This keeps the app running even if you move the project to another PC
    print("Warning: Custom weights not found at the specified path. Using generic model.")
    model = YOLO("yolo11n.pt")

def get_plate_crop(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Using a 0.5 confidence threshold ensures we only crop clear plates
    results = model(img, conf=0.5)

    for result in results:
        if len(result.boxes) == 0:
            continue
            
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Padding logic: Adds a 10px safety buffer around the plate
            h, w, _ = img.shape
            pad = 10
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

            # Return the small, localized crop for PaddleOCR
            plate_crop = img[y1:y2, x1:x2]
            return plate_crop
            
    return None