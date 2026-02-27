# 🚗 Structured-ANPR

An end-to-end Automatic Number Plate Recognition (ANPR) system using YOLOv11 and PaddleOCR.

## 🌟 Overview
This project detects vehicle license plates in images, cleans the detected region using computer vision techniques, and extracts the text with high accuracy.

## 🛠️ Tech Stack
* **Detector:** YOLOv11 (Ultralytics)
* **OCR Engine:** PaddleOCR
* **Image Processing:** OpenCV & NumPy
* **Web UI:** Streamlit

## ⚙️ How It Works
1. **Detection:** A custom-trained **YOLOv11** model locates the plate and crops it with a 10px safety buffer.
2. **Preprocessing:** The crop undergoes **rescaling**, **denoising**, and **deskewing** (auto-rotation up to 25°) to align text for the OCR.
3. **Extraction:** **PaddleOCR** processes the cleaned image to return the alphanumeric plate number.

## 🚀 Quick Start

### 1. Installation
```bash
pip install ultralytics paddleocr paddlepaddle opencv-python streamlit numpy matplotlib
