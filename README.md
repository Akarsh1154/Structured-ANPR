# 🚗 Structured-ANPR
### Automated Number Plate Recognition System

A modular, production-ready end-to-end computer vision pipeline for detecting, preprocessing, and recognizing vehicle license plates. Built with **YOLOv11**, **OpenCV**, and **PaddleOCR**, it delivers high-accuracy OCR even under non-ideal imaging conditions.

---

## ✨ Features

- 🎯 **Precision Detection** — Custom-trained YOLOv11 architecture for localized license plate detection with safety-buffered cropping
- 🖼️ **Intelligent Preprocessing** — Automated deskewing, denoising, and bilinear rescaling to normalize text orientation and clarity
- 🔤 **Robust OCR** — PaddleOCR integration for high-confidence alphanumeric character extraction using CRNN-based sequence recognition
- 🖥️ **Interactive Dashboard** — Streamlit-powered UI providing real-time visualization of the full transformation pipeline

---

## 🛠️ Technical Architecture

The pipeline consists of three sequential stages:

### 1. Localization — YOLOv11
A deep CNN predicts bounding boxes around license plates in the input image. A **10px safety buffer** is applied around the detected region to prevent character edge-clipping during the crop phase.

### 2. Image Optimization — OpenCV
The cropped plate region undergoes a series of enhancement operations:
- **Median Blurring** — Eliminates digital noise and sensor artifacts
- **Geometric Deskewing** — Computes minimum-area bounding rectangle to correct angular tilt (up to ±25°)
- **Cubic Interpolation** — Upscales low-resolution crops for better OCR input quality

### 3. Recognition — PaddleOCR
The restored plate image is passed to PaddleOCR, which uses a **CRNN (Convolutional Recurrent Neural Network)** for robust sequence-based text recognition and outputs the final plate string.

---

## 📁 Project Structure
```
Structured-ANPR/
│
├── model.py            # YOLOv11 inference and plate cropping logic
├── preprocessing.py    # Image restoration and geometric correction suite
├── ocrengine.py        # PaddleOCR wrapper and text extraction
├── app.py              # Streamlit web interface
├── train.py            # Model training script
│
├── datasets/           # Training and evaluation datasets
├── images/             # Sample input images
├── runs/detect/        # YOLO detection run outputs
│
├── yolo11n.pt          # Pre-trained YOLOv11 model weights
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/Akarsh1154/Structured-ANPR.git
cd Structured-ANPR

# Install dependencies
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to access the interactive dashboard.

---

## 🔄 Pipeline Flow
```
Input Image
     │
     ▼
┌─────────────┐
│   YOLOv11   │  ──▶  Detects & crops license plate region
└─────────────┘
     │
     ▼
┌──────────────────┐
│  OpenCV Pipeline │  ──▶  Denoise → Deskew → Upscale
└──────────────────┘
     │
     ▼
┌─────────────┐
│  PaddleOCR  │  ──▶  Extracts alphanumeric plate text
└─────────────┘
     │
     ▼
  Plate Text Output
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv11 (Ultralytics) |
| Image Processing | OpenCV |
| OCR Engine | PaddleOCR |
| Web Interface | Streamlit |
| Language | Python 3 |

---

## 📊 Training Your Own Model

To train a custom YOLOv11 model on your own dataset:
```bash
python train.py
```

Make sure your dataset is structured under the `datasets/` directory in YOLO-compatible format (images + labels).


## 📄 License

This project is open-source. See the repository for license details.

---
