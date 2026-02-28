# Structured-ANPR: Automated License Plate Recognition System

A modular, end-to-end computer vision pipeline designed to detect, preprocess, and recognize vehicle license plates. This project integrates state-of-the-art deep learning models with advanced image processing to deliver high-accuracy OCR results even in non-ideal conditions.

## 🚀 Key Features
* **Precision Detection:** Custom-trained **YOLOv11** architecture for localized plate detection with localized padding.
* **Intelligent Preprocessing:** Automated **deskewing**, **denoising**, and **bilinear rescaling** to normalize text orientation and clarity.
* **Robust OCR:** Integrated with **PaddleOCR** for high-confidence alphanumeric character extraction.
* **Interactive UI:** Streamlit-powered dashboard providing real-time visualization of the transformation pipeline.

## 🛠️ Technical Architecture

### 1. Localization (YOLOv11)
Utilizes a deep CNN to predict bounding boxes. The system applies a 10px safety buffer to ensure no character edge-clipping occurs during the cropping phase.

### 2. Image Optimization (OpenCV)
To maximize OCR accuracy, the cropped plate undergoes:
* **Median Blurring:** To eliminate digital noise.
* **Geometric Deskewing:** Calculation of the minimum area rectangle to correct angular tilt (up to 25°).
* **Interpolation:** Upscaling low-resolution crops using cubic interpolation.

### 3. Recognition (PaddleOCR)
The refined image is passed to the PaddleOCR engine, which utilizes a CRNN (Convolutional Recurrent Neural Network) for sequence-based text recognition.

## 📂 Project Structure
* `model.py`: YOLOv11 inference and cropping logic.
* `preprocessing.py`: Image restoration and geometric correction suite.
* `ocrengine.py`: PaddleOCR implementation.
* `main.py`: Streamlit-based web interface.

## 🚦 Getting Started

### Installation
```bash
pip install -r requirements.txt
