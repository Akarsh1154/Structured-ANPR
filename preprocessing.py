import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def read_image(path):
    img = cv.imread(path)
    # Fix: Always convert to RGB immediately for Streamlit/Paddle compatibility
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def rescale_image(img, scale=2):
    height, width = img.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    return new_img

def to_grayscale(img):
    # Fix: Ensure input is 3-channel before converting
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

def remove_noise(img):
    return cv.medianBlur(img,3)

def binary_threshold(img):
    _, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binary

def deskew_image(img):
    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return img
        
    # Get the rotation rectangle
    rect = cv.minAreaRect(coords)
    angle = rect[-1]
    
    # Logic for OpenCV 4.5+:
    # If width < height, the angle is measured from the vertical
    if rect[1][0] < rect[1][1]:
        angle = 90 + angle
    
    # Limit correction: only deskew if the angle is small (e.g., < 25 degrees)
    # This prevents the "right rotation" error on vertical objects
    if abs(angle) > 25:
        return img 

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated

def preprocess_image(path):
    img = read_image(path)
    img = rescale_image(img, scale=2)
    gray = to_grayscale(img)
    denoised = remove_noise(gray)
    deskew = deskew_image(denoised)
    
    # Final Fix: PaddleOCR and st.image expect 3 channels. 
    # This converts the binary (1-channel) back to a 3-channel grayscale.
    return cv.cvtColor(deskew, cv.COLOR_GRAY2RGB)