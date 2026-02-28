import streamlit as st
import cv2
import numpy as np
import os
from model import get_plate_crop
from preprocessing import preprocess_image
from ocrengine import ocr

def main():
    st.title("🚗 Number Plate Recognition")
    
    if 'history' not in st.session_state:
        st.session_state.history = []

    uploaded_file = st.file_uploader("Upload plate", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 1. Save the original upload
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 2. Get the cropped plate (returns a numpy array)
        number_plate_image = get_plate_crop("temp_image.jpg")
        
        if number_plate_image is None:
            st.error("No number plate detected. Please try another image.")
            return
        
        # Display the crop
        st.image(number_plate_image, caption="Detected Number Plate", use_container_width=True)

        # 3. FIX: Save the cropped array to a temp file so preprocessing can read it
        cv2.imwrite("temp_crop.jpg", number_plate_image)

        # 4. Pass the PATH of the crop to preprocessing
        processed_image = preprocess_image("temp_crop.jpg")
        
        if processed_image is not None:
            st.image(processed_image, caption="Processed Image", use_container_width=True)

            # 5. Run OCR
            result = ocr(processed_image)
            st.markdown(f'<p style="font-size:30px;">Recognized Text: **{result}**</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()