import streamlit as st
from model import get_plate_crop
from preprocessing import preprocess_image
from ocrengine import ocr
import numpy as np
import os
from model import get_plate_crop

def main():
    st.title("🚗 Number Plate Recognition")
    
    if 'history' not in st.session_state:
        st.session_state.history = []

    uploaded_file = st.file_uploader("Upload plate", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        number_plate_image = get_plate_crop("temp_image.jpg")
        if number_plate_image is None:
            st.error("No number plate detected. Please try another image.")
            return
        else:
            st.image(number_plate_image, caption="Detected Number Plate", use_container_width=True)

        processed_image = preprocess_image("temp_image.jpg")
        st.image(processed_image, caption="Processed Image", use_container_width=True)

        result = ocr(processed_image)
        st.markdown(f'<p style="font-size:30px;">Recognized Text: {result}</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main() 