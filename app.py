import streamlit as st
from preprocessing import preprocess_image
from ocrengine import ocr
import numpy as np
import os

def main():
    st.title("🚗 Number Plate Recognition")
    
    if 'history' not in st.session_state:
        st.session_state.history = []

    uploaded_file = st.file_uploader("Upload plate", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        processed_image = preprocess_image("temp_image.jpg")
        st.image(processed_image, caption="Processed Image", use_container_width=True)

        result = ocr(processed_image)
        st.markdown(f'<p style="font-size:30px;">Recognized Text: {result}</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main() 