import re
from paddleocr import PaddleOCR
import streamlit as st  # Assuming streamlit is used for caching here as well

@st.cache_resource
def load_model():
    return PaddleOCR(use_angle_cls=True, lang='en')

def clean_plate_text(text_list):
    """
    Filters out short noise like 'GB' and non-standard characters.
    """
    cleaned_lines = []
    for text in text_list:
        # Remove non-alphanumeric characters
        clean = re.sub(r'[^A-Z0-9 ]', '', text.upper()).strip()
        
        # Filter: Only keep strings longer than 3 characters to ignore 'GB'
        if len(clean) > 3:
            cleaned_lines.append(clean)
            
    return " ".join(cleaned_lines) if cleaned_lines else "No valid plate found"

def ocr(image):
    model = load_model()
    result = model.ocr(image)

    if not result or result[0] is None:
        return "No text found"

    try:
        raw_lines = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5:
                raw_lines.append(text)
        
        # Apply the cleaning logic before returning
        return clean_plate_text(raw_lines)
        
    except (IndexError, TypeError):
        return "Error parsing text"