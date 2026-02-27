import streamlit as st
from paddleocr import PaddleOCR

@st.cache_resource
def load_model():
    return PaddleOCR(use_angle_cls=True, lang='en')

def ocr(image):
    model = load_model()
    result = model.ocr(image)

    if not result or result[0] is None:
        return "No text found"

    try:
        lines = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.5:  # Filter low-confidence results
                lines.append(text)
        return "\n".join(lines) if lines else "No confident text found"
    except (IndexError, TypeError):
        return "Error parsing text"