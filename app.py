import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from tensorflow.keras.models import model_from_json

from utils import (
    merge_contours, preprocess_symbol, make_square, 
    load_models, load_mini_model
)

# App title
st.title("🧠 Handwritten Equation Recognizer")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a handwritten equation", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img = cv2.bitwise_not(img)
    st.image(img, caption="Inverted Image", use_column_width=True)

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = merge_contours(bounding_boxes, x_thresh=15, y_thresh=40)
    sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])

    # Draw detected boxes
    img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in sorted_boxes:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    st.image(img_copy, caption="Detected Boxes", use_column_width=True)

    # Load models (call only once)
    models, class_labels = load_models()
    mini_model, mini_class_labels = load_mini_model()

    expression = ""
    prev_label = ""
    wrong_predictions = []

    for x, y, w, h in sorted_boxes:
        if w < 2 or h < 2 or w * h < 10:
            continue

        cropped = binary[y:y+h, x:x+w]
        input_img, padded_visual = preprocess_symbol(cropped)
        if input_img is None:
            continue

        predictions = [m.predict(input_img, verbose=0) for m in models]
        avg_prediction = np.mean(predictions, axis=0)

        confidence = np.max(avg_prediction)
        predicted_class = np.argmax(avg_prediction)
        predicted_label = class_labels[predicted_class]

        if predicted_label in mini_class_labels and confidence <= 1.0:
            mini_pred = mini_model.predict(input_img, verbose=0)
            mini_class = np.argmax(mini_pred)
            mini_label = mini_class_labels[mini_class]
            if mini_label != predicted_label:
                predicted_label = mini_label

        if confidence < 0.3:
            continue

        if predicted_label in "+-*/" and prev_label in "+-*/":
            continue

        expression += predicted_label
        prev_label = predicted_label

    # Final cleanup
    expression = re.sub(r'^[*/+\-]+', '', expression)
    expression = re.sub(r'[*/+\-]+$', '', expression)

    st.markdown(f"### ✍️ Recognized Expression: `{expression}`")

    try:
        result = eval(expression)
        st.success(f"✅ Result: `{expression} = {result}`")
    except:
        st.error("❌ Failed to evaluate expression")
