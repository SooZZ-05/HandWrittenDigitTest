import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from utils import (
    load_models, merge_contours, preprocess_symbol, predict_symbol,
    resolve_confusion, evaluate_expression
)

# Load models once
ensemble_models, minicorn_model = load_models()

st.set_page_config(page_title="Handwritten Equation Solver", layout="wide")
st.title("🧠 Handwritten Math Expression Recognizer")

uploaded_file = st.file_uploader("Upload an image of a handwritten equation", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    st.subheader("📷 Original Image")
    st.image(image, use_column_width=True)

    # Step 1: Convert to grayscale and find contours
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contours = merge_contours(gray)

    # Step 2: Draw contours
    boxed_image = image.copy()
    for x, y, w, h in contours:
        cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    st.subheader("🟥 Detected Symbols (Bounding Boxes)")
    st.image(boxed_image, use_column_width=True)

    symbols = []
    cnn_inputs = []
    predictions = []
    confidences = []

    # Step 3: Predict each symbol
    for box in contours:
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]
        input_img = preprocess_symbol(roi)
        cnn_inputs.append(input_img)

        label, conf, conf_dict = predict_symbol(input_img, ensemble_models)
        if conf < 0.90 and label in ['1', '2', '7']:
            label = resolve_confusion(input_img, minicorn_model)

        symbols.append(label)
        predictions.append((label, conf_dict))
        confidences.append(conf)

    # Step 4: Show CNN inputs
    st.subheader("📥 CNN Inputs (Preprocessed 28x28)")
    cols = st.columns(min(6, len(cnn_inputs)))
    for i, img in enumerate(cnn_inputs):
        with cols[i % len(cols)]:
            st.image(img.reshape(28, 28), width=50, caption=symbols[i])

    # Step 5: Plot confidence bars
    st.subheader("📊 Prediction Confidence per Symbol")
    for i, (label, conf_dict) in enumerate(predictions):
        fig, ax = plt.subplots(figsize=(4, 2))
        sorted_items = sorted(conf_dict.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_items][:5]
        values = [item[1] for item in sorted_items][:5]
        ax.bar(labels, values, color='skyblue')
        ax.set_ylim([0, 1])
        ax.set_title(f"Symbol {i+1} Prediction: {label}")
        st.pyplot(fig)

    # Step 6: Final expression and result
    expression = ''.join(symbols)
    result = evaluate_expression(expression)

    st.subheader("🧾 Final Expression")
    st.code(expression)

    st.subheader("✅ Evaluated Result")
    st.success(result)
