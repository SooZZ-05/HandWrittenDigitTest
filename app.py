import streamlit as st
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from utils import load_models, load_mini_model, merge_contours, preprocess_symbol, class_labels, mini_class_labels
from PIL import Image
import io

# ---- Load Models Once ----
st.cache_resource

def load_all_models():
    models = load_models()
    mini_model = load_mini_model()
    return models, mini_model

models, mini_model = load_all_models()

# ---- Streamlit Interface ----
st.title("🧮 Handwritten Math Expression Recognizer")
st.write("Upload a handwritten math expression image. The app will detect, classify, and solve the expression.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)

    st.subheader("1. Original Image")
    st.image(img, caption="Inverted grayscale", use_column_width=True, channels="GRAY")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    st.subheader("2. Binarized Image")
    st.image(binary, caption="Binary Thresholded", use_column_width=True, channels="GRAY")

    # Detect contours and draw boxes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = merge_contours(bounding_boxes, x_thresh=15, y_thresh=40)
    sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])

    img_boxed = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in sorted_boxes:
        cv2.rectangle(img_boxed, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.subheader("3. Detected Bounding Boxes")
    st.image(img_boxed, caption="Merged & Sorted Contours", use_column_width=True)

    expression = ""
    prev_label = ""
    wrong_predictions = []

    st.subheader("4. Symbol-wise Detection")
    for idx, (x, y, w, h) in enumerate(sorted_boxes):
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

        if confidence < 0.3 or (predicted_label in "+-*/" and prev_label in "+-*/"):
            continue

        expression += predicted_label
        prev_label = predicted_label

        # Display subplot for each symbol
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(cropped, cmap='gray')
        axes[0].set_title("Cropped")
        axes[1].imshow(padded_visual, cmap='gray')
        axes[1].set_title("CNN Input")
        axes[2].bar(class_labels, avg_prediction[0])
        axes[2].set_title(f"Prediction\n{predicted_label} ({confidence:.2f})")
        axes[2].set_ylim([0, 1.0])
        for ax in axes:
            ax.axis('off')
        st.pyplot(fig)

    # Clean expression
    expression = re.sub(r'^[*/+\-]+', '', expression)
    expression = re.sub(r'[*/+\-]+$', '', expression)

    st.subheader("5. Recognized Expression")
    st.code(expression, language='text')

    st.subheader("6. Final Answer")
    try:
        result = eval(expression)
        st.success(f"{expression} = {result}")
    except Exception as e:
        st.error(f"Failed to evaluate expression: {e}")
