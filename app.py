import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import load_models, merge_contours, preprocess_symbol, mini_model, class_labels, mini_class_labels
import re

# Load models once
models = load_models()
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("🧮 Handwritten Math Expression Recognizer")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Show original image
    image = Image.open(uploaded_file).convert('L')
    img = np.array(image)
    st.subheader("📷 Original Image")
    st.image(img, use_column_width=True)

    # Invert and binarize
    inverted = cv2.bitwise_not(img)
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = merge_contours(bounding_boxes)
    sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])

    # Draw and show bounding boxes
    img_copy = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in sorted_boxes:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.subheader("🔲 Detected Boxes")
    st.image(img_copy, use_column_width=True)

    expression = ""
    prev_label = ""
    wrong_predictions = []

    for x, y, w, h in sorted_boxes:
        if w * h < 10:
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

        # Show subplots for each symbol
        st.markdown("---")
        st.markdown(f"**🧠 Symbol at (x={x}, y={y}, w={w}, h={h})**")
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(cropped, cmap='gray')
        axes[0].set_title("Cropped")

        axes[1].imshow(padded_visual, cmap='gray')
        axes[1].set_title("Input to CNN")

        axes[2].bar(class_labels, avg_prediction[0])
        axes[2].set_ylim([0, 1.0])
        axes[2].set_title(f"Prediction: {predicted_label} ({confidence:.2f})")

        st.pyplot(fig)

        expression += predicted_label
        prev_label = predicted_label

    # Final expression
    expression = re.sub(r'^[*/+\-]+', '', expression)
    expression = re.sub(r'[*/+\-]+$', '', expression)

    st.markdown("## ✍️ Recognized Expression")
    st.code(expression)

    try:
        result = eval(expression)
        st.markdown(f"## ✅ Final Result: `{expression} = {result}`")
    except:
        st.markdown("❌ Failed to evaluate expression.")
