# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import load_models, predict_expression

# Set Streamlit page config
st.set_page_config(page_title="Handwritten Equation Recognizer", layout="wide")

# Class labels
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']
mini_labels = ['1', '2', '7']

# Load models (cached to avoid reloading every time)
@st.cache_resource
def load_all_models():
    model_paths = [(f"models/CNNModel_{i}.json", f"models/CNNModel_{i}.weights.h5") for i in range(10)]
    mini_model_path = "models/MiniCNNModel.json"
    mini_weights_path = "models/MiniCNNModel.weights.h5"
    return load_models(model_paths, mini_model_path, mini_weights_path, mini_labels)

models, mini_model, mini_labels = load_all_models()

# Streamlit UI
st.title("📝 Handwritten Equation Recognizer")
st.markdown("Upload an image containing a handwritten equation and the app will predict and solve it!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.subheader("Original Image")
    st.image(image, use_column_width=True, channels="GRAY")

    # Preprocessing
    inverted = cv2.bitwise_not(image)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    st.subheader("Preprocessed Image")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(inverted, cmap='gray')
    axs[0].set_title('Inverted')
    axs[0].axis('off')

    axs[1].imshow(binary, cmap='gray')
    axs[1].set_title('Binary (Thresholded)')
    axs[1].axis('off')
    st.pyplot(fig)

    # Prediction
    with st.spinner("Predicting..."):
        expression, result, sorted_boxes, prediction_data = predict_expression(binary, models, mini_model, class_labels, mini_labels)

    st.subheader("Detected Expression")
    if expression:
        st.success(f"**{expression}** = **{result}**")
    else:
        st.error("No valid expression detected.")

    # Show detected bounding boxes
    if sorted_boxes:
        st.subheader("Detected Symbols")
        boxed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in sorted_boxes:
            cv2.rectangle(boxed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(boxed_image, channels="BGR", use_column_width=True)

    # Show prediction confidences
    if prediction_data:
        st.subheader("Symbol Predictions and Confidences")
        for idx, data in enumerate(prediction_data):
            st.markdown(f"**Symbol {idx+1}: `{data['label']}` (Confidence: {data['confidence']:.2f})**")

            col1, col2 = st.columns(2)
            with col1:
                st.image(data["cnn_input"], width=150, caption="Preprocessed Input")

            with col2:
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.bar(class_labels, data["confidences"])
                ax.set_ylim(0, 1)
                ax.set_title('Prediction Confidence')
                st.pyplot(fig)
