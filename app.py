import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import model_from_json

from utils import (
    merge_contours, preprocess_symbol, make_square, 
    load_models, load_mini_model
)

st.set_page_config(page_title="Handwritten Equation App", layout="wide")

# App title
st.title("🧠 Handwritten Equation Recognizer")

mode = st.sidebar.selectbox("Choose Input Mode", ["📤 Upload Image", "✍️ Draw on Whiteboard"])

# File uploader
if mode == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image of a handwritten equation", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
        img = cv2.bitwise_not(img)
        st.image(img, caption="Inverted Image", use_container_width=True)
    
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        merged_boxes = merge_contours(bounding_boxes, x_thresh=15, y_thresh=40)
        sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])
    
        # Draw detected boxes
        img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in sorted_boxes:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(img_copy, caption="Detected Boxes", use_container_width=True)
    
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
    
        st.markdown(
            f"<h2 style='font-size: 40px;'>✍️ Recognized Expression: <code>{expression}</code></h2>",
            unsafe_allow_html=True,
        )
        
        try:
            result = eval(expression)
            st.markdown(
                f"<h2 style='font-size: 40px; color: green;'>✅ Result: <code>{expression} = {result}</code></h2>",
                unsafe_allow_html=True
            )
        except:
            st.error("❌ Failed to evaluate expression")

elif mode == "✍️ Draw on Whiteboard":
    st.subheader("Draw your equation below:")

    # Canvas for free drawing
    canvas_result = st_canvas(
        fill_color="#000000", 
        stroke_width=6,
        stroke_color="#000000", 
        background_color="#FFFFFF",
        height=280, 
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert the drawn canvas (RGBA to RGB) and then to grayscale
        drawn_image = (canvas_result.image_data[:, :, :3] * 255).astype(np.uint8)

        # Convert drawn image to grayscale
        gray_img = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2GRAY)

        # Invert the image for better contrast (white on black)
        gray_img = cv2.bitwise_not(gray_img)
        st.image(gray_img, caption="Inverted Drawing", use_container_width=True)

        # Preprocess the image for contour detection
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        merged_boxes = merge_contours(bounding_boxes, x_thresh=15, y_thresh=40)
        sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])

        # Draw detected boxes on the image
        img_copy = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)
        for x, y, w, h in sorted_boxes:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(img_copy, caption="Detected Boxes", use_container_width=True)

        # Load models (call only once)
        models, class_labels = load_models()
        mini_model, mini_class_labels = load_mini_model()

        expression = ""
        prev_label = ""
        wrong_predictions = []

        # Predict and build the expression
        for x, y, w, h in sorted_boxes:
            if w < 2 or h < 2 or w * h < 10:
                continue

            cropped = binary[y:y+h, x:x+w]
            input_img, padded_visual = preprocess_symbol(cropped)
            if input_img is None:
                continue

            # Get predictions from all models
            predictions = [m.predict(input_img, verbose=0) for m in models]
            avg_prediction = np.mean(predictions, axis=0)
            confidence = np.max(avg_prediction)
            predicted_class = np.argmax(avg_prediction)
            predicted_label = class_labels[predicted_class]

            # Handle mini-model corrections
            if predicted_label in mini_class_labels and confidence <= 1.0:
                mini_pred = mini_model.predict(input_img, verbose=0)
                mini_class = np.argmax(mini_pred)
                mini_label = mini_class_labels[mini_class]
                if mini_label != predicted_label:
                    predicted_label = mini_label

            if confidence < 0.3:
                continue

            # Avoid consecutive operators
            if predicted_label in "+-*/" and prev_label in "+-*/":
                continue

            expression += predicted_label
            prev_label = predicted_label

        # Final cleanup
        expression = re.sub(r'^[*/+\-]+', '', expression)
        expression = re.sub(r'[*/+\-]+$', '', expression)

        # Show recognized expression
        st.markdown(
            f"<h2 style='font-size: 40px;'>✍️ Recognized Expression: <code>{expression}</code></h2>",
            unsafe_allow_html=True,
        )

        # Evaluate and show result
        try:
            result = eval(expression)
            st.markdown(
                f"<h2 style='font-size: 40px; color: green;'>✅ Result: <code>{expression} = {result}</code></h2>",
                unsafe_allow_html=True
            )
        except:
            st.error("❌ Failed to evaluate expression")
