import streamlit as st
import cv2
import numpy as np
import re
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import model_from_json

from utils import (
    merge_contours, preprocess_symbol, make_square, 
    load_models, load_mini_model
)

def predict_expression_from_image(gray_img):
    # Invert and binarize
    inverted = cv2.bitwise_not(gray_img)
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = merge_contours(bounding_boxes, x_thresh=15, y_thresh=40)
    sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])

    img_copy = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in sorted_boxes:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    models, class_labels = load_models()
    mini_model, mini_class_labels = load_mini_model()

    expression = ""
    prev_label = ""

    for x, y, w, h in sorted_boxes:
        if w < 2 or h < 2 or w * h < 10:
            continue

        cropped = binary[y:y + h, x:x + w]
        input_img, _ = preprocess_symbol(cropped)
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

    # Final cleanup
    expression = re.sub(r'^[*/+\-]+', '', expression)
    expression = re.sub(r'[*/+\-]+$', '', expression)

    return expression, img_copy


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

    # Draw canvas without auto-updates to avoid mid-drawing re-runs.
    canvas_result = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=6,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=False,  # Disable continuous updates.
    )

    if canvas_result.image_data is not None:
       st.sidebar.success("Drawing detected in canvas state.")
    else:
       st.sidebar.warning("No drawing in canvas state.")
    
    if st.button("Recognize Equation"):
        if canvas_result.image_data is None:
            st.warning("No drawing detected.")
        else:
            # --- Start Detailed Debugging ---
            st.write("--- Debug: Raw Image Data ---")
            st.write(f"Shape: {canvas_result.image_data.shape}")
            st.write(f"Data type: {canvas_result.image_data.dtype}")
            st.write(f"Min value: {canvas_result.image_data.min()}")
            st.write(f"Max value: {canvas_result.image_data.max()}")

            if canvas_result.image_data.shape[2] == 4:
                alpha_channel = canvas_result.image_data[:, :, 3]
                st.write(f"Alpha channel Min: {alpha_channel.min()}")
                st.write(f"Alpha channel Max: {alpha_channel.max()}")
                # Check if *any* pixel is opaque or semi-opaque
                st.write(f"Number of pixels with Alpha > 0: {np.sum(alpha_channel > 0)}")
                # Display Alpha channel itself (should show your drawing in white/gray on black)
                st.image(alpha_channel, caption="Alpha Channel Only", use_container_width=True)
            st.write("--- End Raw Data Debug ---")

            # Display the raw RGBA data again (as you noted, this appears white)
            st.image(canvas_result.image_data, caption="Raw Drawing Data (RGBA)", use_container_width=True)

            # --- Check Scaling ---
            if canvas_result.image_data.max() <= 1:
                st.write("Scaling image data from 0-1 to 0-255 range.")
                rgba_image = (canvas_result.image_data * 255).astype(np.uint8)
            else:
                st.write("Using image data in 0-255 range (or other) directly.")
                rgba_image = canvas_result.image_data.copy()

            # Display after potential scaling
            st.image(rgba_image, caption="🎨 RGBA After Scaling Check", use_container_width=True)

            # --- Check Blending ---
            if rgba_image.shape[2] == 4:
                st.write("Blending RGBA onto white background.")
                alpha = rgba_image[:, :, 3] / 255.0
                # Display normalized alpha again if needed
                # st.image(alpha, caption="Normalized Alpha for Blending", use_container_width=True)
                white_bg = np.ones_like(rgba_image[:, :, :3], dtype=np.uint8) * 255
                blended_image = (alpha[..., None] * rgba_image[:, :, :3] + (1 - alpha[..., None]) * white_bg).astype(np.uint8)
            else:
                st.write("Image is not RGBA, skipping blending.")
                blended_image = rgba_image.copy()

            # Display the result of blending
            st.image(blended_image, caption="🧾 Blended Image (RGB)", use_container_width=True)

            # --- Check Grayscale and Thresholding ---
            st.write("Converting to Grayscale.")
            gray_img = cv2.cvtColor(blended_image, cv2.COLOR_RGB2GRAY)
            st.write(f"Grayscale Min: {gray_img.min()}, Max: {gray_img.max()}")
            st.image(gray_img, caption="🧊 Grayscale Image (Before Threshold)", use_container_width=True)

            st.write("Thresholding Grayscale Image (Threshold=200).")
            _, binary_img_before_not = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
            st.image(binary_img_before_not, caption="Binary Image (After Threshold, Before Inversion)", use_container_width=True)

            st.write("Inverting Binary Image.")
            processed_gray_img = cv2.bitwise_not(binary_img_before_not) # Renamed to avoid confusion
            st.image(processed_gray_img, caption="Processed Drawing for Detection (Final Black & White)", use_container_width=True)
            # --- End Detailed Debugging ---

            # Use the shared prediction function with the final processed image
            expression, result_img = predict_expression_from_image(processed_gray_img) # Pass the B&W image
            st.image(result_img, caption="Detected Boxes (on Original Grayscale)", use_container_width=True)

            st.markdown(
                f"<h2 style='font-size: 40px;'>✍️ Recognized Expression: <code>{expression}</code></h2>",
                unsafe_allow_html=True,
            )
            try:
                result = eval(expression)
                st.markdown(
                    f"<h2 style='font-size: 40px; color: green;'>✅ Result: <code>{expression} = {result}</code></h2>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error("❌ Failed to evaluate expression")
