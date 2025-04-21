import streamlit as st
import cv2
import numpy as np
import re
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import model_from_json

from utils import (
    merge_contours, preprocess_symbol, make_square, 
    load_models, load_mini_model, merge_until_stable
)

def predict_expression_from_image(gray_img):
    # Invert and binarize
    inverted = cv2.bitwise_not(gray_img)
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)

    # Assuming 'binary' is your thresholded image
    kernel = np.ones((3, 3), np.uint8)
    
    # Apply dilation and erosion to separate touching contours
    dilated_image = cv2.dilate(binary, kernel, iterations=1)  # Expands contours
    eroded_image = cv2.erode(dilated_image, kernel, iterations=3)
    
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_once = merge_contours(bounding_boxes, x_thresh=15, y_thresh=0)
    merged_boxes = merge_contours(merged_once, x_thresh=15, y_thresh=0)
    # merged_boxes = merge_contours(bounding_boxes, x_thresh=15, y_thresh=0)
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
    
        st.image(cv2.bitwise_not(img), caption="Inverted Image", use_container_width=True)
        
        # Predict using shared function
        expression, annotated_img = predict_expression_from_image(img)

        st.image(annotated_img, caption="Detected Boxes", use_container_width=True)

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
        # update_streamlit=False,
    )

    if canvas_result.image_data is not None:
        st.sidebar.success("Drawing detected in canvas state.")
    else:
        st.sidebar.warning("No drawing in canvas state.")

    if st.button("Recognize Equation"):
        if canvas_result.image_data is None:
            st.warning("No drawing detected.")
        else:
            # Modify the check for processing:
            if canvas_result.image_data is None or canvas_result.image_data.min() == 255:
                st.error("Drawing not detected in image data (Image is blank white). Cannot process.")
            else:
                # Convert RGBA to Grayscale
                rgba_image = canvas_result.image_data.astype(np.uint8)
                if rgba_image.shape[2] == 4:
                    alpha = rgba_image[:, :, 3] / 255.0
                    white_bg = np.ones_like(rgba_image[:, :, :3], dtype=np.uint8) * 255
                    blended_image = (alpha[..., None] * rgba_image[:, :, :3] + (1 - alpha[..., None]) * white_bg).astype(np.uint8)
                    gray_img = cv2.cvtColor(blended_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = cv2.cvtColor(rgba_image, cv2.COLOR_RGB2GRAY)

                # Apply similar processing steps as the upload image prediction
                inverted = cv2.bitwise_not(gray_img)
                _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)

                # Assuming 'binary' is your thresholded image
                kernel = np.ones((3, 3), np.uint8)
                
                # Apply dilation and erosion to separate touching contours
                dilated_image = cv2.dilate(binary, kernel, iterations=1)  # Expands contours
                eroded_image = cv2.erode(dilated_image, kernel, iterations=3)
                
                contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bounding_boxes = [cv2.boundingRect(c) for c in contours]
                merged_once = merge_contours(bounding_boxes, x_thresh=15, y_thresh=0)
                merged_boxes = merge_contours(merged_once, x_thresh=15, y_thresh=0)
                # merged_boxes = merge_contours(bounding_boxes, x_thresh=15, y_thresh=0)
                sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])

                img_copy = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                for x, y, w, h in sorted_boxes:
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                st.image(img_copy, caption="Detected Boxes (on Drawn Image)", use_container_width=True)

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
