# utils.py
import cv2
import numpy as np
import re
from keras.models import model_from_json

def merge_contours(boxes, x_thresh=15, y_thresh=40):
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        group = [(x1, y1, w1, h1)]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            x2, y2, w2, h2 = boxes[j]
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            horizontal_overlap = (x1 <= x2 + w2 and x2 <= x1 + w1)
            horizontal_close = abs((x1 + w1) - x2) < x_thresh or abs((x2 + w2) - x1) < x_thresh
            vertical_close = abs(cy1 - cy2) < y_thresh
            both_narrow = w1 < 20 and w2 < 20
            stacked_vertically = abs(cx1 - cx2) < 10 and (abs((y1 + h1) - y2) < y_thresh or abs((y2 + h2) - y1) < y_thresh)
            is_divide_candidate = both_narrow and stacked_vertically
            if (horizontal_overlap or horizontal_close) and vertical_close:
                if w1 > 20 and w2 > 20:
                    continue
                group.append((x2, y2, w2, h2))
                used[j] = True
            elif is_divide_candidate:
                group.append((x2, y2, w2, h2))
                used[j] = True
        x_coords = [b[0] for b in group]
        y_coords = [b[1] for b in group]
        x_ends = [b[0] + b[2] for b in group]
        y_ends = [b[1] + b[3] for b in group]
        merged_x = min(x_coords)
        merged_y = min(y_coords)
        merged_w = max(x_ends) - merged_x
        merged_h = max(y_ends) - merged_y
        merged.append((merged_x, merged_y, merged_w, merged_h))
    return merged

def make_square(image, value=0):
    h, w = image.shape
    if h == w:
        return image
    if h > w:
        pad = h - w
        left = pad // 2
        right = pad - left
        return cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=value)
    else:
        pad = w - h
        top = pad // 2
        bottom = pad - top
        return cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=value)

def preprocess_symbol(cropped):
    h, w = cropped.shape[:2]
    if h == 0 or w == 0:
        return None, None
    scale = 20.0 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    padded_visual = padded.copy()
    padded = padded.astype('float32') / 255.0
    padded = padded.reshape(1, 28, 28, 1)
    return padded, padded_visual

def load_models(model_paths, mini_model_path, mini_weights_path, mini_labels):
    models = []
    for json_path, weights_path in model_paths:
        with open(json_path, "r") as f:
            model = model_from_json(f.read())
        model.load_weights(weights_path)
        models.append(model)
    with open(mini_model_path, "r") as f:
        mini_model = model_from_json(f.read())
    mini_model.load_weights(mini_weights_path)
    return models, mini_model, mini_labels

def predict_expression(binary, models, mini_model, class_labels, mini_labels):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = merge_contours(bounding_boxes)
    sorted_boxes = sorted(merged_boxes, key=lambda b: b[0])
    expression = ""
    prediction_data = []

    prev_label = ""
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

        if predicted_label in mini_labels and confidence <= 1.0:
            mini_pred = mini_model.predict(input_img, verbose=0)
            mini_class = np.argmax(mini_pred)
            mini_label = mini_labels[mini_class]
            if mini_label != predicted_label:
                predicted_label = mini_label

        if confidence < 0.3 or (predicted_label in "+-*/" and prev_label in "+-*/"):
            continue

        expression += predicted_label
        prev_label = predicted_label

        prediction_data.append({
            "bbox": (x, y, w, h),
            "cropped": cropped,
            "cnn_input": padded_visual,
            "confidences": avg_prediction[0],
            "label": predicted_label,
            "confidence": confidence
        })

    expression = re.sub(r'^[*/+\-]+', '', expression)
    expression = re.sub(r'[*/+\-]+$', '', expression)

    try:
        result = eval(expression)
    except:
        result = None

    return expression, result, sorted_boxes, prediction_data
