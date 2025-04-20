import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# ---- Label mappings ----
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']
mini_class_labels = ['1', '2', '7']

# ---- Load CNN Ensemble ----
def load_models():
    models = []
    for i in range(10):
        with open(f'models/CNNModel_{i}.json', 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(f'models/CNNModel_{i}.weights.h5')
        models.append(model)
    return models

# ---- Load MiniCNN ----
def load_mini_model():
    with open('models/MiniCNN_127.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('models/MiniCNN_127.weights.h5')
    return model

mini_model = load_mini_model()

# ---- Merge Nearby Contours ----
def merge_contours(bounding_boxes, max_gap=10):
    if not bounding_boxes:
        return []

    # Sort left to right
    bounding_boxes.sort(key=lambda box: box[0])
    merged = [bounding_boxes[0]]

    for box in bounding_boxes[1:]:
        x, y, w, h = box
        last = merged[-1]
        lx, ly, lw, lh = last

        # Merge if close
        if x <= lx + lw + max_gap:
            nx = min(x, lx)
            ny = min(y, ly)
            nw = max(x + w, lx + lw) - nx
            nh = max(y + h, ly + lh) - ny
            merged[-1] = (nx, ny, nw, nh)
        else:
            merged.append(box)
    return merged

# ---- Pad to Square Shape ----
def make_square(img):
    h, w = img.shape
    size = max(h, w)
    square = np.ones((size, size), dtype=np.uint8) * 255
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return square

# ---- Preprocess Individual Symbol ----
def preprocess_symbol(cropped_img):
    if cropped_img.size == 0:
        return None, None

    resized = make_square(cropped_img)
    resized = cv2.resize(resized, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize and reshape for CNN
    input_img = resized.astype("float32") / 255.0
    input_img = np.expand_dims(input_img, axis=(0, -1))  # shape (1, 28, 28, 1)

    return input_img, resized  # return both CNN input and visualization
