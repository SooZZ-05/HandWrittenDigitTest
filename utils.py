import re
import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json
import numpy as np

def merge_until_stable(boxes, x_thresh=15, y_thresh=0):
    prev_len = -1
    curr_boxes = boxes

    while prev_len != len(curr_boxes):
        prev_len = len(curr_boxes)
        curr_boxes = merge_contours(curr_boxes, x_thresh, y_thresh)

    return curr_boxes


def merge_contours(boxes, x_thresh=15, y_thresh=40):
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        x1, y1, w1, h1 = boxes[i]
        group = [(x1, y1, w1, h1)]
        used[i] = True

        x1_left = x1
        x1_right = x1 + w1

        for j in range(len(boxes)):
            if used[j] or i == j:
                continue

            x2, y2, w2, h2 = boxes[j]
            x2_left = x2
            x2_right = x2 + w2
            x2_center = x2 + w2 // 2

            # Condition 1: Box j's horizontal range is inside box i's range
            horizontal_contained = (x2_left >= x1_left and x2_right <= x1_right) or (x1_left >= x2_left and x1_right <= x2_right)

            # Condition 2: Box j's center is within box i's horizontal range
            center_inside = (x2_center >= x1_left and x2_center <= x1_right)

            if horizontal_contained or center_inside:
                group.append((x2, y2, w2, h2))
                used[j] = True

        # Merge grouped boxes
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

def preprocess_symbol(cropped):
    h, w = cropped.shape[:2]
    if h == 0 or w == 0:
        return None, None

    scale = 20.0 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad into 28x28
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    padded_visual = padded.copy()

    padded = padded.astype('float32') / 255.0
    padded = padded.reshape(1, 28, 28, 1)

    return padded, padded_visual

def make_square(image, value=0):
    h, w = image.shape
    if h == w:
        return image
    if h > w:
        pad = (h - w)
        left = pad // 2
        right = pad - left
        return cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=value)
    else:
        pad = (w - h)
        top = pad // 2
        bottom = pad - top
        return cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=value)

def load_models():
    models = []
    class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/']

    for i in range(10):
        json_path = f'models/CNNModel_{i}.json'
        weights_path = f'models/CNNModel_{i}.weights.h5'

        with open(json_path, 'r') as json_file:
            model = model_from_json(json_file.read())
        model.load_weights(weights_path)

        models.append(model)

    return models, class_labels

def load_mini_model():
    with open("models/MiniCNN_127.json", "r") as json_file:
        mini_model = model_from_json(json_file.read())
    mini_model.load_weights("models/MiniCNN_127.weights.h5")
    mini_class_labels = ['1', '2', '7', '9']
    return mini_model, mini_class_labels

def check_expression(expression):
    validEquation = True
    # Pattern to match three consecutive operators
    threeConsecutive = r'([*/+\-])([*/+\-])([*/+\-])'
    invalidDuo = r'([*/+\-])([*/])'
    invalidStart = r'^([*/]|[+\-*/]{2})'
    invalidEnd = r'[*/+\-]$'
    # Search for the pattern in the expression
    if re.search(threeConsecutive, expression):
        validEquation = False
    if re.search(invalidDuo, expression):
        validEquation = False
    if re.search(invalidStart, expression):
        validEquation = False
    if re.search(invalidEnd, expression):
        validEquation = False
    return validEquation

# def check_expression(expr):
#     # Must not start or end with **, *, /, etc.
#     if re.match(r'^[*/]', expr) or re.match(r'[*/+\-]$', expr):
#         return False
#     # No invalid characters
#     if not re.match(r'^[0-9+\-*/^()]+$', expr):
#         return False
#     # Validate ** appears after digit and is followed by +/- or digit
#     tokens = re.split(r'(\*\*|[+\-*/()])', expr)
#     for i, t in enumerate(tokens):
#         if t == '**':
#             if i == 0 or not tokens[i - 1].isdigit():
#                 return False
#             if i + 1 >= len(tokens) or not (tokens[i + 1].isdigit() or tokens[i + 1] in '+-'):
#                 return False
#     return True

