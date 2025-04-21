import cv2
import numpy as np
import os
from tensorflow.keras.models import model_from_json

import numpy as np

def merge_contours(boxes, x_thresh, y_thresh=40):
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        x1, y1, w1, h1 = boxes[i]
        x1_right = x1 + w1
        y1_bottom = y1 + h1

        group = [(x1, y1, w1, h1)]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue

            x2, y2, w2, h2 = boxes[j]
            x2_right = x2 + w2
            y2_bottom = y2 + h2

            # Check vertical alignment (stacked)
            vertically_close = abs(y1_bottom - y2) < y_thresh or abs(y2_bottom - y1) < y_thresh or abs(y1 - y2) < y_thresh

            # Check horizontal containment
            one_within_other = (
                (x2 >= x1 and x2_right <= x1_right) or
                (x1 >= x2 and x1_right <= x2_right)
            )

            if vertically_close and one_within_other:
                group.append((x2, y2, w2, h2))
                used[j] = True

        # Merge all in the group
        if len(group) > 1:
            x_vals = [x for x, _, _, _ in group]
            y_vals = [y for _, y, _, _ in group]
            x_rights = [x + w for x, _, w, _ in group]
            y_bottoms = [y + h for _, y, _, h in group]

            x_min = min(x_vals)
            y_min = min(y_vals)
            x_max = max(x_rights)
            y_max = max(y_bottoms)

            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))
        else:
            merged.append((x1, y1, w1, h1))  # just itself

    return merged



# def merge_contours(boxes, x_thresh=15, y_thresh=40):
#     merged = []
#     used = [False] * len(boxes)

#     for i in range(len(boxes)):
#         if used[i]:
#             continue

#         x1, y1, w1, h1 = boxes[i]
#         group = [(x1, y1, w1, h1)]
#         used[i] = True

#         for j in range(i + 1, len(boxes)):
#             if used[j]:
#                 continue

#             x2, y2, w2, h2 = boxes[j]

#             # Compute centers
#             cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
#             cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

#             # Horizontal overlap check
#             horizontal_overlap = (x1 <= x2 + w2 and x2 <= x1 + w1)
#             horizontal_close = abs((x1 + w1) - x2) < x_thresh or abs((x2 + w2) - x1) < x_thresh

#             # Vertical center closeness
#             vertical_close = abs(cy1 - cy2) < y_thresh

#             # --------- NEW: Divide symbol logic ----------
#             both_narrow = w1 < 20 and w2 < 20
#             stacked_vertically = abs(cx1 - cx2) < 10 and (abs((y1 + h1) - y2) < y_thresh or abs((y2 + h2) - y1) < y_thresh)
#             is_divide_candidate = both_narrow and stacked_vertically
#             # ---------------------------------------------

#             if (horizontal_overlap or horizontal_close) and vertical_close:
#                 if w1 > 20 and w2 > 20:
#                     continue  # skip merging wide boxes
#                 group.append((x2, y2, w2, h2))
#                 used[j] = True

#             elif is_divide_candidate:
#                 group.append((x2, y2, w2, h2))
#                 used[j] = True

#         # Merge grouped boxes
#         x_coords = [b[0] for b in group]
#         y_coords = [b[1] for b in group]
#         x_ends = [b[0] + b[2] for b in group]
#         y_ends = [b[1] + b[3] for b in group]

#         merged_x = min(x_coords)
#         merged_y = min(y_coords)
#         merged_w = max(x_ends) - merged_x
#         merged_h = max(y_ends) - merged_y

#         merged.append((merged_x, merged_y, merged_w, merged_h))

#     return merged

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
    mini_class_labels = ['1', '2', '7']
    return mini_model, mini_class_labels
