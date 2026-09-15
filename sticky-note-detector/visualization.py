import cv2
import numpy as np


# Draw numbered boxes on a copy and add a summary above the image.
def annotate_board(image: np.ndarray, boxes: list, coverage: float) -> np.ndarray:
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    padding = 3

    for number, (left, top, width, height) in enumerate(boxes, start=1):
        cv2.rectangle(annotated, (left, top), (left + width - 1, top + height - 1), (0, 255, 0), 1)

        label = f"Note {number}"
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, 1)

        label_width = text_width + 2 * padding
        label_height = text_height + baseline + 2 * padding
        label_left = max(0, min(left, image.shape[1] - label_width))
        label_top = max(0, top - label_height)
        label_corner = (label_left + label_width - 1, label_top + label_height - 1)
        text_position = (label_left + padding, label_top + padding + text_height)
        cv2.rectangle(annotated, (label_left, label_top), label_corner, (0, 0, 0), -1)
        cv2.putText(annotated, label, text_position, font, font_scale, (255, 255, 255), 1)

    annotated = cv2.copyMakeBorder(annotated, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))

    summary = f"Regions: {len(boxes)} | Coverage: {coverage:.2f}%"

    cv2.putText(annotated, summary, (10, 21), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated
