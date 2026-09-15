import cv2
import numpy as np


# Estimate the background from the image edges and find regions with different colors.
def detect_notes(image: np.ndarray, min_area: int = 300) -> tuple[np.ndarray, list]:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty uint8 BGR image.")

    if min_area < 1:
        raise ValueError("min_area must be positive.")

    smoothed = cv2.GaussianBlur(image, (9, 9), 0)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_pixels = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]))
    background_color = np.median(border_pixels, axis=0)

    color_distance = np.linalg.norm(lab - background_color, axis=2)
    color_distance = np.clip(color_distance, 0, 255).astype(np.uint8)
    _, raw_mask = cv2.threshold(color_distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), dtype=np.uint8)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    keep_label = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep_label[0] = False
    prediction_mask = np.where(keep_label[labels], 255, 0).astype(np.uint8)

    boxes = []

    for row in stats[keep_label]:
        x = int(row[0])
        y = int(row[1])
        width = int(row[2])
        height = int(row[3])

        boxes.append((x, y, width, height))
    boxes.sort(key=lambda box: (box[1], box[0]))

    return prediction_mask, boxes
