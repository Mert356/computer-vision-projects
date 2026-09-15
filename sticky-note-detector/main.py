import os

import cv2
import numpy as np

from detector import detect_notes
from evaluation import coverage_percent, intersection_over_union
from generator import generate_board
from visualization import annotate_board

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
BOARD_PHOTO_PATH = os.path.join(PROJECT_DIR, "images", "boardphoto.png")
SEED = 42
NOTE_COUNT = 6


# Save an image and report a failed write.
def save_image(path: str, image: np.ndarray) -> None:
    if not cv2.imwrite(path, image):
        raise OSError(f"Could not write image: {path}")


# Detect notes, save the images, and return the predicted mask.
def process_board(board: np.ndarray, output_dir: str) -> np.ndarray:
    prediction_mask, boxes = detect_notes(board)
    coverage = coverage_percent(prediction_mask)
    annotated = annotate_board(board, boxes, coverage)

    os.makedirs(output_dir, exist_ok=True)
    save_image(os.path.join(output_dir, "board.png"), board)
    save_image(os.path.join(output_dir, "predicted_mask.png"), prediction_mask)
    save_image(os.path.join(output_dir, "annotated_board.png"), annotated)

    print(f"Detected regions: {len(boxes)}")
    print(f"Estimated coverage: {coverage:.2f}%")
    return prediction_mask


# Run the same detector with the same settings on both images.
def main() -> None:
    print("Generated board")

    board, ground_truth = generate_board(seed=SEED, note_count=NOTE_COUNT)
    prediction_mask = process_board(board, os.path.join(OUTPUT_DIR, "generated"))

    save_image(os.path.join(OUTPUT_DIR, "generated", "ground_truth_mask.png"), ground_truth)

    iou = intersection_over_union(prediction_mask, ground_truth)

    print(f"Generated notes: {NOTE_COUNT}")
    print(f"Mask IoU: {iou:.4f}")

    print("\nCorkboard example")

    board_photo = cv2.imread(BOARD_PHOTO_PATH)
    if board_photo is None:
        raise FileNotFoundError(f"Could not read image: {BOARD_PHOTO_PATH}")

    process_board(board_photo, os.path.join(OUTPUT_DIR, "boardphoto"))
    print("No reference mask is available for the corkboard example, so IoU is not calculated.")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
