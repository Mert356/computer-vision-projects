import cv2
import numpy as np


# Create a colored note with short strokes and rotate it without clipping its corners.
def _create_note(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, rng.integers(20, 50, size=2))
    angle = int(rng.integers(-8, 9))
    hue = int(rng.integers(0, 180))
    note_hsv = np.full((height, width, 3), (hue, 120, 240), dtype=np.uint8)

    stroke_rows = rng.choice(np.arange(4, height - 4, 3), size=3, replace=False)
    for row in stroke_rows:
        length = int(rng.integers(5, width - 10))
        start_x = int(rng.integers(4, width - length - 4))
        cv2.line(note_hsv, (start_x, int(row)), (start_x + length, int(row)), (0, 0, 0), 1)

    note = cv2.cvtColor(note_hsv, cv2.COLOR_HSV2BGR)
    note_mask = np.full((height, width), 255, dtype=np.uint8)

    radians = np.deg2rad(angle)
    rotated_width = int(np.ceil(abs(np.sin(radians)) * height + abs(np.cos(radians)) * width))
    rotated_height = int(np.ceil(abs(np.cos(radians)) * height + abs(np.sin(radians)) * width))

    center = ((width - 1) / 2, (height - 1) / 2)

    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotation[0, 2] += (rotated_width - 1) / 2 - center[0]
    rotation[1, 2] += (rotated_height - 1) / 2 - center[1]

    output_size = (rotated_width, rotated_height)

    rotated_note = cv2.warpAffine(note, rotation, output_size, flags=cv2.INTER_NEAREST)
    rotated_mask = cv2.warpAffine(note_mask, rotation, output_size, flags=cv2.INTER_NEAREST)

    return rotated_note, rotated_mask


# Generate a noisy board with separate notes and a reference mask for evaluation.
def generate_board(seed: int = 42, note_count: int = 6) -> tuple[np.ndarray, np.ndarray]:
    if note_count < 0:
        raise ValueError("note_count must be non-negative.")

    rng = np.random.default_rng(seed)
    height, width = 300, 400
    margin = 10

    board = np.full((height, width, 3), (45, 55, 65), dtype=np.float32)
    board += rng.normal(0, 8, (height, width, 1))
    board = np.clip(board, 0, 255).astype(np.uint8)

    ground_truth = np.zeros((height, width), dtype=np.uint8)
    occupied_boxes = []

    for _ in range(note_count):
        note, note_mask = _create_note(rng)
        note_height, note_width = note.shape[:2]

        for _ in range(1000):
            left = int(rng.integers(margin, width - margin - note_width + 1))
            top = int(rng.integers(margin, height - margin - note_height + 1))
            right, bottom = left + note_width, top + note_height
            overlaps = False
            for x1, y1, x2, y2 in occupied_boxes:
                if not (right < x1 or left > x2 or bottom < y1 or top > y2):
                    overlaps = True
                    break
            if not overlaps:
                break
        else:
            raise RuntimeError("Could not place all notes. Try fewer notes or another seed.")

        occupied_boxes.append((left, top, right, bottom))
        note_pixels = note_mask > 0
        board[top:bottom, left:right][note_pixels] = note[note_pixels]
        ground_truth[top:bottom, left:right][note_pixels] = 255

    horizontal_light = np.linspace(1.0, 0.5, width)[None, :]
    vertical_light = np.linspace(1.0, 0.5, height)[:, None]

    illumination = (horizontal_light + vertical_light) / 2
    board = (board.astype(np.float32) * illumination[..., None]).astype(np.uint8)
    return board, ground_truth
