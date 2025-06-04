from pathlib import Path

import cv2
import numpy as np

def polygon_to_bbox(poly: list[float] | np.ndarray) -> list[float, float, float, float]:
    if isinstance(poly, np.ndarray):
        poly = poly.flatten()
    xs, ys = poly[0::2], poly[1::2]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def crop_image_path(image_path: str | Path, bbox: list[float] | np.ndarray, margin: int = 0) -> np.ndarray | None:
    """Crop a region from an image.

    Args:
        image_path (str | Path): Path to the image file.
        bbox (list[float] | np.ndarray): Bounding box [x1, y1, w, h] where x1, y1 are top-left.
        margin (int, optional): Pixels to expand the crop on each side. Defaults to 0.

    Returns:
        np.ndarray | None: Cropped image array, or None if an error occurs with a message.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: could not read image at '{image_path}'")
        return None

    return crop_image(img, bbox, margin)


def crop_image(img: np.ndarray, bbox: list[float] | np.ndarray, margin: int = 0) -> np.ndarray | None:

    if isinstance(bbox, list):
        x1, y1, w, h = bbox
    elif isinstance(bbox, np.ndarray):
        x1, y1, w, h = bbox.tolist()
    else:
        print(f"Error: unsupported bbox type {type(bbox)}")
        return None

    if int(w) < 1 or int(h) < 1:
        print(f"Error: width and height must be >=1, got w={w}, h={h}")
        return None

    x2 = x1 + w
    y2 = y1 + h
    # Apply margin and constrain within image bounds
    x1m = int(max(0, x1 - margin))
    y1m = int(max(0, y1 - margin))
    x2m = int(min(img.shape[1], x2 + margin))
    y2m = int(min(img.shape[0], y2 + margin))

    if x1m >= x2m or y1m >= y2m:
        print("Error: resulting crop has no area after applying margin")
        return None

    return img[y1m:y2m, x1m:x2m]