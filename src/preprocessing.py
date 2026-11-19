"""Conversion of a canvas drawing into an 8x8 vector matching the `load_digits` format.

The scikit-learn digits are 8x8 images with integer intensities in [0, 16], where
a high value means "ink". The digit is roughly centred and fills most of the
frame. A raw canvas drawing differs (large, off-centre, arbitrary stroke
width), so it is cropped, padded to a square, resized and rescaled.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

TARGET_SIZE = 8
MAX_PIXEL_VALUE = 16.0
# Pixels dimmer than this fraction of the maximum are considered background.
INK_THRESHOLD = 0.08
# Fraction of the 8x8 frame filled by the digit (the training digits leave a margin).
DIGIT_FILL = 6
# Below this share of ink pixels the drawing is considered empty.
MIN_INK_PIXELS = 5


def to_grayscale(image: np.ndarray | Image.Image) -> np.ndarray:
    """Return a float array in [0, 1] where 1 is ink (white stroke on black).

    RGBA input is composited onto a black background so that transparent
    canvas pixels are treated as background.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("L'image doit avoir 2 ou 3 dimensions.")
    arr = arr.astype(np.float64)
    if arr.size == 0:
        raise ValueError("L'image est vide.")
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 4:
            alpha = arr[..., 3:4] / 255.0
            arr = arr[..., :3] * alpha
        elif channels == 1:
            arr = arr[..., :1]
        elif channels != 3:
            raise ValueError("Nombre de canaux non supporté.")
        arr = arr[..., :3].mean(axis=2) if arr.shape[2] >= 3 else arr[..., 0]
    scale = 255.0 if arr.max() > 1.0 else 1.0
    return np.clip(arr / scale, 0.0, 1.0)


def has_ink(gray: np.ndarray) -> bool:
    """True if the grayscale image (ink = 1) contains a drawing."""
    return int((gray > INK_THRESHOLD).sum()) >= MIN_INK_PIXELS


def crop_to_content(gray: np.ndarray) -> np.ndarray:
    """Crop to the bounding box of the ink pixels."""
    mask = gray > INK_THRESHOLD
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Aucun tracé détecté.")
    return gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def pad_to_square(gray: np.ndarray) -> np.ndarray:
    """Pad with background so the aspect ratio is preserved when resizing."""
    h, w = gray.shape
    side = max(h, w)
    out = np.zeros((side, side), dtype=gray.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    out[top : top + h, left : left + w] = gray
    return out


def _resize(gray: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
    img = img.resize((size, size), Image.Resampling.BOX if gray.shape[0] >= size else Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def _center_by_mass(frame: np.ndarray) -> np.ndarray:
    """Shift the image so its centre of mass sits at the frame centre."""
    total = frame.sum()
    if total <= 0:
        return frame
    ys, xs = np.indices(frame.shape)
    cy, cx = (ys * frame).sum() / total, (xs * frame).sum() / total
    centre = (frame.shape[0] - 1) / 2
    dy, dx = int(round(centre - cy)), int(round(centre - cx))
    out = np.zeros_like(frame)
    src_y = slice(max(0, -dy), frame.shape[0] - max(0, dy))
    dst_y = slice(max(0, dy), frame.shape[0] - max(0, -dy))
    src_x = slice(max(0, -dx), frame.shape[1] - max(0, dx))
    dst_x = slice(max(0, dx), frame.shape[1] - max(0, -dx))
    out[dst_y, dst_x] = frame[src_y, src_x]
    return out


def preprocess_image(image: np.ndarray | Image.Image) -> np.ndarray:
    """Return an 8x8 float array with values in [0, 16], digit centred.

    Raises ValueError if the image contains no drawing.
    """
    gray = to_grayscale(image)
    if not has_ink(gray):
        raise ValueError("Le canvas est vide.")
    gray = pad_to_square(crop_to_content(gray))
    # Resize to DIGIT_FILL x DIGIT_FILL, then place in the 8x8 frame (margin of 1).
    inner = _resize(gray, DIGIT_FILL * 4)  # intermediate resolution for smooth downsampling
    inner = _resize(inner, DIGIT_FILL)
    margin = (TARGET_SIZE - DIGIT_FILL) // 2
    frame = np.zeros((TARGET_SIZE, TARGET_SIZE))
    frame[margin : margin + DIGIT_FILL, margin : margin + DIGIT_FILL] = inner
    frame = _center_by_mass(frame)
    peak = frame.max()
    if peak > 0:
        frame = frame / peak  # use the full dynamic range like the dataset
    return np.clip(frame * MAX_PIXEL_VALUE, 0.0, MAX_PIXEL_VALUE)


def to_feature_vector(image: np.ndarray | Image.Image) -> np.ndarray:
    """Preprocessed image flattened to shape (1, 64), ready for the model."""
    return preprocess_image(image).reshape(1, -1)
