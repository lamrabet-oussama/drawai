import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.preprocessing import (
    MAX_PIXEL_VALUE,
    crop_to_content,
    pad_to_square,
    preprocess_image,
    to_feature_vector,
    to_grayscale,
)


def canvas_with_line(box=(100, 40, 130, 240), size=280, width=22):
    """White vertical bar on black, like a drawn '1'."""
    img = Image.new("RGB", (size, size), "black")
    ImageDraw.Draw(img).line([(box[0], box[1]), (box[0], box[3])], fill="white", width=width)
    return np.asarray(img)


def test_output_shape_and_range():
    out = preprocess_image(canvas_with_line())
    assert out.shape == (8, 8)
    assert out.min() >= 0
    assert out.max() == pytest.approx(MAX_PIXEL_VALUE)


def test_feature_vector_shape():
    assert to_feature_vector(canvas_with_line()).shape == (1, 64)


def test_empty_canvas_raises():
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((280, 280, 4), dtype=np.uint8))


def test_tiny_speck_is_treated_as_empty():
    img = np.zeros((280, 280, 3), dtype=np.uint8)
    img[10, 10] = 255
    with pytest.raises(ValueError):
        preprocess_image(img)


def test_rgba_transparent_pixels_are_background():
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[..., :3] = 255  # white colour but fully transparent
    with pytest.raises(ValueError):
        preprocess_image(rgba)


def test_grayscale_conversion_range():
    gray = to_grayscale(canvas_with_line())
    assert gray.ndim == 2
    assert gray.min() >= 0 and gray.max() <= 1


def test_float_input_in_unit_range():
    arr = canvas_with_line().astype(float) / 255.0
    assert preprocess_image(arr).shape == (8, 8)


def test_invalid_dimensions_raise():
    with pytest.raises(ValueError):
        to_grayscale(np.zeros((2, 2, 2, 2)))


def test_crop_and_pad_preserve_aspect_ratio():
    gray = to_grayscale(canvas_with_line())
    cropped = crop_to_content(gray)
    assert cropped.shape[0] > cropped.shape[1]  # tall bar
    squared = pad_to_square(cropped)
    assert squared.shape[0] == squared.shape[1] == cropped.shape[0]


def test_position_invariance():
    a = preprocess_image(canvas_with_line(box=(60, 40, 60, 240)))
    b = preprocess_image(canvas_with_line(box=(220, 40, 220, 240)))
    assert np.abs(a - b).max() < 4  # same shape wherever it is drawn


def test_scale_invariance():
    small = Image.new("RGB", (280, 280), "black")
    ImageDraw.Draw(small).line([(140, 100), (140, 180)], fill="white", width=10)
    big = canvas_with_line(box=(140, 20, 140, 260), width=30)
    a, b = preprocess_image(np.asarray(small)), preprocess_image(big)
    assert np.abs(a - b).mean() < 3


def test_digit_is_roughly_centred():
    out = preprocess_image(canvas_with_line(box=(30, 30, 30, 120)))
    ys, xs = np.indices(out.shape)
    assert abs((ys * out).sum() / out.sum() - 3.5) < 1.0
    assert abs((xs * out).sum() / out.sum() - 3.5) < 1.0


def test_pil_image_accepted():
    assert preprocess_image(Image.fromarray(canvas_with_line())).shape == (8, 8)
