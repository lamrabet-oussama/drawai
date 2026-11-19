import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_digits

from src.prediction import MODEL_PATH, is_blank, load_model, predict
from src.training import simulate_canvas_drawing


@pytest.fixture(scope="module")
def bundle():
    return load_model()


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_load_model_contents(bundle):
    assert hasattr(bundle["model"], "predict_proba")
    assert 0.9 < bundle["metrics"]["test_accuracy"] <= 1.0
    assert bundle["metrics"]["n_samples_total"] == 1797


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")


def test_scores_are_a_distribution(bundle):
    img = simulate_canvas_drawing(load_digits().images[3], np.random.default_rng(0))
    p = predict(bundle["model"], img)
    assert p.scores.shape == (10,)
    assert p.scores.sum() == pytest.approx(1.0)
    assert 0 <= p.confidence <= 1
    assert p.confidence == pytest.approx(p.scores.max())
    assert p.processed.shape == (8, 8)


def test_blank_images_detected():
    assert is_blank(None)
    assert is_blank(np.zeros((280, 280, 4), dtype=np.uint8))
    assert not is_blank(np.full((280, 280, 3), 255, dtype=np.uint8))


def test_predict_blank_raises(bundle):
    with pytest.raises(ValueError):
        predict(bundle["model"], np.zeros((280, 280, 4), dtype=np.uint8))


def test_predict_on_dataset_digits_through_canvas_pipeline(bundle):
    """Held-in sanity check: digits rendered on a canvas are mostly recognised."""
    digits = load_digits()
    rng = np.random.default_rng(123)
    idx = rng.choice(len(digits.target), 200, replace=False)
    ok = sum(
        predict(bundle["model"], simulate_canvas_drawing(digits.images[i], rng)).digit == digits.target[i]
        for i in idx
    )
    assert ok / 200 > 0.8


def _font_digit(digit: int, size: int, width: int, offset=(0, 0)):
    """Independent realistic drawing: a typeface digit rendered with a thick stroke."""
    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf", "cour.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    if font is None:
        pytest.skip("Aucune police TrueType disponible")
    img = Image.new("L", (280, 280), 0)
    ImageDraw.Draw(img).text((60 + offset[0], 20 + offset[1]), str(digit), fill=255, font=font, stroke_width=width, stroke_fill=255)
    return np.asarray(img.convert("RGB"))


def test_typeface_digits_are_recognised(bundle):
    """Digits never seen in training (typeface rendering) with varied size/offset."""
    hits, total = 0, 0
    for digit in range(10):
        for size, width, off in [(200, 2, (0, 0)), (150, 4, (20, 30)), (230, 1, (-20, -10))]:
            hits += predict(bundle["model"], _font_digit(digit, size, width, off)).digit == digit
            total += 1
    assert hits / total >= 0.7, f"{hits}/{total}"
