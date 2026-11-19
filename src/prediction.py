"""Model loading and prediction from a canvas image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from src.preprocessing import has_ink, preprocess_image, to_grayscale

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "digit_classifier.joblib"
# Below this estimated score the drawing is flagged as hard to recognise.
LOW_CONFIDENCE = 0.60


@dataclass
class Prediction:
    digit: int
    confidence: float  # estimated score of the predicted class, in [0, 1]
    scores: np.ndarray  # shape (10,), sums to 1
    processed: np.ndarray  # 8x8 image fed to the model, values in [0, 16]

    @property
    def is_uncertain(self) -> bool:
        return self.confidence < LOW_CONFIDENCE


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load the saved bundle: {"model": estimator, "metrics": dict}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}. Lancez `python -m src.training`.")
    bundle = joblib.load(path)
    if not {"model", "metrics"} <= set(bundle):
        raise ValueError("Fichier de modèle invalide.")
    return bundle


def is_blank(image: np.ndarray | Image.Image) -> bool:
    """True if the image holds no drawing (None, empty, or all background)."""
    if image is None:
        return True
    try:
        return not has_ink(to_grayscale(image))
    except ValueError:
        return True


def predict(model, image: np.ndarray | Image.Image) -> Prediction:
    """Predict the digit drawn in `image`. Raises ValueError for a blank image."""
    processed = preprocess_image(image)
    scores = model.predict_proba(processed.reshape(1, -1))[0]
    classes = np.asarray(model.classes_)
    full = np.zeros(10)
    full[classes] = scores
    digit = int(full.argmax())
    return Prediction(digit=digit, confidence=float(full[digit]), scores=full, processed=processed)
