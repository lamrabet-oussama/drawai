"""Train, compare and save the digit classifier.

Usage:
    python -m src.training

Two lightweight models (RandomForest and SVC) are compared on a validation
split carved out of the training set. The winner is refit with calibrated probabilities and
evaluated once on an independent test set. Because users draw on a large
canvas while the dataset contains 8x8 images, the training set is augmented
with simulated canvas drawings that go through the exact same preprocessing as
the live application.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageFilter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.preprocessing import MAX_PIXEL_VALUE, preprocess_image

RANDOM_STATE = 42
TEST_SIZE = 0.25
VAL_SIZE = 0.2
AUGMENTATIONS_PER_IMAGE = 6
CANVAS_SIZE = 280
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "digit_classifier.joblib"


def simulate_canvas_drawing(digit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn an 8x8 dataset digit into a canvas-like RGB image (white on black).

    The digit is smoothly upscaled, thickened or thinned, slightly rotated,
    rescaled and placed at a random position of a 280x280 canvas.
    """
    img = Image.fromarray((digit / MAX_PIXEL_VALUE * 255).astype(np.uint8), mode="L")
    side = int(rng.integers(90, 230))
    img = img.resize((side, side), Image.Resampling.BICUBIC)
    # Stroke thickness variation.
    thickness = int(rng.choice([1, 3, 5, 7, 9]))
    if thickness > 1:
        img = img.filter(ImageFilter.MaxFilter(thickness))
    img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.5))))
    # Contrast stretch followed by a threshold-like boost, as a canvas stroke is solid.
    arr = np.asarray(img, dtype=np.float64)
    arr = np.clip((arr / max(arr.max(), 1.0) - 0.25) * 3.0, 0, 1) * 255
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img = img.rotate(float(rng.uniform(-12, 12)), resample=Image.Resampling.BILINEAR)
    # Random aspect-ratio change (people draw narrow or wide digits).
    w = max(30, int(img.width * rng.uniform(0.6, 1.2)))
    img = img.resize((w, img.height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    x = int(rng.integers(0, max(1, CANVAS_SIZE - img.width)))
    y = int(rng.integers(0, max(1, CANVAS_SIZE - img.height)))
    canvas.paste(img, (x, y))
    return np.asarray(canvas.convert("RGB"))


def build_augmented(images: np.ndarray, labels: np.ndarray, n_aug: int, seed: int):
    """Original images plus `n_aug` preprocessed canvas simulations per image."""
    rng = np.random.default_rng(seed)
    feats = [images.reshape(len(images), -1)]
    labs = [labels]
    for _ in range(n_aug):
        sim = np.stack([preprocess_image(simulate_canvas_drawing(im, rng)).ravel() for im in images])
        feats.append(sim)
        labs.append(labels)
    return np.vstack(feats), np.concatenate(labs)


def candidate_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "SVC": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE),
    }


def train(model_path: Path = MODEL_PATH, verbose: bool = True) -> dict:
    digits = load_digits()
    images, labels = digits.images, digits.target

    img_train, img_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    # Validation split of the *source* images, so augmented copies never leak across splits.
    img_fit, img_val, y_fit, y_val = train_test_split(
        img_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_STATE
    )

    X_fit, y_fit_aug = build_augmented(img_fit, y_fit, AUGMENTATIONS_PER_IMAGE, seed=1)
    X_val_orig = img_val.reshape(len(img_val), -1)
    X_val_sim, y_val_sim = build_augmented(img_val, y_val, 2, seed=2)
    X_val_sim, y_val_sim = X_val_sim[len(img_val):], y_val_sim[len(img_val):]

    comparison = {}
    for name, model in candidate_models().items():
        model.fit(X_fit, y_fit_aug)
        acc_orig = accuracy_score(y_val, model.predict(X_val_orig))
        acc_sim = accuracy_score(y_val_sim, model.predict(X_val_sim))
        comparison[name] = {
            "validation_accuracy_original": round(float(acc_orig), 4),
            "validation_accuracy_canvas_simulated": round(float(acc_sim), 4),
            "selection_score": round(float((acc_orig + acc_sim) / 2), 4),
        }
        if verbose:
            print(f"[validation] {name}: {comparison[name]}")

    best_name = max(comparison, key=lambda n: comparison[n]["selection_score"])

    # Final model: refit on the whole training set and calibrate the probabilities.
    X_train, y_train_aug = build_augmented(img_train, y_train, AUGMENTATIONS_PER_IMAGE, seed=3)
    final = candidate_models()[best_name]
    calibration = "Platt scaling (SVC probability=True, internal 5-fold CV)"
    if best_name == "RandomForest":  # forest votes are not calibrated probabilities
        final = CalibratedClassifierCV(final, method="sigmoid", cv=3)
        calibration = "sigmoid (CalibratedClassifierCV, cv=3)"
    final.fit(X_train, y_train_aug)

    X_test = img_test.reshape(len(img_test), -1)
    X_test_sim, y_test_sim = build_augmented(img_test, y_test, 2, seed=4)
    X_test_sim, y_test_sim = X_test_sim[len(img_test):], y_test_sim[len(img_test):]

    pred = final.predict(X_test)
    proba = final.predict_proba(X_test)
    pred_sim = final.predict(X_test_sim)
    proba_sim = final.predict_proba(X_test_sim)
    metrics = {
        "model_name": best_name,
        "calibration": calibration,
        "n_samples_total": int(len(images)),
        "n_train_images": int(len(img_train)),
        "n_test_images": int(len(img_test)),
        "n_train_samples_augmented": int(len(X_train)),
        "random_state": RANDOM_STATE,
        "comparison": comparison,
        "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "test_accuracy_canvas_simulated": round(float(accuracy_score(y_test_sim, pred_sim)), 4),
        "test_mean_confidence": round(float(proba.max(axis=1).mean()), 4),
        "test_mean_confidence_canvas_simulated": round(float(proba_sim.max(axis=1).mean()), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    if verbose:
        print(f"[test] selected model: {best_name}")
        print(f"[test] accuracy (original digits): {metrics['test_accuracy']}")
        print(f"[test] accuracy (simulated canvas): {metrics['test_accuracy_canvas_simulated']}")
        print(classification_report(y_test, pred))
        print(confusion_matrix(y_test, pred))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "metrics": metrics}, model_path, compress=3)
    if verbose:
        print(f"Model saved to {model_path} ({model_path.stat().st_size / 1024:.0f} KiB)")
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps({k: v for k, v in result.items() if k not in ("confusion_matrix", "classification_report")}, indent=2))
