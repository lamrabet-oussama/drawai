# DrawAI — Handwritten Digit Recognition

## Project overview

DrawAI is a web application in which you draw a digit (0–9) with the mouse and a real machine-learning model recognises it. The interface is in French and includes a short educational section, which makes the project suitable for demos and for programming workshops with young learners.

- Demo URL: _to be added after deployment_
- Screenshots: _to be added_ (`docs/screenshot-*.png`)

## Features

- Interactive drawing canvas (white stroke on black) with adjustable stroke width
- "Prédire" / "Effacer" buttons
- Large predicted digit, estimated score and a bar chart of the ten class scores
- Explicit messages for an empty canvas and for drawings the model finds hard
- Display of the 8 × 8 image actually fed to the model
- Educational section and technical information (model, test accuracy, dataset size)
- No external API, no API key, no database, no GPU

## Tech stack

Python 3.12 · Streamlit · scikit-learn · NumPy · Pillow · Plotly · Joblib · streamlit-drawable-canvas · pytest

## Machine learning approach

Two lightweight classifiers are compared: `RandomForestClassifier` and `SVC` (RBF kernel). The comparison uses a **validation split** taken from the training set; the winner is refit on the full training set and evaluated **once** on an independent, stratified test set (`random_state=42`, 25 % of the data).

Probabilities: the SVC uses Platt scaling (`probability=True`); a random forest would be wrapped in `CalibratedClassifierCV`. The percentage shown in the app is nevertheless an *estimated score* learned on the training distribution, not a guarantee that the answer is right, and the UI says so.

## Dataset and preprocessing

Dataset: scikit-learn `load_digits` — 1,797 images of 8 × 8 pixels, intensities 0–16.

A canvas drawing differs from these images (large, off-centre, arbitrary stroke width), so `src/preprocessing.py`:

1. converts to grayscale (RGBA is composited on black; ink = bright);
2. crops to the bounding box of the stroke;
3. pads to a square to keep the proportions;
4. downsizes to 6 × 6 and places the result in an 8 × 8 frame with a one-pixel margin;
5. centres the digit by its centre of mass;
6. rescales intensities to the 0–16 range.

To make the model robust to this pipeline, the training set is augmented with *simulated canvas drawings* (random size, position, thickness, blur, rotation and aspect ratio) generated from the training images only, and passed through the same preprocessing. Augmented copies never leak between the train, validation and test splits.

## Model evaluation

Metrics are produced by `python -m src.training` and stored inside `models/digit_classifier.joblib`; the app reads them from there. Run the script to see the comparison, the confusion matrix and the full classification report. Figures from the run that produced the committed model:

| Measure | Value |
| --- | --- |
| Selected model | SVC (RBF, C=10) |
| Validation score, RandomForest / SVC | 0.947 / 0.964 |
| Test accuracy, original digits (450 images) | 98.2 % |
| Test accuracy, simulated canvas drawings of the test images | 97.6 % |
| Mean estimated score on test images | 0.955 |

The "simulated canvas" figure measures the preprocessing pipeline on synthetic drawings, not on real users. It is not a substitute for a study on real hand-drawn digits. The automated tests also check recognition of typeface-rendered digits, which the model has never seen.

## Project structure

```text
drawai/
├── app.py                  # Streamlit interface
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── digit_classifier.joblib   # trained model + metrics
├── src/
│   ├── preprocessing.py    # canvas image -> 8x8 vector
│   ├── prediction.py       # model loading and prediction
│   └── training.py         # model comparison, evaluation, saving
└── tests/
    ├── test_preprocessing.py
    └── test_prediction.py
```

## Local installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install pytest   # only to run the tests
```

## Running the application

```bash
streamlit run app.py
```

Retrain the model (optional, takes about a minute):

```bash
python -m src.training
```

Run the tests:

```bash
python -m pytest
```

## Deployment on Streamlit Community Cloud

1. Push the repository to GitHub (keep `models/digit_classifier.joblib` committed).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Choose "New app", select the repository and branch, and set the main file to `app.py`.
4. Deploy. Dependencies are installed from `requirements.txt`; no secret or setting is needed.

The pinned `streamlit==1.38.0` is required by `streamlit-drawable-canvas==0.9.3`, which breaks on later Streamlit versions. `pyarrow` is required by Streamlit custom components.

## Limitations and future improvements

- 8 × 8 images hold little detail; look-alike digits (1/7, 4/9, 3/5, 8/9) can be confused.
- Only isolated digits are supported; letters or symbols still get an answer.
- Real-user accuracy has not been measured; a collected set of real drawings would give a more reliable estimate.
- Possible improvements: train on a higher-resolution dataset such as MNIST (with a small model), add feedback buttons to collect corrected labels, support touch input on mobile, and replace the pinned canvas component with a maintained alternative.
