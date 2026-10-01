"""
The EMG classifier used by the app, with everything needed to apply it.

`python -m experiments.train_models` writes models/emg_model.joblib: a dict
with the classifier, its feature set, window length and whether it expects
per-user calibration. The thesis model (a bare RandomForestClassifier on the
thesis features, 200-sample windows, no calibration; attached to the
thesis-2026 release as rf_emg_best.joblib) is also accepted.

Calibration: before use, the user is recorded going through the gestures
once; the recording's labels are not used. Each feature is then standardised
with that recording's mean and standard deviation, which removes much of the
person-to-person difference in signal amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np

from emg_ai_arm.acquisition.replay_csv_8ch import load_session
from emg_ai_arm.features.emg_features import extract
from emg_ai_arm.utils.config import MODELS_DIR

MODEL_FILE = MODELS_DIR / "emg_model.joblib"
THESIS_MODEL_FILE = MODELS_DIR / "rf_emg_best.joblib"


@dataclass
class EMGModel:
    classifier: object
    feature_set: str
    window: int
    classes: np.ndarray
    calibration: bool
    description: str = ""

    def features(self, windows):
        return extract(windows, self.feature_set)


def load_emg_model(path=None) -> EMGModel:
    if path is None:
        path = MODEL_FILE if MODEL_FILE.exists() else THESIS_MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m experiments.train_models`")
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return EMGModel(**obj)
    return EMGModel(classifier=obj, feature_set="thesis", window=200,
                    classes=np.asarray(obj.classes_), calibration=False,
                    description="thesis model (rf_emg_best.joblib)")


def calibration_windows(signal, window):
    """All half-overlapping windows of an unlabelled recording."""
    starts = np.arange(0, len(signal) - window + 1, window // 2)
    return signal[starts[:, None] + np.arange(window)[None, :]]


def calibration_stats(features):
    return features.mean(axis=0), features.std(axis=0) + 1e-8


def calibrate_from_file(model: EMGModel, path):
    """Feature mean/std of a recording, ignoring its labels."""
    signal, _ = load_session(path)
    return calibration_stats(model.features(calibration_windows(signal, model.window)))
