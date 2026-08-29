"""
80-feature extractor for the 8-channel research RF (rf_emg_best.joblib).

This is a SEPARATE module from extract_features.py (which is the
7-feature/channel extractor used by the 2-channel placeholder pipeline).
Do not merge them - the research RF was trained on exactly this function,
verified against the Colab notebook (untitled3.py) that produced
X_train_features / X_val_features / X_test_features.

Feature order per channel (10 features, matches training exactly):
    mean, std, RMS, energy, min, max, peak-to-peak, median,
    mean absolute value, zero crossings

Input:  window of shape (200, 8)
Output: feature vector of shape (80,)
"""

from __future__ import annotations

import numpy as np

N_CHANNELS = 8
WINDOW_SAMPLES = 200

FEATURE_NAMES_PER_CHANNEL = (
    "mean", "std", "rms", "energy", "min", "max",
    "peak_to_peak", "median", "mean_abs", "zero_crossings",
)


def extract_emg_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 10 features per channel:
    mean, std, RMS, energy, min, max,
    peak-to-peak, median, mean absolute value, zero crossings

    Input:  (200, 8)
    Output: (80,)
    """

    window = np.asarray(window, dtype=np.float32)

    if window.ndim != 2:
        raise ValueError(
            f"Expected one window with shape (200, 8), got {window.shape}"
        )

    if window.shape[1] != 8:
        raise ValueError(
            f"Expected 8 channels, got shape {window.shape}"
        )

    features = []

    for ch in range(8):
        signal = window[:, ch]

        mean = np.mean(signal)
        std = np.std(signal)

        rms = np.sqrt(np.mean(signal ** 2))
        energy = np.sum(signal ** 2)

        minimum = np.min(signal)
        maximum = np.max(signal)

        peak_to_peak = maximum - minimum

        median = np.median(signal)

        mean_abs = np.mean(np.abs(signal))

        zero_crossings = np.sum(
            np.diff(np.signbit(signal))
        )

        features.extend([
            mean,
            std,
            rms,
            energy,
            minimum,
            maximum,
            peak_to_peak,
            median,
            mean_abs,
            zero_crossings
        ])

    return np.array(features, dtype=np.float32)


def feature_names(n_channels: int = N_CHANNELS) -> list[str]:
    return [
        f"CH{c}_{name}"
        for c in range(n_channels)
        for name in FEATURE_NAMES_PER_CHANNEL
    ]


FEATURE_NAMES = feature_names()

__all__ = [
    "extract_emg_features",
    "feature_names",
    "FEATURE_NAMES",
    "N_CHANNELS",
    "WINDOW_SAMPLES",
]
