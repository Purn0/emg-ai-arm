"""
Replay a recorded 8-channel session from the UCI "EMG data for Gestures"
dataset (tab-separated: time, channel1..channel8, class) as a stream of
windows.

Windows do not overlap and take the label of their first sample, as in the
thesis. Windows labelled 7 (extended palm) are skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_CHANNELS = 8
CHANNEL_COLS = [f"channel{i}" for i in range(1, N_CHANNELS + 1)]


def load_session(path):
    df = pd.read_csv(path, sep="\t").dropna()
    missing = [c for c in CHANNEL_COLS + ["class"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[CHANNEL_COLS].values.astype(np.float32), df["class"].values.astype(np.int64)


def load_windows(path, window: int = 200):
    signal, labels = load_session(path)
    n = len(signal) // window
    windows = signal[: n * window].reshape(n, window, N_CHANNELS)
    wlabels = labels[: n * window : window]
    keep = wlabels <= 6
    return windows[keep], wlabels[keep]


def replay(path, window: int = 200):
    """Yield (window, label) pairs once through the session."""
    windows, labels = load_windows(Path(path), window)
    if len(windows) == 0:
        raise ValueError(f"no complete {window}-sample windows in {path}")
    for w, lab in zip(windows, labels):
        yield w, int(lab)
