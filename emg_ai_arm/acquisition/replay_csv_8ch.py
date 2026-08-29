"""
Replay real 8-channel EMG recordings (UCI "EMG data for gestures" raw
format) through the GUI as a live stream, for the research RF pipeline
(rf_emg_best.joblib, 80 features, classes 0-6).

Raw file format (tab-separated):
    time  channel1  channel2  ...  channel8  class

Windows are built the same way as in training: non-overlapping
200-sample windows, label taken from the first sample of the window.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 200
N_CHANNELS = 8

CHANNEL_COLS = [f"channel{i}" for i in range(1, N_CHANNELS + 1)]


def load_windows(path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep="\t")

    missing = [c for c in CHANNEL_COLS + ["class"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Replay file missing expected columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    signal = df[CHANNEL_COLS].values.astype(np.float32)
    labels = df["class"].values.astype(np.int64)

    n_windows = len(signal) // WINDOW_SIZE
    windows = []
    wlabels = []
    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE
        windows.append(signal[start:end])
        wlabels.append(labels[start])

    return np.asarray(windows, dtype=np.float32), np.asarray(wlabels, dtype=np.int64)


def replay(path, playback_sec: float = 0.0):
    """
    playback_sec defaults to 0.0 because StreamWorker already paces
    playback in real time via its own chunk-streaming loop.
    """
    path = Path(path)
    windows, labels = load_windows(path)

    if len(windows) == 0:
        raise ValueError(f"No complete {WINDOW_SIZE}-sample windows found in {path}")

    while True:
        for w, lab in zip(windows, labels):
            yield w, int(lab)
            time.sleep(playback_sec)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "emg_ai_arm/data/raw/subject05_session1.txt"
    for i, (w, lab) in enumerate(replay(p, playback_sec=0.0)):
        print(f"{i:03d} label={lab} shape={w.shape}")
        if i >= 10:
            break
