"""
Sliding-window segmentation for EMG classification.

Given a continuous (N, C) recording, produce overlapping fixed-length
windows of shape (M, win_samples, C) suitable for feeding feature
extractors or a 1-D CNN.

For streaming use, `WindowBuffer` keeps a rolling buffer and emits a
window every time `step_samples` new samples arrive.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Offline windowing
# --------------------------------------------------------------------------- #

def make_windows(
    x: np.ndarray,
    win_samples: int,
    step_samples: int,
    labels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Slice a recording into overlapping windows.

    Parameters
    ----------
    x : ndarray of shape (N, C)
        Continuous signal.
    win_samples : int
        Window length in samples (e.g. fs * 0.2 for a 200-ms window).
    step_samples : int
        Hop length in samples (e.g. fs * 0.1 for 50% overlap).
    labels : optional ndarray of shape (N,)
        Per-sample label. If supplied, the majority label inside each
        window is returned alongside the windows.

    Returns
    -------
    windows : ndarray of shape (M, win_samples, C)
    labels  : ndarray of shape (M,) or None
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n, c = x.shape
    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("win_samples and step_samples must be positive")
    if n < win_samples:
        raise ValueError(f"Signal too short: {n} < win_samples={win_samples}")

    n_windows = 1 + (n - win_samples) // step_samples
    windows = np.empty((n_windows, win_samples, c), dtype=np.float32)
    for i in range(n_windows):
        start = i * step_samples
        windows[i] = x[start : start + win_samples]

    win_labels = None
    if labels is not None:
        labels = np.asarray(labels)
        win_labels = np.empty(n_windows, dtype=labels.dtype)
        for i in range(n_windows):
            start = i * step_samples
            seg = labels[start : start + win_samples]
            # majority vote
            vals, counts = np.unique(seg, return_counts=True)
            win_labels[i] = vals[np.argmax(counts)]

    return windows, win_labels


def windows_per_second(fs: float, step_samples: int) -> float:
    """Convenience: predictions per second given a hop length."""
    return fs / float(step_samples)


# --------------------------------------------------------------------------- #
# Streaming windowing
# --------------------------------------------------------------------------- #

class WindowBuffer:
    """
    Rolling buffer that emits a window every `step_samples` samples.

    Usage
    -----
    >>> wb = WindowBuffer(win_samples=40, step_samples=20, n_channels=2)
    >>> for chunk in serial_stream():     # chunk shape (k, 2)
    ...     for window in wb.push(chunk):
    ...         features = extract_features(window)
    ...         predict(features)
    """

    def __init__(self, win_samples: int, step_samples: int, n_channels: int):
        if step_samples > win_samples:
            raise ValueError("step_samples cannot exceed win_samples")
        self.win = win_samples
        self.step = step_samples
        self.c = n_channels
        self.buf = deque(maxlen=win_samples)
        self._since_last_emit = 0

    def push(self, chunk: np.ndarray) -> Iterator[np.ndarray]:
        """Yield zero or more windows after appending `chunk`."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.shape[1] != self.c:
            raise ValueError(
                f"Expected {self.c} channels, got {chunk.shape[1]}"
            )
        for row in chunk:
            self.buf.append(row)
            self._since_last_emit += 1
            if len(self.buf) == self.win and self._since_last_emit >= self.step:
                self._since_last_emit = 0
                yield np.stack(self.buf, axis=0)


__all__ = ["make_windows", "WindowBuffer", "windows_per_second"]
