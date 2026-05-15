"""
Envelope extraction.

Surface EMG is a high-frequency oscillating signal; classifiers usually
operate on its slow-moving amplitude envelope. Two envelope strategies
are supplied:

* `moving_average_envelope` — rectify then average over a window.
* `rms_envelope`            — sliding RMS, smoother and more standard in
                              prosthetics literature.

Both return the same shape as the input.
"""

from __future__ import annotations

import numpy as np


def _moving_window_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Symmetric box-car smoothing of a 1-D signal."""
    if win < 1:
        return x.astype(np.float32)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    # mode='same' keeps the length; pad with the signal mean to avoid edge dips
    padded = np.pad(x, win, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[win:-win].astype(np.float32)


def moving_average_envelope(
    x: np.ndarray,
    fs: float,
    window_ms: float = 100.0,
) -> np.ndarray:
    """
    Full-wave rectification followed by moving-average smoothing.

    Parameters
    ----------
    x : ndarray, shape (N,) or (N, C)
        Filtered EMG (after bandpass + notch).
    fs : float
        Sample rate in Hz.
    window_ms : float
        Smoothing window length in milliseconds.

    Returns
    -------
    ndarray, same shape as x, values >= 0.
    """
    x = np.asarray(x, dtype=np.float32)
    rectified = np.abs(x)
    win = max(1, int(round(window_ms * 1e-3 * fs)))
    if rectified.ndim == 1:
        return _moving_window_1d(rectified, win)
    out = np.empty_like(rectified)
    for c in range(rectified.shape[1]):
        out[:, c] = _moving_window_1d(rectified[:, c], win)
    return out


def rms_envelope(
    x: np.ndarray,
    fs: float,
    window_ms: float = 100.0,
) -> np.ndarray:
    """
    Sliding root-mean-square envelope. Use this if you want a smoother
    estimate; many EMG papers prefer RMS over rectified-and-averaged.
    """
    x = np.asarray(x, dtype=np.float32)
    squared = x ** 2
    win = max(1, int(round(window_ms * 1e-3 * fs)))
    if squared.ndim == 1:
        smoothed = _moving_window_1d(squared, win)
        return np.sqrt(smoothed)
    out = np.empty_like(squared)
    for c in range(squared.shape[1]):
        out[:, c] = _moving_window_1d(squared[:, c], win)
    return np.sqrt(out)


__all__ = ["moving_average_envelope", "rms_envelope"]
