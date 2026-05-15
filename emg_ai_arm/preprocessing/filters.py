"""
Signal-conditioning filters for surface EMG.

Three operations are exposed:

* `remove_dc`       — subtract the per-channel mean.
* `bandpass_filter` — 4th-order Butterworth, default 20–450 Hz (sEMG band).
* `notch_filter`    — IIR notch, default 50 Hz (Bangladesh mains; use 60 Hz for North America).

All functions are stateless: they take a 1-D or (N, C) array and return the
same shape. For a streaming pipeline use `LiveBandpass`, which keeps the
filter state between calls so chunks don't introduce edge artefacts.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt


# --------------------------------------------------------------------------- #
# Stateless / offline helpers
# --------------------------------------------------------------------------- #

def remove_dc(x: np.ndarray) -> np.ndarray:
    """Subtract per-channel mean. Accepts (N,) or (N, C)."""
    x = np.asarray(x, dtype=np.float32)
    return x - x.mean(axis=0, keepdims=True)


def _butter_bandpass_sos(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if not 0 < low < high < 1:
        raise ValueError(
            f"Invalid band: lowcut={lowcut}, highcut={highcut}, fs={fs}"
        )
    return butter(order, [low, high], btype="bandpass", output="sos")


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    order: int = 4,
) -> np.ndarray:
    """
    Offline, zero-phase Butterworth bandpass. Use this for *recorded* data.
    For real-time use `LiveBandpass`.
    """
    sos = _butter_bandpass_sos(lowcut, highcut, fs, order=order)
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


def notch_filter(
    x: np.ndarray,
    fs: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """
    Offline, zero-phase IIR notch. Default 50 Hz (BD mains).
    """
    b, a = iirnotch(freq, quality, fs)
    # Convert to SOS for stability on cascaded use
    from scipy.signal import tf2sos
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Streaming / real-time helpers
# --------------------------------------------------------------------------- #

class LiveBandpass:
    """
    Stateful bandpass for chunked streaming data.

    Usage
    -----
    >>> bp = LiveBandpass(fs=1000, n_channels=2)
    >>> while True:
    ...     chunk = read_serial_chunk()           # shape (n_samples, 2)
    ...     filtered = bp(chunk)
    """

    def __init__(
        self,
        fs: float,
        n_channels: int,
        lowcut: float = 20.0,
        highcut: float = 450.0,
        order: int = 4,
    ):
        self.sos = _butter_bandpass_sos(lowcut, highcut, fs, order=order)
        zi_single = sosfilt_zi(self.sos)  # (n_sections, 2)
        # broadcast initial conditions to every channel
        self.zi = np.repeat(zi_single[:, :, None], n_channels, axis=2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y.astype(np.float32)


class LiveNotch:
    """Stateful 50-Hz notch for streaming data."""

    def __init__(
        self,
        fs: float,
        n_channels: int,
        freq: float = 50.0,
        quality: float = 30.0,
    ):
        from scipy.signal import tf2sos
        b, a = iirnotch(freq, quality, fs)
        self.sos = tf2sos(b, a)
        zi_single = sosfilt_zi(self.sos)
        self.zi = np.repeat(zi_single[:, :, None], n_channels, axis=2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y.astype(np.float32)


__all__ = [
    "remove_dc",
    "bandpass_filter",
    "notch_filter",
    "LiveBandpass",
    "LiveNotch",
]
