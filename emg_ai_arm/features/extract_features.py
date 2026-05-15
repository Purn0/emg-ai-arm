"""
Hand-engineered features for sEMG classification.

This implementation matches the Hudgins time-domain (TD) set plus two
frequency-domain features. Per channel the features are:

    MAV   – Mean Absolute Value
    RMS   – Root Mean Square
    WL    – Waveform Length
    ZC    – Zero Crossings   (with deadband threshold)
    SSC   – Slope Sign Changes (with deadband threshold)
    MNF   – Mean Frequency
    MDF   – Median Frequency

Use `FEATURE_NAMES` to label columns in plots and reports.
"""

from __future__ import annotations

import numpy as np

# Sample rate is needed for MNF/MDF; fake_emg uses 200 Hz envelopes which makes
# frequency-domain features less informative, but we compute them anyway for
# pipeline parity with raw-EMG runs.
DEFAULT_FS = 200.0

# Small voltage threshold used by ZC and SSC to ignore noise crossings.
# Tune this once you have real data; 0.01 is a reasonable starting point for
# signals scaled to ~[0, 1].
ZC_THRESHOLD = 0.01
SSC_THRESHOLD = 0.01


def _mav(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _wl(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def _zc(x: np.ndarray, threshold: float = ZC_THRESHOLD) -> float:
    """Zero crossings with a deadband, per Hudgins (1993)."""
    if x.size < 2:
        return 0.0
    a = x[:-1]
    b = x[1:]
    sign_change = (a * b) < 0
    above_thresh = np.abs(a - b) >= threshold
    return float(np.sum(sign_change & above_thresh))


def _ssc(x: np.ndarray, threshold: float = SSC_THRESHOLD) -> float:
    """Slope sign changes with a deadband, per Hudgins (1993)."""
    if x.size < 3:
        return 0.0
    a = x[:-2]
    b = x[1:-1]
    c = x[2:]
    cond = ((b - a) * (b - c)) > 0
    above_thresh = (np.abs(b - a) >= threshold) | (np.abs(b - c) >= threshold)
    return float(np.sum(cond & above_thresh))


def _power_spectrum(x: np.ndarray, fs: float):
    """One-sided power spectrum via FFT."""
    n = x.size
    if n < 4:
        return np.array([0.0]), np.array([0.0])
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x)) ** 2
    return freqs, spec


def _mnf(x: np.ndarray, fs: float) -> float:
    """Mean Frequency: sum(f * P) / sum(P)."""
    freqs, spec = _power_spectrum(x, fs)
    s = spec.sum()
    if s <= 0:
        return 0.0
    return float(np.sum(freqs * spec) / s)


def _mdf(x: np.ndarray, fs: float) -> float:
    """Median Frequency: frequency that splits the spectrum's total power in half."""
    freqs, spec = _power_spectrum(x, fs)
    total = spec.sum()
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(spec)
    idx = int(np.searchsorted(cumulative, total / 2.0))
    idx = min(idx, freqs.size - 1)
    return float(freqs[idx])


# Order of features per channel; keep in sync with FEATURE_NAMES.
_PER_CHANNEL_FUNCS = ("MAV", "RMS", "WL", "ZC", "SSC", "MNF", "MDF")


def feature_names(n_channels: int) -> list[str]:
    """Return human-readable names for the flattened feature vector."""
    return [f"CH{c}_{name}"
            for c in range(n_channels)
            for name in _PER_CHANNEL_FUNCS]


def extract_features(window: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Compute the full feature vector for one window.

    Parameters
    ----------
    window : ndarray of shape (SAMPLES, C)
        One windowed segment of EMG (envelope or raw).
    fs : float
        Sample rate, used for MNF and MDF.

    Returns
    -------
    ndarray of shape (7 * C,), dtype float32.
    """
    window = np.asarray(window, dtype=np.float32)
    if window.ndim == 1:
        window = window[:, None]
    feats = []
    for c in range(window.shape[1]):
        ch = window[:, c]
        feats.extend([
            _mav(ch),
            _rms(ch),
            _wl(ch),
            _zc(ch),
            _ssc(ch),
            _mnf(ch, fs),
            _mdf(ch, fs),
        ])
    return np.asarray(feats, dtype=np.float32)


# Convenience: expose the names that match `extract_features` for a 2-channel setup.
FEATURE_NAMES = feature_names(n_channels=2)

__all__ = [
    "extract_features",
    "feature_names",
    "FEATURE_NAMES",
    "DEFAULT_FS",
]
