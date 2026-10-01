"""
Time-domain feature sets for 8-channel sEMG windows.

All functions take windows shaped (n_windows, n_samples, n_channels), or a
single window (n_samples, n_channels), and return one row of features per
window, grouped by channel (channel 0's features first).

Signals are in the units of the UCI "EMG data for Gestures" files, which
store the MYO bracelet's raw 8-bit samples (-128..127) multiplied by 1e-5.
Thresholds below are given in raw MYO levels.

Feature sets
------------
thesis   the 10 features of the thesis model: mean, std, RMS, energy, min,
         max, peak-to-peak, median, mean absolute value, sign changes.
hudgins  MAV, waveform length, zero crossings, slope sign changes
         (Hudgins et al., 1993).
ls4      L-scale, maximum fractal length, mean square root, Willison
         amplitude: the set Phinyomark et al. (2018) propose for low
         sampling rate armbands, called "TD4" in their paper (named ls4
         here to avoid confusion with Hudgins' set).
ls9      ls4 + zero crossings, RMS, integrated absolute value, DASDV,
         variance: their "TD9".

The threshold for zero crossings, slope sign changes and Willison amplitude
(2 raw levels, just above the 1-level quantisation step) is our choice;
Phinyomark et al. do not prescribe one for these recordings.
"""

from __future__ import annotations

import numpy as np

RAW_SCALE = 1e5        # file units -> raw MYO levels
THRESHOLD_LEVELS = 2   # amplitude threshold for ZC, SSC and WAMP, in raw levels
_EPS = 1e-12


def _as_batch(windows):
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 2:
        w = w[None]
    if w.ndim != 3:
        raise ValueError(f"expected (n, samples, channels) or (samples, channels), got {w.shape}")
    return w


def _stack(per_feature):
    """list of (n, channels) arrays -> (n, channels * n_features), grouped by channel."""
    return np.stack(per_feature, axis=2).reshape(per_feature[0].shape[0], -1)


def thesis_features(windows):
    """Exactly the 80 features of extract_features_8ch.extract_emg_features."""
    w = _as_batch(windows).astype(np.float32)
    mn, mx = w.min(axis=1), w.max(axis=1)
    feats = [
        w.mean(axis=1),
        w.std(axis=1),
        np.sqrt(np.mean(w ** 2, axis=1)),
        np.sum(w ** 2, axis=1),
        mn,
        mx,
        mx - mn,
        np.median(w, axis=1),
        np.mean(np.abs(w), axis=1),
        np.sum(np.diff(np.signbit(w), axis=1), axis=1).astype(np.float32),
    ]
    return _stack(feats).astype(np.float32)


def _zero_crossings(x, thr):
    a, b = x[:, :-1], x[:, 1:]
    return np.sum((a * b < 0) & (np.abs(a - b) >= thr), axis=1)


def _slope_sign_changes(x, thr):
    prev, mid, nxt = x[:, :-2], x[:, 1:-1], x[:, 2:]
    return np.sum((mid - prev) * (mid - nxt) >= thr, axis=1)


def _l_scale(x):
    """Second sample L-moment of each channel."""
    n = x.shape[1]
    s = np.sort(x, axis=1)
    weights = (np.arange(n) / (n - 1))[None, :, None]
    b0 = s.mean(axis=1)
    b1 = np.mean(weights * s, axis=1)
    return 2 * b1 - b0


def hudgins_features(windows):
    x = _as_batch(windows) * RAW_SCALE
    d = np.diff(x, axis=1)
    thr = THRESHOLD_LEVELS
    return _stack([
        np.mean(np.abs(x), axis=1),
        np.sum(np.abs(d), axis=1),
        _zero_crossings(x, thr),
        _slope_sign_changes(x, thr ** 2),
    ])


def ls4_features(windows):
    x = _as_batch(windows) * RAW_SCALE
    d = np.diff(x, axis=1)
    return _stack([
        _l_scale(x),
        np.log10(np.sqrt(np.sum(d ** 2, axis=1)) + _EPS),
        np.mean(np.sqrt(np.abs(x)), axis=1),
        np.sum(np.abs(d) >= THRESHOLD_LEVELS, axis=1),
    ])


def ls9_features(windows):
    x = _as_batch(windows) * RAW_SCALE
    d = np.diff(x, axis=1)
    n = x.shape[1]
    return _stack([
        _l_scale(x),
        np.log10(np.sqrt(np.sum(d ** 2, axis=1)) + _EPS),
        np.mean(np.sqrt(np.abs(x)), axis=1),
        np.sum(np.abs(d) >= THRESHOLD_LEVELS, axis=1),
        _zero_crossings(x, THRESHOLD_LEVELS),
        np.sqrt(np.mean(x ** 2, axis=1)),
        np.sum(np.abs(x), axis=1),
        np.sqrt(np.mean(d ** 2, axis=1)),
        np.sum(x ** 2, axis=1) / (n - 1),
    ])


FEATURE_SETS = {
    "thesis": thesis_features,
    "hudgins": hudgins_features,
    "ls4": ls4_features,
    "ls9": ls9_features,
}


def extract(windows, feature_set="ls9"):
    return FEATURE_SETS[feature_set](windows)
