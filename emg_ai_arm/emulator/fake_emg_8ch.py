"""
Synthetic 8-channel signal for smoke-testing the pipeline without data.

This is not physiological EMG: each "class" is a noise floor plus smooth
bursts on a fixed set of channels, and its amplitude is about 1000 times
larger than the UCI recordings. Models trained on real EMG do not classify
it meaningfully; use it only to check that the GUI, threads and arm run.
"""

from __future__ import annotations

import numpy as np

N_CHANNELS = 8
N_CLASSES = 7

_CLASS_ACTIVE_CHANNELS = {
    0: [],
    1: [0, 1],
    2: [2, 3],
    3: [4, 5],
    4: [6, 7],
    5: [0, 2, 4, 6],
    6: [1, 3, 5, 7],
}


def generate_window(label: int, length: int, rng: np.random.Generator) -> np.ndarray:
    window = np.abs(rng.normal(0.02, 0.01, size=(length, N_CHANNELS)))
    edge = length // 5
    ramp = np.ones(length)
    ramp[:edge] = np.linspace(0, 1, edge)
    ramp[-edge:] = np.linspace(1, 0, edge)
    for ch in _CLASS_ACTIVE_CHANNELS[int(label)]:
        window[:, ch] += rng.uniform(0.25, 0.65) * ramp + rng.normal(0, 0.02, size=length)
    return window.astype(np.float32)


def stream_sequence(length: int = 200, seed: int | None = None):
    """Endless (window, label) pairs: rest, class 1, rest, class 2, ..."""
    rng = np.random.default_rng(seed)
    sequence = []
    for cls in range(1, N_CLASSES):
        sequence += [0, cls]
    while True:
        for label in sequence:
            yield generate_window(label, length, rng), label
