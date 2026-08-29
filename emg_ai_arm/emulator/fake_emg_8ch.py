"""
Synthetic 8-channel EMG signal generator for demo purposes ONLY.

IMPORTANT: This is a clearly-labeled SYNTHETIC/SIMULATED signal source,
NOT physiological EMG. It exists as a mechanical fallback demo source
when no real recording is selected. For the actual thesis defense, prefer
the "replay (CSV)" source, which streams a real held-out subject
recording through the same pipeline.

Produces windows shaped (200, 8) to match the research pipeline:
    - 200 samples per window (matches rf_emg_best.joblib training)
    - 8 channels
    - 7 classes (0=rest, 1-6=synthetic "gesture" bursts)
"""

from __future__ import annotations

import numpy as np

FS = 200
WIN_SEC = 1.0
SAMPLES = int(FS * WIN_SEC)   # 200
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


def generate_window(label: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, int]:
    if rng is None:
        rng = np.random.default_rng()

    window = np.abs(rng.normal(0.02, 0.01, size=(SAMPLES, N_CHANNELS))).astype(np.float32)

    active = _CLASS_ACTIVE_CHANNELS.get(int(label), [])
    for ch in active:
        burst_level = rng.uniform(0.25, 0.65)
        ramp = np.ones(SAMPLES, dtype=np.float32)
        edge = SAMPLES // 5
        ramp[:edge] = np.linspace(0, 1, edge)
        ramp[-edge:] = np.linspace(1, 0, edge)
        window[:, ch] += burst_level * ramp
        window[:, ch] += rng.normal(0, 0.02, size=SAMPLES)

    return window.astype(np.float32), int(label)


def stream_sequence():
    rng = np.random.default_rng()
    seq = []
    for cls in range(N_CLASSES):
        if cls != 0:
            seq.append(0)
        seq.append(cls)
    seq.append(0)

    while True:
        for lab in seq:
            yield generate_window(lab, rng)


if __name__ == "__main__":
    for i, (w, lab) in enumerate(stream_sequence()):
        print(f"{i:02d} label={lab} shape={w.shape} mean={w.mean():.3f}")
        if i >= 14:
            break
