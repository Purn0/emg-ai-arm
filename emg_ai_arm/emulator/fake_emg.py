import os
import time
import numpy as np
from pathlib import Path


FS = 200               # Hz
WIN_SEC = 0.2          # 200 ms
SAMPLES = int(FS * WIN_SEC)
N_CHANNELS = 2

LABELS = {
    0: "REST",
    1: "CH1",
    2: "CH2",
    3: "BOTH"          # used for mode switch
}

def generate_window(label: int, seed=None):
    """
    Returns window shape: (SAMPLES, 2), values >= 0 (envelope-like).
    """
    if seed is not None:
        np.random.seed(seed)

    # baseline noise (small positive)
    ch1 = np.abs(np.random.normal(0.02, 0.01, SAMPLES))
    ch2 = np.abs(np.random.normal(0.02, 0.01, SAMPLES))

    def burst(level_low, level_high):
        return np.random.uniform(level_low, level_high)

    if label == 1:
        ch1 += burst(0.25, 0.60)
    elif label == 2:
        ch2 += burst(0.25, 0.60)
    elif label == 3:
        ch1 += burst(0.35, 0.75)
        ch2 += burst(0.35, 0.75)

    window = np.vstack([ch1, ch2]).T  # (SAMPLES, 2)
    return window, label

def make_dataset(n_per_class=2000, filename="fake_windows.npz"):
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "data" / "windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.makedirs(out_dir, exist_ok=True)

    X = []
    y = []

    for label in LABELS.keys():
        for _ in range(n_per_class):
            w, lab = generate_window(label)
            X.append(w)
            y.append(lab)

    X = np.stack(X, axis=0)  # (N, SAMPLES, 2)
    y = np.array(y, dtype=np.int64)

    # shuffle
    idx = np.random.permutation(len(y))
    X = X[idx]
    y = y[idx]

    path = out_dir / filename
    np.savez_compressed(str(path), X=X, y=y, fs=FS, win_sec=WIN_SEC)
    return str(path)

def stream_sequence():
    """
    Generator for a realistic streaming sequence:
    rest -> CH1 -> rest -> CH2 -> rest -> BOTH (mode switch) -> ...
    """
    seq = [0]*15 + [1]*10 + [0]*10 + [2]*10 + [0]*10 + [3]*8 + [0]*15
    while True:
        for lab in seq:
            w, _ = generate_window(lab)
            yield w, lab

if __name__ == "__main__":
    path = make_dataset(n_per_class=1500)
    print("Saved:", path)

