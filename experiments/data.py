"""
Loading and windowing the UCI "EMG data for Gestures" dataset.

Download it with `python -m experiments.download_data`; by default it is
expected in data/uci_emg/EMG_data_for_gestures-master (36 subject folders,
two sessions each). Set EMG_DATA to use another location.

Each file has a millisecond timestamp, eight channels and a label:
0 unmarked, 1 hand at rest, 2 fist, 3 wrist flexion, 4 wrist extension,
5 radial deviation, 6 ulnar deviation, 7 extended palm (not performed by
every subject, never used here).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "uci_emg" / "EMG_data_for_gestures-master"
CHANNELS = [f"channel{i}" for i in range(1, 9)]
SUBJECTS = list(range(1, 37))
CLASS_NAMES = ["unmarked", "rest", "fist", "wrist flexion", "wrist extension",
               "radial deviation", "ulnar deviation"]


def data_dir():
    return Path(os.environ.get("EMG_DATA", DEFAULT_DATA))


def session_paths(root=None):
    """{(subject, session): path}; session 1 or 2 from the file name prefix."""
    root = Path(root or data_dir())
    out = {}
    for s in SUBJECTS:
        for path in sorted(glob.glob(str(root / f"{s:02d}" / "*.txt"))):
            out[(s, int(os.path.basename(path)[0]))] = path
    if len(out) != 72:
        raise FileNotFoundError(f"expected 72 session files under {root}, found {len(out)}; "
                                "run `python -m experiments.download_data`")
    return out


def load_session(path):
    df = pd.read_csv(path, sep="\t").dropna()
    return df[CHANNELS].values.astype(np.float32), df["class"].values.astype(np.int64)


def make_windows(signal, labels, length, step, label_rule="first"):
    """
    Cut a session into windows. label_rule "first" labels a window by its
    first sample (the thesis convention); "majority" by its most frequent
    sample label. Windows labelled 7 are dropped.
    """
    starts = np.arange(0, len(signal) - length + 1, step)
    idx = starts[:, None] + np.arange(length)[None, :]
    if label_rule == "first":
        y = labels[starts]
    elif label_rule == "majority":
        y = np.array([np.bincount(row, minlength=8).argmax() for row in labels[idx]])
    else:
        raise ValueError(label_rule)
    keep = y <= 6
    return signal[idx[keep]], y[keep]


def subject_folds(n_folds=6, seed=42):
    """Subjects shuffled once and split into equal folds."""
    perm = np.random.default_rng(seed).permutation(len(SUBJECTS)) + 1
    return [sorted(int(s) for s in fold) for fold in perm.reshape(n_folds, -1)]
