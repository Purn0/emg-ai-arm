"""
Train the two models the app uses and write them to emg_ai_arm/models/.

EMG: the configuration named by --emg-config (see experiments/benchmark.py),
trained on every subject except subject 05, whose recording the app
replays, so the replay stays a test on an unseen person.

Camera: Random Forest (500 trees, seed 42) on the eight-gesture landmark
dataset of https://github.com/Purn0/gesture-control-project, downloaded
unless --gestures-csv points to a local copy.

    python -m experiments.train_models
"""

from __future__ import annotations

import argparse
import io
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from emg_ai_arm.emg_model import MODEL_FILE
from emg_ai_arm.utils.config import MODELS_DIR
from experiments import benchmark, data

GESTURES_URL = "https://raw.githubusercontent.com/Purn0/gesture-control-project/main/data/gestures.csv"
REPLAY_SUBJECT = 5


def train_emg(config_name):
    c = benchmark.CONFIGS[config_name]
    if c["task"] != "all7" or c["model"] == "cnn" or c["smooth"] != 1:
        raise SystemExit(f"{config_name}: the app needs a 7-class, feature-based, unsmoothed configuration")
    paths = data.session_paths()
    subjects = [s for s in data.SUBJECTS if s != REPLAY_SUBJECT]
    X, y, _ = benchmark.build_set(subjects, paths, c, c["length"] // 2)
    X, y = benchmark.rebalance(X, y, c["sampling"], np.random.default_rng(benchmark.SEED))
    model = benchmark.make_model(c).fit(X, y)
    bundle = {
        "classifier": model,
        "feature_set": c["features"],
        "window": c["length"],
        "classes": np.asarray(model.classes_),
        "calibration": c["norm"] == "calib",
        "description": f"{config_name}, trained on all subjects except {REPLAY_SUBJECT:02d}",
    }
    joblib.dump(bundle, MODEL_FILE, compress=3)
    print(f"EMG model ({config_name}, {len(y)} windows) -> {MODEL_FILE}")


def train_camera(csv_path):
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        with urllib.request.urlopen(GESTURES_URL) as response:
            df = pd.read_csv(io.BytesIO(response.read()))
    X, y = df.drop(columns="label"), df["label"]
    model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1).fit(X, y)
    path = MODELS_DIR / "gesture_model.pkl"
    joblib.dump(model, path, compress=3)
    print(f"camera model ({len(df)} samples, {y.nunique()} gestures) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emg-config", default="app")
    ap.add_argument("--gestures-csv", default=None)
    args = ap.parse_args()
    train_emg(args.emg_config)
    train_camera(args.gestures_csv)


if __name__ == "__main__":
    main()
