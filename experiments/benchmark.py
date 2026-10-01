"""
Subject-wise evaluation of sEMG gesture classifiers on the UCI dataset.

Every configuration uses the same six folds of six subjects (seed 42): each
fold's model is trained on the other 30 subjects only, so every subject is
tested exactly once by a model that never saw any of its recordings.
Training windows overlap by 50%; test windows do not overlap, as in the live
replay. Scores are computed per subject (both sessions together) and
averaged; the pooled confusion matrix is kept as well.

Tasks
  all7       classes 0-6, "unmarked" included (the thesis setting; what the
             live pipeline sees).
  idle6      as all7, but "unmarked" and "hand at rest" form one class
             ("idle"): both mean that the arm should not move.
  gestures6  classes 1-6, windows labelled "unmarked" left out (the setting
             used by most work on this dataset).

Options (see CONFIGS)
  features   thesis | hudgins | ls4 | ls9 | raw (CNN input)
  length     window length in samples (the files have ~967 samples/s)
  label      first (thesis) | majority
  norm       none | calib: z-score each session's features with statistics
             from the same subject's *other* session, used unlabelled, i.e.
             a calibration recording made before use.
  sampling   downsample (class 0 -> 4,000 windows + balanced class weights,
             the thesis recipe) | weights (balanced class weights) | none
  model      rf | et | lda | lgbm | xgb | cnn
  smooth     majority vote over the last k predictions (k=1: off)

    python -m experiments.benchmark                 # every configuration
    python -m experiments.benchmark ls9_calib_rf    # selected ones
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from emg_ai_arm.emg_model import calibration_stats as feature_calibration_stats
from emg_ai_arm.emg_model import calibration_windows
from emg_ai_arm.features.emg_features import extract
from experiments import data

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"
TASK_LABELS = {
    "all7": [0, 1, 2, 3, 4, 5, 6],
    "idle6": [0, 2, 3, 4, 5, 6],
    "gestures6": [1, 2, 3, 4, 5, 6],
}

BASE = dict(task="all7", features="thesis", length=200, label="first", norm="none",
            sampling="downsample", model="rf", smooth=1)


def cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


# Every configuration keeps the thesis windowing (200 samples, label of the
# first sample) unless its name says otherwise, and changes one thing at a time.
CONFIGS = {
    # class imbalance (thesis features, Random Forest)
    "thesis": cfg(),                                        # the thesis recipe
    "thesis_original": cfg(sampling="none"),
    "thesis_balanced": cfg(sampling="balanced"),
    "thesis_weights": cfg(sampling="weights"),
    # feature sets
    "hudgins": cfg(features="hudgins", sampling="weights"),
    "ls4": cfg(features="ls4", sampling="weights"),
    "ls9": cfg(features="ls9", sampling="weights"),
    # per-user calibration
    "thesis_calib": cfg(sampling="weights", norm="calib"),
    "ls9_calib": cfg(features="ls9", sampling="weights", norm="calib"),
    "ls9_calib_w250": cfg(features="ls9", sampling="weights", norm="calib", length=250, label="majority"),
    # classifiers on ls9 + calibration
    "ls9_calib_et": cfg(features="ls9", sampling="weights", norm="calib", model="et"),
    "ls9_calib_lda": cfg(features="ls9", sampling="weights", norm="calib", model="lda"),
    "ls9_calib_lgbm": cfg(features="ls9", sampling="weights", norm="calib", model="lgbm"),
    "ls9_calib_xgb": cfg(features="ls9", sampling="weights", norm="calib", model="xgb"),
    "cnn": cfg(features="raw", sampling="weights", model="cnn"),
    "cnn_calib": cfg(features="raw", sampling="weights", norm="calib", model="cnn"),
    # temporal smoothing of the live output
    "ls9_calib_smooth3": cfg(features="ls9", sampling="weights", norm="calib", smooth=3),
    "ls9_calib_smooth5": cfg(features="ls9", sampling="weights", norm="calib", smooth=5),
}
# "unmarked" and "hand at rest" merged into one idle class
for _name in ("thesis_weights", "ls9_calib", "ls9_calib_lda", "ls9_calib_smooth5"):
    CONFIGS["idle_" + _name] = dict(CONFIGS[_name], task="idle6")
# the six labelled gestures only (the usual benchmark setting; smoothing is
# left out because removing the unmarked windows breaks the time order)
for _name in ("thesis_weights", "hudgins", "ls4", "ls9", "thesis_calib", "ls9_calib", "ls9_calib_et",
              "ls9_calib_lda", "ls9_calib_lgbm", "ls9_calib_xgb", "cnn", "cnn_calib"):
    CONFIGS["g6_" + _name] = dict(CONFIGS[_name], task="gestures6")
# the model the app uses (experiments/train_models.py)
CONFIGS["app"] = CONFIGS["ls9_calib"]


# --------------------------------------------------------------------------- data

_signal_cache = {}
_feature_cache = {}


def _session(key, paths):
    if key not in _signal_cache:
        _signal_cache[key] = data.load_session(paths[key])
    return _signal_cache[key]


def session_features(key, paths, c, step):
    """Features (or raw windows) and labels of one session, cached."""
    ck = (key, c["features"], c["length"], step, c["label"])
    if ck not in _feature_cache:
        sig, lab = _session(key, paths)
        W, y = data.make_windows(sig, lab, c["length"], step, c["label"])
        X = (W * 1e5).astype(np.float32) if c["features"] == "raw" else extract(W, c["features"])
        _feature_cache[ck] = (X, y)
    return _feature_cache[ck]


def calibration_stats(subject, session, paths, c):
    """
    Mean/std of the subject's *other* session, computed from all its
    half-overlapping windows without looking at labels, exactly as the app
    does with a calibration recording (emg_model.calibrate_from_file).
    """
    key = ("calibration", subject, 3 - session, c["features"], c["length"])
    if key not in _feature_cache:
        signal, _ = _session((subject, 3 - session), paths)
        W = calibration_windows(signal, c["length"])
        if c["features"] == "raw":
            X = (W * 1e5).astype(np.float32)
            _feature_cache[key] = (X.mean(axis=(0, 1)), X.std(axis=(0, 1)) + 1e-8)
        else:
            _feature_cache[key] = feature_calibration_stats(extract(W, c["features"]))
    return _feature_cache[key]


def build_set(subjects, paths, c, step):
    Xs, ys, groups = [], [], []
    for s in subjects:
        for sess in (1, 2):
            X, y = session_features((s, sess), paths, c, step)
            if c["norm"] == "calib":
                mu, sd = calibration_stats(s, sess, paths, c)
                X = (X - mu) / sd
            if c["task"] == "gestures6":
                keep = y > 0
                X, y = X[keep], y[keep]
            elif c["task"] == "idle6":
                y = np.where(y == 1, 0, y)
            Xs.append(X)
            ys.append(y)
            groups.append(np.full(len(y), s * 10 + sess))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)


def rebalance(X, y, sampling, rng):
    if sampling == "downsample":
        idx0 = np.where(y == 0)[0]
        keep0 = rng.choice(idx0, size=min(4000, len(idx0)), replace=False)
        idx = np.sort(np.concatenate([keep0, np.where(y != 0)[0]]))
        return X[idx], y[idx]
    if sampling == "balanced":
        n = min(Counter(y).values())
        idx = np.sort(np.concatenate([rng.choice(np.where(y == k)[0], n, replace=False) for k in np.unique(y)]))
        return X[idx], y[idx]
    return X, y


# --------------------------------------------------------------------------- models

def make_model(c):
    weighted = c["sampling"] in ("weights", "downsample")
    cw = "balanced" if weighted else None
    if c["model"] == "rf":
        return RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight=cw,
                                      random_state=SEED, n_jobs=-1)
    if c["model"] == "et":
        return ExtraTreesClassifier(n_estimators=300, max_features="sqrt", class_weight=cw,
                                    random_state=SEED, n_jobs=-1)
    if c["model"] == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    if c["model"] == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight=cw,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=SEED, n_jobs=-1, verbose=-1)
    if c["model"] == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=-1)
    raise ValueError(c["model"])


def fit_predict(c, Xtr, ytr, Xte):
    if c["model"] == "cnn":
        from experiments.cnn import fit_predict_cnn
        return fit_predict_cnn(Xtr, ytr, Xte, seed=SEED, weighted=c["sampling"] != "none")
    classes = np.unique(ytr)
    remap = {k: i for i, k in enumerate(classes)}
    yi = np.array([remap[k] for k in ytr])
    model = make_model(c)
    if c["model"] == "xgb" and c["sampling"] != "none":
        model.fit(Xtr, yi, sample_weight=compute_sample_weight("balanced", yi))
    else:
        model.fit(Xtr, yi)
    return classes[model.predict(Xte)]


def smooth(pred, groups, k):
    """Causal majority vote over the last k predictions of each session."""
    if k <= 1:
        return pred
    out = pred.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for j, i in enumerate(idx):
            out[i] = Counter(pred[idx[max(0, j - k + 1):j + 1]]).most_common(1)[0][0]
    return out


# --------------------------------------------------------------------------- run

def run(name, c, paths):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    labels = TASK_LABELS[c["task"]]
    per_subject, all_y, all_p = {}, [], []
    for fold in data.subject_folds():
        train = [s for s in data.SUBJECTS if s not in fold]
        Xtr, ytr, _ = build_set(train, paths, c, c["length"] // 2)
        Xtr, ytr = rebalance(Xtr, ytr, c["sampling"], rng)
        Xte, yte, gte = build_set(fold, paths, c, c["length"])
        pred = smooth(fit_predict(c, Xtr, ytr, Xte), gte, c["smooth"])
        for s in fold:
            m = gte // 10 == s
            per_subject[s] = {
                "accuracy": float(accuracy_score(yte[m], pred[m])),
                "macro_f1": float(f1_score(yte[m], pred[m], labels=labels, average="macro", zero_division=0)),
                "windows": int(m.sum()),
            }
        all_y.append(yte)
        all_p.append(pred)
    y, p = np.concatenate(all_y), np.concatenate(all_p)
    acc = np.array([v["accuracy"] for v in per_subject.values()])
    f1 = np.array([v["macro_f1"] for v in per_subject.values()])
    prec, rec, f1c, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    result = {
        "name": name, "config": c, "labels": labels,
        "accuracy_mean": float(acc.mean()), "accuracy_sd": float(acc.std(ddof=1)),
        "macro_f1_mean": float(f1.mean()), "macro_f1_sd": float(f1.std(ddof=1)),
        "macro_f1_min": float(f1.min()), "macro_f1_max": float(f1.max()),
        "pooled_accuracy": float(accuracy_score(y, p)),
        "pooled_macro_f1": float(f1_score(y, p, labels=labels, average="macro")),
        "windows": int(len(y)),
        "per_class": {"precision": prec.tolist(), "recall": rec.tolist(), "f1": f1c.tolist(), "support": sup.tolist()},
        "confusion": confusion_matrix(y, p, labels=labels).tolist(),
        "per_subject": per_subject,
        "seconds": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=1))
    print(f"{name:26s} acc {100 * acc.mean():5.1f} +- {100 * acc.std(ddof=1):4.1f}   "
          f"macro-F1 {f1.mean():.3f} +- {f1.std(ddof=1):.3f}   ({result['seconds']} s)", flush=True)
    return result


def main(argv):
    paths = data.session_paths()
    names = argv or [n for n in CONFIGS if n != "app"]
    for name in names:
        run(name, CONFIGS[name], paths)


if __name__ == "__main__":
    main(sys.argv[1:])
