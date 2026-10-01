"""
Score an EMG model on a replayed session exactly as the app does: the same
windows (non-overlapping, label of the first sample), the same features,
calibration from the subject's other session if the model uses it, and one
predict_proba per window.

    python -m experiments.replay_check                          # app model, subject 05
    python -m experiments.replay_check --model path/to/rf_emg_best.joblib
    python -m experiments.replay_check --model ... --all-subjects

--all-subjects scores every subject and reports the recall on the six
labelled gestures (classes 1-6). A Random Forest recognises the gestures of
the subjects it was trained on almost perfectly, so this shows which
subjects a model has seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from emg_ai_arm.acquisition.replay_csv_8ch import load_windows
from emg_ai_arm.emg_model import calibrate_from_file, load_emg_model
from experiments import data


def score_session(model, path, calibration_path):
    windows, labels = load_windows(path, model.window)
    X = model.features(windows)
    if model.calibration:
        mean, std = calibrate_from_file(model, calibration_path)
        X = (X - mean) / std
    probs = model.classifier.predict_proba(X)
    return labels, np.asarray(model.classes)[probs.argmax(axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--subject", type=int, default=5)
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--all-subjects", action="store_true")
    ap.add_argument("--json", type=Path, default=None, help="also save the result here")
    args = ap.parse_args()

    model = load_emg_model(args.model)
    paths = data.session_paths()
    print(f"model: {model.description} ({model.feature_set} features, {model.window}-sample windows, "
          f"calibration: {model.calibration})")

    key, other = (args.subject, args.session), (args.subject, 3 - args.session)
    y, p = score_session(model, paths[key], paths[other])
    print(f"subject {args.subject:02d} session {args.session}: {len(y)} windows, "
          f"accuracy {100 * accuracy_score(y, p):.1f}%, macro-F1 {f1_score(y, p, average='macro'):.3f}")
    cm = confusion_matrix(y, p, labels=list(range(7)))
    print(cm)
    if args.json:
        args.json.write_text(json.dumps({
            "model": model.description, "subject": args.subject, "session": args.session,
            "windows": int(len(y)), "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro")), "confusion": cm.tolist(),
        }, indent=1))

    if args.all_subjects:
        print("\nsubject  gesture recall (classes 1-6)")
        for s in data.SUBJECTS:
            ys, ps = [], []
            for sess in (1, 2):
                a, b = score_session(model, paths[(s, sess)], paths[(s, 3 - sess)])
                ys.append(a)
                ps.append(b)
            ys, ps = np.concatenate(ys), np.concatenate(ps)
            active = ys > 0
            print(f"  {s:02d}     {np.mean(ps[active] == ys[active]):.3f}")


if __name__ == "__main__":
    main()
