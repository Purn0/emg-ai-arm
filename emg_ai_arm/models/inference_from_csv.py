import time
import joblib
import numpy as np

from emg_ai_arm.features.extract_features import extract_features
from emg_ai_arm.control.state_machine import Controller
from emg_ai_arm.utils.config import MODELS_DIR, RAW_DIR
from emg_ai_arm.acquisition.replay_csv import replay

FS = 200
WIN_SEC = 0.2
SAMPLES = int(FS * WIN_SEC)

def strength_from_window(w: np.ndarray) -> float:
    # for envelope windows this works well
    return float(max(w[:, 0].mean(), w[:, 1].mean()))

def main():
    model_path = MODELS_DIR / "rf_model.joblib"
    clf = joblib.load(str(model_path))

    ctrl = Controller()
    csv_path = RAW_DIR / "fake_stream.csv"

    print("Model:", model_path)
    print("CSV:", csv_path)
    print("Running CSV -> model -> controller (Ctrl+C to stop)\n")

    for w, true_lab in replay(csv_path):
        x = extract_features(w).reshape(1, -1)
        pred = int(clf.predict(x)[0])
        strength = strength_from_window(w)

        cmd = ctrl.update(pred, strength)
        print(f"true={true_lab} pred={pred} strength={strength:.2f} -> {cmd}")

        # replay() already sleeps WIN_SEC, so no sleep here

if __name__ == "__main__":
    main()
