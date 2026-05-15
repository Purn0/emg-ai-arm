import joblib
import numpy as np

from emg_ai_arm.features.extract_features import extract_features
from emg_ai_arm.control.state_machine import Controller
from emg_ai_arm.utils.config import MODELS_DIR
from emg_ai_arm.acquisition.serial_reader import serial_windows

def strength_from_window(w: np.ndarray) -> float:
    # for raw ADC, normalize roughly to 0..1
    # NOTE: you can improve this once you know your baseline
    ch1 = w[:, 0].mean() / 1023.0
    ch2 = w[:, 1].mean() / 1023.0
    return float(max(ch1, ch2))

def main():
    clf = joblib.load(str(MODELS_DIR / "rf_model.joblib"))
    ctrl = Controller()

    PORT = "COM3"  # change later
    print("Listening on", PORT, "(Ctrl+C to stop)\n")

    for w, _ in serial_windows(PORT):
        x = extract_features(w).reshape(1, -1)
        pred = int(clf.predict(x)[0])
        strength = strength_from_window(w)
        cmd = ctrl.update(pred, strength)
        print(f"pred={pred} strength={strength:.2f} -> {cmd}")

if __name__ == "__main__":
    main()
