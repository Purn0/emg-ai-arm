import time
import joblib
import numpy as np

from emg_ai_arm.emulator.fake_emg import stream_sequence
from emg_ai_arm.features.extract_features import extract_features
from emg_ai_arm.control.state_machine import Controller


MODEL_PATH = "emg_ai_arm/models/rf_model.joblib"


def strength_from_window(w: np.ndarray) -> float:
    # Simple strength estimate: max of channel means
    return float(max(w[:,0].mean(), w[:,1].mean()))

def main():
    clf = joblib.load(MODEL_PATH)
    ctrl = Controller()

    gen = stream_sequence()
    period = 0.2  # seconds, matches window size

    print("Running real-time fake EMG inference. Ctrl+C to stop.\n")

    while True:
        w, true_lab = next(gen)
        x = extract_features(w).reshape(1, -1)
        pred = int(clf.predict(x)[0])
        strength = strength_from_window(w)

        cmd = ctrl.update(pred, strength)
        print(f"pred={pred} strength={strength:.2f} -> {cmd}")

        time.sleep(period)

if __name__ == "__main__":
    main()
