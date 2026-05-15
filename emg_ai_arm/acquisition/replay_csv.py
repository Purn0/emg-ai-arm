import time
import csv
import numpy as np
from emg_ai_arm.utils.config import RAW_DIR

FS = 200
WIN_SEC = 0.2
SAMPLES = int(FS * WIN_SEC)

def replay(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ch1 = float(row["ch1"])
            ch2 = float(row["ch2"])
            lab = int(row["label"])

            w = np.column_stack([
                np.full(SAMPLES, ch1, dtype=float),
                np.full(SAMPLES, ch2, dtype=float)
            ])

            yield w, lab
            time.sleep(WIN_SEC)

def main():
    path = RAW_DIR / "fake_stream.csv"
    print("Replaying:", path)

    for i, (w, lab) in enumerate(replay(path)):
        print(f"{i:03d} label={lab} ch1={w[:,0].mean():.3f} ch2={w[:,1].mean():.3f}")
        if i >= 20:
            break

if __name__ == "__main__":
    main()
