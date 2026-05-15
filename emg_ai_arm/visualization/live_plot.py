import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from emg_ai_arm.emulator.fake_emg import stream_sequence

FS = 200
WIN_SEC = 0.2
SAMPLES = int(FS * WIN_SEC)

def main():
    gen = stream_sequence()

    history_seconds = 8
    history_len = FS * history_seconds

    ch1_hist = deque([0.0] * history_len, maxlen=history_len)
    ch2_hist = deque([0.0] * history_len, maxlen=history_len)

    plt.ion()
    fig, ax = plt.subplots()

    while True:
        w, lab = next(gen)  # w: (SAMPLES, 2)

        # push samples into scrolling buffer
        for i in range(SAMPLES):
            ch1_hist.append(float(w[i, 0]))
            ch2_hist.append(float(w[i, 1]))

        ax.clear()
        ax.plot(np.array(ch1_hist), label="CH1")
        ax.plot(np.array(ch2_hist), label="CH2")
        ax.set_title(f"Live stream (fake) | label={lab}")
        ax.set_ylim(0, 1.2)
        ax.legend()
        plt.pause(0.01)

if __name__ == "__main__":
    main()
