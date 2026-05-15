import time
from emg_ai_arm.utils.config import RAW_DIR
from emg_ai_arm.emulator.fake_emg import stream_sequence

WIN_SEC = 0.2

def main():
    out_path = RAW_DIR / "fake_stream.csv"
    gen = stream_sequence()

    start = time.time()
    n_windows = 300  # 60 seconds

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("t_ms,ch1,ch2,label\n")
        f.flush()

        for i in range(n_windows):
            w, lab = next(gen)
            ch1 = float(w[:, 0].mean())
            ch2 = float(w[:, 1].mean())
            t_ms = int((time.time() - start) * 1000)
            f.write(f"{t_ms},{ch1:.6f},{ch2:.6f},{lab}\n")

            if i % 20 == 0:
                f.flush()

            time.sleep(WIN_SEC)

    print("Saved:", out_path)
    print("Rows:", n_windows)

if __name__ == "__main__":
    main()
