import time
import serial
from emg_ai_arm.utils.config import RAW_DIR

def main(port="COM3", baud=115200, seconds=60):
    out_path = RAW_DIR / "serial_stream.csv"

    ser = serial.Serial(port, baudrate=baud, timeout=1)
    time.sleep(2)

    start = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("t_ms,ch1,ch2\n")
        f.flush()

        while (time.time() - start) < seconds:
            line = ser.readline().decode(errors="ignore").strip()
            # accept only well-formed lines
            if line.count(",") == 2:
                f.write(line + "\n")

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
