import time
import numpy as np
import serial

FS = 200
WIN_SEC = 0.2
SAMPLES = int(FS * WIN_SEC)

def _parse_line(line: str):
    # expected: t_ms,ch1,ch2
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        t_ms = int(parts[0])
        ch1 = float(parts[1])
        ch2 = float(parts[2])
        return t_ms, ch1, ch2
    except ValueError:
        return None

def serial_windows(port: str, baud: int = 115200):
    """
    Yields: (window, None)
    window shape: (SAMPLES, 2)
    Values are raw ADC by default; you can normalize later.
    """
    ser = serial.Serial(port, baudrate=baud, timeout=1)
    time.sleep(2)  # let Arduino reset

    buf = []

    while True:
        line = ser.readline().decode(errors="ignore")
        parsed = _parse_line(line)
        if parsed is None:
            continue

        _, ch1, ch2 = parsed
        buf.append((ch1, ch2))

        if len(buf) >= SAMPLES:
            w = np.array(buf[-SAMPLES:], dtype=float)  # (SAMPLES,2)
            yield w, None
