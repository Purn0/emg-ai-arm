"""
Background worker that produces (window, label) pairs at a steady rate
and forwards them to the GUI via Qt signals.

A single worker is shared by all three tabs (signal viewer, trainer,
inference) so we only read the data source once.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from emg_ai_arm.emulator.fake_emg import stream_sequence


class StreamWorker(QThread):
    """
    Emits a new EMG window on the `window_ready` signal.

    Parameters
    ----------
    source : "fake" or "serial"
        Where to read from. "serial" requires a working SerialEMG.
    interval_sec : float
        Time between windows. Defaults to 0.2 s (matches fake_emg).
    """

    window_ready = pyqtSignal(np.ndarray, int)  # (window, true_label_or_-1)
    error = pyqtSignal(str)

    def __init__(
        self,
        source: str = "fake",
        interval_sec: float = 0.2,
        port: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.interval_sec = float(interval_sec)
        self.port = port
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        gen = None
        try:
            if self.source == "fake":
                gen = stream_sequence()
            elif self.source == "serial":
                # Lazy import so users without pyserial can still run the GUI
                from emg_ai_arm.acquisition.serial_reader import serial_windows
                if not self.port:
                    raise RuntimeError("Serial source requires a COM port.")
                gen = serial_windows(self.port)
            else:
                raise ValueError(f"Unknown source: {self.source!r}")

            while self._running:
                t0 = time.perf_counter()
                window, label = next(gen)
                # Serial reader yields (window, None); coerce to int sentinel
                lab_int = int(label) if label is not None else -1
                self.window_ready.emit(window.astype(np.float32), lab_int)

                # Pace the loop to the requested interval (for the emulator).
                # Serial naturally paces itself; the sleep just smooths jitter.
                elapsed = time.perf_counter() - t0
                remaining = self.interval_sec - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except StopIteration:
            pass
        except Exception as e:
            self.error.emit(str(e))
