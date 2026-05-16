"""Background worker producing fast sample chunks and slow classification windows."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from emg_ai_arm.emulator.fake_emg import stream_sequence, FS, SAMPLES


CHUNK_SAMPLES = 5


class StreamWorker(QThread):
    samples_ready = pyqtSignal(np.ndarray)
    window_ready = pyqtSignal(np.ndarray, int)
    error = pyqtSignal(str)

    def __init__(self, source="fake", port=None, parent=None):
        super().__init__(parent)
        self.source = source
        self.port = port
        self._running = False
        self.fs = FS
        self.win_samples = SAMPLES

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        gen = None
        try:
            if self.source == "fake":
                gen = stream_sequence()
            elif self.source == "serial":
                from emg_ai_arm.acquisition.serial_reader import serial_windows
                if not self.port:
                    raise RuntimeError("Serial source requires a COM port.")
                gen = serial_windows(self.port)
            else:
                raise ValueError("Unknown source: " + repr(self.source))

            sample_period = 1.0 / float(self.fs)
            chunk_period = sample_period * CHUNK_SAMPLES

            while self._running:
                window, label = next(gen)
                lab_int = int(label) if label is not None else -1
                window = window.astype(np.float32)

                n = window.shape[0]
                for start in range(0, n, CHUNK_SAMPLES):
                    if not self._running:
                        break
                    end = min(start + CHUNK_SAMPLES, n)
                    chunk = window[start:end]
                    t0 = time.perf_counter()
                    self.samples_ready.emit(chunk.copy())
                    elapsed = time.perf_counter() - t0
                    remaining = chunk_period - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

                if self._running:
                    self.window_ready.emit(window, lab_int)

        except StopIteration:
            pass
        except Exception as e:
            self.error.emit(str(e))
