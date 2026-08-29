"""Background worker producing fast sample chunks and classification windows."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from emg_ai_arm.emulator.fake_emg_8ch import stream_sequence, FS, SAMPLES


CHUNK_SAMPLES = 5


class StreamWorker(QThread):
    samples_ready = pyqtSignal(np.ndarray)
    window_ready = pyqtSignal(np.ndarray, int)
    error = pyqtSignal(str)

    def __init__(self, source="fake", port=None, replay_path=None, parent=None):
        super().__init__(parent)

        self.source = source
        self.port = port
        self.replay_path = replay_path

        self._running = False

        # FS/SAMPLES come from the 8-channel research pipeline (matches
        # rf_emg_best.joblib: 200-sample windows). Both "fake" (synthetic
        # demo signal) and "replay" (real held-out subject recording) use
        # this same window size and pacing.
        self.fs = FS
        self.win_samples = SAMPLES

    def stop(self):
        self._running = False

    def _create_generator(self):
        """Create the selected EMG input generator."""

        if self.source == "fake":
            # Synthetic 8-channel demo signal. NOT physiological EMG -
            # used only when no real recording is selected for replay.
            return stream_sequence()

        elif self.source == "serial":
            from emg_ai_arm.acquisition.serial_reader import serial_windows

            if not self.port:
                raise RuntimeError(
                    "Serial source requires a COM port."
                )

            return serial_windows(self.port)

        elif self.source == "replay":
            from emg_ai_arm.acquisition.replay_csv_8ch import replay

            if not self.replay_path:
                raise RuntimeError(
                    "Replay source requires a recording file."
                )

            path = Path(self.replay_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Replay file not found: {path}"
                )

            return replay(path)

        else:
            raise ValueError(
                f"Unknown EMG source: {self.source!r}"
            )

    def run(self):
        self._running = True

        try:
            gen = self._create_generator()

            sample_period = 1.0 / float(self.fs)
            chunk_period = sample_period * CHUNK_SAMPLES

            while self._running:

                window, label = next(gen)

                window = np.asarray(
                    window,
                    dtype=np.float32
                )

                if window.ndim != 2:
                    raise ValueError(
                        f"Invalid EMG window shape: {window.shape}"
                    )

                if window.shape[0] == 0:
                    continue

                lab_int = (
                    int(label)
                    if label is not None
                    else -1
                )

                n = window.shape[0]

                # Stream the window in small chunks
                for start in range(0, n, CHUNK_SAMPLES):

                    if not self._running:
                        break

                    end = min(
                        start + CHUNK_SAMPLES,
                        n
                    )

                    chunk = window[start:end]

                    t0 = time.perf_counter()

                    self.samples_ready.emit(
                        chunk.copy()
                    )

                    elapsed = (
                        time.perf_counter() - t0
                    )

                    remaining = (
                        chunk_period - elapsed
                    )

                    if remaining > 0:
                        time.sleep(remaining)

                # Send the complete window
                if self._running:
                    self.window_ready.emit(
                        window,
                        lab_int
                    )

        except StopIteration:
            # Replay reached the end of the file.
            self._running = False

        except Exception as e:
            self._running = False
            self.error.emit(str(e))
