"""Background thread that streams EMG windows from a replay or synthetic source."""

from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from emg_ai_arm.acquisition.replay_csv_8ch import replay
from emg_ai_arm.emulator.fake_emg_8ch import stream_sequence

# Display pacing: samples are sent to the plot at this rate. The UCI files
# hold about 967 samples per second, so a replay runs about 5x slower than
# the recording, which keeps the arm motion easy to follow.
PLAYBACK_RATE = 200
CHUNK_SAMPLES = 5


class StreamWorker(QThread):
    samples_ready = pyqtSignal(np.ndarray)
    window_ready = pyqtSignal(np.ndarray, int)
    finished_stream = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, source="replay", replay_path=None, window=200, parent=None):
        super().__init__(parent)
        self.source = source
        self.replay_path = replay_path
        self.window = int(window)
        self._running = False

    def stop(self):
        self._running = False

    def _windows(self):
        if self.source == "replay":
            if not self.replay_path:
                raise RuntimeError("the replay source needs a recording file")
            return replay(self.replay_path, self.window)
        if self.source == "synthetic":
            return stream_sequence(self.window)
        raise ValueError(f"unknown EMG source: {self.source!r}")

    def run(self):
        """Stream until the source ends (then emit finished_stream), stop() or an error."""
        self._running = True
        chunk_period = CHUNK_SAMPLES / PLAYBACK_RATE
        try:
            for window, label in self._windows():
                if not self._running:
                    return
                for start in range(0, len(window), CHUNK_SAMPLES):
                    if not self._running:
                        return
                    t0 = time.perf_counter()
                    self.samples_ready.emit(window[start:start + CHUNK_SAMPLES].copy())
                    remaining = chunk_period - (time.perf_counter() - t0)
                    if remaining > 0:
                        time.sleep(remaining)
                if self._running:
                    self.window_ready.emit(window, int(label))
            self.finished_stream.emit()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._running = False
