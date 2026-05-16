"""
Tab 1 - live two-channel EMG signal viewer.

Subscribes to the worker's fast `samples_ready` signal and appends new
samples to a rolling buffer. A QTimer-driven redraw at 30 fps keeps the
visual update rate independent of how fast samples arrive, so the plot
always feels smooth.
"""

from __future__ import annotations

from collections import deque
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

FS = 200
HISTORY_SECONDS = 8
HISTORY_LEN = FS * HISTORY_SECONDS

REFRESH_HZ = 30


class SignalTab(QWidget):
    """Displays a scrolling 2-channel EMG envelope at 30 fps."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Dark-mode-friendly plot styling
        pg.setConfigOptions(antialias=True, background="#1e1e22", foreground="#e8e8ea")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ---- Header row ----
        header = QHBoxLayout()
        title = QLabel("Live EMG envelope - 2 channels @ 200 Hz")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch(1)

        self.fps_label = QLabel("- fps")
        self.fps_label.setFont(QFont("Consolas", 10))
        self.fps_label.setStyleSheet("color: #8a8a90;")
        header.addWidget(self.fps_label)
        layout.addLayout(header)

        # ---- Plot ----
        self.plot = pg.PlotWidget()
        self.plot.setYRange(0.0, 1.2)
        self.plot.setLabel("left", "Amplitude")
        self.plot.setLabel("bottom", "Sample (last 8 s)")
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.addLegend(offset=(12, 12))

        self.curve_ch1 = self.plot.plot(pen=pg.mkPen("#4ec9b0", width=2), name="CH1")
        self.curve_ch2 = self.plot.plot(pen=pg.mkPen("#ce9178", width=2), name="CH2")
        layout.addWidget(self.plot)

        # ---- Status row ----
        self.status = QLabel("Status: idle")
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.status.setStyleSheet("color: #b0b0b6;")
        layout.addWidget(self.status)

        # ---- Buffers ----
        self._ch1 = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self._ch2 = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

        # ---- FPS tracking ----
        self._frames = 0
        self._frame_t0 = time.perf_counter()

        # ---- Redraw timer (30 fps regardless of sample arrival rate) ----
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / REFRESH_HZ))
        self._timer.timeout.connect(self._redraw)
        self._timer.start()

    # ------------------------------------------------------------------- #
    # Slots
    # ------------------------------------------------------------------- #

    def on_samples(self, chunk: np.ndarray) -> None:
        """Worker delivers small chunks (5 samples) at ~40 Hz."""
        n = chunk.shape[0]
        for i in range(n):
            self._ch1.append(float(chunk[i, 0]))
            self._ch2.append(float(chunk[i, 1]))

    def on_window(self, window: np.ndarray, true_label: int) -> None:
        """Slow per-window callback - just used to update the status row."""
        label_text = f"true label = {true_label}" if true_label >= 0 else ""
        self.status.setText(
            f"CH1 mean={window[:, 0].mean():.3f}    "
            f"CH2 mean={window[:, 1].mean():.3f}    {label_text}"
        )

    # ------------------------------------------------------------------- #
    # Internal
    # ------------------------------------------------------------------- #

    def _redraw(self) -> None:
        self.curve_ch1.setData(np.fromiter(self._ch1, dtype=np.float32))
        self.curve_ch2.setData(np.fromiter(self._ch2, dtype=np.float32))

        # update fps counter once per second
        self._frames += 1
        now = time.perf_counter()
        if now - self._frame_t0 >= 1.0:
            fps = self._frames / (now - self._frame_t0)
            self.fps_label.setText(f"{fps:.0f} fps")
            self._frames = 0
            self._frame_t0 = now
