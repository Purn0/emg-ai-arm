"""
Tab 1 — live two-channel EMG signal viewer.

A scrolling pyqtgraph plot fed by the shared StreamWorker. Roughly the
same scrolling-buffer behaviour as `visualization/live_plot.py`, but
embedded in the Qt event loop instead of matplotlib.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget, QLabel

FS = 200
HISTORY_SECONDS = 8
HISTORY_LEN = FS * HISTORY_SECONDS


class SignalTab(QWidget):
    """Displays a scrolling 2-channel EMG envelope."""

    def __init__(self, parent=None):
        super().__init__(parent)

        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Live EMG envelope — 2 channels @ 200 Hz")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        self.plot = pg.PlotWidget()
        self.plot.setYRange(0.0, 1.2)
        self.plot.setLabel("left", "Amplitude")
        self.plot.setLabel("bottom", "Sample (last 8 s)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        legend = self.plot.addLegend(offset=(10, 10))

        self.curve_ch1 = self.plot.plot(pen=pg.mkPen("#1f77b4", width=2), name="CH1")
        self.curve_ch2 = self.plot.plot(pen=pg.mkPen("#d62728", width=2), name="CH2")
        layout.addWidget(self.plot)

        self.status = QLabel("Status: idle")
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.status)

        self._ch1 = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self._ch2 = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)

    # ------------------------------------------------------------------- #
    # Slots
    # ------------------------------------------------------------------- #

    def on_window(self, window: np.ndarray, true_label: int) -> None:
        """Called by the shared StreamWorker for every new window."""
        n = window.shape[0]
        for i in range(n):
            self._ch1.append(float(window[i, 0]))
            self._ch2.append(float(window[i, 1]))
        self.curve_ch1.setData(np.fromiter(self._ch1, dtype=np.float32))
        self.curve_ch2.setData(np.fromiter(self._ch2, dtype=np.float32))

        label_text = f"true label = {true_label}" if true_label >= 0 else ""
        self.status.setText(
            f"CH1 mean={window[:, 0].mean():.3f}    "
            f"CH2 mean={window[:, 1].mean():.3f}    {label_text}"
        )
