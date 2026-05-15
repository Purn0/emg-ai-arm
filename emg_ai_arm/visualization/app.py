"""
Main PyQt6 entry point for the EMG-AI-Arm GUI.

Usage
-----
    python -m emg_ai_arm.visualization.app

Three tabs share a single background StreamWorker:

    1. Live signal     — scrolling 2-channel plot.
    2. Trainer         — cue-based labelled-window recording + RF training.
    3. Inference + Arm — runs a trained model and animates the virtual arm.

A top toolbar lets you pick the data source (fake emulator / Arduino
serial) and start / stop the stream. Everything that talks to the
stream listens to `worker.window_ready` and reacts in its own slot.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from emg_ai_arm.visualization.inference_tab import InferenceTab
from emg_ai_arm.visualization.signal_tab import SignalTab
from emg_ai_arm.visualization.stream_worker import StreamWorker
from emg_ai_arm.visualization.trainer_tab import TrainerTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG-AI-Arm — control panel")
        self.resize(1180, 760)

        # ---- Tabs ----
        self.tab_signal = SignalTab()
        self.tab_trainer = TrainerTab()
        self.tab_inference = InferenceTab()

        tabs = QTabWidget()
        tabs.addTab(self.tab_signal, "1 · Live signal")
        tabs.addTab(self.tab_trainer, "2 · Trainer")
        tabs.addTab(self.tab_inference, "3 · Inference + arm")
        self.setCentralWidget(tabs)

        # ---- Top bar: source selector + start/stop ----
        toolbar = QToolBar("Source")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        toolbar.addWidget(QLabel("  Source: "))
        self.combo_source = QComboBox()
        self.combo_source.addItems(["fake (emulator)", "serial (Arduino)"])
        toolbar.addWidget(self.combo_source)

        self.btn_start = QPushButton("Start stream")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        toolbar.addWidget(self.btn_start)
        toolbar.addWidget(self.btn_stop)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Pick a source and press Start.")

        # ---- Worker (created on Start) ----
        self.worker: StreamWorker | None = None

        # ---- Signals ----
        self.btn_start.clicked.connect(self._start_stream)
        self.btn_stop.clicked.connect(self._stop_stream)

    # ------------------------------------------------------------------- #
    # Stream lifecycle
    # ------------------------------------------------------------------- #

    def _start_stream(self) -> None:
        src = "fake" if self.combo_source.currentIndex() == 0 else "serial"
        port = None
        if src == "serial":
            port, ok = QInputDialog.getText(
                self, "Serial port",
                "Arduino COM port (e.g. COM7):",
                text="COM3",
            )
            if not ok or not port:
                return

        self.worker = StreamWorker(source=src, port=port)
        self.worker.window_ready.connect(self.tab_signal.on_window)
        self.worker.window_ready.connect(self.tab_trainer.on_window)
        self.worker.window_ready.connect(self.tab_inference.on_window)
        self.worker.error.connect(self._on_worker_error)

        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_source.setEnabled(False)
        self.status.showMessage(f"Streaming from {src}…")

    def _stop_stream(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo_source.setEnabled(True)
        self.status.showMessage("Stopped.")

    def _on_worker_error(self, msg: str) -> None:
        self._stop_stream()
        QMessageBox.critical(self, "Stream error", msg)

    def closeEvent(self, event):
        self._stop_stream()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
