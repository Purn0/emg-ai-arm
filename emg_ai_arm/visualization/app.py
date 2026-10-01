"""EMG-AI-Arm GUI: python -m emg_ai_arm.visualization.app"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QLabel, QMainWindow, QMessageBox, QPushButton,
    QStatusBar, QTabWidget, QToolBar,
)

from emg_ai_arm.camera.camera_worker import CameraWorker
from emg_ai_arm.emg_model import calibrate_from_file
from emg_ai_arm.utils.config import RAW_DIR
from emg_ai_arm.visualization.inference_tab import InferenceTab
from emg_ai_arm.visualization.signal_tab import SignalTab
from emg_ai_arm.visualization.stream_worker import StreamWorker

# Subject 05 of the UCI dataset. The app's EMG model is trained without this
# subject, so the replay is a test on an unseen person; session 2 serves as
# the unlabelled calibration recording when the model needs one.
REPLAY_FILE = RAW_DIR / "subject05_session1.txt"
CALIBRATION_FILE = RAW_DIR / "subject05_session2.txt"

SOURCES = [
    ("Replay: recorded EMG (subject 05)", "replay"),
    ("Synthetic EMG (smoke test only)", "synthetic"),
    ("Camera", "camera"),
]

DARK_STYLESHEET = """
QTabWidget::pane { border: 1px solid #2a2a30; top: -1px; }
QTabBar::tab {
    background: #1e1e22; color: #b0b0b6;
    padding: 8px 18px; border: 1px solid #2a2a30; border-bottom: none;
}
QTabBar::tab:selected {
    background: #26262b; color: #e8e8ea;
    border-bottom: 2px solid #4ec9b0;
}
QPushButton {
    background: #2a2a30; color: #e8e8ea;
    border: 1px solid #3a3a44; padding: 6px 14px; border-radius: 4px;
}
QPushButton:hover { background: #34343c; }
QPushButton:disabled { color: #6a6a72; background: #232328; }
QComboBox {
    background: #26262b; color: #e8e8ea;
    border: 1px solid #3a3a44; padding: 4px 8px; border-radius: 3px;
}
QGroupBox {
    border: 1px solid #2a2a30; border-radius: 5px;
    margin-top: 12px; padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #b0b0b6;
}
QToolBar { background: #1a1a1f; border: none; spacing: 6px; padding: 4px; }
QStatusBar { background: #1a1a1f; color: #b0b0b6; }
QLabel { color: #d6d6dc; }
"""


def apply_dark_palette(app):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, "#1e1e22"), (QPalette.ColorRole.WindowText, "#e8e8ea"),
        (QPalette.ColorRole.Base, "#26262b"), (QPalette.ColorRole.AlternateBase, "#2a2a30"),
        (QPalette.ColorRole.Text, "#e8e8ea"), (QPalette.ColorRole.Button, "#2a2a30"),
        (QPalette.ColorRole.ButtonText, "#e8e8ea"), (QPalette.ColorRole.Highlight, "#4ec9b0"),
        (QPalette.ColorRole.HighlightedText, "#1a1a1f"), (QPalette.ColorRole.ToolTipBase, "#1a1a1f"),
        (QPalette.ColorRole.ToolTipText, "#e8e8ea"),
    ):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(DARK_STYLESHEET)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG-AI-Arm")
        self.resize(1280, 820)

        self.tab_signal = SignalTab()
        self.tab_inference = InferenceTab()
        tabs = QTabWidget()
        tabs.addTab(self.tab_signal, "Live signal")
        tabs.addTab(self.tab_inference, "Inference + arm")
        self.setCentralWidget(tabs)

        toolbar = QToolBar("Source")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        toolbar.addWidget(QLabel("   Source: "))
        self.combo_source = QComboBox()
        self.combo_source.addItems([label for label, _ in SOURCES])
        toolbar.addWidget(self.combo_source)
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        toolbar.addWidget(self.btn_start)
        toolbar.addWidget(self.btn_stop)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Pick a source and press Start.")

        self.worker = None
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)

    def _start(self):
        label, source = SOURCES[self.combo_source.currentIndex()]
        self.tab_signal.reset()
        try:
            if source == "camera":
                self.tab_inference.start_stream()
                self.worker = CameraWorker()
                self.worker.prediction_ready.connect(self.tab_inference.process_gesture)
                self.worker.frame_ready.connect(self.tab_inference.update_camera_frame)
            else:
                self.worker = self._emg_worker(source, label)
        except Exception as e:
            QMessageBox.critical(self, "Cannot start", str(e))
            return
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_source.setEnabled(False)
        self.status.showMessage(f"Streaming: {label}")

    def _emg_worker(self, source, label):
        model = self.tab_inference.emg_model()
        calibration = None
        if source == "replay":
            needed = [REPLAY_FILE] + ([CALIBRATION_FILE] if model.calibration else [])
            missing = [str(p) for p in needed if not p.exists()]
            if missing:
                raise FileNotFoundError("Missing " + ", ".join(missing) +
                                        "; run `python -m experiments.download_data`.")
            if model.calibration:
                calibration = calibrate_from_file(model, CALIBRATION_FILE)
        self.tab_inference.start_stream(calibration)
        worker = StreamWorker(source=source, replay_path=REPLAY_FILE, window=model.window)
        worker.samples_ready.connect(self.tab_signal.on_samples)
        worker.window_ready.connect(self.tab_signal.on_window)
        worker.window_ready.connect(
            lambda window, true_label: self.tab_inference.process_emg_window(window, true_label, label))
        worker.finished_stream.connect(self._on_replay_finished)
        return worker

    def _on_replay_finished(self):
        self._stop()
        self.status.showMessage("Replay finished. " + self.tab_inference.score_label.text())

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo_source.setEnabled(True)
        self.status.showMessage("Stopped.")

    def _on_worker_error(self, msg):
        self._stop()
        QMessageBox.critical(self, "Stream error", msg)

    def closeEvent(self, event):
        self._stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
