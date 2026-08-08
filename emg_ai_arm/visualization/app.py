"""Main PyQt6 entry point for the EMG-AI-Arm GUI."""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QPushButton, QStatusBar, QTabWidget, QToolBar,
)

from emg_ai_arm.visualization.inference_tab import InferenceTab
from emg_ai_arm.visualization.signal_tab import SignalTab
from emg_ai_arm.visualization.stream_worker import StreamWorker
from emg_ai_arm.visualization.trainer_tab import TrainerTab
from emg_ai_arm.camera.camera_worker import CameraWorker

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
QComboBox, QSpinBox {
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
QProgressBar {
    background: #26262b; border: 1px solid #3a3a44;
    border-radius: 4px; text-align: center; color: #e8e8ea;
}
QProgressBar::chunk { background: #4ec9b0; border-radius: 3px; }
QToolBar { background: #1a1a1f; border: none; spacing: 6px; padding: 4px; }
QStatusBar { background: #1a1a1f; color: #b0b0b6; }
QLabel { color: #d6d6dc; }
"""


def apply_dark_palette(app):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#1e1e22"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#e8e8ea"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#26262b"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#2a2a30"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#e8e8ea"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#2a2a30"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#e8e8ea"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#4ec9b0"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#1a1a1f"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1a1a1f"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#e8e8ea"))
    app.setPalette(pal)
    app.setStyleSheet(DARK_STYLESHEET)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG-AI-Arm  -  control panel")
        self.resize(1280, 820)

        self.tab_signal = SignalTab()
        self.tab_trainer = TrainerTab()
        self.tab_inference = InferenceTab()

        tabs = QTabWidget()
        tabs.addTab(self.tab_signal, "1  Live signal")
        tabs.addTab(self.tab_trainer, "2  Trainer")
        tabs.addTab(self.tab_inference, "3  Inference + arm")
        self.setCentralWidget(tabs)

        toolbar = QToolBar("Source")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        toolbar.addWidget(QLabel("   Source: "))
        self.combo_source = QComboBox()
        self.combo_source.addItems([
            "fake (emulator)",
            "serial (Arduino)",
            "camera (vision)"
        ])
        toolbar.addWidget(self.combo_source)

        self.btn_start = QPushButton("Start stream")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        toolbar.addWidget(self.btn_start)
        toolbar.addWidget(self.btn_stop)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Pick a source and press Start.")

        self.worker = None

        self.btn_start.clicked.connect(self._start_stream)
        self.btn_stop.clicked.connect(self._stop_stream)

    def _start_stream(self):
        index = self.combo_source.currentIndex()

        if index == 0:
            src = "fake"

        elif index == 1:
            src = "serial"

        else:
            src = "camera"
        port = None
        if src == "serial":
            port, ok = QInputDialog.getText(
                self, "Serial port",
                "Arduino COM port (e.g. COM7):",
                text="COM3",
            )
            if not ok or not port:
                return

        if src == "camera":
            self.worker = CameraWorker()
        else:
            self.worker = StreamWorker(source=src, port=port)
        if src == "camera":

            self.worker.prediction_ready.connect(
                self.tab_inference.process_prediction
            )
            self.worker.frame_ready.connect(
                self.tab_inference.update_camera_frame
            )
            self.worker.error.connect(self._on_worker_error)

        else:

            self.worker.samples_ready.connect(
                self.tab_signal.on_samples
            )

            self.worker.window_ready.connect(
                self.tab_signal.on_window
            )

            self.worker.window_ready.connect(
                self.tab_trainer.on_window
            )

            self.worker.window_ready.connect(
                self.tab_inference.on_window
            )

            self.worker.error.connect(
                self._on_worker_error
            )
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.combo_source.setEnabled(False)
        self.status.showMessage("Streaming from " + src + " ...")

    def _stop_stream(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.combo_source.setEnabled(True)
        self.status.showMessage("Stopped.")

    def _on_worker_error(self, msg):
        self._stop_stream()
        QMessageBox.critical(self, "Stream error", msg)

    def closeEvent(self, event):
        self._stop_stream()
        event.accept()


def main():
    app = QApplication(sys.argv)
    apply_dark_palette(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
