"""
Tab 3 — real-time inference dashboard.

Loads a trained Random Forest from disk and runs it on every window
emitted by the shared stream worker. The classifier's prediction is
fed into the state-machine controller, and the resulting command drives
the 2-D virtual arm widget.

Lets you preview the entire control pipeline without any hardware
hooked up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from emg_ai_arm.control.state_machine import Controller
from emg_ai_arm.features.extract_features import extract_features
from emg_ai_arm.utils.config import MODELS_DIR
from emg_ai_arm.visualization.arm_widget import ArmWidget


CLASS_NAMES = {0: "REST", 1: "CH1", 2: "CH2", 3: "BOTH"}


class InferenceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- Left: arm visualisation ----
        self.arm = ArmWidget()

        # ---- Right: predictions + controls ----
        right = QVBoxLayout()

        load_box = QGroupBox("Model")
        lb = QHBoxLayout(load_box)
        self.model_label = QLabel("No model loaded")
        self.btn_load = QPushButton("Load model (.joblib)")
        self.btn_default = QPushButton("Load default (rf_model.joblib)")
        self.btn_reset = QPushButton("Reset arm pose")
        lb.addWidget(self.model_label, 1)
        lb.addWidget(self.btn_default)
        lb.addWidget(self.btn_load)
        right.addWidget(load_box)

        pred_box = QGroupBox("Live prediction")
        pb = QVBoxLayout(pred_box)
        self.pred_label = QLabel("Prediction: —")
        self.pred_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.strength_label = QLabel("Strength: 0.00")
        self.cmd_label = QLabel("Command: REST")
        self.cmd_label.setFont(QFont("Consolas", 12))
        self.mode_label = QLabel("Mode: GRIP")
        self.mode_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        for w in (self.pred_label, self.mode_label, self.strength_label, self.cmd_label):
            pb.addWidget(w)
        right.addWidget(pred_box)

        hist_box = QGroupBox("Recent commands")
        hb = QVBoxLayout(hist_box)
        self.history = QLabel("")
        self.history.setFont(QFont("Consolas", 9))
        self.history.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history.setMinimumHeight(140)
        hb.addWidget(self.history)
        right.addWidget(hist_box)

        right.addWidget(self.btn_reset)
        right.addStretch(1)

        # ---- Overall layout ----
        h = QHBoxLayout(self)
        h.addWidget(self.arm, 2)
        right_widget = QWidget()
        right_widget.setLayout(right)
        h.addWidget(right_widget, 1)

        # ---- State ----
        self._clf = None
        self._ctrl = Controller()
        self._history: list[str] = []

        # ---- Signals ----
        self.btn_load.clicked.connect(self._load_dialog)
        self.btn_default.clicked.connect(self._load_default)
        self.btn_reset.clicked.connect(self._reset)

        # Try to auto-load
        self._load_default()

    # ------------------------------------------------------------------- #
    # Model loading
    # ------------------------------------------------------------------- #

    def _load_default(self) -> None:
        path = MODELS_DIR / "rf_model.joblib"
        if path.exists():
            self._load_path(path)
        else:
            self.model_label.setText(
                "No default model found. Train one with "
                "`python -m emg_ai_arm.models.train_ml`."
            )

    def _load_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a model",
            str(MODELS_DIR),
            "Joblib model (*.joblib);;All files (*.*)",
        )
        if path:
            self._load_path(Path(path))

    def _load_path(self, path: Path) -> None:
        try:
            import joblib
            self._clf = joblib.load(str(path))
            self.model_label.setText(f"Loaded: {path.name}")
        except Exception as e:
            self._clf = None
            self.model_label.setText(f"Load failed: {e}")

    # ------------------------------------------------------------------- #
    # Reset
    # ------------------------------------------------------------------- #

    def _reset(self) -> None:
        self._ctrl = Controller()
        self.arm.reset_pose()
        self.mode_label.setText("Mode: GRIP")
        self.cmd_label.setText("Command: REST")
        self._history.clear()
        self.history.setText("")

    # ------------------------------------------------------------------- #
    # Slot — stream worker calls this for every new window
    # ------------------------------------------------------------------- #

    def on_window(self, window: np.ndarray, _true: int) -> None:
        if self._clf is None:
            return
        feats = extract_features(window).reshape(1, -1)
        try:
            pred = int(self._clf.predict(feats)[0])
        except Exception:
            return
        strength = float(max(window[:, 0].mean(), window[:, 1].mean()))

        cmd = self._ctrl.update(pred, strength)

        # Boost the speed passed to the arm so motion is clearly visible
        # under the emulator. Real EMG strength values tend to be larger,
        # so this multiplier can be reduced later when real data arrives.
        arm_speed = max(0.6, min(1.0, strength * 1.8))

        self.pred_label.setText(f"Prediction: {pred} ({CLASS_NAMES.get(pred, '?')})")
        self.strength_label.setText(f"Strength: {strength:.2f}")
        self.cmd_label.setText(f"Command: {cmd}")
        self.mode_label.setText(f"Mode: {self._ctrl.mode_names[self._ctrl.mode]}")

        self.arm.set_mode(self._ctrl.mode_names[self._ctrl.mode])
        self.arm.set_command(cmd)
        self.arm.apply_command(cmd, speed=arm_speed)

        # Maintain a small rolling command log
        self._history.append(cmd)
        self._history = self._history[-12:]
        self.history.setText("\n".join(reversed(self._history)))
