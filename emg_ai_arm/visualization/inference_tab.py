"""
Inference tab: camera preview, simulated arm, live prediction and command
history. Camera gestures and EMG windows both end up as a RobotCommand.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from emg_ai_arm.control.gesture_mapper import gesture_to_command
from emg_ai_arm.control.robot_command import RobotCommand
from emg_ai_arm.emg_model import load_emg_model
from emg_ai_arm.visualization.arm_widget import ArmWidget

# Labels of the UCI "EMG data for Gestures" dataset (class 7 is not used).
EMG_CLASS_NAMES = {
    0: "Unmarked",
    1: "Hand at rest",
    2: "Fist",
    3: "Wrist flexion",
    4: "Wrist extension",
    5: "Radial deviation",
    6: "Ulnar deviation",
}

# An EMG prediction arrives once per window, a camera prediction once per
# frame; applying each EMG command a few times keeps the arm's motion visible.
EMG_REPEATS = 3


class InferenceTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.arm = ArmWidget()
        self.camera_label = QLabel("Camera preview")
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left = QVBoxLayout()
        left.addWidget(self.camera_label)
        left.addWidget(self.arm)

        pred_box = QGroupBox("Live prediction")
        pred_layout = QVBoxLayout(pred_box)
        self.pred_label = QLabel("Prediction: —")
        self.pred_label.setWordWrap(True)
        self.pred_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.mode_label = QLabel("Source: —")
        self.mode_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.strength_label = QLabel("Confidence: 0.00")
        self.cmd_label = QLabel("Command: STOP")
        self.cmd_label.setFont(QFont("Consolas", 12))
        self.score_label = QLabel("")
        for w in (self.pred_label, self.mode_label, self.strength_label, self.cmd_label, self.score_label):
            pred_layout.addWidget(w)

        hist_box = QGroupBox("Recent commands")
        hist_layout = QVBoxLayout(hist_box)
        self.history = QLabel("")
        self.history.setFont(QFont("Consolas", 9))
        self.history.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history.setMinimumHeight(140)
        hist_layout.addWidget(self.history)

        self.btn_reset = QPushButton("Reset arm pose")
        self.btn_reset.clicked.connect(self._reset)

        right = QVBoxLayout()
        right.addWidget(pred_box)
        right.addWidget(hist_box)
        right.addWidget(self.btn_reset)
        right.addStretch(1)

        layout = QHBoxLayout(self)
        left_widget, right_widget = QWidget(), QWidget()
        left_widget.setLayout(left)
        right_widget.setLayout(right)
        layout.addWidget(left_widget, 2)
        layout.addWidget(right_widget, 1)

        self._history: list[str] = []
        self._emg_model = None
        self._calibration = None
        self._correct = 0
        self._scored = 0

    # ------------------------------------------------------------------ EMG model

    def emg_model(self):
        """The EMG model bundle (loaded on first use)."""
        if self._emg_model is None:
            self._emg_model = load_emg_model()
        return self._emg_model

    def start_stream(self, calibration=None):
        """Called when any source starts. calibration = (mean, std) of the EMG features, or None."""
        self._calibration = calibration
        self._correct = 0
        self._scored = 0
        self.score_label.setText("")

    # ------------------------------------------------------------------ helpers

    def _reset(self):
        self.arm.reset_pose()
        self.pred_label.setText("Prediction: —")
        self.strength_label.setText("Confidence: 0.00")
        self.cmd_label.setText("Command: STOP")
        self._history.clear()
        self.history.setText("")

    def _show(self, source, prediction, confidence, cmd, history_entry):
        self.pred_label.setText(f"Prediction: {prediction}")
        self.mode_label.setText(f"Source: {source}")
        self.strength_label.setText(f"Confidence: {confidence:.2f}")
        self.cmd_label.setText(f"Command: {cmd.name}")
        self.arm.set_mode(source)
        self.arm.set_command(cmd.name)
        self._history = (self._history + [history_entry])[-12:]
        self.history.setText("\n".join(reversed(self._history)))

    # ------------------------------------------------------------------ camera

    def process_gesture(self, gesture: str, confidence: float):
        cmd = gesture_to_command(gesture)
        self._show("Camera", gesture, confidence, cmd, cmd.name)
        if cmd != RobotCommand.STOP:
            self.arm.apply_command(cmd, speed=max(0.6, min(1.0, confidence * 1.8)))

    def update_camera_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    # ------------------------------------------------------------------ EMG

    def process_emg_window(self, window, true_label: int, source: str = "EMG"):
        try:
            bundle = self.emg_model()
            features = bundle.features(np.asarray(window)[None])
            if self._calibration is not None:
                mean, std = self._calibration
                features = (features - mean) / std
            probs = bundle.classifier.predict_proba(features)[0]
            best = int(np.argmax(probs))
            pred = int(bundle.classes[best])
            confidence = float(probs[best])
        except Exception as e:
            self.pred_label.setText("Prediction: —")
            self.cmd_label.setText(f"EMG error: {e}")
            self.mode_label.setText(f"Source: {source}")
            return

        cmd = RobotCommand(pred)  # EMG classes 0-6 map onto the first seven commands
        name = EMG_CLASS_NAMES.get(pred, f"class {pred}")
        entry = cmd.name
        if true_label is not None and true_label >= 0:
            true_name = EMG_CLASS_NAMES.get(true_label, f"class {true_label}")
            name += f"\n(true: {true_name})"
            entry += f" (true: {true_name})"
            self._scored += 1
            self._correct += int(pred == true_label)
            self.score_label.setText(
                f"Correct so far: {self._correct}/{self._scored} "
                f"({100 * self._correct / self._scored:.1f}%)")
        self._show(source, name, confidence, cmd, entry)
        if cmd != RobotCommand.STOP:
            speed = max(0.6, min(1.0, confidence * 1.5))
            for _ in range(EMG_REPEATS):
                self.arm.apply_command(cmd, speed=speed)
