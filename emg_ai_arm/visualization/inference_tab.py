"""
Tab 3 — real-time camera gesture inference dashboard.

The camera worker performs:
    camera -> MediaPipe -> gesture_model.pkl -> gesture name

This tab receives:
    gesture + confidence + camera frame

and sends:
    gesture -> RobotCommand -> virtual arm
"""

from __future__ import annotations

import cv2

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from emg_ai_arm.control.gesture_mapper import gesture_to_command
from emg_ai_arm.control.robot_command import RobotCommand
from emg_ai_arm.visualization.arm_widget import ArmWidget


class InferenceTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---------------------------------------------------------
        # Left side: camera + virtual arm
        # ---------------------------------------------------------

        self.arm = ArmWidget()

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.camera_label.setText("Camera Preview")

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.camera_label)
        left_layout.addWidget(self.arm)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # ---------------------------------------------------------
        # Right side: prediction + controls
        # ---------------------------------------------------------

        right = QVBoxLayout()

        # Model information
        model_box = QGroupBox("Camera Model")
        model_layout = QHBoxLayout(model_box)

        self.model_label = QLabel(
            "Using camera gesture model"
        )

        self.btn_load = QPushButton(
            "Load gesture model"
        )

        self.btn_reset = QPushButton(
            "Reset arm pose"
        )

        model_layout.addWidget(
            self.model_label,
            1
        )
        model_layout.addWidget(
            self.btn_load
        )

        right.addWidget(model_box)

        # Live prediction
        pred_box = QGroupBox("Live prediction")
        pred_layout = QVBoxLayout(pred_box)

        self.pred_label = QLabel(
            "Prediction: —"
        )
        self.pred_label.setFont(
            QFont(
                "Segoe UI",
                18,
                QFont.Weight.Bold
            )
        )

        self.strength_label = QLabel(
            "Confidence: 0.00"
        )

        self.cmd_label = QLabel(
            "Command: STOP"
        )
        self.cmd_label.setFont(
            QFont(
                "Consolas",
                12
            )
        )

        self.mode_label = QLabel(
            "Mode: CAMERA"
        )
        self.mode_label.setFont(
            QFont(
                "Segoe UI",
                14,
                QFont.Weight.Bold
            )
        )

        pred_layout.addWidget(
            self.pred_label
        )
        pred_layout.addWidget(
            self.mode_label
        )
        pred_layout.addWidget(
            self.strength_label
        )
        pred_layout.addWidget(
            self.cmd_label
        )

        right.addWidget(pred_box)

        # Command history
        hist_box = QGroupBox(
            "Recent commands"
        )

        hist_layout = QVBoxLayout(
            hist_box
        )

        self.history = QLabel("")
        self.history.setFont(
            QFont(
                "Consolas",
                9
            )
        )

        self.history.setAlignment(
            Qt.AlignmentFlag.AlignTop
        )

        self.history.setMinimumHeight(
            140
        )

        hist_layout.addWidget(
            self.history
        )

        right.addWidget(hist_box)

        right.addWidget(
            self.btn_reset
        )

        right.addStretch(1)

        # ---------------------------------------------------------
        # Overall layout
        # ---------------------------------------------------------

        layout = QHBoxLayout(self)

        layout.addWidget(
            left_widget,
            2
        )

        right_widget = QWidget()
        right_widget.setLayout(right)

        layout.addWidget(
            right_widget,
            1
        )

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------

        self._history: list[str] = []

        # ---------------------------------------------------------
        # Signals
        # ---------------------------------------------------------

        self.btn_load.clicked.connect(
            self._load_model_dialog
        )

        self.btn_reset.clicked.connect(
            self._reset
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model_dialog(self):
        """
        Optional manual model selection.

        The CameraWorker normally owns the actual gesture model,
        so this is mainly informational for now.
        """

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose gesture model",
            "",
            "Pickle model (*.pkl);;All files (*.*)",
        )

        if path:
            self.model_label.setText(
                f"Selected: {path}"
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset(self):

        self.arm.reset_pose()

        self.pred_label.setText(
            "Prediction: —"
        )

        self.strength_label.setText(
            "Confidence: 0.00"
        )

        self.cmd_label.setText(
            "Command: STOP"
        )

        self.mode_label.setText(
            "Mode: CAMERA"
        )

        self._history.clear()

        self.history.setText("")

    # ------------------------------------------------------------------
    # Camera gesture
    # ------------------------------------------------------------------

    def process_gesture(
        self,
        gesture: str,
        strength: float
    ):

        # Convert gesture name into RobotCommand.
        cmd = gesture_to_command(
            gesture
        )

        # ------------------------------------------------------
        # UI
        # ------------------------------------------------------

        self.pred_label.setText(
            f"Prediction: {gesture}"
        )

        self.strength_label.setText(
            f"Confidence: {strength:.2f}"
        )

        self.cmd_label.setText(
            f"Command: {cmd.name}"
        )

        self.mode_label.setText(
            "Mode: CAMERA"
        )

        # ------------------------------------------------------
        # Virtual arm
        # ------------------------------------------------------

        self.arm.set_mode(
            "CAMERA"
        )

        self.arm.set_command(
            cmd.name
        )

        # Confidence controls movement speed.
        arm_speed = max(
            0.6,
            min(
                1.0,
                strength * 1.8
            )
        )

        if cmd != RobotCommand.STOP:

            self.arm.apply_command(
                cmd,
                speed=arm_speed
            )

        # ------------------------------------------------------
        # History
        # ------------------------------------------------------

        self._history.append(
            cmd.name
        )

        self._history = self._history[-12:]

        self.history.setText(
            "\n".join(
                reversed(
                    self._history
                )
            )
        )

    # ------------------------------------------------------------------
    # Camera frame
    # ------------------------------------------------------------------

    def update_camera_frame(
        self,
        frame
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb.shape

        img = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format.Format_RGB888
        )

        pix = QPixmap.fromImage(
            img
        )

        self.camera_label.setPixmap(
            pix.scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )