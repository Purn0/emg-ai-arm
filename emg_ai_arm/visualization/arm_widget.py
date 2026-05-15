"""
2-D schematic of the 6-DoF arm + claw, drawn with QPainter.

Joints animate smoothly toward target angles set by the inference tab.
The intent is to make the control loop visible without wiring up the
actual servos. Once you have hardware, the same `apply_command()` API
can write angles to the Arduino instead.

Coordinate system:
    base_x, base_y is the centre of the base servo.
    The arm extends upward and to the right by default.
    Angles are in degrees, joint positions in pixels.

Joints
------
0: base_yaw        (left/right of arm in the horizontal plane — drawn as a small triangle)
1: shoulder_pitch  (up/down of arm)
2: elbow_pitch
3: wrist_pitch     (up/down of wrist)
4: wrist_roll      (rotation, drawn as a rotating ring)
5: claw            (open/close)
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PyQt6.QtWidgets import QWidget


# Default servo angles (degrees) and limits.
DEFAULTS = {
    "base":     90.0,
    "shoulder": 90.0,
    "elbow":    90.0,
    "wrist_p":  90.0,
    "wrist_r":  90.0,
    "claw":     30.0,  # 0 = fully open, 60 = closed
}

LIMITS = {
    "base":     (0.0, 180.0),
    "shoulder": (20.0, 160.0),
    "elbow":    (10.0, 170.0),
    "wrist_p":  (20.0, 160.0),
    "wrist_r":  (0.0, 180.0),
    "claw":     (0.0, 60.0),
}

# Segment lengths in pixels.
L_SHOULDER = 110.0
L_ELBOW = 95.0
L_WRIST = 50.0
L_CLAW = 35.0


def _clip(name: str, value: float) -> float:
    lo, hi = LIMITS[name]
    return max(lo, min(hi, value))


class ArmWidget(QWidget):
    """QWidget that draws a side-view arm with an animated claw."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 360)
        self._current = dict(DEFAULTS)
        self._target = dict(DEFAULTS)

        # 30 fps animation toward target
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

        self._mode_text = "GRIP"
        self._cmd_text = "REST"

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def set_mode(self, mode: str) -> None:
        self._mode_text = mode

    def set_command(self, cmd: str) -> None:
        self._cmd_text = cmd

    def apply_command(self, cmd: str, speed: float = 0.3) -> None:
        """
        Translate a Controller command string into an angle delta and
        update the target pose. `speed` ∈ [0, 1] scales the step.
        """
        step = 6.0 * max(0.1, min(1.0, speed))  # degrees per command

        c = cmd.upper()
        # Strip the trailing speed= part if present.
        c = c.split(" SPEED=")[0]

        if c.startswith("GRIP_CLOSE"):
            self._target["claw"] = _clip("claw", self._target["claw"] + step)
        elif c.startswith("GRIP_OPEN"):
            self._target["claw"] = _clip("claw", self._target["claw"] - step)
        elif c.startswith("WRIST_LEFT"):
            self._target["wrist_r"] = _clip("wrist_r", self._target["wrist_r"] - step)
        elif c.startswith("WRIST_RIGHT"):
            self._target["wrist_r"] = _clip("wrist_r", self._target["wrist_r"] + step)
        elif c.startswith("ARM_UP"):
            self._target["shoulder"] = _clip("shoulder", self._target["shoulder"] - step)
            self._target["elbow"] = _clip("elbow", self._target["elbow"] + step * 0.5)
        elif c.startswith("ARM_DOWN"):
            self._target["shoulder"] = _clip("shoulder", self._target["shoulder"] + step)
            self._target["elbow"] = _clip("elbow", self._target["elbow"] - step * 0.5)
        # REST / HOLD / MODE_SWITCH have no effect on pose.

    def reset_pose(self) -> None:
        self._target = dict(DEFAULTS)

    # ------------------------------------------------------------------- #
    # Animation
    # ------------------------------------------------------------------- #

    def _tick(self) -> None:
        # Exponential easing toward target.
        alpha = 0.15
        for k in self._current:
            self._current[k] += alpha * (self._target[k] - self._current[k])
        self.update()

    # ------------------------------------------------------------------- #
    # Drawing
    # ------------------------------------------------------------------- #

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#fafafa"))

        # ---- Floor and base ----
        floor_y = h - 50
        p.setPen(QPen(QColor("#bbb"), 1))
        p.drawLine(0, floor_y, w, floor_y)

        base_x = w * 0.30
        base_y = floor_y
        p.setBrush(QBrush(QColor("#444")))
        p.setPen(QPen(QColor("#222"), 2))
        p.drawRect(QRectF(base_x - 35, base_y - 18, 70, 18))

        # Yaw indicator (small triangle on the base showing left/right)
        base_yaw = self._current["base"]
        yaw_dx = (base_yaw - 90.0) / 90.0 * 25.0
        p.setBrush(QBrush(QColor("#888")))
        p.drawPolygon(
            QPointF(base_x + yaw_dx, base_y - 22),
            QPointF(base_x + yaw_dx - 8, base_y - 8),
            QPointF(base_x + yaw_dx + 8, base_y - 8),
        )

        # ---- Forward kinematics (side view) ----
        sh_x, sh_y = base_x, base_y - 20

        sh_angle = math.radians(self._current["shoulder"])
        el_angle = sh_angle + math.radians(180.0 - self._current["elbow"])
        wr_angle = el_angle + math.radians(self._current["wrist_p"] - 90.0)

        el_x = sh_x + L_SHOULDER * math.cos(-sh_angle)
        el_y = sh_y + L_SHOULDER * math.sin(-sh_angle)

        wr_x = el_x + L_ELBOW * math.cos(-el_angle)
        wr_y = el_y + L_ELBOW * math.sin(-el_angle)

        claw_x = wr_x + L_WRIST * math.cos(-wr_angle)
        claw_y = wr_y + L_WRIST * math.sin(-wr_angle)

        # Segments
        p.setPen(QPen(QColor("#1f77b4"), 8, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sh_x, sh_y), QPointF(el_x, el_y))

        p.setPen(QPen(QColor("#2ca02c"), 7, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(el_x, el_y), QPointF(wr_x, wr_y))

        p.setPen(QPen(QColor("#ff7f0e"), 6, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(wr_x, wr_y), QPointF(claw_x, claw_y))

        # Joints
        def _joint(cx, cy, r=7, color="#222"):
            p.setBrush(QBrush(QColor(color)))
            p.setPen(QPen(QColor("#111"), 1))
            p.drawEllipse(QPointF(cx, cy), r, r)

        _joint(sh_x, sh_y, 9)
        _joint(el_x, el_y, 7)
        _joint(wr_x, wr_y, 6)

        # Wrist roll ring (rotated tick)
        roll = math.radians(self._current["wrist_r"])
        p.setPen(QPen(QColor("#555"), 2))
        p.setBrush(QBrush(Qt.GlobalColor.transparent))
        p.drawEllipse(QPointF(wr_x, wr_y), 12, 12)
        tick_x = wr_x + 12 * math.cos(roll)
        tick_y = wr_y - 12 * math.sin(roll)
        p.setPen(QPen(QColor("#d62728"), 2))
        p.drawLine(QPointF(wr_x, wr_y), QPointF(tick_x, tick_y))

        # Claw — two opening fingers
        claw_open = self._current["claw"]
        finger_len = L_CLAW
        spread = math.radians(35.0 - claw_open / 2.0)  # closes as value grows

        for sign in (-1, +1):
            ax = -wr_angle + sign * spread
            fx = claw_x + finger_len * math.cos(ax)
            fy = claw_y + finger_len * math.sin(ax)
            p.setPen(QPen(QColor("#ff7f0e"), 5, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(claw_x, claw_y), QPointF(fx, fy))

        # ---- Heads-up text ----
        p.setPen(QPen(QColor("#222")))
        p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        p.drawText(QPointF(12, 22), f"Mode: {self._mode_text}")
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(QPointF(12, 42), f"Cmd:  {self._cmd_text}")

        p.setFont(QFont("Consolas", 9))
        y0 = h - 18
        readout = (
            f"base={self._current['base']:.0f}  "
            f"sh={self._current['shoulder']:.0f}  "
            f"el={self._current['elbow']:.0f}  "
            f"wp={self._current['wrist_p']:.0f}  "
            f"wr={self._current['wrist_r']:.0f}  "
            f"claw={self._current['claw']:.0f}"
        )
        p.drawText(QPointF(12, y0), readout)
