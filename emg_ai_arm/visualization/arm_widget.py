"""Semi-3D isometric visualisation of the 6-DoF arm + claw.

Each link is drawn as a filled trapezoidal "tube" projected from 3D,
with disc-shaped servo housings at every joint to give it the look of
a real hobby kit arm. The right side of the widget shows live joint
angle gauges, one per servo.

DEFAULTS are set so the arm sits in a curled "C-shape" rest pose, the
way a typical 6-DoF servo arm powers up. Tune DEFAULTS to match your
specific physical arm if it parks in a different home position.

Joints / servos
---------------
0: base_yaw        (rotation around the vertical Z axis)
1: shoulder_pitch  (up/down of upper arm)
2: elbow_pitch     (forearm hinge)
3: wrist_pitch     (claw up/down)
4: wrist_roll      (claw rotation)
5: claw            (open/close)
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QTimer, Qt
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPen, QPolygonF,
)
from PyQt6.QtWidgets import QWidget


JOINTS = ("base", "shoulder", "elbow", "wrist_p", "wrist_r", "claw")

# Rest position - tuned to look like a curled / hunched hobby arm.
# shoulder=135 means upper arm rises ~45 deg from horizontal.
# elbow=35  means a deep negative bend so the forearm tucks back-down.
# wrist_p=55 lets the claw drop further down.
DEFAULTS = {
    "base":     90.0,
    "shoulder": 210.0,
    "elbow":    0,
    "wrist_p":  55.0,
    "wrist_r":  90.0,
    "claw":     25.0,
}

LIMITS = {
    "base":     (25.0, 155.0),
    "shoulder": (160.0, 210.0),   # was (55.0, 145.0)
    "elbow":    (0, 40.0),     # was (35.0, 145.0)
    "wrist_p":  (15.0, 95.0),     # was (45.0, 135.0)
    "wrist_r":  (20.0, 160.0),
    "claw":     (0.0, 60.0),
}
LABELS = {
    "base":     "Base yaw",
    "shoulder": "Shoulder",
    "elbow":    "Elbow",
    "wrist_p":  "Wrist pitch",
    "wrist_r":  "Wrist roll",
    "claw":     "Claw open",
}

# Per-joint accent colours (used in gauges + link tinting)
COLORS = {
    "base":     "#7aa2f7",
    "shoulder": "#4ec9b0",
    "elbow":    "#9ece6a",
    "wrist_p":  "#e0af68",
    "wrist_r":  "#bb9af7",
    "claw":     "#f7768e",
}

# Link sizes in 3D units (scaled to pixels on draw).
L_BASE   = 0.32      # base pillar height
L_UPPER  = 1.00
L_FORE   = 0.85
L_WRIST  = 0.35
L_CLAW   = 0.32

W_UPPER  = 0.22      # link tube widths (sideways thickness in iso)
W_FORE   = 0.18
W_WRIST  = 0.14


def _clip(name, value):
    lo, hi = LIMITS[name]
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------- #
# Forward kinematics + isometric projection
# --------------------------------------------------------------------------- #

def _fk(angles):
    yaw = math.radians(angles["base"] - 90.0)
    sh = math.radians(angles["shoulder"] - 90.0)
    el = math.radians(angles["elbow"] - 90.0)
    wp = math.radians(angles["wrist_p"] - 90.0)

    sh_abs = sh
    el_abs = sh_abs + el
    wp_abs = el_abs + wp

    base = np.array([0.0, 0.0, 0.0])
    shoulder = np.array([0.0, 0.0, L_BASE])

    upper_dir = np.array([math.cos(sh_abs), 0.0, math.sin(sh_abs)])
    elbow = shoulder + L_UPPER * upper_dir

    fore_dir = np.array([math.cos(el_abs), 0.0, math.sin(el_abs)])
    wrist = elbow + L_FORE * fore_dir

    wrist_dir = np.array([math.cos(wp_abs), 0.0, math.sin(wp_abs)])
    claw_root = wrist + L_WRIST * wrist_dir

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    def rot(p):
        return np.array([
            cos_y * p[0] - sin_y * p[1],
            sin_y * p[0] + cos_y * p[1],
            p[2],
        ])

    # ---------------------------------------------------------
    # Wrist roll
    # ---------------------------------------------------------

    roll = math.radians(
        angles["wrist_r"] - 90.0
    )

    # Local sideways direction before wrist roll.
    local_side = np.array([
        0.0,
        1.0,
        0.0
    ])

    # Rotate local_side around the wrist direction.
    axis = wrist_dir / max(
        np.linalg.norm(wrist_dir),
        1e-9
    )

    cos_r = math.cos(roll)
    sin_r = math.sin(roll)

    # Rodrigues' rotation formula.
    rolled_side = (
            local_side * cos_r
            + np.cross(axis, local_side) * sin_r
            + axis
            * np.dot(axis, local_side)
            * (1.0 - cos_r)
    )

    rolled_side = rolled_side / max(
        np.linalg.norm(rolled_side),
        1e-9
    )

    return {
        "base": rot(base),
        "shoulder": rot(shoulder),
        "elbow": rot(elbow),
        "wrist": rot(wrist),
        "claw_root": rot(claw_root),
        "upper_dir": rot(upper_dir),
        "fore_dir": rot(fore_dir),
        "wrist_dir": rot(wrist_dir),

        # NEW
        "gripper_side": rot(rolled_side),

        "yaw": yaw,
        "wp_abs": wp_abs,
    }


def _iso(p3, ox, oy, scale):
    cos30 = math.cos(math.radians(30))
    sin30 = math.sin(math.radians(30))
    sx = (p3[0] - p3[1]) * cos30 * scale + ox
    sy = (p3[0] + p3[1]) * sin30 * scale - p3[2] * scale + oy
    return QPointF(sx, sy)


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #

def _shade(hex_color, factor):
    """Darken or lighten a hex colour by factor (0..2)."""
    c = QColor(hex_color)
    h, s, v, a = c.getHsv()
    v = max(0, min(255, int(v * factor)))
    out = QColor()
    out.setHsv(h, s, v, a)
    return out


# --------------------------------------------------------------------------- #
# Widget
# --------------------------------------------------------------------------- #

class ArmWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(620, 500)
        self._current = dict(DEFAULTS)
        self._target = dict(DEFAULTS)

        self._easing = 0.18
        self._mode_text = "GRIP"
        self._cmd_text = "REST"

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def set_mode(self, mode):
        self._mode_text = mode

    def set_command(self, cmd):
        self._cmd_text = cmd

    def apply_command(self, cmd, speed=0.5):
        step = 4.0 * max(0.25, min(1.0, speed))

        # Accept RobotCommand enum
        # as well as the old string command format.
        if hasattr(cmd, "name"):
            c = cmd.name.upper()
        else:
            c = str(cmd).upper().split(" SPEED=")[0]

        # ---------------------------------------------------------
        # Gripper — continuous movement
        # ---------------------------------------------------------
        if c.startswith("GRIP_CLOSE"):
            self._target["claw"] = _clip(
                "claw",
                self._target["claw"] + step
            )

        elif c.startswith("GRIP_OPEN"):
            self._target["claw"] = _clip(
                "claw",
                self._target["claw"] - step
            )

        # ---------------------------------------------------------
        # Wrist roll — continuous movement
        # ---------------------------------------------------------
        elif c.startswith("WRIST_CW"):
            self._target["wrist_r"] = _clip(
                "wrist_r",
                self._target["wrist_r"] + step
            )

        elif c.startswith("WRIST_CCW"):
            self._target["wrist_r"] = _clip(
                "wrist_r",
                self._target["wrist_r"] - step
            )

        # ---------------------------------------------------------
        # Base yaw — continuous movement
        # ---------------------------------------------------------
        elif c.startswith("BASE_LEFT"):
            self._target["base"] = _clip(
                "base",
                self._target["base"] - step
            )

        elif c.startswith("BASE_RIGHT"):
            self._target["base"] = _clip(
                "base",
                self._target["base"] + step
            )

        # ---------------------------------------------------------
        # Arm up — weighted 3-servo movement
        # ---------------------------------------------------------
        elif c.startswith("ARM_UP"):
            self._target["shoulder"] = _clip("shoulder", self._target["shoulder"] + step * 0.35)
            self._target["elbow"] = _clip("elbow", self._target["elbow"] + step * 0.35)
            self._target["wrist_p"] = _clip("wrist_p", self._target["wrist_p"] + step * 1.0)

        elif c.startswith("ARM_DOWN"):
            self._target["shoulder"] = _clip("shoulder", self._target["shoulder"] - step * 0.35)
            self._target["elbow"] = _clip("elbow", self._target["elbow"] - step * 0.35)
            self._target["wrist_p"] = _clip("wrist_p", self._target["wrist_p"] - step * 1.0)

    def reset_pose(self):
        self._target = dict(DEFAULTS)

    def _tick(self):
        a = self._easing
        for k in self._current:
            self._current[k] += a * (self._target[k] - self._current[k])
        self.update()

    # ------------------------------------------------------------------- #
    # Painting
    # ------------------------------------------------------------------- #

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#f5f5f5"))

        split = int(w * 0.66)
        self._draw_iso_scene(p, 0, 0, split, h)
        self._draw_gauges(p, split, 0, w - split, h)

    # --- iso scene -------------------------------------------------------

    def _draw_iso_scene(self, p, x, y, w, h):
        cx = x + w // 2
        cy = y + int(h * 0.66)
        scale = min(w, h) * 0.20

        self._draw_grid(p, cx, cy, scale)
        self._draw_axes(p, cx, cy, scale)

        pts = _fk(self._current)
        self._draw_base(p, pts, cx, cy, scale)
        self._draw_arm(p, pts, cx, cy, scale)
        self._draw_gripper(p, pts, cx, cy, scale)
        self._draw_hud(p)

    def _draw_grid(self, p, cx, cy, scale):
        p.setPen(QPen(QColor("#d0d0d5"), 1))
        n = 4
        step = 0.5
        for i in range(-n, n + 1):
            a = _iso(np.array([i * step, -n * step, 0]), cx, cy, scale)
            b = _iso(np.array([i * step,  n * step, 0]), cx, cy, scale)
            p.drawLine(a, b)
            a = _iso(np.array([-n * step, i * step, 0]), cx, cy, scale)
            b = _iso(np.array([ n * step, i * step, 0]), cx, cy, scale)
            p.drawLine(a, b)

    def _draw_axes(self, p, cx, cy, scale):
        origin = _iso(np.array([0, 0, 0]), cx, cy, scale)
        for vec, col, label in (
            (np.array([2.0, 0, 0]),   "#7aa2f7", "X"),
            (np.array([0,   2.0, 0]), "#9ece6a", "Y"),
            (np.array([0,   0, 1.7]), "#f7768e", "Z"),
        ):
            end = _iso(vec, cx, cy, scale)
            p.setPen(QPen(QColor(col), 1.5))
            p.drawLine(origin, end)
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.setPen(QPen(QColor(col)))
            p.drawText(QPointF(end.x() + 4, end.y() + 4), label)

    # ---- Base -----------------------------------------------------------

    def _draw_base(self, p, pts, cx, cy, scale):
        # Cylindrical pillar drawn as side rectangle + ellipse caps.
        # Side colour darker than top.
        base_radius = 0.36
        top_h = 0.05
        body_h = L_BASE - top_h

        # Bottom ellipse
        bot_c = _iso(np.array([0, 0, 0]), cx, cy, scale)
        rx = base_radius * math.cos(math.radians(30)) * scale
        ry = base_radius * math.sin(math.radians(30)) * scale + 1
        p.setBrush(QBrush(QColor("#2d2d34")))
        p.setPen(QPen(QColor("#1a1a1f"), 1))
        p.drawEllipse(bot_c, rx, ry)

        # Side wall via polygon
        left_bot = _iso(np.array([base_radius, -base_radius, 0]), cx, cy, scale)
        right_bot = _iso(np.array([-base_radius, base_radius, 0]), cx, cy, scale)
        left_top = _iso(np.array([base_radius, -base_radius, body_h]),
                        cx, cy, scale)
        right_top = _iso(np.array([-base_radius, base_radius, body_h]),
                         cx, cy, scale)

        # Use a vertical gradient for some shading
        grad = QLinearGradient(0, left_top.y(), 0, left_bot.y())
        grad.setColorAt(0, QColor("#3a3a44"))
        grad.setColorAt(1, QColor("#202024"))
        p.setBrush(QBrush(grad))
        p.setPen(QPen(QColor("#1a1a1f"), 1))
        side = QPolygonF([left_bot, right_bot, right_top, left_top])
        p.drawPolygon(side)

        # Top disc (slightly raised platform with bolt holes)
        top_c = _iso(np.array([0, 0, body_h]), cx, cy, scale)
        p.setBrush(QBrush(QColor("#4a4a55")))
        p.setPen(QPen(QColor("#2a2a30"), 1.2))
        p.drawEllipse(top_c, rx, ry)

        # Bolts on top disc
        p.setBrush(QBrush(QColor("#1c1c20")))
        for ang in (30, 150, 270):
            br = base_radius * 0.75
            bx = br * math.cos(math.radians(ang))
            by = br * math.sin(math.radians(ang))
            bp = _iso(np.array([bx, by, body_h + 0.01]), cx, cy, scale)
            p.drawEllipse(bp, 2.5, 2.5)

        # Shoulder yoke (two upright plates surrounding the shoulder axle)
        yoke_h = 0.18
        yoke_t = 0.06
        yoke_w = 0.30
        for side_sign in (-1, +1):
            y_off = side_sign * (yoke_w / 2 + yoke_t / 2)
            corners = [
                np.array([-yoke_t/2, y_off, body_h]),
                np.array([ yoke_t/2, y_off, body_h]),
                np.array([ yoke_t/2, y_off, body_h + yoke_h]),
                np.array([-yoke_t/2, y_off, body_h + yoke_h]),
            ]
            poly = QPolygonF([_iso(c, cx, cy, scale) for c in corners])
            p.setBrush(QBrush(QColor("#3a3a44")))
            p.setPen(QPen(QColor("#1a1a1f"), 1))
            p.drawPolygon(poly)

        # Shoulder axle visible between yokes
        shoulder_screen = _iso(pts["shoulder"], cx, cy, scale)
        p.setBrush(QBrush(QColor(COLORS["shoulder"])))
        p.setPen(QPen(QColor("#0c0c10"), 1.5))
        p.drawEllipse(shoulder_screen, 10, 10)
        p.setBrush(QBrush(QColor("#1c1c20")))
        p.drawEllipse(shoulder_screen, 3, 3)

    # ---- Arm links ------------------------------------------------------

    def _link_tube(self, p, start_3d, end_3d, width, color, yaw, cx, cy, scale):
        """Draw a rectangular link between two 3D points with given side width."""
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.0]) * (width / 2)

        c1 = start_3d + side
        c2 = start_3d - side
        c3 = end_3d   - side
        c4 = end_3d   + side

        # 2D corners
        s1 = _iso(c1, cx, cy, scale)
        s2 = _iso(c2, cx, cy, scale)
        s3 = _iso(c3, cx, cy, scale)
        s4 = _iso(c4, cx, cy, scale)

        # Highlight side (top edge of tube): brighter
        bright = _shade(color, 1.20)
        dark = _shade(color, 0.55)

        # Use a left-to-right gradient on the polygon for a 3D feel
        center_x = (s1.x() + s2.x() + s3.x() + s4.x()) / 4
        center_y = (s1.y() + s2.y() + s3.y() + s4.y()) / 4
        grad = QLinearGradient(center_x - 30, center_y - 30,
                               center_x + 30, center_y + 30)
        grad.setColorAt(0.0, bright)
        grad.setColorAt(0.5, QColor(color))
        grad.setColorAt(1.0, dark)

        poly = QPolygonF([s1, s2, s3, s4])
        p.setBrush(QBrush(grad))
        p.setPen(QPen(_shade(color, 0.4), 1.2))
        p.drawPolygon(poly)

        # Subtle centreline highlight along the link's length
        mid_start = QPointF((s1.x() + s2.x()) / 2, (s1.y() + s2.y()) / 2)
        mid_end = QPointF((s3.x() + s4.x()) / 2, (s3.y() + s4.y()) / 2)
        p.setPen(QPen(_shade(color, 1.4), 1.0))
        p.drawLine(mid_start, mid_end)

    def _draw_plate_link(self, p, start_3d, end_3d,
                         width, thickness, cx, cy, scale):
        """
        Draw a flat mechanical arm plate between two 3D joint positions.
        Designed to resemble the black aluminum plates used on the physical arm.
        """
        direction = end_3d - start_3d
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return

        direction = direction / length

        # Horizontal sideways vector.
        side = np.array([
            -direction[1],
            direction[0],
            0.0
        ])

        side_len = np.linalg.norm(side)

        if side_len < 1e-6:
            side = np.array([0.0, 1.0, 0.0])
        else:
            side /= side_len

        side *= width / 2.0

        # Slight vertical thickness.
        thickness_vec = np.array([0.0, 0.0, thickness / 2.0])

        corners = [
            start_3d + side + thickness_vec,
            start_3d - side + thickness_vec,
            end_3d - side + thickness_vec,
            end_3d + side + thickness_vec,
        ]

        bottom = [
            start_3d + side - thickness_vec,
            start_3d - side - thickness_vec,
            end_3d - side - thickness_vec,
            end_3d + side - thickness_vec,
        ]

        top_poly = QPolygonF([
            _iso(c, cx, cy, scale) for c in corners
        ])

        bottom_poly = QPolygonF([
            _iso(c, cx, cy, scale) for c in bottom
        ])

        black = QColor("#202124")
        dark = QColor("#111214")
        edge = QColor("#08090a")

        # Main top surface
        p.setBrush(QBrush(black))
        p.setPen(QPen(edge, 1.2))
        p.drawPolygon(top_poly)

        # Bottom / thickness surface
        p.setBrush(QBrush(dark))
        p.setPen(QPen(edge, 1.0))
        p.drawPolygon(bottom_poly)

        # Center highlight
        s = _iso(start_3d, cx, cy, scale)
        e = _iso(end_3d, cx, cy, scale)

        p.setPen(QPen(QColor("#3a3b3e"), 1.0))
        p.drawLine(s, e)

    def _draw_servo_box(self, p, center_3d,
                        width, height, depth,
                        cx, cy, scale,
                        axle_radius=8):
        """
        Draw a compact black rectangular hobby-servo housing.
        """

        # The box is centered around the joint.
        x = width / 2.0
        y = depth / 2.0
        z = height / 2.0

        pts = [
            np.array([-x, -y, -z]),
            np.array([x, -y, -z]),
            np.array([x, y, -z]),
            np.array([-x, y, -z]),

            np.array([-x, -y, z]),
            np.array([x, -y, z]),
            np.array([x, y, z]),
            np.array([-x, y, z]),
        ]

        pts = [center_3d + v for v in pts]

        top = QPolygonF([
            _iso(pts[i], cx, cy, scale)
            for i in (4, 5, 6, 7)
        ])

        front = QPolygonF([
            _iso(pts[i], cx, cy, scale)
            for i in (0, 1, 5, 4)
        ])

        side = QPolygonF([
            _iso(pts[i], cx, cy, scale)
            for i in (1, 2, 6, 5)
        ])

        # Draw darker faces first.
        p.setBrush(QBrush(QColor("#151619")))
        p.setPen(QPen(QColor("#08090a"), 1.0))
        p.drawPolygon(front)

        p.setBrush(QBrush(QColor("#1c1d20")))
        p.drawPolygon(side)

        p.setBrush(QBrush(QColor("#292a2e")))
        p.drawPolygon(top)

        # Axle
        c = _iso(center_3d, cx, cy, scale)

        p.setBrush(QBrush(QColor("#55565a")))
        p.setPen(QPen(QColor("#08090a"), 1.2))
        p.drawEllipse(c, axle_radius, axle_radius)

        p.setBrush(QBrush(QColor("#111214")))
        p.drawEllipse(c, axle_radius * 0.35, axle_radius * 0.35)

        # Servo mounting screws
        screw_offset = min(width, depth) * 0.30

        for sx, sy in (
                (-screw_offset, -screw_offset),
                (screw_offset, -screw_offset),
                (-screw_offset, screw_offset),
                (screw_offset, screw_offset),
        ):
            screw = _iso(
                center_3d + np.array([sx, sy, z + 0.01]),
                cx, cy, scale
            )

            p.setBrush(QBrush(QColor("#66676b")))
            p.setPen(QPen(QColor("#0a0a0b"), 0.8))
            p.drawEllipse(screw, 2.0, 2.0)

    def _servo_housing(self, p, pos_3d, color, radius_px, cx, cy, scale):
        c = _iso(pos_3d, cx, cy, scale)
        # outer ring
        p.setBrush(QBrush(_shade(color, 0.7)))
        p.setPen(QPen(QColor("#0c0c10"), 1.5))
        p.drawEllipse(c, radius_px, radius_px)
        # inner disc
        p.setBrush(QBrush(QColor(color)))
        p.setPen(QPen(_shade(color, 0.5), 1))
        p.drawEllipse(c, radius_px * 0.65, radius_px * 0.65)
        # axle dot
        p.setBrush(QBrush(QColor("#1c1c20")))
        p.setPen(QPen(QColor("#1c1c20")))
        p.drawEllipse(c, radius_px * 0.20, radius_px * 0.20)

    def _draw_arm(self, p, pts, cx, cy, scale):
        """
        Mechanical rendering of the main six-DOF arm.

        Physical structure:
            shoulder servo
            -> upper arm plates
            -> elbow servo
            -> forearm plates
            -> wrist pitch servo
            -> wrist section
            -> wrist roll
        """

        # ---------------------------------------------------------------
        # Dimensions
        # ---------------------------------------------------------------

        upper_width = 0.32
        upper_thickness = 0.075

        fore_width = 0.28
        fore_thickness = 0.070

        wrist_width = 0.20
        wrist_thickness = 0.060

        # ---------------------------------------------------------------
        # Shoulder servo housing
        # ---------------------------------------------------------------

        self._draw_servo_box(
            p,
            pts["shoulder"],
            width=0.42,
            height=0.32,
            depth=0.28,
            cx=cx,
            cy=cy,
            scale=scale,
            axle_radius=10,
        )

        # Shoulder side brackets.
        shoulder = pts["shoulder"]

        bracket_offset = 0.20

        for sign in (-1, 1):
            offset = np.array([
                0.0,
                sign * bracket_offset,
                0.0
            ])

            bracket_start = shoulder + offset
            bracket_end = pts["elbow"] + offset

            self._draw_plate_link(
                p,
                bracket_start,
                bracket_end,
                width=upper_width,
                thickness=upper_thickness,
                cx=cx,
                cy=cy,
                scale=scale,
            )

        # ---------------------------------------------------------------
        # Elbow servo
        # ---------------------------------------------------------------

        self._draw_servo_box(
            p,
            pts["elbow"],
            width=0.40,
            height=0.30,
            depth=0.26,
            cx=cx,
            cy=cy,
            scale=scale,
            axle_radius=9,
        )

        # ---------------------------------------------------------------
        # Forearm plates
        # ---------------------------------------------------------------

        elbow = pts["elbow"]
        wrist = pts["wrist"]

        bracket_offset = 0.16

        for sign in (-1, 1):
            offset = np.array([
                0.0,
                sign * bracket_offset,
                0.0
            ])

            self._draw_plate_link(
                p,
                elbow + offset,
                wrist + offset,
                width=fore_width,
                thickness=fore_thickness,
                cx=cx,
                cy=cy,
                scale=scale,
            )

        # ---------------------------------------------------------------
        # Wrist pitch servo
        # ---------------------------------------------------------------

        self._draw_servo_box(
            p,
            pts["wrist"],
            width=0.34,
            height=0.25,
            depth=0.23,
            cx=cx,
            cy=cy,
            scale=scale,
            axle_radius=8,
        )

        # ---------------------------------------------------------------
        # Wrist extension
        # ---------------------------------------------------------------

        wrist_dir = pts["wrist_dir"]

        wrist_end = pts["claw_root"]

        self._draw_plate_link(
            p,
            pts["wrist"],
            wrist_end,
            width=wrist_width,
            thickness=wrist_thickness,
            cx=cx,
            cy=cy,
            scale=scale,
        )

        # ---------------------------------------------------------------
        # Wrist-roll housing
        # ---------------------------------------------------------------

        wr_center = _iso(
            pts["claw_root"],
            cx,
            cy,
            scale
        )

        p.setBrush(QBrush(QColor("#252629")))
        p.setPen(QPen(QColor("#08090a"), 1.5))

        p.drawEllipse(
            wr_center,
            13,
            13
        )

        # Roll plate
        p.setBrush(QBrush(QColor("#3a3b3f")))
        p.drawEllipse(
            wr_center,
            9,
            9
        )

        # Roll indicator
        roll = math.radians(
            self._current["wrist_r"] - 90.0
        )

        tick = QPointF(
            wr_center.x() + 9 * math.cos(roll),
            wr_center.y() - 9 * math.sin(roll),
        )

        p.setPen(
            QPen(
                QColor("#d0d0d0"),
                2
            )
        )

        p.drawLine(wr_center, tick)

    # ---- Gripper --------------------------------------------------------

    def _draw_gripper(self, p, pts, cx, cy, scale):
        """
        Mechanical two-finger gripper.

        claw = 0   -> open
        claw = 60  -> closed
        """

        forward = pts["wrist_dir"]

        # Wrist-roll-controlled sideways direction.
        side = pts["gripper_side"]

        # Make absolutely sure the side vector is perpendicular
        # to the gripper's forward direction.
        side = side - np.dot(side, forward) * forward

        side_norm = np.linalg.norm(side)

        if side_norm > 1e-9:
            side = side / side_norm
        else:
            side = np.array([
                0.0,
                1.0,
                0.0
            ])

        # ---------------------------------------------------------------
        # Gripper mounting plate
        # ---------------------------------------------------------------

        root = pts["claw_root"]

        palm_back = root
        palm_front = root + 0.12 * forward

        self._draw_plate_link(
            p,
            palm_back,
            palm_front,
            width=0.24,
            thickness=0.08,
            cx=cx,
            cy=cy,
            scale=scale,
        )

        # ---------------------------------------------------------------
        # Continuous claw opening
        # ---------------------------------------------------------------

        claw = self._current["claw"]

        # 38 degrees when fully open.
        # 4 degrees when nearly closed.
        spread = math.radians(
            38.0 - claw * (34.0 / 60.0)
        )

        # ---------------------------------------------------------------
        # Fingers
        # ---------------------------------------------------------------

        for sign in (-1, 1):
            pivot = (
                    palm_front
                    + sign * 0.11 * side
            )

            cos_s = math.cos(spread)
            sin_s = math.sin(spread)

            finger_dir = (
                    cos_s * forward
                    + sign * sin_s * side
            )

            # Main finger
            finger_end = (
                    pivot
                    + 0.27 * finger_dir
            )

            self._draw_plate_link(
                p,
                pivot,
                finger_end,
                width=0.075,
                thickness=0.045,
                cx=cx,
                cy=cy,
                scale=scale,
            )

            # Finger tip
            tip_end = (
                    finger_end
                    + 0.12 * finger_dir
            )

            self._draw_plate_link(
                p,
                finger_end,
                tip_end,
                width=0.055,
                thickness=0.035,
                cx=cx,
                cy=cy,
                scale=scale,
            )

            # Pivot
            pivot_screen = _iso(
                pivot,
                cx,
                cy,
                scale
            )

            p.setBrush(QBrush(QColor("#4c4d51")))
            p.setPen(QPen(QColor("#090909"), 1.0))

            p.drawEllipse(
                pivot_screen,
                4,
                4
            )

    # ---- HUD ------------------------------------------------------------

    def _draw_hud(self, p):
        p.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        p.setPen(QPen(QColor("#202024")))
        p.drawText(QPointF(14, 26), "Mode: " + self._mode_text)
        p.setFont(QFont("Segoe UI", 11))
        p.setPen(QPen(QColor("#55555a")))
        p.drawText(QPointF(14, 48), "Cmd:  " + self._cmd_text)

    # ---- Gauges panel ---------------------------------------------------

    def _draw_gauges(self, p, x, y, w, h):
        p.setPen(QPen(QColor("#d0d0d5"), 1))
        p.drawLine(x, y + 8, x, y + h - 8)

        p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        p.setPen(QPen(QColor("#202024")))
        p.drawText(QPointF(x + 16, y + 28), "Servo angles")

        pad_x = 16
        gauge_x = x + pad_x
        gauge_w = w - 2 * pad_x
        block_h = 60
        start_y = y + 50

        for i, name in enumerate(JOINTS):
            by = start_y + i * block_h

            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.setPen(QPen(QColor("#333338")))
            p.drawText(QPointF(gauge_x, by + 12), LABELS[name])

            p.setFont(QFont("Consolas", 9))
            p.setPen(QPen(QColor("#5a5a60")))
            val_text = "%5.1f deg" % self._current[name]
            p.drawText(QPointF(gauge_x + gauge_w - 60, by + 12), val_text)

            track_y = by + 22
            track_h = 11
            p.setBrush(QBrush(QColor("#22222a")))
            p.setPen(QPen(QColor("#22222a")))
            p.drawRoundedRect(QRectF(gauge_x, track_y, gauge_w, track_h),
                              4, 4)

            lo, hi = LIMITS[name]
            frac = (self._current[name] - lo) / max(hi - lo, 1e-6)
            frac = max(0.0, min(1.0, frac))
            fill_w = max(2, int(gauge_w * frac))
            p.setBrush(QBrush(QColor(COLORS[name])))
            p.setPen(QPen(QColor(COLORS[name])))
            p.drawRoundedRect(QRectF(gauge_x, track_y, fill_w, track_h),
                              4, 4)

            t_frac = (self._target[name] - lo) / max(hi - lo, 1e-6)
            t_frac = max(0.0, min(1.0, t_frac))
            tick_x = gauge_x + int(gauge_w * t_frac)
            p.setPen(QPen(QColor("#e8e8ea"), 1.5))
            p.drawLine(tick_x, track_y - 2, tick_x, track_y + track_h + 2)
