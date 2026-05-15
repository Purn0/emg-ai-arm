"""
Tab 2 — cue-based training session recorder.

User picks how many trials per class and the duration of each phase.
The tab then runs through a sequence of trials. For each trial the user
sees:

    READY  →  CUE: <class>  →  RECORDING  →  REST

During the RECORDING phase, every window the stream worker delivers is
labelled with the current class and appended to the in-memory dataset.

When the session ends, the dataset is saved to
`data/windows/session_<timestamp>.npz`, and a "Train RF" button kicks
off a quick fit so the user can immediately see whether the data is
separable.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from emg_ai_arm.utils.config import WINDOWS_DIR

# Class IDs match the emulator and the controller.
CLASS_NAMES = {0: "REST", 1: "CH1 (close / left / up)",
               2: "CH2 (open / right / down)", 3: "BOTH (mode switch)"}


class TrainerTab(QWidget):
    """Run cue-based trials and append labelled windows to a dataset."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- Top: parameters ----
        params = QGridLayout()
        params.addWidget(QLabel("Trials per class:"), 0, 0)
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(1, 50)
        self.spin_trials.setValue(5)
        params.addWidget(self.spin_trials, 0, 1)

        params.addWidget(QLabel("Action duration (s):"), 0, 2)
        self.spin_action = QSpinBox()
        self.spin_action.setRange(1, 10)
        self.spin_action.setValue(3)
        params.addWidget(self.spin_action, 0, 3)

        params.addWidget(QLabel("Rest duration (s):"), 0, 4)
        self.spin_rest = QSpinBox()
        self.spin_rest.setRange(1, 10)
        self.spin_rest.setValue(2)
        params.addWidget(self.spin_rest, 0, 5)

        params.addWidget(QLabel("Class order:"), 1, 0)
        self.combo_order = QComboBox()
        self.combo_order.addItems(["randomised", "fixed 0,1,2,3"])
        params.addWidget(self.combo_order, 1, 1)

        # ---- Middle: cue display ----
        self.cue_label = QLabel("Press Start to begin")
        self.cue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cue_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.cue_label.setMinimumHeight(120)
        self.cue_label.setStyleSheet(
            "background: #f6f6f6; border: 2px solid #ddd; border-radius: 12px;"
        )

        self.phase_label = QLabel("")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setFont(QFont("Segoe UI", 14))

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        # ---- Bottom: controls + summary ----
        self.btn_start = QPushButton("Start session")
        self.btn_stop = QPushButton("Abort")
        self.btn_stop.setEnabled(False)
        self.btn_save = QPushButton("Save dataset")
        self.btn_save.setEnabled(False)
        self.btn_train = QPushButton("Train Random Forest on this session")
        self.btn_train.setEnabled(False)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(self.btn_save)
        ctrl_row.addWidget(self.btn_train)

        self.summary = QLabel("0 windows collected.")
        self.summary.setAlignment(Qt.AlignmentFlag.AlignRight)

        # ---- Layout ----
        v = QVBoxLayout(self)
        v.addLayout(params)
        v.addSpacing(8)
        v.addWidget(self.cue_label)
        v.addWidget(self.phase_label)
        v.addWidget(self.progress)
        v.addStretch(1)
        v.addLayout(ctrl_row)
        v.addWidget(self.summary)

        # ---- State ----
        self._trials: list[int] = []
        self._idx = 0
        self._phase: str = "idle"           # "ready", "action", "rest", "done"
        self._phase_t0 = 0.0
        self._action_sec = 3.0
        self._rest_sec = 2.0
        self._X: list[np.ndarray] = []
        self._y: list[int] = []

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)

        # Signals
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._abort)
        self.btn_save.clicked.connect(self._save_dataset)
        self.btn_train.clicked.connect(self._train_rf)

    # ------------------------------------------------------------------- #
    # Public slot — called from the shared stream worker
    # ------------------------------------------------------------------- #

    def on_window(self, window: np.ndarray, _true_label: int) -> None:
        if self._phase == "action":
            cls = self._trials[self._idx]
            self._X.append(window.copy())
            self._y.append(int(cls))
            self.summary.setText(f"{len(self._X)} windows collected.")

    # ------------------------------------------------------------------- #
    # Session control
    # ------------------------------------------------------------------- #

    def _start(self) -> None:
        n = int(self.spin_trials.value())
        self._action_sec = float(self.spin_action.value())
        self._rest_sec = float(self.spin_rest.value())

        order = []
        for c in (0, 1, 2, 3):
            order.extend([c] * n)
        if self.combo_order.currentText().startswith("rand"):
            rng = np.random.default_rng()
            rng.shuffle(order)
        self._trials = order
        self._idx = 0
        self._X.clear()
        self._y.clear()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_train.setEnabled(False)

        self._enter_phase("ready")
        self._tick.start(50)

    def _abort(self) -> None:
        self._tick.stop()
        self._phase = "idle"
        self.cue_label.setText("Aborted")
        self.cue_label.setStyleSheet(
            "background: #fff3f3; border: 2px solid #d62728; border-radius: 12px;"
        )
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(len(self._X) > 0)

    def _enter_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_t0 = time.perf_counter()

        if phase == "ready":
            cls = self._trials[self._idx]
            self.cue_label.setText(f"Get ready: {CLASS_NAMES[cls]}")
            self._set_cue_colour("#fff8d6", "#caa400")
            self.phase_label.setText(
                f"Trial {self._idx + 1}/{len(self._trials)} — countdown 1.0 s"
            )

        elif phase == "action":
            cls = self._trials[self._idx]
            self.cue_label.setText(f"GO!  →  {CLASS_NAMES[cls]}")
            self._set_cue_colour("#dfffe0", "#2ca02c")
            self.phase_label.setText(
                f"Hold contraction for {self._action_sec:.0f} s"
            )

        elif phase == "rest":
            self.cue_label.setText("Rest")
            self._set_cue_colour("#eef1ff", "#5566cc")
            self.phase_label.setText(f"Relax for {self._rest_sec:.0f} s")

        elif phase == "done":
            self._tick.stop()
            self.cue_label.setText("Session complete")
            self._set_cue_colour("#f0fff0", "#2ca02c")
            self.phase_label.setText(
                f"Collected {len(self._X)} windows across "
                f"{len(set(self._y))} classes."
            )
            self.progress.setValue(100)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_save.setEnabled(len(self._X) > 0)
            self.btn_train.setEnabled(len(self._X) > 0)

    def _set_cue_colour(self, bg: str, border: str) -> None:
        self.cue_label.setStyleSheet(
            f"background: {bg}; border: 2px solid {border}; "
            "border-radius: 12px;"
        )

    def _on_tick(self) -> None:
        if self._phase == "idle" or self._phase == "done":
            return

        now = time.perf_counter()
        elapsed = now - self._phase_t0

        if self._phase == "ready":
            self.progress.setValue(int(elapsed / 1.0 * 100))
            if elapsed >= 1.0:
                self._enter_phase("action")

        elif self._phase == "action":
            self.progress.setValue(int(elapsed / self._action_sec * 100))
            if elapsed >= self._action_sec:
                self._enter_phase("rest")

        elif self._phase == "rest":
            self.progress.setValue(int(elapsed / self._rest_sec * 100))
            if elapsed >= self._rest_sec:
                self._idx += 1
                if self._idx >= len(self._trials):
                    self._enter_phase("done")
                else:
                    self._enter_phase("ready")

    # ------------------------------------------------------------------- #
    # Saving + training
    # ------------------------------------------------------------------- #

    def _save_dataset(self) -> Path:
        if not self._X:
            return Path()
        X = np.stack(self._X, axis=0)
        y = np.asarray(self._y, dtype=np.int64)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = WINDOWS_DIR / f"session_{stamp}.npz"
        np.savez_compressed(str(out), X=X, y=y, fs=200, win_sec=0.2)
        self.summary.setText(f"Saved {len(y)} windows to {out.name}")
        return out

    def _train_rf(self) -> None:
        # Save first, then call train_ml with the new dataset.
        path = self._save_dataset()
        if not path:
            return
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report
            from sklearn.model_selection import train_test_split
            from emg_ai_arm.features.extract_features import extract_features

            data = np.load(str(path))
            Xw, y = data["X"], data["y"]
            X = np.vstack([extract_features(w) for w in Xw])
            if len(np.unique(y)) < 2:
                self.summary.setText("Need at least 2 distinct classes.")
                return
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            clf = RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced"
            )
            clf.fit(X_tr, y_tr)
            acc = clf.score(X_te, y_te)
            self.summary.setText(f"RF test accuracy: {acc * 100:.1f}%  "
                                 f"({len(y_te)} test windows)")
        except Exception as e:
            self.summary.setText(f"Training failed: {e}")
