"""
Camera gesture recognizer: MediaPipe Hands landmarks -> Random Forest.

The model and landmark dataset come from
https://github.com/Purn0/gesture-control-project (eight gestures, 42
features: 21 (x, y) landmarks relative to the wrist, divided by the largest
absolute coordinate). `python -m experiments.train_models` trains it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from emg_ai_arm.utils.config import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "gesture_model.pkl"


@dataclass
class GestureResult:
    gesture_name: str
    score: float
    handedness: str


def normalize_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
    coords -= coords[0]
    scale = np.max(np.abs(coords))
    if scale > 0:
        coords /= scale
    return coords.flatten()


class MLGestureRecognizer:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, confidence_threshold: float = 0.6):
        """Predictions whose top class probability is below the threshold become "Unknown"."""
        self.model = joblib.load(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process_with_raw(self, frame):
        raw = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = []
        for idx, hand in enumerate(raw.multi_hand_landmarks or []):
            features = pd.DataFrame([normalize_landmarks(hand)], columns=self.model.feature_names_in_)
            # One forest pass: predict() is the argmax of predict_proba().
            probs = self.model.predict_proba(features)[0]
            best = int(np.argmax(probs))
            score = float(probs[best])
            name = str(self.model.classes_[best]) if score >= self.confidence_threshold else "Unknown"
            handedness = "Unknown"
            if raw.multi_handedness and idx < len(raw.multi_handedness):
                handedness = raw.multi_handedness[idx].classification[0].label
            results.append(GestureResult(name, score, handedness))
        return raw, results

    def draw_landmarks(self, frame, raw):
        for hand in raw.multi_hand_landmarks or []:
            self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
