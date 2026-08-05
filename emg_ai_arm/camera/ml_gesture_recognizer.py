import os

import joblib
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Column names the model was trained with (train_model.py assigns f0..f41).
# We re-use them at predict time so sklearn doesn't emit a warning.
_FEATURE_COLUMNS = [f"f{i}" for i in range(42)]


@dataclass
class GestureResult:
    gesture_name: str
    score: float
    handedness: str
    landmarks: Optional[List[Tuple[int, int]]] = None


# Resolve the default model path relative to the project root so the
# recognizer works regardless of the caller's current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CAMERA_MODEL_DIR = os.path.join(_PROJECT_ROOT, "camera_models")
_DEFAULT_MODEL_PATH = os.path.join(
    _CAMERA_MODEL_DIR,
    "gesture_model.pkl"
)


class MLGestureRecognizer:
    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH,
                 confidence_threshold: float = 0.0):
        """
        model_path:           absolute path to the trained joblib pickle.
        confidence_threshold: predictions with max probability below this
                              are reported as "Unknown". Keeps low-confidence
                              junk from firing commands.
        """
        self.model = joblib.load(model_path)
        self.confidence_threshold = float(confidence_threshold)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def normalize_landmarks(self, hand_landmarks):
        coords = np.array(
            [[lm.x, lm.y] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )

        wrist = coords[0]
        coords = coords - wrist

        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = coords / max_val

        return coords.flatten().tolist()

    def process_with_raw(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_results = self.hands.process(rgb)
        parsed = []

        if raw_results.multi_hand_landmarks:
            image_h, image_w, _ = frame.shape

            for idx, hand_landmarks in enumerate(raw_results.multi_hand_landmarks):
                landmark_points = []
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * image_w)
                    y = int(lm.y * image_h)
                    landmark_points.append((x, y))

                features = self.normalize_landmarks(hand_landmarks)

                # Wrap features in a DataFrame with the same column names
                # used at training time. This silences sklearn's
                # "X does not have valid feature names" warning.
                features_df = pd.DataFrame([features], columns=_FEATURE_COLUMNS)

                predicted_label = self.model.predict(features_df)[0]
                score = 1.0
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(features_df)[0]
                    score = float(np.max(probs))

                # Confidence gate: below threshold -> Unknown
                if score < self.confidence_threshold:
                    gesture_name = "Unknown"
                else:
                    gesture_name = predicted_label

                handedness = "Unknown"
                if raw_results.multi_handedness and idx < len(raw_results.multi_handedness):
                    handedness = raw_results.multi_handedness[idx].classification[0].label

                parsed.append(
                    GestureResult(
                        gesture_name=gesture_name,
                        score=score,
                        handedness=handedness,
                        landmarks=landmark_points
                    )
                )

        return raw_results, parsed

    def draw_landmarks(self, frame, raw_results):
        if raw_results.multi_hand_landmarks:
            for hand_landmarks in raw_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
