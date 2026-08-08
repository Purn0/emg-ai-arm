from __future__ import annotations

import cv2

from PyQt6.QtCore import QThread, pyqtSignal

from emg_ai_arm.camera.ml_gesture_recognizer import MLGestureRecognizer
from emg_ai_arm.camera.gesture_mapper import gesture_to_prediction


class CameraWorker(QThread):
    prediction_ready = pyqtSignal(int, float)
    frame_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)

        self.camera_index = camera_index
        self.running = False

    def stop(self):
        self.running = False

    def run(self):

        self.running = True

        recognizer = MLGestureRecognizer()

        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.error.emit("Could not open camera.")
            return

        try:

            while self.running:

                ok, frame = cap.read()

                if not ok:
                    continue

                raw, results = recognizer.process_with_raw(frame)

                recognizer.draw_landmarks(frame, raw)

                if results:

                    result = results[0]

                    pred = gesture_to_prediction(result.gesture_name)

                    strength = float(result.score)

                    self.prediction_ready.emit(pred, strength)

                self.frame_ready.emit(frame)

        except Exception as e:

            self.error.emit(str(e))

        finally:

            cap.release()