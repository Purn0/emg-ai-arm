"""
Camera frame acquisition.

Responsibilities
----------------
- Open webcam
- Capture frames
- Return raw images

This module should NOT:
- classify gestures
- draw landmarks
- interact with the GUI
"""

class CameraDetector:
    """Placeholder for future MediaPipe integration."""

    def __init__(self):
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False