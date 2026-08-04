# Camera Module

This module provides computer vision input for the AI Robotic Arm Framework.

Responsibilities:
- Capture frames from webcam.
- Detect hand landmarks using MediaPipe.
- Normalize landmarks.
- Classify gestures.
- Produce the same logical commands used by the EMG pipeline.

The module should not know anything about:
- GUI
- Simulation
- EMG
- Serial communication