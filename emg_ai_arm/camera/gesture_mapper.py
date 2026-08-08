"""
Maps camera gesture names to the same prediction IDs used by
the EMG Random Forest model.

EMG predictions:
0 = REST
1 = CH1
2 = CH2
3 = BOTH
"""

GESTURE_TO_PREDICTION = {
    "Closed_Fist": 1,
    "Open_Palm": 2,
    "Thumb_Up": 3,

    # Unknown gestures become REST
    "Unknown": 0,
    None: 0,
}


def gesture_to_prediction(name: str) -> int:
    return GESTURE_TO_PREDICTION.get(name, 0)