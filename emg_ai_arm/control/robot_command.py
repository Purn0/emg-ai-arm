from enum import Enum


class RobotCommand(Enum):
    NONE = 0

    HAND_OPEN = 1
    HAND_CLOSE = 2

    THUMB_UP = 3
    VICTORY = 4
    POINT = 5

    UNKNOWN = 99