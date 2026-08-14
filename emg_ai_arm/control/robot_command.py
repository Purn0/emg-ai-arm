from enum import Enum


class RobotCommand(Enum):
    STOP = 0

    ARM_UP = 1
    ARM_DOWN = 2

    GRIP_OPEN = 3
    GRIP_CLOSE = 4

    WRIST_CW = 5
    WRIST_CCW = 6

    BASE_LEFT = 7
    BASE_RIGHT = 8