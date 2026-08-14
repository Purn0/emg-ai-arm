from emg_ai_arm.control.robot_command import RobotCommand


GESTURE_TO_COMMAND = {
    # Vertical arm movement
    "Thumb_Up": RobotCommand.ARM_UP,
    "Thumb_Down": RobotCommand.ARM_DOWN,

    # Gripper
    "Open_Palm": RobotCommand.GRIP_OPEN,
    "Closed_Fist": RobotCommand.GRIP_CLOSE,

    # Wrist rotation
    "Rock": RobotCommand.WRIST_CW,
    "Call_Me": RobotCommand.WRIST_CCW,

    # Base rotation
    "Victory": RobotCommand.BASE_LEFT,
    "Pointing_Up": RobotCommand.BASE_RIGHT,
}


def gesture_to_command(name: str) -> RobotCommand:
    return GESTURE_TO_COMMAND.get(name, RobotCommand.STOP)