from emg_ai_arm.control.robot_command import RobotCommand


GESTURE_TO_COMMAND = {
    "Open_Palm": RobotCommand.HAND_OPEN,
    "Closed_Fist": RobotCommand.HAND_CLOSE,
    "Victory": RobotCommand.VICTORY,
    "Thumb_Up": RobotCommand.THUMB_UP,
    "Pointing_Up": RobotCommand.POINT,
}