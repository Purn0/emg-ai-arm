import cv2

from emg_ai_arm.camera.ml_gesture_recognizer import MLGestureRecognizer

recognizer = MLGestureRecognizer()

cap = cv2.VideoCapture(0)

while True:

    ok, frame = cap.read()

    if not ok:
        break

    raw, results = recognizer.process_with_raw(frame)

    recognizer.draw_landmarks(frame, raw)

    for result in results:

        cv2.putText(
            frame,
            f"{result.gesture_name} ({result.score:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()