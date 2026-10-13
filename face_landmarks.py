"""Standalone landmark demo. Escape exits; the camera is always released."""
from face_app.capture import open_capture
from face_app.config import PROJECT_ROOT


def face_landmarks():
    import cv2
    import dlib

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(PROJECT_ROOT / "shape_predictor_68_face_landmarks.dat"))
    try:
        with open_capture(0) as capture:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Камера перестала выдавать кадры.")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for face in detector(gray):
                    landmarks = predictor(gray, face)
                    for index in range(68):
                        point = landmarks.part(index)
                        cv2.circle(frame, (point.x, point.y), 3, (255, 0, 0), -1)
                cv2.imshow("Face landmarks — Esc to stop", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_landmarks()
