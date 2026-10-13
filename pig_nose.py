"""Optional standalone filter demo; importing it never activates the camera."""
from math import hypot

from face_app.capture import open_capture
from face_app.config import PROJECT_ROOT


def overlay_nose(frame, nose, center, width, cv):
    """Clip both the destination and overlay when the face touches a frame edge."""
    import numpy as np

    height = int(width * 0.77)
    if width <= 0 or height <= 0:
        return
    resized = cv.resize(nose, (width, height))
    x, y = int(center[0] - width / 2), int(center[1] - height / 2)
    left, top = max(0, x), max(0, y)
    right, bottom = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
    if right <= left or bottom <= top:
        return
    cropped = resized[top - y:bottom - y, left - x:right - x]
    area = frame[top:bottom, left:right]
    if cropped.shape[2] == 4:
        alpha = cropped[:, :, 3:4].astype(np.float32) / 255
        area[:] = (alpha * cropped[:, :, :3] + (1 - alpha) * area).astype(np.uint8)
    else:
        mask = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY) > 25
        area[mask] = cropped[mask]


def main():
    import cv2
    import dlib

    nose = cv2.imread(str(PROJECT_ROOT / "faces/pig/pig-nose.png"), cv2.IMREAD_UNCHANGED)
    if nose is None:
        raise FileNotFoundError("Для демо нужен файл faces/pig/pig-nose.png.")
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
                    left, right = landmarks.part(31), landmarks.part(35)
                    center = landmarks.part(30)
                    width = int(hypot(left.x - right.x, left.y - right.y) * 1.7)
                    overlay_nose(frame, nose, (center.x, center.y), width, cv2)
                cv2.imshow("Pig nose — Esc to stop", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
