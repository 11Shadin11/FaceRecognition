import cv2
import os
import face_recognition


class DataSetGen:

    def __init__(self, directory):
        self.direct = directory
        self.face_loc = []
        # self.face_enc = []
        self.video_capture()

    def video_capture(self):
        video_path = f'faces/{self.direct}/dataset_{self.direct}.avi'

        video_cap = cv2.VideoCapture(video_path)

        fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        while True:
            ret, frame = video_cap.read()

            frame_count = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ret:
                break

            self.face_loc = face_recognition.face_locations(frame)

            if self.face_loc and frame_count % fps == 0:
                cv2.imwrite(f'faces/{self.direct}/{frame_count}.jpg', frame)

        video_cap.release()
        cv2.destroyAllWindows()
        os.remove(video_path)
