import cv2
import os
import face_recognition


class DataSetGen:

    def __init__(self, directory):
        self.direct = directory
        self.face_locMy = []
        self.face_encMy = []
        self.face_loc = []
        self.face_enc = []
        self.video_capture()

    def video_capture(self):
        # video_path = f'faces/{self.direct}/dataset_{self.direct}.avi'
        video_path = f'faces/dataset_{self.direct}.avi'

        # img = face_recognition.load_image_file(f'faces/{self.direct}/{self.direct}.jpg')
        # self.face_locMy = face_recognition.face_locations(img)
        #
        # if self.face_locMy:
        #     self.face_encMy = face_recognition.face_encodings(img, self.face_locMy)[0]
        # else:
        #     print("Лицо не найдено на изображении")

        video_cap = cv2.VideoCapture(video_path)
        count = 0
        asd = 0
        ermo_count: int = 0

        while True:
            ret, frame = video_cap.read()
            fps = video_cap.get(cv2.CAP_PROP_FPS)

            if not ret:
                break

            print(asd)
            asd += 1
            self.face_loc = face_recognition.face_locations(frame)
            if self.face_loc:
                self.face_enc = face_recognition.face_encodings(frame, self.face_loc)[0]
            else:
                continue

            # result = face_recognition.compare_faces(self.face_encMy, self.face_loc)
            print(f'video_cap.get(1) - {video_cap.get(1)}')
            frame_id = int(round(video_cap.get(1)))
            # cv2.imshow('frame', frame)
            # cv2.waitKey(20)

            # if frame_id % multiplier == 0 and True in result:
            print(frame_id)
            print(int(fps))
            if frame_id % int(fps) == 0:
                print("take screen")
                cv2.imwrite(f'faces/{self.direct}/{count}.jpg', frame)
                count += 1
            # if frame_id > ermo_count:
            #     print(f'Tyta')
            #     ermo_count = ermo_count + 30
            #     print(f'ermo - {ermo_count}')
            #     cv2.imwrite(f'faces/{self.direct}/{count}.jpg', frame)
            #     count += 1

        video_cap.release()
        cv2.destroyAllWindows()
        os.remove(video_path)
