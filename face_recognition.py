import os, sys
import cv2
import face_recognition
import numpy as np
import math
import json
import generate_data_set
from tkinter import *
from tkinter import ttk



def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    line_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(line_val * 100, 2)) + '%'
    else:
        value = (line_val + ((1.0 - line_val) * math.pow((line_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def show_init_window():
    window = Tk()
    window.title('Init Window')
    window.geometry("300x500")

    btn_dataset = ttk.Button(text='Пополнить Dataset', command=generate_data_set.generate_data_from_camera())
    btn_video_recognition = ttk.Button(text='Начать распознавание', command=start_init)

    btn_dataset.pack()
    btn_video_recognition.pack()

    window.mainloop()


def start_init():
    fr = FaceRecognition()
    fr.run_recognition()


class FaceRecognition:
    face_locations = []
    face_encodings = []
    know_face_encodings = []
    know_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        with open("faces_name_list.json", "r") as fh:
            faces_dataset = json.load(fh)

        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)
            if face_encoding:
                face_encoding = face_encoding[0]
            else:
                continue

            self.know_face_encodings.append(face_encoding)
            name = faces_dataset.get(image)
            self.know_face_names.append(name)

    def run_recognition(self):

        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Камера не найдена')

        while True:
            ret, frame = video_capture.read()
            # os.mkdir("faces/Shadin")
            # for i in range(10):
                # generate_data_set.generate_data_from_camera(frame=frame, i=i)

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Нахождение всех лиц в обьективе
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.know_face_encodings, face_encoding)
                    name = 'Неизвестный'
                    confidence = '0%'

                    face_distances = face_recognition.face_distance(self.know_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.know_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    show_init_window()
#     fr = FaceRecognition()
#     fr.run_recognition()
