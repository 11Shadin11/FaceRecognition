import itertools
import json
import math
import os
import sys

import cv2
import face_recognition

import dialog_window_module


def face_confidence(face_distance, face_match_threshold=0.6):
    range_p = (1.0 - face_match_threshold)
    line_val = (1.0 - face_distance) / (range_p * 2.0)

    if face_distance > face_match_threshold:
        return str(round(line_val * 100, 2)) + '%'
    else:
        value = (line_val + ((1.0 - line_val) * math.pow((line_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def show_init_window():
    dialog_window_module.open_init_window()


def start_init():
    fr = FaceRecognition()
    fr.run_recognition()


class FaceRecognition:
    face_locations = []
    face_encodings = []
    known_faces = []

    def __init__(self):
        self.known_faces = []
        self.encode_faces()

    def encode_faces(self):
        for folder in os.listdir('faces'):
            for image in os.listdir(f'faces/{folder}'):
                face_image = face_recognition.load_image_file(f'faces/{folder}/{image}')
                face_encoding = face_recognition.face_encodings(face_image)
                if face_encoding:
                    face_encoding = face_encoding[0]

                self.known_faces.append({'name': folder, 'encoding': face_encoding})
    def run_recognition(self):

        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Камера не найдена')

        while True:
            ret, frame = video_capture.read()

            if ret:
                self.face_locations = face_recognition.face_locations(frame)
                self.face_encodings = face_recognition.face_encodings(frame, self.face_locations)

                for face_encoding, face_location in zip(self.face_encodings, self.face_locations):
                    result = face_recognition.compare_faces([face['encoding'] for face in self.known_faces], face_encoding)

                    face_distances = face_recognition.face_distance([face['encoding'] for face in self.known_faces], face_encoding)

                    if True in result:
                        idx = result.index(True)
                        name = self.known_faces[idx]['name']
                        confidence = face_confidence(face_distances[idx])
                        match = f'{name} - {confidence}'
                    else:
                        match = "Undefined"

                    left_top = (face_location[3], face_location[0])
                    right_bottom = (face_location[1], face_location[2])

                    color = [0, 255, 0]

                    cv2.rectangle(frame, left_top, right_bottom, color, 4)
                    cv2.putText(
                        frame,
                        match,
                        (face_location[3] + 6, face_location[2] - 6),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (0, 0, 0),
                        1
                    )

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord(" "):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    show_init_window()
