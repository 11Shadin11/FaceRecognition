import math
import sys
import threading
import time

import tkinter as tk
from tkinter import ttk
import pickle
import os

import cv2
import face_recognition

import dialog_window_module


def face_confidence(face_distance, face_match_threshold=0.6):
    range_p = (1.0 - face_match_threshold)
    line_val = (1.0 - face_distance) / (range_p * 2.0)

    if face_distance > face_match_threshold:
        return round(line_val * 100, 2)
    else:
        value = (line_val + ((1.0 - line_val) * math.pow((line_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)


def show_init_window():
    dialog_window_module.open_init_window()


def start_init():
    fr = FaceRecognition()
    # fr.run_recognition()


class FaceRecognition:
    face_locations = []
    face_encodings = []
    known_faces = []

    def __init__(self):
        self.known_faces = []
        self.encode_faces()

    def encode_faces(self):
        if os.path.isfile('faces/encoded_faces.pkl'):
            with open('faces/encoded_faces.pkl', 'rb') as f:
                self.known_faces = pickle.load(f)
                self.run_recognition()
                # self.test('faces/Karina/80.jpg')
                # test_dataset_path = 'faces/test'
                # accuracy = self.calculate_accuracy(test_dataset_path)
                # print(f'Точность распознавания: {accuracy}%')
        else:
            total_images = sum([len(files) for r, d, files in os.walk('faces/')])
            progress_window = tk.Toplevel()
            progress_window.title("Encoding Faces")
            progress_bar = ttk.Progressbar(progress_window, length=500, mode='determinate', maximum=total_images)
            progress_bar.pack(padx=10, pady=10)
            progress_label = tk.Label(progress_window, text="0%")
            progress_label.pack()

            threading.Thread(target=self.process_faces, args=(progress_bar, progress_label, progress_window)).start()

    def process_faces(self, progress_bar, progress_label, progress_window):
        image_count = 0
        for folder in os.listdir('faces'):
            for image in os.listdir(f'faces/{folder}'):
                face_image = face_recognition.load_image_file(f'faces/{folder}/{image}')
                face_encoding = face_recognition.face_encodings(face_image)
                if face_encoding:
                    face_encoding = face_encoding[0]

                self.known_faces.append({'name': folder, 'encoding': face_encoding})

                image_count += 1
                progress = image_count / progress_bar['maximum'] * 100
                progress_bar['value'] = image_count
                progress_label['text'] = f"{progress:.2f}%"
                progress_window.update()

        with open('faces/encoded_faces.pkl', 'wb') as f:
            pickle.dump(self.known_faces, f)

        progress_window.destroy()
        self.run_recognition()

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
                        match = f'{name} - {confidence}%'
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
                        (255, 255, 255),
                        1
                    )

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord(" "):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def test(self, img_path):
        print(img_path)
        frame = cv2.imread(img_path)
        frame_img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)[:, :, ::-1]
        if frame_img is None:
            sys.exit('Фотография не найдена')

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            result = face_recognition.compare_faces([face['encoding'] for face in self.known_faces], face_encoding)

            face_distances = face_recognition.face_distance([face['encoding'] for face in self.known_faces],
                                                            face_encoding)

            if True in result:
                idx = result.index(True)
                name = self.known_faces[idx]['name']
                confidence = face_confidence(face_distances[idx])
                match = f'{name} - {confidence}%'
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
                (255, 255, 255),
                1
            )
        frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Face Recognition from image', frame_display)

    def calculate_accuracy(self, test_dataset_path):
        correct_matches_prc = 0
        total_matches = 0

        for filename in os.listdir(test_dataset_path):
            test_image_path = os.path.join(test_dataset_path, filename)

            total_time = 0
            test_image = face_recognition.load_image_file(test_image_path)
            face_locations = face_recognition.face_locations(test_image)
            face_encodings = face_recognition.face_encodings(test_image, face_locations)
            for face_encoding, face_location in zip(face_encodings, face_locations):
                start_time = time.time()
                result = face_recognition.compare_faces([face['encoding'] for face in self.known_faces], face_encoding)

                face_distances = face_recognition.face_distance([face['encoding'] for face in self.known_faces],
                                                                face_encoding)

                if True in result:
                    idx = result.index(True)
                    confidence = face_confidence(face_distances[idx])
                    correct_matches_prc += int(confidence)
                    end_time = time.time()
                total_matches += 1
                total_time += end_time - start_time

        accuracy = correct_matches_prc / total_matches
        print(f'Среднее скорость обработки кадра: {total_time}')
        return accuracy


if __name__ == '__main__':
    show_init_window()
