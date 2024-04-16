import cv2
import os
import face_recognition
import tkinter as tk
from tkinter import ttk
import threading


class DataSetGen:

    def __init__(self, dts_path, download_file_path):
        self.download_file_path = download_file_path
        self.file_directory = download_file_path if download_file_path else f'faces/{dts_path}/dataset_{dts_path}.avi'
        self.sel_dts = dts_path
        self.face_loc = []
        self.video_capture()

    def video_capture(self):

        video_cap = cv2.VideoCapture(self.file_directory)

        fps = int(video_cap.get(cv2.CAP_PROP_FPS)) if not self.download_file_path else 20

        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_window = tk.Toplevel()
        progress_window.title("Обработка видео")

        progress_bar = ttk.Progressbar(progress_window, length=500, mode='determinate', maximum=total_frames)
        progress_bar.pack(padx=10, pady=10)

        progress_label = tk.Label(progress_window, text="0%")
        progress_label.pack()

        threading.Thread(
            target=self.process_video,
            args=(video_cap, fps, total_frames, progress_bar, progress_label, progress_window)
        ).start()

    def process_video(self, video_cap, fps, total_frames, progress_bar, progress_label, progress_window):
        for frame_cnt in range(total_frames):
            ret, frame = video_cap.read()

            frame_count = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))

            if not ret:
                break

            if frame_count % fps != 0:
                continue

            self.face_loc = face_recognition.face_locations(frame)

            if self.face_loc:
                cv2.imwrite(f'faces/{self.sel_dts}/{frame_count}.jpg', frame)

            progress_bar['value'] = frame_cnt
            progress = frame_cnt / total_frames * 100
            progress_label['text'] = f"{progress:.2f}%"
            progress_window.update()

        video_cap.release()
        cv2.destroyAllWindows()
        progress_window.destroy()

        if os.path.isfile('faces/encoded_faces.pkl'):
            os.remove('faces/encoded_faces.pkl')
        if not self.download_file_path:
            os.remove(self.file_directory)
