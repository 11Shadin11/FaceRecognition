import cv2
import os
import face_recognition
from PIL import Image, ImageDraw



def generate_data_from_camera(directory):
    video_path = f'faces/{directory}/dataset_{directory}.avi'
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # rgb_small_frame = small_frame[:, :, ::-1]

        # Нахождение всех лиц в обьективе
        # face_locations = face_recognition.face_locations(rgb_small_frame)
        # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        if ret:
            frame_count += 1
            if frame_count > 20:
                # for face_location in face_locations:
                #     top, right, bottom, left = face_location
                #     face_img = frame[top:bottom, left:right]
                #     pil_image = Image.fromarray(face_img)
                #     pil_image.save(f'faces/{directory}/frame_{frame_count}.jpg')
                frame_name = f'frame_{frame_count}.jpg'
                cv2.imwrite(f'faces/{directory}/{frame_name}', frame)
        else:
            break

    cap.release()
