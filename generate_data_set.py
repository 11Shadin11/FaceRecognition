import cv2
import os


def generate_data_from_camera():
    print('asdasdasdsadsadsadasdsada')

    name = input('write your name: ')

    video = cv2.VideoCapture(0)

    os.mkdir(f'faces/{name}')
    for i in range(10):
        ret, frame = video.read()
        cv2.imwrite('faces/Shadin/img_' + str(i) + '.png', frame)
