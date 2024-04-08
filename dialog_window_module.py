from tkinter import *
from tkinter import ttk
import generate_data_set
import os
import cv2
import faces_recognition

window = Tk()


def open_init_window():
    window.title('😁😁')
    window.geometry("300x150")

    label = Label(text="Выберите действие")

    # btn_dataset = ttk.Button(text='Пополнить Dataset', command=generate_data_set.generate_data_from_camera())
    btn_dataset = ttk.Button(text='Пополнить Dataset', command=open_dataset_window)
    btn_video_recognition = ttk.Button(text='Начать распознавание', command=faces_recognition.start_init)

    label.pack(pady=30)
    btn_dataset.pack(expand=True, side=LEFT)
    btn_video_recognition.pack(expand=True, side=RIGHT)

    window.mainloop()


def open_dataset_window():
    dataset_window = Toplevel(window)
    dataset_window.title("Пополнение dataset")
    dataset_window.geometry("300x300")

    def_folder = StringVar()
    folders = []
    for item in os.listdir('faces'):
        if os.path.isdir(os.path.join('faces', item)):
            folders.append(item)
    if folders:
        def_folder.set(folders[0])

    label_dataset = Label(dataset_window, text="Выберите dataset, который нужно пополнить")
    label_dataset.pack(anchor=CENTER, padx=6, pady=6)
    combobox = ttk.Combobox(dataset_window, text=folders[0], values=folders)
    combobox.pack(anchor=CENTER, padx=6, pady=6)

    recording_btn = ttk.Button(dataset_window, text='Recording', command=lambda: start_recording(combobox.get()))
    recording_btn.pack()


def start_recording(directory):

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if not os.path.exists(f'faces/{directory}'):
        os.mkdir(f'faces/{directory}')

    out = cv2.VideoWriter(f'faces/{directory}/dataset_{directory}.avi', fourcc, fps, (width, height))

    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 5:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # generate_data_set.video_capture(directory)
    generate_data_set.DataSetGen(directory)
