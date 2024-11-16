from tkinter import *
import tkinter as tk
from tkinter import ttk, filedialog
import generate_data_set
import os
import cv2
import faces_recognition

window = Tk()


def open_init_window():
    window.title('😁😁')
    window.geometry("300x150")

    label = Label(text="Выберите действие")

    btn_dataset = ttk.Button(text='Пополнить Dataset', command=open_dataset_window)
    btn_video_recognition = ttk.Button(text='Начать распознавание', command=faces_recognition.start_init)

    label.pack(pady=30)
    btn_dataset.pack(expand=True, side=LEFT)
    btn_video_recognition.pack(expand=True, side=RIGHT)

    window.mainloop()


recording_btn = None
video_download_btn = None
rb1 = None
rb2 = None


def open_dataset_window():

    global recording_btn
    global video_download_btn
    global rb1
    global rb2

    dataset_window = Toplevel(window)
    dataset_window.title("Пополнение dataset")
    dataset_window.geometry("300x300")

    folders = []
    for item in os.listdir('faces'):
        if os.path.isdir(os.path.join('faces', item)):
            folders.append(item)

    label_dataset = Label(dataset_window, text="Выберите dataset, который нужно пополнить")
    label_dataset.pack(anchor=CENTER, padx=6, pady=6)

    combobox = ttk.Combobox(dataset_window, values=folders)
    combobox.pack(anchor=CENTER, padx=6, pady=6)

    selected = tk.StringVar(value='0')

    button_frame = tk.Frame(dataset_window)
    button_frame.pack()

    rb1 = ttk.Radiobutton(button_frame, text='Загрузить видео файл', variable=selected, value='0')
    rb2 = ttk.Radiobutton(button_frame, text='Записать видео', variable=selected, value='1')

    video_download_btn = ttk.Button(
        button_frame,
        text='Загрузить видео',
        command=lambda: open_file_dialog(combobox.get(), dataset_window)
    )

    recording_btn = ttk.Button(
        button_frame,
        text='Recording',
        state="disabled",
        command=lambda: start_recording(combobox.get(), dataset_window)
    )

    rb1.grid(row=0, column=0, sticky="w")
    video_download_btn.grid(row=0, column=1, sticky="e")

    rb2.grid(row=1, column=0, sticky="w")
    recording_btn.grid(row=1, column=1, sticky="e")

    rb1.grid_remove()
    video_download_btn.grid_remove()

    rb2.grid_remove()
    recording_btn.grid_remove()

    combobox.bind('<<ComboboxSelected>>', lambda event: update_interface(selected.get(), combobox.get()))
    combobox.bind('<KeyRelease>', lambda event: update_interface(selected.get(), combobox.get()))
    selected.trace('w', lambda *args: update_interface(selected.get(), combobox.get()))


def update_interface(selected_button, selected_dir):
    global recording_btn
    global video_download_btn
    global rb1
    global rb2

    if len(selected_dir) == 0:
        return

    rb1.grid()
    video_download_btn.grid()
    rb2.grid()
    recording_btn.grid()

    if selected_button == '0':
        video_download_btn.config(state="normal")
        recording_btn.config(state="disabled")
    else:
        recording_btn.config(state="normal")
        video_download_btn.config(state="disabled")


def open_file_dialog(sel_dts, window_name):
    filename = filedialog.askopenfilename()
    generate_data_set.DataSetGen(sel_dts, filename)
    window_name.destroy()
    print(f"Выбранный файл: {filename}")


def start_recording(directory, window_name):
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

    generate_data_set.DataSetGen(directory, None)
    window_name.destroy()
