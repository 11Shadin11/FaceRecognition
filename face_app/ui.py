"""Tk owns every widget; task threads only publish data for after() to consume."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from queue import Empty
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy  # Prime array decoding before the first frame arrives through IPC.

from .service import FaceService
from .storage import FaceStore
from .overlays import FaceOverlay, MAX_OVERLAY_AGE
from .theme import apply_theme
from .tasks import TaskRunner, TaskStatus
from .backup import create_backup
from .preferences import Preferences
from .profiles_ui import ProfilesWindow
from .preview import make_preview


class FaceApplication:
    def __init__(self, window, config):
        self.window = window
        self.config = config
        self.store = FaceStore(config.data_dir)
        self.preferences = Preferences(self.store.root)
        self.profiles_window = None
        self.service = FaceService(self.store, config)
        self.runner = TaskRunner()
        self.closing = False
        self.close_deadline = None
        self.rendered_frame = None
        self.photo = None
        self.name = tk.StringVar()
        self.camera_number = tk.StringVar(value=str(config.camera_index))
        self.status_text = tk.StringVar()
        self.people_count_text = tk.StringVar(value="0")
        self.sample_count_text = tk.StringVar(value="0")
        self.palette, self.font_family = apply_theme(window)
        self.activity = None
        self.last_outcome = None
        self.sample_count = 0
        self._button_states = None
        self._result_state = None
        self._camera_state = None
        self._canvas_size = None
        self._preview_message = None
        self._displayed_progress = None
        self._progress_visible = False
        self.window.title("Распознавание лиц")
        self.window.geometry(f"{min(1120, window.winfo_screenwidth() - 60)}x{min(760, window.winfo_screenheight() - 100)}")
        self.window.minsize(920, 660)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda event: self.stop())
        self._build_widgets()
        # Load Pillow's Tk bridge before the event loop and the first live frame.
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (1, 1)), master=self.window)
        self.photo = None
        self._refresh_people()
        self.window.after(40, self._poll)

    def _build_widgets(self):
        self.content = ttk.Frame(self.window, padding=22)
        self.content.pack(fill="both", expand=True)
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)
        self._build_header(self.content)
        body = ttk.Frame(self.content)
        body.grid(row=1, sticky="nsew", pady=(16, 12))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self._build_camera_panel(left)
        self._build_result_panel(left)
        sidebar = ttk.Frame(body, width=300)
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.columnconfigure(0, weight=1)
        self._build_profile_panel(sidebar)
        self._build_enrollment_panel(sidebar)
        self._build_legacy_panel(sidebar)
        self.progress = ttk.Progressbar(self.content, maximum=1, style="App.Horizontal.TProgressbar")
        self.progress.grid(row=2, sticky="ew")
        self.progress.grid_remove()
        self.status_label = ttk.Label(self.content, textvariable=self.status_text, style="Muted.TLabel", wraplength=860)
        self.status_label.grid(row=3, sticky="ew", pady=(10, 0))
        self.content.bind("<Configure>", self._resize_text)

    def _build_header(self, parent):
        header = ttk.Frame(parent)
        header.grid(row=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        icon = tk.Canvas(header, width=48, height=48, background=self.palette.background, highlightthickness=0)
        icon.grid(row=0, column=0, rowspan=2, padx=(0, 14))
        icon.create_oval(1, 1, 47, 47, fill="#29204e", outline="")
        for points in ((13, 20, 13, 13, 20, 13), (28, 13, 35, 13, 35, 20),
                       (35, 28, 35, 35, 28, 35), (20, 35, 13, 35, 13, 28)):
            icon.create_line(*points, fill=self.palette.cyan, width=2)
        icon.create_oval(20, 18, 28, 26, fill=self.palette.violet, outline="")
        ttk.Label(header, text="Распознавание лиц", style="Title.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="Локальный эксперимент. Всё начинается с одного лица.",
                  style="Muted.TLabel").grid(row=1, column=1, sticky="w", pady=(4, 0))
        self.camera_badge = tk.Label(header, text="●  Камера выключена", background=self.palette.raised,
                                     foreground=self.palette.muted, font=(self.font_family, 10, "bold"), padx=12, pady=8)
        self.camera_badge.grid(row=0, column=2, rowspan=2, padx=(12, 0))

    def _build_camera_panel(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.grid(row=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)
        header = ttk.Frame(card, style="Surface.TFrame")
        header.grid(row=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Камера", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Источник №", style="CardMuted.TLabel").grid(row=0, column=1, padx=(10, 6))
        self.camera_box = ttk.Spinbox(header, from_=0, to=99, width=3, textvariable=self.camera_number,
                                     font=(self.font_family, 10), command=self._remember_camera)
        self.camera_box.grid(row=0, column=2)
        self.camera_box.bind("<FocusOut>", lambda event: self._remember_camera())
        self.camera_box.bind("<Return>", lambda event: self._remember_camera())
        self.preview_canvas = tk.Canvas(card, height=260, width=380, background=self.palette.preview,
                                        highlightthickness=0)
        self.preview_canvas.grid(row=1, sticky="nsew")
        self.preview_canvas.bind("<Configure>", self._resize_result_text)
        self.preview_image = self.preview_canvas.create_image(0, 0, anchor="center")
        self.preview_message = self.preview_canvas.create_text(0, 0, text="Камера выключена", justify="center",
                                                               fill=self.palette.muted, font=(self.font_family, 12))
        self.face_overlay = FaceOverlay(self.preview_canvas)
        controls = ttk.Frame(card, style="Surface.TFrame")
        controls.grid(row=2, sticky="ew", pady=(12, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        self.recognize_button = ttk.Button(controls, text="▶  Распознавать", style="Primary.TButton", command=self.start_recognition)
        self.recognize_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="■  Остановить", style="Stop.TButton", command=self.stop)
        self.stop_button.grid(row=0, column=1, sticky="ew")
        ttk.Label(card, text="Esc — остановить текущую задачу", style="CardMuted.TLabel").grid(row=3, sticky="w", pady=(10, 0))

    def _build_result_panel(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.grid(row=1, sticky="ew", pady=(14, 0))
        card.columnconfigure(0, weight=1)
        self.result_eyebrow = ttk.Label(card, text="РЕЗУЛЬТАТ", style="Eyebrow.TLabel")
        self.result_eyebrow.grid(row=0, sticky="w")
        self.result_title = ttk.Label(card, text="Добавьте первого человека", style="Result.TLabel", wraplength=480)
        self.result_title.grid(row=1, sticky="ew", pady=(6, 4))
        self.result_hint = ttk.Label(card, text="Запишите образцы, затем запустите распознавание.",
                                     style="CardMuted.TLabel", wraplength=480)
        self.result_hint.grid(row=2, sticky="ew")

    def _build_profile_panel(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.grid(row=0, sticky="ew", pady=(0, 10))
        ttk.Label(card, text="ВАША БАЗА", style="Eyebrow.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        for column, (variable, label, color) in enumerate(((self.people_count_text, "профилей", self.palette.green),
                                                          (self.sample_count_text, "образцов", self.palette.violet))):
            card.columnconfigure(column, weight=1, uniform="stats")
            ttk.Label(card, textvariable=variable, style="Card.TLabel", foreground=color,
                      font=(self.font_family, 22, "bold")).grid(row=1, column=column, sticky="w", padx=(0, 22))
            ttk.Label(card, text=label, style="CardMuted.TLabel").grid(row=2, column=column, sticky="w")
        self.manage_button = ttk.Button(card, text="Профили и образцы", style="Compact.TButton", command=self.manage_profiles)
        self.manage_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_enrollment_panel(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.grid(row=1, sticky="ew", pady=(0, 10))
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Добавить образцы", style="CardTitle.TLabel").grid(row=0, sticky="w")
        ttk.Label(card, text="Имя или существующий профиль", style="CardMuted.TLabel").grid(row=1, sticky="w", pady=(8, 4))
        self.people_box = ttk.Combobox(card, textvariable=self.name, width=25, font=(self.font_family, 11))
        self.people_box.grid(row=2, sticky="ew")
        self.camera_button = ttk.Button(card, text="●  Записать с камеры", style="Compact.Cyan.TButton", command=self.start_enrollment)
        self.camera_button.grid(row=3, sticky="ew", pady=(8, 6))
        imports = ttk.Frame(card, style="Surface.TFrame")
        imports.grid(row=4, sticky="ew")
        imports.columnconfigure(0, weight=1)
        imports.columnconfigure(1, weight=1)
        self.photo_button = ttk.Button(imports, text="Фото", style="Compact.TButton", command=self.import_photos)
        self.photo_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.video_button = ttk.Button(imports, text="Видео", style="Compact.TButton", command=self.import_video)
        self.video_button.grid(row=0, column=1, sticky="ew")
        ttk.Label(card, text="8 секунд · одно лицо в кадре\nХорошее освещение и чёткая картинка",
                  style="CardMuted.TLabel", font=(self.font_family, 9), wraplength=265, justify="left").grid(row=5, sticky="w", pady=(8, 0))

    def _build_legacy_panel(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.grid(row=2, sticky="ew")
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Перенос и резервная копия", style="Card.TLabel", foreground=self.palette.amber,
                  font=(self.font_family, 11, "bold")).grid(row=0, sticky="w")
        actions = ttk.Frame(card, style="Surface.TFrame")
        actions.grid(row=1, sticky="ew", pady=(8, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.legacy_button = ttk.Button(actions, text="Старая база", style="Compact.TButton", command=self.import_legacy)
        self.legacy_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.backup_button = ttk.Button(actions, text="Копия ZIP", style="Compact.TButton", command=self.backup)
        self.backup_button.grid(row=0, column=1, sticky="ew")

    def _resize_text(self, event):
        self.status_label.configure(wraplength=max(200, event.width - 44))

    def _resize_result_text(self, event):
        width = max(200, event.width - 4)
        self.result_title.configure(wraplength=width)
        self.result_hint.configure(wraplength=width)

    def _refresh_people(self):
        people = self.store.people()
        self.people_box["values"] = [person.name for person in people]
        self.sample_count = sum(person.sample_count for person in people)
        self.people_count_text.set(str(len(people)))
        self.sample_count_text.set(str(self.sample_count))

    def _start(self, target, uses_camera=False, activity="import"):
        if self.closing or self.profiles_window is not None or not self.runner.can_start:
            return
        if uses_camera:
            try:
                index = int(self.camera_number.get())
                self.preferences.save_camera(index)
                self.config = replace(self.config, camera_index=index)
            except (ValueError, OSError) as error:
                messagebox.showinfo("Выбор камеры", str(error), parent=self.window)
                return
            self.service.config = self.config
        self.activity = activity
        self.last_outcome = None
        self.runner.set_status(TaskStatus("Подготовка…"))
        self.progress["value"] = 0
        self.runner.start(target)
        self._update_buttons()

    def start_recognition(self):
        self._start(self.service.recognize_camera, uses_camera=True, activity="recognition")

    def _selected_name(self):
        name = self.name.get().strip()
        if not name:
            messagebox.showinfo("Добавление человека", "Введите имя или выберите существующий профиль.", parent=self.window)
            return None
        return name

    def start_enrollment(self):
        name = self._selected_name()
        if name:
            self._start(lambda context: self.service.collect_samples(context, name, self.config.camera_index),
                        uses_camera=True, activity="enrollment")

    def import_video(self):
        name = self._selected_name()
        if not name:
            return
        path = filedialog.askopenfilename(parent=self.window, title="Выберите видео с одним человеком",
                                         filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*")])
        if path:
            self._start(lambda context: self.service.collect_samples(context, name, path))

    def import_legacy(self):
        self._start(self.service.import_legacy)

    def import_photos(self):
        name = self._selected_name()
        if not name:
            return
        paths = filedialog.askopenfilenames(parent=self.window, title="Фотографии одного человека",
                                           filetypes=[("Фотографии", "*.jpg *.jpeg *.png")])
        if paths:
            self._start(lambda context: self.service.import_photos(context, name, paths))

    def manage_profiles(self):
        if self.runner.can_start and not self.closing and self.profiles_window is None:
            self.profiles_window = ProfilesWindow(self)
            self._update_buttons()

    def backup(self):
        path = filedialog.asksaveasfilename(parent=self.window, title="Новая резервная копия",
                                           initialfile=f"face-backup-{datetime.now():%Y%m%d-%H%M%S}.zip",
                                           defaultextension=".zip", filetypes=[("ZIP", "*.zip")])
        if path:
            self._start(lambda context: create_backup(self.store, path, context), activity="backup")

    def _remember_camera(self):
        if self.runner.can_start and not self.closing:
            try:
                self.preferences.save_camera(int(self.camera_number.get()))
            except (OSError, ValueError):
                self.status_text.set("Не удалось сохранить выбор камеры. Проверьте номер (0–99) и каталог данных.")

    def stop(self):
        if self.runner.busy:
            self.face_overlay.clear()
            self.runner.set_status(TaskStatus("Остановка задачи…"))
            self.runner.cancel()
            self.stop_button.configure(state="disabled")

    def _update_buttons(self):
        state = "disabled" if not self.runner.can_start or self.closing or self.profiles_window is not None else "normal"
        stop_state = "normal" if self.runner.busy and not self.runner.cancel_event.is_set() else "disabled"
        if (state, stop_state) == self._button_states:
            return
        for button in (self.recognize_button, self.camera_button, self.video_button, self.photo_button,
                       self.legacy_button, self.manage_button, self.backup_button):
            button.configure(state=state)
        self.people_box.configure(state=state)
        self.camera_box.configure(state=state)
        self.stop_button.configure(state=stop_state)
        self._button_states = (state, stop_state)

    def _update_result(self, status, now):
        color = self.palette.violet
        heading = "РЕЗУЛЬТАТ"
        if self.runner.busy and self.runner.cancel_event.is_set():
            title, hint = "Завершаем задачу…", "Останавливаем обработку и освобождаем ресурсы."
        elif self.runner.busy and self.activity == "recognition":
            result = status.recognition
            if result is None:
                title, hint = "Подготовка распознавания", "Загружаем модель и подключаем камеру."
            elif now - result.captured_at > MAX_OVERLAY_AGE:
                title, hint = "Ожидание нового результата", "Предыдущий результат устарел."
            elif not result.faces:
                title, hint = "Лицо не найдено", "Смотрите в камеру при хорошем освещении."
            elif len(result.faces) > 1:
                title, hint = f"В кадре: {len(result.faces)} лиц", "Имена и результаты показаны рядом с каждым лицом."
                color = self.palette.cyan
            else:
                face = result.faces[0]
                if face.status == "known":
                    heading, title, hint, color = "СОВПАДЕНИЕ НАЙДЕНО", face.name, "Лицо совпало с сохранённым профилем.", self.palette.green
                elif face.status == "ambiguous":
                    title, hint, color = "Неоднозначное совпадение", "В базе есть несколько близких совпадений.", self.palette.amber
                else:
                    title, hint, color = "Неизвестный человек", "Лицо найдено, совпадения в базе нет.", self.palette.red
        elif self.runner.busy:
            if self.activity == "backup":
                title, hint = "Создание резервной копии", "Копируем базу и её фотографии в один архив."
            else:
                title = "Запись образцов" if self.activity == "enrollment" else "Импорт образцов"
                hint = "Смотрите в камеру. Запись длится 8 секунд." if self.activity == "enrollment" else "Проверяем кадры и добавляем подходящие образцы."
            color = self.palette.cyan
        elif self.last_outcome == "error":
            title, hint, color = "Задача не завершена", "Подробности ниже. Проверьте источник и попробуйте снова.", self.palette.red
        elif self.sample_count:
            title, hint = "Готово к распознаванию", "Нажмите «Распознавать», чтобы проверить совпадение."
        else:
            title, hint = "Добавьте первого человека", "Введите имя справа и запишите образцы с камеры."
        key = (heading, title, hint, color)
        if key != self._result_state:
            self.result_eyebrow.configure(text=heading, foreground=color)
            self.result_title.configure(text=title, foreground=color)
            self.result_hint.configure(text=hint)
            self._result_state = key

    def _update_camera_badge(self, frame):
        stream = self.runner.stream
        if self.runner.busy and self.runner.cancel_event.is_set():
            key = ("●  Остановка", self.palette.amber, "#392f24")
        elif stream is not None:
            if isinstance(stream.source, int):
                fresh = frame is not None and time.monotonic() - frame.captured_at <= 1
                key = ("●  Прямой эфир", self.palette.green, "#17372f") if fresh else ("●  Ожидание кадра", self.palette.cyan, "#183240")
            else:
                key = ("●  Импорт видео", self.palette.amber, "#392f24")
        elif self.runner.busy:
            key = ("●  Подготовка", self.palette.violet, "#29204e")
        else:
            key = ("●  Камера выключена", self.palette.muted, self.palette.raised)
        if key != self._camera_state:
            self.camera_badge.configure(text=key[0], foreground=key[1], background=key[2])
            self._camera_state = key

    def _render_preview(self, frame, recognition=None):
        width = max(1, self.preview_canvas.winfo_width())
        height = max(1, self.preview_canvas.winfo_height())
        if (width, height) != self._canvas_size:
            for item in (self.preview_image, self.preview_message):
                self.preview_canvas.coords(item, width / 2, height / 2)
            self.preview_canvas.itemconfigure(self.preview_message, width=max(100, width - 40))
            self._canvas_size = (width, height)
        if frame is None or (isinstance(getattr(self.runner.stream, "source", None), int)
                             and time.monotonic() - frame.captured_at > 1):
            if self.runner.busy:
                message = ("Подключаем источник…" if self.activity in {"recognition", "enrollment"}
                           else "Обработка данных…\n\nКамера выключена")
            else:
                message = "Камера выключена\n\nЗапишите образцы или запустите распознавание"
            if message != self._preview_message or self.photo is not None:
                self.preview_canvas.itemconfigure(self.preview_image, image="")
                self.preview_canvas.itemconfigure(self.preview_message, state="normal", text=message)
                self._preview_message = message
            self.photo = None
            self.rendered_frame = None
            self.face_overlay.clear()
            return
        key = (frame.sequence, frame.captured_at, width, height)
        if key != self.rendered_frame:
            image = make_preview(frame.image, (width, height))
            self.photo = ImageTk.PhotoImage(image, master=self.window)
            self.preview_canvas.itemconfigure(self.preview_image, image=self.photo)
            self.preview_canvas.itemconfigure(self.preview_message, state="hidden")
            self._preview_message = None
            self.rendered_frame = key
        image_width, image_height = self.photo.width(), self.photo.height()
        image_rect = ((width - image_width) / 2, (height - image_height) / 2, image_width, image_height)
        self.face_overlay.render(recognition, frame, image_rect, time.monotonic())

    def _poll(self):
        while True:
            try:
                kind, message = self.runner.events.get_nowait()
            except Empty:
                break
            self.runner.set_status(TaskStatus(message, 1 if kind == "done" else 0))
            self.last_outcome = kind
            self._refresh_people()
            if kind == "error" and not self.closing:
                messagebox.showerror("Не удалось выполнить задачу", message, parent=self.window)
        status = self.runner.status()
        message = status.message
        if status.recognized_at is not None and time.monotonic() - status.recognized_at > 1:
            message = "Результат предыдущего кадра устарел. Ожидание новой проверки…"
        if message != self.status_text.get():
            self.status_text.set(message)
        progress = max(0, min(1, status.progress or 0))
        progress_visible = self.runner.busy and status.progress is not None
        if progress_visible != self._progress_visible:
            self.progress.grid() if progress_visible else self.progress.grid_remove()
            self._progress_visible = progress_visible
        if progress != self._displayed_progress:
            self.progress["value"] = progress
            self._displayed_progress = progress
        self._update_buttons()
        self._update_result(status, time.monotonic())
        recognition = None if self.runner.cancel_event.is_set() else status.recognition
        frame = self.runner.preview() if self.runner.busy else None
        self._update_camera_badge(frame)
        self._render_preview(frame, recognition)
        if self.closing:
            if (self.runner.camera_closed() and self.service.model_closed
                    and self.profiles_window is None
                    and (not self.runner.busy or time.monotonic() >= self.close_deadline)):
                self.window.destroy()
                return
        self.window.after(40, self._poll)

    def close(self):
        if not self.closing:
            self.closing = True
            self.close_deadline = time.monotonic() + 4
            self.stop()
            if self.profiles_window is not None:
                self.profiles_window.close()
            threading.Thread(target=self.service.close, daemon=True).start()
            self._update_buttons()
