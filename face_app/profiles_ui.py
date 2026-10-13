"""Profile management: one background operation, Tk rendering on the main thread."""
from pathlib import Path
from queue import Empty, Queue
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from PIL import Image, ImageOps, ImageTk

from .storage import MAX_IMAGE_BYTES
from .tasks import TaskRunner


def load_preview(path, size=(460, 360)):
    """Decode just the selected photo, with the same file/pixel limits as import."""
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Изображение недоступно или превышает 20 МиБ.")
    with path.open("rb") as source, Image.open(source) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("Изображение превышает 16 миллионов пикселей.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.contain(image, size, method=Image.Resampling.BILINEAR)


class ProfilesWindow:
    def __init__(self, application):
        self.application = application
        self.store = application.store
        self.runner = TaskRunner()
        self.results = Queue(maxsize=1)
        self.callback = None
        self.closing = False
        self.people = []
        self.samples = []
        self.person_id = None
        self.sample_id = None
        self.photo = None
        self.window = tk.Toplevel(application.window)
        self.window.title("Профили и образцы")
        self.window.geometry("860x620")
        self.window.minsize(740, 540)
        self.window.configure(background=application.palette.background)
        self.window.transient(application.window)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda event: self.close())
        self._build()
        self.window.grab_set()
        self.reload()
        self.window.after(40, self._poll)

    def _listbox(self, parent):
        colors = self.application.palette
        box = tk.Listbox(parent, background=colors.panel, foreground=colors.text,
                         selectbackground=colors.primary, selectforeground=colors.text,
                         font=(self.application.font_family, 11), exportselection=False,
                         highlightthickness=0, borderwidth=0, width=22)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=box.yview)
        box.configure(yscrollcommand=scroll.set)
        box.grid(row=1, column=0, sticky="nsew", pady=8)
        scroll.grid(row=1, column=1, sticky="ns", pady=8)
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)
        return box

    def _build(self):
        content = ttk.Frame(self.window, padding=20)
        content.pack(fill="both", expand=True)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(1, weight=1)
        ttk.Label(content, text="Профили и образцы", style="Title.TLabel").grid(row=0, columnspan=2, sticky="w", pady=(0, 16))
        lists = ttk.Frame(content)
        lists.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        lists.rowconfigure(0, weight=1)
        lists.rowconfigure(1, weight=1)
        for row, title in enumerate(("Профили", "Сохранённые фото")):
            panel = ttk.Frame(lists, style="Card.TFrame", padding=12)
            panel.grid(row=row, sticky="nsew", pady=(0, 10))
            ttk.Label(panel, text=title, style="CardTitle.TLabel").grid(row=0, sticky="w")
            box = self._listbox(panel)
            if row == 0:
                self.people_list = box
                box.bind("<<ListboxSelect>>", self._select_person)
            else:
                self.samples_list = box
                box.bind("<<ListboxSelect>>", self._select_sample)
        preview = ttk.Frame(content, style="Card.TFrame", padding=14)
        preview.grid(row=1, column=1, sticky="nsew")
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.image_label = ttk.Label(preview, text="Выберите профиль и фото", style="CardMuted.TLabel", anchor="center")
        self.image_label.grid(row=0, sticky="nsew")
        self.detail = ttk.Label(preview, text="", style="CardMuted.TLabel", wraplength=420)
        self.detail.grid(row=1, sticky="w", pady=12)
        actions = ttk.Frame(preview, style="Surface.TFrame")
        actions.grid(row=2, sticky="ew")
        self.rename_button = ttk.Button(actions, text="Переименовать профиль", command=self.rename)
        self.delete_sample_button = ttk.Button(actions, text="Удалить выбранное фото", style="Stop.TButton", command=self.delete_sample)
        self.delete_person_button = ttk.Button(actions, text="Удалить профиль целиком", style="Stop.TButton", command=self.delete_person)
        for row, button in enumerate((self.rename_button, self.delete_sample_button, self.delete_person_button)):
            button.grid(row=row, sticky="ew", pady=(0, 8))
        actions.columnconfigure(0, weight=1)
        self.status = tk.StringVar(value="Загрузка базы…")
        ttk.Label(content, textvariable=self.status, style="Muted.TLabel", wraplength=780).grid(row=2, columnspan=2, sticky="ew", pady=(12, 0))

    def _run(self, operation, callback):
        if self.closing or not self.runner.can_start:
            return
        self.callback = callback

        def task(context):
            context.check_cancelled()
            result = operation()
            self.results.put_nowait(result)
            return "Готово."

        self.runner.start(task)
        self.status.set("Обработка…")
        self._buttons()

    def reload(self):
        self._run(lambda: (self.store.people(), self.store.samples(self.person_id) if self.person_id else []), self._loaded)

    def _loaded(self, data):
        self.people, self.samples = data
        # Tk Listbox ignores inserts/deletes while disabled. Re-enable it for
        # the main-thread update; _buttons restores the correct interaction state.
        self.people_list.configure(state="normal")
        self.samples_list.configure(state="normal")
        self.people_list.delete(0, "end")
        self.samples_list.delete(0, "end")
        person_index = None
        for index, person in enumerate(self.people):
            self.people_list.insert("end", f"{person.name} · {person.sample_count}")
            if person.id == self.person_id:
                person_index = index
        if person_index is None:
            self.person_id = None
            self.samples = []
        else:
            self.people_list.selection_set(person_index)
            self.people_list.see(person_index)
        for index, sample in enumerate(self.samples):
            self.samples_list.insert("end", f"Фото {index + 1} · {sample.id[:8]}")
        self.sample_id = None
        self.photo = None
        self.image_label.configure(image="", text="Выберите фото" if self.person_id else "Выберите профиль")
        self.detail.configure(text=f"Образцов в профиле: {len(self.samples)}" if self.person_id else "")
        self.status.set("Готово. Удаление требует подтверждения и не отменяется.")

    def _select_person(self, event=None):
        selected = self.people_list.curselection()
        if not self.runner.can_start or not selected:
            return
        person_id = self.people[selected[0]].id
        if person_id != self.person_id:
            self.person_id = person_id
            self.reload()

    def _select_sample(self, event=None):
        selected = self.samples_list.curselection()
        if not self.runner.can_start or not selected:
            return
        sample = self.samples[selected[0]]
        self.sample_id = sample.id
        size = (max(1, min(460, self.image_label.winfo_width())),
                max(1, min(360, self.image_label.winfo_height())))
        self._run(lambda: load_preview(sample.path, size), self._show_photo)

    def _show_photo(self, image):
        self.photo = ImageTk.PhotoImage(image, master=self.window)
        self.image_label.configure(image=self.photo, text="")
        self.detail.configure(text="Проверьте, что на фото нужный человек и лицо достаточно чёткое.")
        self.status.set("Фото загружено.")

    def rename(self):
        person = next((person for person in self.people if person.id == self.person_id), None)
        if person is None or not self.runner.can_start:
            return
        name = simpledialog.askstring("Переименовать профиль", "Новое имя:", initialvalue=person.name, parent=self.window)
        if name is not None:
            self._run(lambda: self.store.rename_person(person.id, name), lambda result: self.reload())

    def delete_sample(self):
        if self.sample_id and self.runner.can_start and messagebox.askyesno("Удалить фото?", "Удалить выбранный образец из базы?\nИсходные импортированные файлы останутся на диске.", parent=self.window):
            sample_id = self.sample_id
            self._run(lambda: self.store.delete_sample(sample_id), lambda result: self.reload())

    def delete_person(self):
        person = next((person for person in self.people if person.id == self.person_id), None)
        if person and self.runner.can_start and messagebox.askyesno("Удалить профиль?", f"Удалить профиль «{person.name}» и все его образцы ({person.sample_count})?\nОтменить удаление нельзя.", parent=self.window):
            self._run(lambda: self.store.delete_person(person.id), lambda result: self.reload())

    def _buttons(self):
        idle = self.runner.can_start and not self.closing
        self.people_list.configure(state="normal" if idle else "disabled")
        self.samples_list.configure(state="normal" if idle else "disabled")
        for button in (self.rename_button, self.delete_person_button):
            button.configure(state="normal" if idle and self.person_id else "disabled")
        self.delete_sample_button.configure(state="normal" if idle and self.sample_id else "disabled")

    def _poll(self):
        try:
            if self.runner.busy:
                raise Empty
            kind, message = self.runner.events.get_nowait()
        except Empty:
            pass
        else:
            if kind == "done":
                result = self.results.get_nowait()
                if not self.closing:
                    self.callback(result)
            elif kind == "error" and not self.closing:
                self.photo = None
                self.image_label.configure(image="", text="Не удалось выполнить действие")
                self.status.set(message)
                messagebox.showerror("Ошибка", message, parent=self.window)
        self._buttons()
        if self.closing and not self.runner.busy:
            self.window.grab_release()
            self.window.destroy()
            self.application.profiles_window = None
            self.application._refresh_people()
            return
        self.window.after(40, self._poll)

    def close(self):
        # Let an atomic storage mutation finish; never kill it halfway through.
        self.closing = True
        self.status.set("Завершение операции…")
        self._buttons()
