"""Exercise the real Tk interface using disposable data, without a camera."""
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import tkinter as tk
    from PIL import Image
    from face_app.config import AppConfig
    from face_app.preferences import Preferences
    from face_app.ui import FaceApplication

    with tempfile.TemporaryDirectory(prefix="face-interface-check-") as directory:
        window = tk.Tk()
        window.withdraw()
        app = FaceApplication(window, AppConfig(data_dir=Path(directory) / "data"))
        album = BytesIO()
        Image.new("RGB", (640, 480), "#42678b").save(album, format="JPEG")
        person = app.store.get_or_create_person("Alice")
        other = app.store.get_or_create_person("Bob")
        for profile, payload in ((person, album.getvalue()), (person, album.getvalue() + b"extra"), (other, album.getvalue())):
            app.store.add_sample(profile.id, payload, [0.1] * 128, "test")
        app._refresh_people()
        window.geometry("920x660")
        window.deiconify()
        window.update()

        def until(condition, timeout=5):
            deadline = time.monotonic() + timeout
            while not condition():
                window.update()
                if time.monotonic() > deadline:
                    raise AssertionError("Tk operation timed out")
                time.sleep(.01)
            window.update()

        widgets = (app.manage_button, app.photo_button, app.video_button, app.backup_button,
                   app.legacy_button, app.status_label, app.recognize_button, app.camera_button)
        for widget in widgets:
            right = widget.winfo_rootx() - window.winfo_rootx() + widget.winfo_width()
            bottom = widget.winfo_rooty() - window.winfo_rooty() + widget.winfo_height()
            assert right <= window.winfo_width() and bottom <= window.winfo_height(), (widget, right, bottom)
        assert app.preview_canvas.winfo_height() >= 80
        assert app.progress.winfo_manager() == ""
        app.camera_number.set("1")
        app._remember_camera()
        assert Preferences(app.store.root).camera() == 1

        release = threading.Event()

        def job(context):
            context.status("Импорт", .5)
            release.wait(2)

        app._start(job)
        until(lambda: app.progress.winfo_manager() == "grid")
        assert str(app.manage_button.cget("state")) == "disabled"
        release.set()
        until(lambda: app.runner.can_start and app.progress.winfo_manager() == "")

        app.manage_profiles()
        viewer = app.profiles_window
        until(lambda: viewer.runner.can_start and len(viewer.people) == 2)
        viewer.people_list.selection_set(0)
        viewer._select_person()
        until(lambda: viewer.runner.can_start and len(viewer.samples) == 2)
        viewer.samples_list.selection_set(0)
        viewer._select_sample()
        until(lambda: viewer.runner.can_start and viewer.photo is not None)
        assert 1 < viewer.photo.width() <= 460
        with patch("face_app.profiles_ui.simpledialog.askstring", return_value="Алиса"):
            viewer.rename()
        until(lambda: viewer.runner.can_start and any(p.name == "Алиса" for p in viewer.people))
        viewer.samples_list.selection_set(0)
        viewer._select_sample()
        until(lambda: viewer.runner.can_start and viewer.photo is not None)
        with patch("face_app.profiles_ui.messagebox.askyesno", return_value=True):
            viewer.delete_sample()
        until(lambda: viewer.runner.can_start and len(viewer.samples) == 1)
        with patch("face_app.profiles_ui.messagebox.askyesno", return_value=True):
            viewer.delete_person()
        until(lambda: viewer.runner.can_start and len(viewer.people) == 1)
        assert app.store.samples(other.id)[0].path.read_bytes() == album.getvalue()
        viewer.close()
        until(lambda: app.profiles_window is None)
        assert app.sample_count == 1
        app.close()
        window.mainloop()
        print("Tk: minimum layout, conditional progress, camera preferences, photo preview, rename/delete, refresh and close passed.")


if __name__ == "__main__":
    main()
