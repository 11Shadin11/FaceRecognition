"""Small validated local preferences; atomic writes, no camera probing."""
import json
import os
from pathlib import Path
import tempfile


class Preferences:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "settings.json"

    def camera(self, default=0):
        try:
            if self.path.is_symlink() or self.path.stat().st_size > 4096:
                return default
            value = json.loads(self.path.read_text(encoding="utf-8")).get("camera_index")
            return value if type(value) is int and 0 <= value <= 99 else default
        except (OSError, ValueError, TypeError, AttributeError):
            return default

    def save_camera(self, index):
        if type(index) is not int or not 0 <= index <= 99:
            raise ValueError("Номер камеры должен быть целым числом от 0 до 99.")
        if self.path.is_symlink():
            raise ValueError("Файл настроек не должен быть ссылкой.")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, mode="w", encoding="utf-8", delete=False) as output:
                temporary = Path(output.name)
                json.dump({"camera_index": index}, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
