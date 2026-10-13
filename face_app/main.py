"""Entry point: parse settings, create a window, and run only the Tk event loop."""
import argparse
from pathlib import Path
import sys

from .config import AppConfig
from .preferences import Preferences


def main(argv=None):
    parser = argparse.ArgumentParser(description="Локальное распознавание лиц")
    parser.add_argument("--camera", type=int, default=None, help="Номер камеры (по умолчанию последний выбранный)")
    parser.add_argument("--data-dir", type=Path, default=AppConfig.data_dir)
    parser.add_argument("--legacy-dir", type=Path, default=AppConfig.legacy_dir)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args(argv)
    try:
        config = AppConfig(data_dir=args.data_dir, legacy_dir=args.legacy_dir,
                           camera_index=args.camera if args.camera is not None else Preferences(args.data_dir).camera(),
                           match_threshold=args.threshold)
        from PIL import ImageTk
        import tkinter as tk
        from .ui import FaceApplication

        if tk.TkVersion < 8.6:
            raise RuntimeError("Нужен Python с Tk 8.6 или новее. Системный Python macOS с Tk 8.5 не поддерживается.")
        window = tk.Tk()
        application = None
        try:
            application = FaceApplication(window, config)
            window.mainloop()
        finally:
            if application is not None:
                application.runner.cancel()
                application.service.close()
            try:
                window.destroy()
            except tk.TclError:
                pass  # WM_DELETE_WINDOW may already have destroyed the interpreter.
    except Exception as error:
        print(f"Не удалось запустить приложение: {error}", file=sys.stderr)
        if isinstance(error, ImportError):
            print("Установите зависимости по инструкции в README.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
