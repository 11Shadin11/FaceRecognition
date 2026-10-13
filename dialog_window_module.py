"""Compatibility launcher. The actual interface lives in face_app.ui."""


def open_init_window():
    from face_app.main import main
    return main()


if __name__ == "__main__":
    open_init_window()
