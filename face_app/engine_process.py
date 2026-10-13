"""A persistent inference process keeps dlib's native calls away from Tk's GIL.

Only one trusted local request is outstanding. Cancellation or a timeout closes
the process; the next job creates a fresh model. No data files use pickle.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

from .recognition import FaceEngine, InvalidSample


ALLOWED_METHODS = {"read_image", "encode_sample", "image_bytes", "recognize",
                   "set_known_faces", "encode_image_file", "prepare_sample",
                   "prepare_sample_file"}


def _serve_model(config, connection, engine_factory):
    try:
        engine = engine_factory(config)
        connection.send(("ready", engine.model_version))
        while True:
            method, args = connection.recv()
            if method == "close":
                return
            if method not in ALLOWED_METHODS:
                raise ValueError("Неизвестная операция модели.")
            try:
                connection.send(("result", getattr(engine, method)(*args)))
            except Exception as error:
                connection.send(("error", (type(error).__name__, str(error))))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send(("error", (type(error).__name__, str(error))))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class ModelProcess:
    def __init__(self, config, engine_factory=FaceEngine):
        context = mp.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(target=_serve_model,
                                       args=(config, child_connection, engine_factory), daemon=True)
        self.timeout = config.inference_timeout
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.closed = False
        self.model_version = None
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            child_connection.close()
            self.closed = True
            raise
        child_connection.close()

    def set_cancel_event(self, cancel):
        self.cancel = cancel
        if self.model_version is None:
            with self.lock:
                kind, value = self._receive()
                if kind != "ready":
                    self._shutdown()
                    self._raise_error(value)
                self.model_version = value

    def _receive(self):
        deadline = time.monotonic() + self.timeout
        while not self.connection.poll(0.05):
            if self.cancel.is_set():
                self._shutdown()
                raise RuntimeError("Вычисление отменено.")
            if not self.process.is_alive() or time.monotonic() >= deadline:
                self._shutdown()
                raise RuntimeError("Модель не отвечает. Попробуйте запустить задачу повторно.")
        if self.cancel.is_set():
            self._shutdown()
            raise RuntimeError("Вычисление отменено.")
        try:
            return self.connection.recv()
        except (EOFError, OSError) as error:
            self._shutdown()
            raise RuntimeError("Соединение с процессом модели потеряно.") from error

    @staticmethod
    def _raise_error(value):
        name, message = value
        error_type = {"InvalidSample": InvalidSample, "ValueError": ValueError,
                      "FileNotFoundError": FileNotFoundError, "OSError": OSError,
                      "ModuleNotFoundError": ImportError, "ImportError": ImportError}.get(name, RuntimeError)
        raise error_type(message)

    def _request(self, method, *args):
        with self.lock:
            if self.closed or self.cancel.is_set():
                raise RuntimeError("Процесс модели закрыт или задача отменена.")
            try:
                self.connection.send((method, args))
            except (BrokenPipeError, OSError) as error:
                self._shutdown()
                raise RuntimeError("Соединение с процессом модели потеряно.") from error
            kind, value = self._receive()
            if kind == "error":
                self._raise_error(value)
            return value

    def set_known_faces(self, people, encodings):
        return self._request("set_known_faces", people, encodings)

    def recognize(self, frame):
        return self._request("recognize", frame)

    def read_image(self, path):
        return self._request("read_image", path)

    def encode_image_file(self, path):
        return self._request("encode_image_file", path)

    def prepare_sample(self, frame):
        return self._request("prepare_sample", frame)

    def prepare_sample_file(self, path):
        return self._request("prepare_sample_file", path)

    def _shutdown(self):
        if self.closed:
            return
        self.connection.close()
        self.process.join(timeout=0.3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.6)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=0.3)
        self.closed = True

    def close(self):
        with self.lock:
            if not self.closed:
                try:
                    self.connection.send(("close", ()))
                except (BrokenPipeError, OSError):
                    pass
                self._shutdown()
