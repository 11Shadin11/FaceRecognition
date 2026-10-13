"""One application job at a time; workers communicate through bounded mailboxes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from queue import Queue
import threading

from .capture import CaptureStream
from .models import RecognitionResult


CANCELLED_MESSAGE = "Остановлено. Сохранённые данные остаются в базе."


class Cancelled(Exception):
    pass


@dataclass(frozen=True)
class TaskStatus:
    message: str
    progress: float | None = None
    recognized_at: float | None = None
    recognition: RecognitionResult | None = None


class TaskContext:
    def __init__(self, runner, cancel):
        self.runner = runner
        self.cancel = cancel

    def check_cancelled(self):
        if self.cancel.is_set():
            raise Cancelled()

    def status(self, message, progress=None, recognized_at=None, recognition=None):
        if self.cancel.is_set():
            return
        self.runner.set_status(TaskStatus(message, progress, recognized_at, recognition))

    def preview(self, frame):
        with self.runner.lock:
            self.runner.last_preview = frame

    @contextmanager
    def capture(self, source, config):
        self.check_cancelled()
        stream = CaptureStream(source, config)
        try:
            stream.start()
            with self.runner.lock:
                self.runner.stream = stream
            self.check_cancelled()
            yield stream
        finally:
            stream.close()
            with self.runner.lock:
                if self.runner.stream is stream:
                    self.runner.stream = None


class TaskRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = Queue()
        self.status_value = TaskStatus("Добавьте фотографии человека или импортируйте старую базу.")
        self.last_preview = None
        self.stream = None
        self.thread = None
        self.cancel_event = threading.Event()

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    @property
    def can_start(self):
        return not self.busy and self.events.empty()

    def start(self, target):
        if not self.can_start:
            raise RuntimeError("Дождитесь завершения текущей задачи.")
        # The UI consumes each terminal event before enabling another task.
        self.cancel_event = threading.Event()
        with self.lock:
            self.last_preview = None
        context = TaskContext(self, self.cancel_event)
        self.thread = threading.Thread(target=self._run, args=(target, context), daemon=True)
        self.thread.start()

    def _run(self, target, context):
        try:
            message = target(context)
            if context.cancel.is_set():
                self.events.put(("cancelled", CANCELLED_MESSAGE))
            else:
                self.events.put(("done", message or "Готово."))
        except Cancelled:
            self.events.put(("cancelled", CANCELLED_MESSAGE))
        except Exception as error:
            if context.cancel.is_set():
                self.events.put(("cancelled", CANCELLED_MESSAGE))
            else:
                logging.exception("Application task failed")
                self.events.put(("error", str(error)))

    def set_status(self, status):
        with self.lock:
            self.status_value = status

    def status(self):
        with self.lock:
            return self.status_value

    def preview(self):
        with self.lock:
            stream, fallback = self.stream, self.last_preview
        return stream.latest() if stream is not None and isinstance(stream.source, int) else fallback

    def cancel(self):
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        with self.lock:
            stream = self.stream
        if stream is not None:
            stream.request_stop()
            # Never join a process or a worker from a Tk event handler.
            threading.Thread(target=stream.close, daemon=True).start()

    def camera_closed(self):
        with self.lock:
            return self.stream is None or self.stream.closed
