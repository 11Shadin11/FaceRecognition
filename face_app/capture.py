"""OpenCV capture in a spawned process, with bounded buffers and shutdown.

Tk never calls a camera driver. A blocked native read can be terminated without
blocking the interface. Live frames may be dropped; sampled file frames are not.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import multiprocessing as mp
from queue import Empty, Full
import threading
import time


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    captured_at: float
    image: object


class CameraFrameHealth:
    """Allow camera warm-up, but report a sustained almost-black image clearly."""

    def __init__(self, source):
        self.source = source
        self.black_since = None

    def check(self, image, now):
        # A sparse sample avoids scanning every pixel on every captured frame.
        pixels = image[::16, ::16]
        if float(pixels.mean()) < 2 and int(pixels.max()) < 8:
            if self.black_since is None:
                self.black_since = now
            elif now - self.black_since >= 3:
                raise RuntimeError(f"Камера №{self.source} выдаёт почти чёрное изображение. "
                                   "Проверьте номер камеры, шторку и освещение. "
                                   "При подключённом iPhone встроенная камера может иметь другой номер.")
        else:
            self.black_since = None


@contextmanager
def open_capture(source, width=640, height=480, cv_module=None):
    if cv_module is None:
        import cv2 as cv_module
    capture = cv_module.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise RuntimeError("Не удалось открыть камеру или видео. Проверьте источник и разрешения.")
        if isinstance(source, int):
            capture.set(cv_module.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv_module.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv_module.CAP_PROP_FPS, 30)
            capture.set(cv_module.CAP_PROP_BUFFERSIZE, 1)
        yield capture
    finally:
        capture.release()


def _put_frame(frames, item, stop, live):
    if live:
        try:
            frames.put_nowait(item)
        except Full:
            try:
                frames.get_nowait()
            except Empty:
                pass
            try:
                frames.put_nowait(item)
            except Full:
                pass
    else:
        while not stop.is_set():
            try:
                frames.put(item, timeout=0.1)
                return
            except Full:
                continue


def _capture_process(source, config, frames, messages, stop):
    import cv2

    cv2.setNumThreads(1)
    live = isinstance(source, int)
    health = CameraFrameHealth(source) if live else None
    # Waiting for a queue feeder during shutdown could itself hang on large frames.
    if live:
        frames.cancel_join_thread()
    try:
        with open_capture(source, config.camera_width, config.camera_height) as capture:
            reported_fps = capture.get(cv2.CAP_PROP_FPS)
            fps = reported_fps if math.isfinite(reported_fps) and 1 <= reported_fps <= 240 else 30.0
            step = max(1, round(fps * config.sample_interval))
            frame_limit = math.ceil(fps * config.max_video_seconds)
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not live and width * height > 16_000_000:
                raise ValueError("Разрешение видео превышает 16 мегапикселей.")
            sequence = 0
            next_live_frame = time.monotonic()
            while not stop.is_set():
                if not live and sequence >= frame_limit:
                    break
                ok, image = capture.read()
                if not ok or image is None:
                    if live:
                        raise RuntimeError("Камера перестала выдавать кадры.")
                    break
                sequence += 1
                if health is not None:
                    health.check(image, time.monotonic())
                if live or (sequence - 1) % step == 0:
                    if live and image.shape[1] > config.camera_width:
                        scaled_height = max(1, round(image.shape[0] * config.camera_width / image.shape[1]))
                        image = cv2.resize(image, (config.camera_width, scaled_height))
                    _put_frame(frames, CapturedFrame(sequence, time.monotonic(), image), stop, live)
                if live:
                    next_live_frame = max(next_live_frame + 1 / 30, time.monotonic())
                    stop.wait(max(0, next_live_frame - time.monotonic()))
        messages.put(("done", None))
    except Exception as error:
        messages.put(("error", str(error)))
    finally:
        # File consumers may stop early (e.g. the sample limit); do not wait for them.
        if stop.is_set():
            frames.cancel_join_thread()


class CaptureStream:
    def __init__(self, source, config):
        self.source = source
        self.config = config
        context = mp.get_context("spawn")
        self.frames = context.Queue(maxsize=1 if isinstance(source, int) else 2)
        self.messages = context.Queue()
        self.stop_event = context.Event()
        self.process = context.Process(target=_capture_process,
                                       args=(source, config, self.frames, self.messages, self.stop_event),
                                       daemon=True)
        self._latest = None
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._reader = None
        self._closed = threading.Event()
        self.error = None
        self.done = False
        self._started = False

    def start(self):
        self.process.start()
        self._started = True
        if isinstance(self.source, int):
            self._reader = threading.Thread(target=self._read_live_frames, daemon=True)
            self._reader.start()
        return self

    def _check_messages(self):
        while True:
            try:
                kind, message = self.messages.get_nowait()
            except Empty:
                break
            if kind == "error":
                self.error = message
            self.done = True
        if self._started and not self.process.is_alive() and not self.done:
            # Queue messages can arrive just after the process exits.
            try:
                kind, message = self.messages.get(timeout=0.05)
                self.done = True
                if kind == "error":
                    self.error = message
            except Empty:
                self.error = "Процесс захвата неожиданно завершился."
                self.done = True

    def _read_live_frames(self):
        while not self.stop_event.is_set():
            try:
                try:
                    frame = self.frames.get(timeout=0.1)
                    with self._lock:
                        self._latest = frame
                except Empty:
                    pass
                self._check_messages()
            except (EOFError, OSError, ValueError):
                if not self.stop_event.is_set():
                    self.error = "Соединение с процессом камеры потеряно."
                return
            if self.done:
                return

    def latest(self):
        with self._lock:
            return self._latest

    def live_frames(self, cancel):
        """Yield new live frames; distinguish initial permission wait from a stalled camera."""
        last_sequence = -1
        waiting_since = time.monotonic()
        while not cancel.is_set():
            if self.error:
                raise RuntimeError(self.error)
            frame = self.latest()
            if frame is None or frame.sequence == last_sequence:
                timeout = 15 if frame is None else 5
                if time.monotonic() - waiting_since > timeout:
                    raise RuntimeError("Камера не выдаёт новые кадры. Проверьте источник и разрешения.")
                cancel.wait(0.03)
                continue
            waiting_since = time.monotonic()
            last_sequence = frame.sequence
            yield frame

    def file_frames(self, cancel):
        while not cancel.is_set() and not self.stop_event.is_set():
            try:
                yield self.frames.get(timeout=0.1)
            except Empty:
                self._check_messages()
                if self.error:
                    raise RuntimeError(self.error)
                if self.done and not self.process.is_alive():
                    return

    def request_stop(self):
        self.stop_event.set()

    def close(self):
        """Call off the UI thread; shutdown has bounded waits even for broken drivers."""
        with self._close_lock:
            if self._closed.is_set():
                return
            self.request_stop()
            if self._started:
                self.process.join(timeout=0.6)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=0.6)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=0.3)
            if self._reader is not None:
                self._reader.join(timeout=0.3)
            for queue in (self.frames, self.messages):
                queue.cancel_join_thread()
                queue.close()
            self._closed.set()

    @property
    def closed(self):
        return self._closed.is_set()
