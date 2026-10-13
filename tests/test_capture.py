from contextlib import contextmanager
import multiprocessing as mp
from queue import Queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from face_app.capture import CameraFrameHealth, CaptureStream, CapturedFrame, _capture_process, _put_frame, open_capture
from face_app.config import AppConfig


def blocked_capture(*args):
    # Simulates a native read that does not observe cancellation.
    threading.Event().wait()


def finite_file_capture(source, config, frames, messages, stop):
    for sequence in range(5):
        _put_frame(frames, CapturedFrame(sequence, time.monotonic(), sequence), stop, False)
    messages.put(("done", None))


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return self.opened

    def set(self, *args):
        pass

    def get(self, property_id):
        return 0  # Missing FPS and dimensions.

    def read(self):
        self.read_count += 1
        return (True, b"frame") if self.read_count <= 4 else (False, None)

    def release(self):
        self.released = True


class CaptureTests(unittest.TestCase):
    def test_sustained_black_camera_has_actionable_error(self):
        import numpy as np
        health = CameraFrameHealth(0)
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        health.check(black, 10)
        health.check(black, 12.9)
        with self.assertRaisesRegex(RuntimeError, "Камера №0.*номер камеры"):
            health.check(black, 13)

    def test_camera_warmup_and_recovery_reset_black_frame_timer(self):
        import numpy as np
        health = CameraFrameHealth(1)
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        valid = np.full((32, 32, 3), 50, dtype=np.uint8)
        health.check(black, 0)
        health.check(valid, 2)
        health.check(black, 4)
        health.check(black, 6)
        health.check(valid, 6.5)
        self.assertIsNone(health.black_since)

    def fake_cv(self, capture):
        return SimpleNamespace(VideoCapture=lambda source: capture, CAP_PROP_FPS=1,
                               CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
                               CAP_PROP_BUFFERSIZE=4, setNumThreads=lambda value: None)

    def test_open_failure_still_releases_capture(self):
        capture = FakeCapture(opened=False)
        with self.assertRaises(RuntimeError):
            with open_capture(0, cv_module=self.fake_cv(capture)):
                pass
        self.assertTrue(capture.released)

    def test_processing_failure_releases_capture(self):
        capture = FakeCapture()
        with self.assertRaises(ValueError):
            with open_capture(0, cv_module=self.fake_cv(capture)):
                raise ValueError("simulated failure")
        self.assertTrue(capture.released)

    def test_missing_video_fps_uses_fallback_and_reaches_eof(self):
        capture = FakeCapture()
        frames, messages = Queue(), Queue()
        with patch.dict("sys.modules", {"cv2": self.fake_cv(capture)}):
            _capture_process("video", AppConfig(), frames, messages, threading.Event())
        self.assertTrue(capture.released)
        self.assertEqual(messages.get_nowait()[0], "done")
        self.assertEqual(frames.get_nowait().sequence, 1)

    def test_live_buffer_keeps_latest_instead_of_growing(self):
        frames = Queue(maxsize=1)
        for sequence in range(10):
            _put_frame(frames, sequence, threading.Event(), True)
        self.assertEqual(frames.qsize(), 1)
        self.assertEqual(frames.get_nowait(), 9)

    def test_file_buffer_stops_waiting_when_cancelled(self):
        frames = Queue(maxsize=1)
        frames.put("existing")
        stop = threading.Event()
        worker = threading.Thread(target=_put_frame, args=(frames, "new", stop, False))
        worker.start()
        stop.set()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(frames.get_nowait(), "existing")

    def test_blocked_capture_process_is_terminated_within_bound(self):
        stream = CaptureStream("fake-file", AppConfig())
        stream.process.close()
        stream.process = mp.get_context("spawn").Process(target=blocked_capture, daemon=True)
        stream.start()
        started = time.monotonic()
        stream.close()
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertFalse(stream.process.is_alive())
        self.assertTrue(stream.closed)
        stream.close()  # Idempotent cleanup.

    def test_file_queue_delivers_all_sampled_frames_before_eof(self):
        stream = CaptureStream("fake-file", AppConfig())
        stream.process.close()
        stream.process = mp.get_context("spawn").Process(
            target=finite_file_capture,
            args=(stream.source, stream.config, stream.frames, stream.messages, stream.stop_event),
            daemon=True)
        try:
            stream.start()
            frames = list(stream.file_frames(threading.Event()))
            self.assertEqual([frame.sequence for frame in frames], list(range(5)))
        finally:
            stream.close()
