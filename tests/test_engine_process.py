import threading
import time
import unittest

from face_app.config import AppConfig
from face_app.models import FaceMatch
from face_app.engine_process import ModelProcess
from face_app.recognition import InvalidSample


class FakeProcessEngine:
    model_version = "fake-process-model-v1"

    def __init__(self, config):
        self.people = []

    def set_known_faces(self, people, encodings):
        self.people = people

    def recognize(self, frame):
        return self.people

    def prepare_sample(self, frame):
        if frame == b"block":
            # CPU work retains this process's GIL and deliberately ignores cancel.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                pass
        if frame == b"invalid":
            raise InvalidSample("Нужно одно лицо.")
        return (0.1,) * 128, frame


class ModelProcessTests(unittest.TestCase):
    def setUp(self):
        self.engine = ModelProcess(AppConfig(), FakeProcessEngine)
        self.addCleanup(self.engine.close)
        self.cancel = threading.Event()
        self.engine.set_cancel_event(self.cancel)

    def test_rpc_returns_valid_samples_and_preserves_domain_errors(self):
        encoding, data = self.engine.prepare_sample(b"photo")
        self.assertEqual(len(encoding), 128)
        self.assertEqual(data, b"photo")
        with self.assertRaises(InvalidSample):
            self.engine.prepare_sample(b"invalid")
        self.assertFalse(self.engine.closed)
        self.assertEqual(self.engine.prepare_sample(b"next")[1], b"next")

    def test_face_geometry_and_identity_survive_process_boundary(self):
        face = FaceMatch("known", "a", "Алиса", 0.2, (10, 20, 100, 120))
        self.engine.set_known_faces([face], [])
        self.assertEqual(self.engine.recognize(b"frame"), [face])

    def test_cancel_interrupts_non_cooperative_inference(self):
        timer = threading.Timer(0.1, self.cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        with self.assertRaises(RuntimeError):
            self.engine.prepare_sample(b"block")
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(self.engine.closed)
        self.assertFalse(self.engine.process.is_alive())

    def test_timeout_stops_hung_model(self):
        self.engine.timeout = 0.15
        with self.assertRaises(RuntimeError):
            self.engine.prepare_sample(b"block")
        self.assertTrue(self.engine.closed)

    def test_model_is_reused_between_jobs(self):
        pid = self.engine.process.pid
        self.cancel.set()
        self.engine.set_cancel_event(threading.Event())
        self.engine.prepare_sample(b"next-job")
        self.assertEqual(self.engine.process.pid, pid)

    def test_close_is_idempotent(self):
        self.engine.close()
        self.engine.close()
        self.assertTrue(self.engine.closed)
