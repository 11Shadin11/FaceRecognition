from pathlib import Path
from contextlib import contextmanager
import time
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from face_app.config import AppConfig
from face_app.capture import CapturedFrame
from face_app.recognition import InvalidSample
from face_app.models import FaceMatch
from face_app.service import FaceService
from face_app.storage import FaceStore
from face_app.tasks import Cancelled, TaskContext, TaskRunner


class FakeEngine:
    model_version = "model-v1"

    def __init__(self, config):
        self.encoded = 0
        self.index = []

    def read_image(self, path):
        return path.read_bytes()

    def encode_sample(self, image):
        self.encoded += 1
        if image in (b"no-face", b"two-faces"):
            raise InvalidSample("Нужно ровно одно лицо.")
        return (len(image) / 100,) * 128

    def image_bytes(self, image):
        return image

    def encode_image_file(self, path):
        return self.encode_sample(self.read_image(path))

    def prepare_sample(self, image):
        return self.encode_sample(image), self.image_bytes(image)

    def prepare_sample_file(self, path):
        return self.prepare_sample(self.read_image(path))

    def set_known_faces(self, people, encodings):
        self.index = list(zip(people, encodings))


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.config = AppConfig(data_dir=root / "data", legacy_dir=root / "faces")
        self.store = FaceStore(self.config.data_dir)
        self.service = FaceService(self.store, self.config, FakeEngine)
        self.runner = TaskRunner()
        self.context = TaskContext(self.runner, self.runner.cancel_event)

    def legacy_photo(self, name, filename, payload):
        folder = self.config.legacy_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        path.write_bytes(payload)
        return path

    def test_legacy_import_preserves_originals_and_rejects_bad_faces(self):
        valid = self.legacy_photo("Alice", "1.jpg", b"good")
        self.legacy_photo("Alice", "2.jpg", b"no-face")
        self.legacy_photo("Alice", "3.jpg", b"two-faces")
        self.service.import_legacy(self.context)
        self.assertEqual(len(self.store.samples()), 1)
        self.assertEqual(valid.read_bytes(), b"good")
        self.service.import_legacy(self.context)
        self.assertEqual(len(self.store.samples()), 1)

    def test_unchanged_cached_samples_are_not_encoded_again(self):
        self.legacy_photo("Alice", "1.jpg", b"good")
        self.service.import_legacy(self.context)
        engine = self.service.engine(self.context)
        engine.encoded = 0
        self.assertEqual(self.service.prepare_index(self.context), (1, 0))
        self.assertEqual(engine.encoded, 0)

    def test_changed_photo_is_encoded_again(self):
        self.legacy_photo("Alice", "1.jpg", b"good")
        self.service.import_legacy(self.context)
        sample = self.store.samples()[0]
        sample.path.write_bytes(b"changed")
        self.service.prepare_index(self.context)
        self.assertEqual(self.store.samples()[0].encoding, (0.07,) * 128)

    def test_model_change_and_corrupt_cache_trigger_rebuild(self):
        self.legacy_photo("Alice", "1.jpg", b"good")
        self.service.import_legacy(self.context)
        with self.store.connection() as db:
            db.execute("UPDATE samples SET encoding_json = '[]', model_version = 'old'")
        self.assertEqual(self.service.prepare_index(self.context), (1, 0))
        self.assertEqual(self.store.samples()[0].model_version, "model-v1")

    def test_changed_invalid_and_missing_photos_do_not_enter_index(self):
        self.legacy_photo("Alice", "1.jpg", b"good")
        self.legacy_photo("Alice", "2.jpg", b"other-good")
        self.service.import_legacy(self.context)
        samples = self.store.samples()
        samples[0].path.write_bytes(b"two-faces")
        samples[1].path.unlink()
        self.assertEqual(self.service.prepare_index(self.context), (0, 2))
        self.assertEqual(self.service.engine(self.context).index, [])

    def test_cancelled_import_does_not_write_or_load_model(self):
        self.legacy_photo("Alice", "1.jpg", b"good")
        self.runner.cancel_event.set()
        with self.assertRaises(Cancelled):
            self.service.import_legacy(self.context)
        self.assertEqual(self.store.samples(), [])
        self.assertIsNone(self.service._engine)

    def test_photo_import_rejects_bad_images_and_deduplicates_per_person(self):
        good = self.legacy_photo("input", "1.jpg", b"good")
        bad = self.legacy_photo("input", "2.jpg", b"two-faces")
        result = self.service.import_photos(self.context, "Alice", [good, bad])
        self.assertIn("добавлено 1", result)
        self.assertIn("отклонено 1", result)
        self.service.import_photos(self.context, "Alice", [good])
        self.assertEqual(len(self.store.samples()), 1)
        self.assertEqual(good.read_bytes(), b"good")

    def test_failed_photo_import_does_not_create_empty_profile(self):
        bad = self.legacy_photo("input", "bad.jpg", b"no-face")
        with self.assertRaises(ValueError):
            self.service.import_photos(self.context, "Alice", [bad])
        self.assertEqual(self.store.people(), [])

    def test_photo_import_checks_limit_and_cancellation_before_loading(self):
        with self.assertRaises(ValueError):
            self.service.import_photos(self.context, "Alice", [Path("a.jpg")] * 101)
        self.assertIsNone(self.service._engine)
        self.runner.cancel_event.set()
        with self.assertRaises(Cancelled):
            self.service.import_photos(self.context, "Alice", [Path("a.jpg")])
        self.assertEqual(self.store.people(), [])

    def fake_file_capture(self, payloads):
        state = SimpleNamespace(closed=False)

        @contextmanager
        def capture(source, config):
            try:
                yield SimpleNamespace(file_frames=lambda cancel: (
                    CapturedFrame(index, time.monotonic(), payload)
                    for index, payload in enumerate(payloads)))
            finally:
                state.closed = True

        return capture, state

    def test_video_collection_keeps_only_valid_unique_samples(self):
        capture, state = self.fake_file_capture([b"good", b"no-face", b"two-faces", b"good", b"other"])
        with patch.object(self.context, "capture", capture):
            message = self.service.collect_samples(self.context, "Alice", "file.avi")
        self.assertTrue(state.closed)
        self.assertEqual(len(self.store.samples()), 2)
        self.assertIn("отклонено 2", message)

    def test_cancelled_collection_keeps_accepted_samples_and_closes_source(self):
        capture, state = self.fake_file_capture([b"first", b"second"])
        original = self.store.add_sample

        def save_then_cancel(*args, **kwargs):
            saved = original(*args, **kwargs)
            self.runner.cancel_event.set()
            return saved

        with patch.object(self.context, "capture", capture), \
                patch.object(self.store, "add_sample", save_then_cancel):
            with self.assertRaises(Cancelled):
                self.service.collect_samples(self.context, "Alice", "file.avi")
        self.assertTrue(state.closed)
        self.assertEqual(len(self.store.samples()), 1)

    def test_empty_video_reports_error_instead_of_success(self):
        capture, state = self.fake_file_capture([])
        with patch.object(self.context, "capture", capture):
            with self.assertRaises(ValueError):
                self.service.collect_samples(self.context, "Alice", "empty.avi")
        self.assertTrue(state.closed)
        self.assertEqual(self.store.samples(), [])

    def test_live_recognition_publishes_geometry_and_status_together(self):
        face = FaceMatch("known", "a", "Alice", .2, (10, 20, 100, 120))
        image = SimpleNamespace(shape=(480, 640, 3))
        frame = CapturedFrame(42, time.monotonic(), image)
        engine = SimpleNamespace(recognize=lambda image: [face])

        @contextmanager
        def capture(*args):
            yield SimpleNamespace(live_frames=lambda cancel: iter([frame]))

        with patch.object(self.service, "prepare_index", return_value=(1, 0)), \
                patch.object(self.service, "engine", return_value=engine), \
                patch.object(self.context, "capture", capture):
            self.service.recognize_camera(self.context)
        status = self.runner.status()
        self.assertEqual(status.recognition.faces, (face,))
        self.assertEqual(status.recognition.sequence, 42)
        self.assertEqual(status.recognition.frame_size, (640, 480))
        self.assertEqual(status.recognition.captured_at, status.recognized_at)
        self.assertIn("Alice", status.message)
