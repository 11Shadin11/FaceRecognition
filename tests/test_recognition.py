import importlib.util
from pathlib import Path
import threading
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from face_app.config import AppConfig
from face_app.capture import CaptureStream
from face_app.models import Person
from face_app.recognition import FaceEngine, InvalidSample


NATIVE_AVAILABLE = all(importlib.util.find_spec(name) for name in ("cv2", "numpy", "PIL"))


@unittest.skipUnless(NATIVE_AVAILABLE, "Install NumPy, OpenCV and Pillow to test image operations")
class RecognitionTests(unittest.TestCase):
    def setUp(self):
        import cv2  # Native modules must remain loaded across sys.modules patches.
        import numpy as np
        self.np = np
        self.locations = [(10, 100, 100, 10)]
        self.received_rgb = None

        def locations(rgb, **kwargs):
            self.received_rgb = rgb
            return self.locations

        self.api = SimpleNamespace(face_locations=locations,
                                   face_encodings=lambda rgb, boxes, **kwargs: [np.zeros(128) for box in boxes])
        with patch.dict("sys.modules", {"face_recognition": self.api}), \
                patch("face_app.recognition.version", return_value="test"):
            self.engine = FaceEngine(AppConfig(min_sharpness=0, min_face_size=10))

    def test_bgr_input_is_converted_to_contiguous_rgb(self):
        frame = self.np.full((120, 120, 3), [20, 80, 160], dtype=self.np.uint8)
        self.engine.recognize(frame)
        self.np.testing.assert_array_equal(self.received_rgb[0, 0], [160, 80, 20])
        self.assertTrue(self.received_rgb.flags.c_contiguous)

    def test_recognition_uses_nearest_sample_in_prebuilt_index(self):
        a, b = self.np.zeros(128), self.np.zeros(128)
        a[0], b[0] = 0.59, 0.30
        self.engine.set_known_faces([Person("a", "Alice"), Person("b", "Bob")], [a, b])
        matches = self.engine.recognize(self.np.full((120, 120, 3), 128, dtype=self.np.uint8))
        self.assertEqual(matches[0].person_id, "b")

    def test_empty_index_returns_unknown(self):
        matches = self.engine.recognize(self.np.full((120, 120, 3), 128, dtype=self.np.uint8))
        self.assertEqual(matches[0].status, "unknown")

    def test_bounds_return_to_original_frame_and_keep_identity_pairing(self):
        self.locations = [(10, 200, 110, 20), (-5, 490, 275, 400)]
        self.api.face_encodings = lambda *args, **kwargs: [self.np.zeros(128), self.np.ones(128)]
        self.engine.set_known_faces([Person("a", "Alice"), Person("b", "Bob")],
                                    [self.np.zeros(128), self.np.ones(128)])
        with patch.object(self.api, "face_locations", wraps=self.api.face_locations) as detector:
            matches = self.engine.recognize(self.np.full((361, 641, 3), 128, dtype=self.np.uint8))
        self.assertEqual(detector.call_count, 1)
        self.assertEqual([match.name for match in matches], ["Alice", "Bob"])
        self.assertEqual(matches[0].bounds, (27, 13, 267, 147))
        self.assertEqual(matches[1].bounds, (534, 0, 641, 361))

    def test_multiple_or_no_faces_cannot_be_enrolled(self):
        frame = self.np.full((120, 120, 3), 128, dtype=self.np.uint8)
        for boxes in ([], [(10, 60, 60, 10), (60, 110, 110, 60)]):
            self.locations = boxes
            with self.assertRaises(InvalidSample):
                self.engine.encode_sample(frame)

    def test_resize_reduces_analysis_width(self):
        frame = self.np.full((720, 1280, 3), 128, dtype=self.np.uint8)
        self.engine.recognize(frame)
        self.assertEqual(self.received_rgb.shape, (270, 480, 3))

    def test_valid_image_roundtrip_and_missing_image_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.jpg"
            frame = self.np.full((120, 120, 3), 128, dtype=self.np.uint8)
            path.write_bytes(self.engine.image_bytes(frame))
            loaded = self.engine.read_image(path)
            self.assertEqual(loaded.shape, frame.shape)
            self.assertEqual(len(self.engine.encode_sample(loaded)), 128)
            with self.assertRaises(OSError):
                self.engine.read_image(path.with_name("missing.jpg"))

    def test_nose_overlay_clips_edges_and_handles_zero_size(self):
        from pig_nose import overlay_nose

        frame = self.np.zeros((30, 30, 3), dtype=self.np.uint8)
        nose = self.np.full((20, 20, 4), 255, dtype=self.np.uint8)
        overlay_nose(frame, nose, (0, 0), 20, self.engine.cv)
        self.assertTrue(frame[0, 0].any())
        overlay_nose(frame, nose, (-100, -100), 20, self.engine.cv)
        overlay_nose(frame, nose, (0, 0), 0, self.engine.cv)

    def test_decoder_dimension_limit_is_reported_as_invalid_sample(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.jpg"
            path.write_bytes(b"invalid")
            with patch("PIL.Image.open", side_effect=Image.DecompressionBombError("oversized")):
                with self.assertRaises(InvalidSample):
                    self.engine.read_image(path)

    def test_real_video_decoding_in_spawned_process_preserves_sampled_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.avi"
            cv = self.engine.cv
            writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"MJPG"), 30, (120, 120))
            self.assertTrue(writer.isOpened())
            try:
                for number in range(60):
                    writer.write(self.np.full((120, 120, 3), number, dtype=self.np.uint8))
            finally:
                writer.release()
            stream = CaptureStream(str(path), AppConfig())
            try:
                stream.start()
                frames = list(stream.file_frames(threading.Event()))
                self.assertEqual([frame.sequence for frame in frames], [1, 23, 45])
                self.assertEqual(frames[-1].image.shape, (120, 120, 3))
            finally:
                stream.close()
