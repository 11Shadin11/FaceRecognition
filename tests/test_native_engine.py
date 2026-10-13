"""Optional model smoke tests; synthetic images do not measure real-world accuracy."""
import importlib.util
import unittest
import threading
from unittest.mock import patch

from face_app.config import AppConfig
from face_app.models import Person
from face_app.recognition import FaceEngine
from face_app.engine_process import ModelProcess


MODEL_AVAILABLE = all(importlib.util.find_spec(name) for name in
                      ("cv2", "numpy", "PIL", "face_recognition", "dlib"))


@unittest.skipUnless(MODEL_AVAILABLE, "Install requirements.txt for native model smoke tests")
class NativeEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np

        cls.np = np
        cls.engine = FaceEngine(AppConfig())

    def test_real_detector_handles_frame_without_faces(self):
        frame = self.np.zeros((240, 320, 3), dtype=self.np.uint8)
        self.assertEqual(self.engine.recognize(frame), [])

    def test_real_encoder_and_matching_pipeline_on_controlled_region(self):
        # Check dlib's RGB/NumPy binding without biometric photos in the repository.
        frame = self.np.random.default_rng(42).integers(40, 210, (240, 320, 3), dtype=self.np.uint8)
        with patch.object(self.engine.api, "face_locations", return_value=[(40, 200, 200, 40)]):
            encoding = self.engine.encode_sample(frame)
            self.assertEqual(len(encoding), 128)
            self.engine.set_known_faces([Person("test", "Synthetic sample")], [encoding])
            result = self.engine.recognize(frame)[0]
        self.assertEqual(result.status, "known")
        self.assertLess(result.distance, 0.001)

    def test_real_model_can_run_in_a_separate_process(self):
        engine = ModelProcess(AppConfig())
        try:
            engine.set_cancel_event(threading.Event())
            result = engine.recognize(self.np.zeros((240, 320, 3), dtype=self.np.uint8))
            self.assertEqual(result, [])
        finally:
            engine.close()
        self.assertTrue(engine.closed)
