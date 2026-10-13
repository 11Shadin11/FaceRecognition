import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from face_app.evaluation import evaluate, load_cases
from face_app.models import FaceMatch
from face_app.storage import FaceStore


class EvaluationEngine:
    model_version = "v2"

    def __init__(self):
        self.responses = {}
        self.encoded = 0

    def encode_image_file(self, path):
        self.encoded += 1
        return (0.2,) * 128

    def set_known_faces(self, people, encodings):
        self.index_size = len(encodings)

    def read_image(self, path):
        return path.read_bytes()

    def recognize(self, image):
        return self.responses[image]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = FaceStore(self.root / "data")
        self.alice = self.store.get_or_create_person("Alice")
        self.store.add_sample(self.alice.id, b"registration", [0.1] * 128, "old-model")
        self.engine = EvaluationEngine()

    def case(self, number, expected, faces):
        path = self.root / f"test-{number}.jpg"
        payload = f"case-{number}".encode()
        path.write_bytes(payload)
        self.engine.responses[payload] = faces
        return path, expected

    def test_counts_distinguish_wrong_identity_rejection_and_unknown_acceptance(self):
        alice = FaceMatch("known", self.alice.id, "Alice", .2)
        bob = FaceMatch("known", "other", "Bob", .2)
        unknown = FaceMatch("unknown", distance=.7)
        ambiguous = FaceMatch("ambiguous", distance=.4)
        cases = [self.case(1, "Alice", [alice]), self.case(2, "Alice", [bob]),
                 self.case(3, "Alice", [ambiguous]), self.case(4, None, [alice]),
                 self.case(5, None, [unknown]), self.case(6, "Alice", []),
                 self.case(7, "Alice", [alice, unknown])]
        before = self.store.database.read_bytes()
        readonly = FaceStore(self.store.root, read_only=True)
        report = evaluate(readonly, self.engine, cases)
        self.assertEqual(report["counts"], {"known_total": 5, "known_correct": 1,
                         "known_wrong_person": 1, "known_rejected": 3,
                         "unknown_total": 2, "unknown_false_accept": 1,
                         "no_face": 1, "multiple_faces": 1, "errors": 0})
        self.assertEqual(self.engine.encoded, 1)
        self.assertEqual(readonly.samples()[0].model_version, "old-model")
        self.assertEqual(self.store.database.read_bytes(), before)
        with self.assertRaises(sqlite3.OperationalError):
            readonly.rename_person(self.alice.id, "Changed")

    def test_direct_registration_and_copied_registration_are_rejected(self):
        sample = self.store.samples()[0]
        with self.assertRaises(ValueError):
            evaluate(self.store, self.engine, [(sample.path, "Alice")])
        copied = self.root / "copied.jpg"
        copied.write_bytes(sample.path.read_bytes())
        with self.assertRaises(ValueError):
            evaluate(self.store, self.engine, [(copied, "Alice")])

    def test_missing_test_photo_is_reported_explicitly(self):
        report = evaluate(self.store, self.engine, [(self.root / "missing.jpg", None)])
        self.assertEqual(report["counts"]["errors"], 1)
        self.assertEqual(report["cases"][0]["status"], "error")

    def test_manifest_validates_labels_duplicates_and_relative_paths(self):
        manifest = self.root / "cases.json"
        manifest.write_text(json.dumps({"cases": [{"image": "fresh.jpg", "expected": "  ALICE  "},
                                                {"image": "stranger.jpg", "expected": None}]}))
        cases = load_cases(manifest, self.store.people())
        self.assertEqual(cases, [(self.root / "fresh.jpg", "Alice"), (self.root / "stranger.jpg", None)])
        for rows in ([], [{"image": "fresh.jpg", "expected": "typo"}],
                     [{"image": "fresh.jpg", "expected": None}] * 2,
                     [{"image": "fresh.jpg", "expected": 123}]):
            manifest.write_text(json.dumps({"cases": rows}))
            with self.assertRaises(ValueError):
                load_cases(manifest, self.store.people())

    def test_readonly_store_does_not_create_missing_database(self):
        missing = self.root / "missing-data"
        with self.assertRaises(sqlite3.OperationalError):
            FaceStore(missing, read_only=True)
        self.assertFalse(missing.exists())
