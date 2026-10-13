from pathlib import Path
import sqlite3
import tempfile
import unittest

from face_app.storage import FaceStore, legacy_images


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = FaceStore(self.root / "data")
        self.person = self.store.get_or_create_person("Alice")

    def add_sample(self, payload=b"photo", source=None):
        return self.store.add_sample(self.person.id, payload, [0.1] * 128, "v1", source)

    def test_names_are_display_data_not_paths(self):
        person = self.store.get_or_create_person("../outside")
        self.store.add_sample(person.id, b"photo", [0.0] * 128, "v1")
        sample = self.store.samples()[0]
        self.assertTrue(sample.path.is_relative_to(self.store.images_dir))
        self.assertEqual(sample.path.parent.name, person.id)
        self.assertFalse((self.root / "outside").exists())

    def test_person_names_are_normalized(self):
        same = self.store.get_or_create_person("  ALICE  ")
        self.assertEqual(same.id, self.person.id)
        for name in ("", "a" * 81, "a\n"):
            # Leading/trailing whitespace is trimmed; embedded controls are forbidden.
            invalid = "a\nb" if name == "a\n" else name
            with self.assertRaises(ValueError):
                self.store.get_or_create_person(invalid)

    def test_samples_do_not_overwrite_and_duplicates_are_ignored(self):
        self.assertTrue(self.add_sample(b"first"))
        self.assertTrue(self.add_sample(b"second"))
        self.assertFalse(self.add_sample(b"first"))
        samples = self.store.samples()
        self.assertEqual(len(samples), 2)
        self.assertEqual({s.path.read_bytes() for s in samples}, {b"first", b"second"})

    def test_source_import_is_idempotent(self):
        self.assertTrue(self.add_sample(source="legacy/source:hash"))
        self.assertFalse(self.add_sample(b"changed-output", "legacy/source:hash"))
        self.assertEqual(len(self.store.samples()), 1)

    def test_corrupt_json_is_treated_as_invalid_cache(self):
        self.add_sample()
        with self.store.connection() as db:
            db.execute("UPDATE samples SET encoding_json = ?", ('{"bad":"cache"}',))
        self.assertIsNone(self.store.samples()[0].encoding)

    def test_insert_failure_removes_new_image_and_preserves_existing(self):
        self.add_sample(b"existing")
        with self.store.connection() as db:
            db.execute("CREATE TRIGGER reject_sample BEFORE INSERT ON samples "
                       "BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.add_sample(b"new")
        self.assertEqual([s.path.read_bytes() for s in self.store.samples()], [b"existing"])
        self.assertEqual(len(list(self.store.images_dir.rglob("*.jpg"))), 1)

    def test_symlink_and_database_path_traversal_are_rejected(self):
        external = self.root / "outside"
        external.mkdir()
        link = self.root / "linked-data"
        link.symlink_to(external, target_is_directory=True)
        with self.assertRaises(ValueError):
            FaceStore(link)
        self.add_sample()
        with self.store.connection() as db:
            db.execute("UPDATE samples SET filename = '../outside.jpg'")
        with self.assertRaises(ValueError):
            self.store.samples()

    def test_invalid_embedding_does_not_write_image(self):
        with self.assertRaises(ValueError):
            self.store.add_sample(self.person.id, b"image", [], "v1")
        self.assertEqual(self.store.samples(), [])
        self.assertEqual(list(self.store.images_dir.rglob("*.jpg")), [])

    def test_legacy_scan_excludes_test_data_video_pickle_and_links(self):
        legacy = self.root / "faces"
        for name in ("Alice", "test", "tests", "pig", ".hidden"):
            folder = legacy / name
            folder.mkdir(parents=True)
            (folder / "photo.jpg").write_bytes(b"photo")
        (legacy / "Alice" / "video.avi").write_bytes(b"video")
        (legacy / "encoded_faces.pkl").write_bytes(b"never load")
        (legacy / "Alice" / "link.jpg").symlink_to(legacy / "Alice" / "photo.jpg")
        self.assertEqual(legacy_images(legacy), [("Alice", legacy / "Alice" / "photo.jpg")])

    def test_unknown_database_version_is_not_overwritten(self):
        with self.store.connection() as db:
            db.execute("PRAGMA user_version = 99")
        with self.assertRaises(ValueError):
            FaceStore(self.store.root)
