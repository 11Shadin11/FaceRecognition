import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from face_app.backup import create_backup
from face_app.preferences import Preferences
from face_app.storage import FaceStore
from face_app.tasks import Cancelled, TaskContext, TaskRunner


class ManagementTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = FaceStore(self.root / "data")
        self.alice = self.store.get_or_create_person("Alice")
        self.bob = self.store.get_or_create_person("Bob")
        for person, payload in ((self.alice, b"alice-1"), (self.alice, b"alice-2"), (self.bob, b"bob")):
            self.store.add_sample(person.id, payload, [0.1] * 128, "v1")
        self.runner = TaskRunner()
        self.context = TaskContext(self.runner, self.runner.cancel_event)

    def test_rename_preserves_id_photos_and_encodings_and_rejects_collision(self):
        before = self.store.samples(self.alice.id)
        self.store.rename_person(self.alice.id, "  Алиса  ")
        self.assertEqual(before, self.store.samples(self.alice.id))
        person = next(p for p in self.store.people() if p.id == self.alice.id)
        self.assertEqual((person.name, person.sample_count), ("Алиса", 2))
        with self.assertRaises(ValueError):
            self.store.rename_person(self.alice.id, "BOB")
        self.assertEqual(len(self.store.people()), 2)

    def test_deleting_one_sample_preserves_other_people_and_photos(self):
        selected = self.store.samples(self.alice.id)[0]
        self.store.delete_sample(selected.id)
        self.assertFalse(selected.path.exists())
        self.assertEqual(len(self.store.samples(self.alice.id)), 1)
        self.assertEqual(self.store.samples(self.bob.id)[0].path.read_bytes(), b"bob")
        self.assertEqual(list(self.store.root.glob(".deleting-*")), [])

    def test_deleting_profile_removes_only_its_photos(self):
        alice_paths = [sample.path for sample in self.store.samples(self.alice.id)]
        self.store.delete_person(self.alice.id)
        self.assertEqual(self.store.samples(self.alice.id), [])
        self.assertTrue(all(not path.exists() for path in alice_paths))
        self.assertEqual([p.id for p in self.store.people()], [self.bob.id])
        self.assertEqual(self.store.samples(self.bob.id)[0].path.read_bytes(), b"bob")

    def test_failed_database_delete_restores_all_staged_images(self):
        before = self.store.samples()
        with self.store.connection() as db:
            db.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON people "
                       "BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.delete_person(self.alice.id)
        self.assertEqual(self.store.samples(), before)
        self.assertTrue(all(sample.path.is_file() for sample in before))
        self.assertEqual(list(self.store.root.glob(".deleting-*")), [])

    def test_partial_file_move_failure_restores_previous_images(self):
        samples = self.store.samples(self.alice.id)
        original = os.replace
        calls = 0

        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk failure")
            return original(source, destination)

        with patch("face_app.storage.os.replace", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.store.delete_person(self.alice.id)
        self.assertEqual(self.store.samples(self.alice.id), samples)
        self.assertTrue(all(sample.path.is_file() for sample in samples))

    def test_symlink_is_not_deleted_and_missing_photo_can_be_removed(self):
        sample = self.store.samples(self.alice.id)[0]
        external = self.root / "external.jpg"
        external.write_bytes(b"outside")
        sample.path.unlink()
        sample.path.symlink_to(external)
        with self.assertRaises(ValueError):
            self.store.delete_sample(sample.id)
        self.assertEqual(external.read_bytes(), b"outside")
        sample.path.unlink()
        self.store.delete_sample(sample.id)
        self.assertEqual(len(self.store.samples(self.alice.id)), 1)

    def test_directory_in_place_of_photo_is_not_moved_or_erased(self):
        sample = self.store.samples(self.alice.id)[0]
        sample.path.unlink()
        sample.path.mkdir()
        (sample.path / "keep.txt").write_text("preserve")
        with self.assertRaises(ValueError):
            self.store.delete_person(self.alice.id)
        self.assertEqual((sample.path / "keep.txt").read_text(), "preserve")
        self.assertEqual(len(self.store.samples(self.alice.id)), 2)

    def test_backup_restores_database_and_exact_photos_without_extra_files(self):
        (self.store.root / "unrelated.txt").write_text("exclude me")
        archive = self.root / "backup.zip"
        create_backup(self.store, archive, self.context)
        self.assertEqual(archive.stat().st_mode & 0o777, 0o600)
        with zipfile.ZipFile(archive) as backup:
            self.assertIsNone(backup.testzip())
            manifest = json.loads(backup.read("manifest.json"))
            self.assertEqual((manifest["people"], manifest["samples"]), (2, 3))
            self.assertNotIn("data/unrelated.txt", backup.namelist())
            for name, digest in manifest["images"].items():
                self.assertEqual(hashlib.sha256(backup.read(name)).hexdigest(), digest)
            backup.extractall(self.root / "restored")  # Only this test's own archive.
        restored = FaceStore(self.root / "restored" / "data")
        self.assertEqual(restored.people(), self.store.people())
        self.assertEqual({s.path.read_bytes() for s in restored.samples()}, {b"alice-1", b"alice-2", b"bob"})
        with restored.connection() as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_backup_does_not_overwrite_existing_files_or_write_into_data(self):
        archive = self.root / "backup.zip"
        archive.write_bytes(b"previous")
        with self.assertRaises(ValueError):
            create_backup(self.store, archive, self.context)
        self.assertEqual(archive.read_bytes(), b"previous")
        with self.assertRaises(ValueError):
            create_backup(self.store, self.store.root / "copy.zip", self.context)

    def test_failed_or_cancelled_backup_never_publishes_partial_archive(self):
        archive = self.root / "backup.zip"
        sample = self.store.samples()[0]
        sample.path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            create_backup(self.store, archive, self.context)
        self.assertFalse(archive.exists())
        self.assertEqual(list(self.root.glob(".backup-*")), [])
        self.runner.cancel_event.set()
        with self.assertRaises(Cancelled):
            create_backup(self.store, archive, self.context)
        self.assertFalse(archive.exists())

    def test_backup_cancelled_mid_copy_cleans_temporary_files(self):
        archive = self.root / "backup.zip"
        original = self.context.status

        def cancel_after_first(*args):
            original(*args)
            self.runner.cancel_event.set()

        with patch.object(self.context, "status", side_effect=cancel_after_first):
            with self.assertRaises(Cancelled):
                create_backup(self.store, archive, self.context)
        self.assertFalse(archive.exists())
        self.assertEqual(len(self.store.samples()), 3)
        self.assertEqual(list(self.root.glob(".backup-*")), [])

    def test_backup_snapshot_blocks_concurrent_image_writes_until_complete(self):
        copying = threading.Event()
        release = threading.Event()
        written = threading.Event()
        errors = []
        archive = self.root / "backup.zip"

        def status(*args):
            copying.set()
            if not release.wait(3):
                raise RuntimeError("test synchronization timeout")

        def backup():
            try:
                create_backup(self.store, archive, self.context)
            except Exception as error:
                errors.append(error)

        def write():
            try:
                self.store.add_sample(self.bob.id, b"concurrent", [0.1] * 128, "v1")
                written.set()
            except Exception as error:
                errors.append(error)

        with patch.object(self.context, "status", side_effect=status):
            worker = threading.Thread(target=backup)
            worker.start()
            try:
                self.assertTrue(copying.wait(2))
                writer = threading.Thread(target=write)
                writer.start()
                self.assertFalse(written.wait(.1))
            finally:
                release.set()
                worker.join(3)
                if 'writer' in locals():
                    writer.join(3)
        self.assertEqual(errors, [])
        self.assertTrue(written.is_set())
        self.assertEqual(len(self.store.samples()), 4)
        with zipfile.ZipFile(archive) as output:
            self.assertEqual(json.loads(output.read("manifest.json"))["samples"], 3)


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.preferences = Preferences(temporary.name)

    def test_camera_is_remembered_and_corruption_falls_back(self):
        self.assertEqual(self.preferences.camera(), 0)
        self.preferences.save_camera(1)
        self.assertEqual(self.preferences.camera(), 1)
        for payload in ('broken', '[1]', '{"camera_index": true}', '{"camera_index": -1}', '{"camera_index": 100}'):
            self.preferences.path.write_text(payload)
            self.assertEqual(self.preferences.camera(default=2), 2)

    def test_invalid_camera_and_symlinks_do_not_write(self):
        for value in (-1, 100, True, 1.2):
            with self.assertRaises(ValueError):
                self.preferences.save_camera(value)
        self.assertFalse(self.preferences.path.exists())
        target = self.preferences.path.parent / "other.json"
        target.write_text("untouched")
        self.preferences.path.symlink_to(target)
        with self.assertRaises(ValueError):
            self.preferences.save_camera(1)
        self.assertEqual(target.read_text(), "untouched")
        self.assertEqual(self.preferences.camera(), 0)
