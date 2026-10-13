import threading
import unittest
from unittest.mock import patch

from face_app.tasks import TaskRunner


class TaskTests(unittest.TestCase):
    def test_single_worker_and_cooperative_cancel(self):
        runner = TaskRunner()
        started = threading.Event()

        def job(context):
            started.set()
            context.cancel.wait(2)
            context.check_cancelled()

        runner.start(job)
        self.assertTrue(started.wait(1))
        with self.assertRaises(RuntimeError):
            runner.start(job)
        runner.cancel()
        runner.thread.join(timeout=2)
        self.assertFalse(runner.busy)
        self.assertEqual(runner.events.get_nowait()[0], "cancelled")

    def test_failures_are_reported_and_next_task_can_run(self):
        runner = TaskRunner()

        def broken(context):
            raise ValueError("broken source")

        with patch("face_app.tasks.logging.exception"):
            runner.start(broken)
            runner.thread.join(timeout=1)
        self.assertEqual(runner.events.get_nowait(), ("error", "broken source"))
        runner.start(lambda context: "recovered")
        runner.thread.join(timeout=1)
        self.assertEqual(runner.events.get_nowait(), ("done", "recovered"))

    def test_terminal_event_must_be_consumed_before_next_task(self):
        runner = TaskRunner()
        runner.start(lambda context: "finished")
        runner.thread.join(timeout=1)
        with self.assertRaises(RuntimeError):
            runner.start(lambda context: "next")
        runner.events.get_nowait()
        self.assertTrue(runner.can_start)

    def test_importing_launchers_and_demos_has_no_camera_or_tk_side_effects(self):
        import importlib

        with patch("tkinter.Tk", side_effect=AssertionError("unexpected window")), \
                patch("face_app.capture.open_capture", side_effect=AssertionError("unexpected camera")):
            for name in ("faces_recognition", "dialog_window_module", "generate_data_set",
                         "face_landmarks", "pig_nose"):
                importlib.reload(importlib.import_module(name))
