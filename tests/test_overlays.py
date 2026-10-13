"""Overlay geometry, freshness and bounded Canvas allocation without a display."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from face_app.models import FaceMatch, RecognitionResult
from face_app.overlays import FaceOverlay, project_bounds


class FakeFont:
    def metrics(self, name):
        return 16

    def measure(self, text):
        return len(text) * 8


class FakeCanvas:
    def __init__(self):
        self.items = {}
        self.updates = 0

    def create_rectangle(self, *coords, **options):
        return self._create(coords, options)

    def create_text(self, *coords, **options):
        return self._create(coords, options)

    def _create(self, coords, options):
        number = len(self.items) + 1
        self.items[number] = {"coords": coords, **options}
        return number

    def coords(self, item, *coords):
        self.items[item]["coords"] = coords
        self.updates += 1

    def itemconfigure(self, item, **options):
        self.items[item].update(options)
        self.updates += 1


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.canvas = FakeCanvas()
        self.overlay = FaceOverlay(self.canvas, FakeFont())
        self.frame = SimpleNamespace(sequence=11, captured_at=10.1,
                                     image=SimpleNamespace(shape=(480, 640, 3)))
        self.known = FaceMatch("known", "a", "Алиса", .2, (160, 120, 480, 360))
        self.result = RecognitionResult(10, 10, (640, 480), (self.known,))
        # 4:3 camera image fitted into a wider Canvas.
        self.rect = (80, 0, 640, 480)

    def test_projection_includes_letterboxing_and_resize(self):
        self.assertEqual(project_bounds(self.known.bounds, (640, 480), (100, 50, 320, 240)),
                         (180, 110, 340, 230))

    def test_multiple_faces_have_own_labels_and_colors(self):
        unknown = FaceMatch("unknown", bounds=(0, 0, 100, 100))
        ambiguous = FaceMatch("ambiguous", bounds=(500, 360, 640, 480))
        result = replace(self.result, faces=(self.known, unknown, ambiguous))
        self.overlay.render(result, self.frame, self.rect, 10.2)
        groups = self.overlay.items
        self.assertEqual([self.canvas.items[group[2]]["text"] for group in groups],
                         ["Алиса", "Неизвестный человек", "Неоднозначное совпадение"])
        self.assertEqual([self.canvas.items[group[0]]["outline"] for group in groups],
                         ["#44d07b", "#ff6b6b", "#f4ba54"])

    def test_stale_absent_future_or_wrong_resolution_results_are_hidden(self):
        for result, now in ((self.result, 10.6), (replace(self.result, faces=()), 10.2),
                            (replace(self.result, sequence=12), 10.2),
                            (replace(self.result, frame_size=(1280, 720)), 10.2), (None, 10.2)):
            with self.subTest(result=result):
                self.overlay.render(self.result, self.frame, self.rect, 10.2)
                self.overlay.render(result, self.frame, self.rect, now)
                self.assertTrue(all(item["state"] == "hidden" for item in self.canvas.items.values()))

    def test_canvas_items_are_reused_and_unchanged_results_do_no_work(self):
        self.overlay.render(self.result, self.frame, self.rect, 10.2)
        updates = self.canvas.updates
        self.overlay.render(self.result, self.frame, self.rect, 10.3)
        self.assertEqual(self.canvas.updates, updates)
        for sequence in range(500):
            result = replace(self.result, sequence=sequence, captured_at=sequence)
            frame = SimpleNamespace(sequence=sequence + 1, captured_at=sequence + .1, image=self.frame.image)
            self.overlay.render(result, frame, self.rect, sequence + .2)
        self.assertEqual(len(self.canvas.items), 3)

    def test_long_names_and_edge_badges_stay_inside_displayed_image(self):
        face = replace(self.known, name="Очень длинное имя " * 100, bounds=(600, 0, 640, 80))
        self.overlay.render(replace(self.result, faces=(face,)), self.frame, (80, 20, 200, 150), 10.2)
        box, badge, caption = self.overlay.items[0]
        left, top, right, bottom = self.canvas.items[badge]["coords"]
        self.assertTrue(80 <= left < right <= 280)
        self.assertTrue(20 <= top < bottom <= 170)
        self.assertTrue(self.canvas.items[caption]["text"].endswith("…"))


if __name__ == "__main__":
    unittest.main()
