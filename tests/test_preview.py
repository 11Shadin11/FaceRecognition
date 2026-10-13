import unittest

import numpy as np

from face_app.preview import make_preview


class PreviewTests(unittest.TestCase):
    def test_red_and_blue_are_not_swapped_and_source_is_unchanged(self):
        frame = np.array([[[255, 0, 0], [0, 0, 255]], [[0, 255, 0], [25, 50, 75]]], dtype=np.uint8)
        before = frame.copy()
        image = make_preview(frame, (2, 2))
        self.assertEqual(list(image.getdata()), [(0, 0, 255), (255, 0, 0), (0, 255, 0), (75, 50, 25)])
        np.testing.assert_array_equal(frame, before)

    def test_cropped_noncontiguous_frame_keeps_orientation_and_aspect_ratio(self):
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        frame[0, 0] = (255, 0, 0)
        cropped = frame[::2, ::2]
        self.assertFalse(cropped.flags.c_contiguous)
        image = make_preview(cropped, (4, 2))
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(image.getpixel((3, 1)), (0, 0, 0))
        self.assertEqual(make_preview(cropped, (8, 8)).size, (8, 4))
