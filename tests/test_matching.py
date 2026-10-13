import math
import unittest

from face_app.config import AppConfig
from face_app.models import Person, select_match, validate_encoding


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.alice = Person("a", "Alice")
        self.bob = Person("b", "Bob")

    def test_nearest_match_wins_regardless_of_order(self):
        result = select_match([self.alice, self.bob], [0.59, 0.30])
        self.assertEqual(result.person_id, "b")
        self.assertAlmostEqual(result.distance, 0.30)

    def test_nearest_outside_threshold_is_unknown(self):
        self.assertEqual(select_match([self.alice], [0.61]).status, "unknown")

    def test_close_different_people_are_ambiguous(self):
        self.assertEqual(select_match([self.alice, self.bob], [0.30, 0.32]).status, "ambiguous")

    def test_multiple_samples_of_same_person_are_not_ambiguous(self):
        result = select_match([self.alice, self.alice, self.bob], [0.31, 0.30, 0.50])
        self.assertEqual(result.person_id, "a")

    def test_empty_database_is_unknown(self):
        self.assertEqual(select_match([], []).status, "unknown")

    def test_invalid_encodings_and_distances_are_rejected(self):
        for encoding in ([], [math.nan] * 128, [math.inf] * 128, [True] * 128, ["1"] * 128):
            with self.subTest(encoding=encoding[:1]), self.assertRaises(ValueError):
                validate_encoding(encoding)
        for distance in (math.nan, -1, math.inf):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                select_match([self.alice], [distance])

    def test_mismatched_index_is_rejected(self):
        with self.assertRaises(ValueError):
            select_match([self.alice], [])

    def test_invalid_settings_fail_before_camera_start(self):
        for kwargs in ({"camera_index": -1}, {"match_threshold": 1}, {"detection_width": 0},
                       {"recognition_interval": 0}, {"ambiguity_margin": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AppConfig(**kwargs)
