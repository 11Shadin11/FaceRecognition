"""Face inference and matching, independent of windows and camera ownership."""
from __future__ import annotations

from dataclasses import replace
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

from .models import select_match, validate_encoding


class InvalidSample(ValueError):
    """An image cannot safely be enrolled under a single person's name."""


class FaceEngine:
    def __init__(self, config):
        # Native libraries and model loading belong to the model process, not Tk.
        import cv2
        import face_recognition
        import numpy as np

        self.cv = cv2
        self.api = face_recognition
        self.np = np
        self.config = config
        self.cv.setNumThreads(1)
        self.model_version = (
            f"rgb-hog-small-v1:width={config.detection_width}:"
            f"min-face={config.min_face_size}:sharpness={config.min_sharpness}:"
            + ":".join(version(package) for package in
                       ("face-recognition", "face-recognition-models", "dlib", "numpy", "Pillow"))
        )
        self.people = []
        self.matrix = np.empty((0, 128), dtype=np.float32)

    def set_known_faces(self, people, encodings):
        if len(people) != len(encodings):
            raise ValueError("Некорректный индекс лиц.")
        self.people = list(people)
        self.matrix = self.np.asarray([validate_encoding(e) for e in encodings],
                                      dtype=self.np.float32).reshape((-1, 128))

    def _rgb_for_detection(self, frame):
        height, width = frame.shape[:2]
        if width > self.config.detection_width:
            new_height = max(1, round(height * self.config.detection_width / width))
            frame = self.cv.resize(frame, (self.config.detection_width, new_height))
        return self.np.ascontiguousarray(self.cv.cvtColor(frame, self.cv.COLOR_BGR2RGB))

    def read_image(self, path):
        """Check decoded dimensions before materializing a potentially large image."""
        from PIL import Image, ImageOps
        from .storage import MAX_IMAGE_BYTES

        path = Path(path)
        if path.is_symlink() or path.stat().st_size > MAX_IMAGE_BYTES:
            raise InvalidSample("Файл является ссылкой или превышает 20 МиБ.")
        try:
            with Image.open(BytesIO(path.read_bytes())) as image:
                if image.width * image.height > 16_000_000:
                    raise InvalidSample("Разрешение изображения превышает 16 мегапикселей.")
                rgb = self.np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        except Image.DecompressionBombError as error:
            raise InvalidSample("Разрешение изображения превышает безопасный предел.") from error
        return self.cv.cvtColor(rgb, self.cv.COLOR_RGB2BGR)

    def encode_sample(self, frame):
        rgb = self._rgb_for_detection(frame)
        locations = self.api.face_locations(rgb, number_of_times_to_upsample=0, model="hog")
        if len(locations) != 1:
            raise InvalidSample("Нужно ровно одно лицо в кадре.")
        top, right, bottom, left = locations[0]
        scale = frame.shape[1] / rgb.shape[1]
        if min(right - left, bottom - top) * scale < self.config.min_face_size:
            raise InvalidSample("Подойдите ближе к камере: лицо слишком маленькое.")
        face = rgb[top:bottom, left:right]
        if not face.size:
            raise InvalidSample("Лицо выходит за границы кадра.")
        gray = self.cv.cvtColor(face, self.cv.COLOR_RGB2GRAY)
        if self.cv.Laplacian(gray, self.cv.CV_64F).var() < self.config.min_sharpness:
            raise InvalidSample("Кадр размыт: удерживайте голову неподвижно.")
        if not 35 <= float(gray.mean()) <= 220:
            raise InvalidSample("Измените освещение: лицо слишком тёмное или светлое.")
        encodings = self.api.face_encodings(rgb, locations, model="small")
        if len(encodings) != 1:
            raise InvalidSample("Не удалось получить описание лица.")
        return validate_encoding(encodings[0])

    def image_bytes(self, frame):
        ok, encoded = self.cv.imencode(".jpg", frame)
        if not ok:
            raise InvalidSample("Не удалось сохранить изображение.")
        return encoded.tobytes()

    def encode_image_file(self, path):
        return self.encode_sample(self.read_image(path))

    def prepare_sample(self, frame):
        return self.encode_sample(frame), self.image_bytes(frame)

    def prepare_sample_file(self, path):
        return self.prepare_sample(self.read_image(path))

    def recognize(self, frame):
        rgb = self._rgb_for_detection(frame)
        locations = self.api.face_locations(rgb, number_of_times_to_upsample=0, model="hog")
        encodings = self.api.face_encodings(rgb, locations, model="small")
        matches = []
        height, width = frame.shape[:2]
        scale_x, scale_y = width / rgb.shape[1], height / rgb.shape[0]
        for location, encoding in zip(locations, encodings):
            query = self.np.asarray(validate_encoding(encoding), dtype=self.np.float32)
            distances = self.np.linalg.norm(self.matrix - query, axis=1)
            match = select_match(self.people, distances,
                                 self.config.match_threshold, self.config.ambiguity_margin)
            top, right, bottom, left = location
            bounds = (max(0, min(width, round(left * scale_x))),
                      max(0, min(height, round(top * scale_y))),
                      max(0, min(width, round(right * scale_x))),
                      max(0, min(height, round(bottom * scale_y))))
            matches.append(replace(match, bounds=bounds))
        return matches
