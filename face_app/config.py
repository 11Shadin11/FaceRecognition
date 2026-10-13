from dataclasses import dataclass
from pathlib import Path
import math


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    legacy_dir: Path = PROJECT_ROOT / "faces"
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    detection_width: int = 480
    recognition_interval: float = 0.2
    inference_timeout: float = 30.0
    match_threshold: float = 0.6
    ambiguity_margin: float = 0.04
    enrollment_seconds: float = 8.0
    sample_interval: float = 0.75
    max_samples_per_import: int = 100
    max_video_seconds: float = 300.0
    min_face_size: int = 70
    min_sharpness: float = 40.0

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (
            self.match_threshold, self.ambiguity_margin, self.recognition_interval,
            self.enrollment_seconds, self.sample_interval, self.max_video_seconds,
            self.min_sharpness,
            self.inference_timeout,
        )):
            raise ValueError("Настройки должны содержать конечные числа.")
        if self.camera_index < 0:
            raise ValueError("Номер камеры должен быть неотрицательным.")
        if not 0 < self.match_threshold < 1 or self.ambiguity_margin < 0:
            raise ValueError("Некорректный порог распознавания.")
        if min(self.camera_width, self.camera_height, self.detection_width,
               self.min_face_size, self.max_samples_per_import) <= 0:
            raise ValueError("Размеры и лимиты должны быть положительными.")
        if min(self.recognition_interval, self.enrollment_seconds,
               self.sample_interval, self.max_video_seconds, self.inference_timeout) <= 0:
            raise ValueError("Интервалы должны быть положительными.")
