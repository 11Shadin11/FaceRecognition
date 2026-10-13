"""Small data objects shared by the application; no GUI or native imports."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


ENCODING_SIZE = 128


def validate_encoding(values) -> tuple[float, ...]:
    if len(values) != ENCODING_SIZE:
        raise ValueError("Описание лица должно содержать 128 чисел.")
    if any(isinstance(value, (bool, str, bytes)) for value in values):
        raise ValueError("Описание лица содержит нечисловые значения.")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Описание лица содержит NaN или бесконечность.")
    return result


@dataclass(frozen=True)
class Person:
    id: str
    name: str
    sample_count: int = 0


@dataclass(frozen=True)
class Sample:
    id: str
    person_id: str
    path: Path
    content_hash: str
    encoding: tuple[float, ...] | None
    model_version: str | None


@dataclass(frozen=True)
class FaceMatch:
    status: str
    person_id: str | None = None
    name: str | None = None
    distance: float | None = None
    # Original camera pixels, in (left, top, right, bottom) order.
    bounds: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class RecognitionResult:
    """One analyzed frame; its geometry and identities must travel together."""
    sequence: int
    captured_at: float
    frame_size: tuple[int, int]
    faces: tuple[FaceMatch, ...]


def select_match(people, distances, threshold=0.6, ambiguity_margin=0.04):
    """Rank people by their best sample, rejecting ties between identities."""
    if len(people) != len(distances):
        raise ValueError("Количество образцов и расстояний не совпадает.")
    nearest_by_person = {}
    for person, distance in zip(people, distances):
        distance = float(distance)
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Некорректное расстояние до лица.")
        previous = nearest_by_person.get(person.id)
        if previous is None or distance < previous[0]:
            nearest_by_person[person.id] = (distance, person)
    ranked = sorted(nearest_by_person.values(), key=lambda item: item[0])
    if not ranked:
        return FaceMatch("unknown")
    distance, person = ranked[0]
    if distance > threshold:
        return FaceMatch("unknown", distance=distance)
    if (len(ranked) > 1 and ranked[1][0] <= threshold
            and ranked[1][0] - distance < ambiguity_margin):
        return FaceMatch("ambiguous", distance=distance)
    return FaceMatch("known", person.id, person.name, distance)
