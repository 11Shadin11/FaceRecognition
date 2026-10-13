"""Offline evaluation on separately labeled photos; never enroll or write to DB."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

from .config import AppConfig
from .engine_process import ModelProcess
from .storage import FaceStore, MAX_IMAGE_BYTES, file_hash, normalize_name


def load_cases(manifest, people):
    manifest = Path(manifest)
    if manifest.is_symlink() or manifest.stat().st_size > 1024 * 1024:
        raise ValueError("Разметка должна быть обычным JSON-файлом до 1 МиБ.")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
        raise ValueError("Нужен список cases от 1 до 10000 проверочных фотографий.")
    names = {person.name.casefold(): person.name for person in people}
    cases = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("image"), str) or "expected" not in row:
            raise ValueError("Каждый пример должен содержать image и expected (имя либо null).")
        path = manifest.parent / row["image"]
        if path.resolve() in seen:
            raise ValueError("Один файл повторяется в проверочной выборке.")
        seen.add(path.resolve())
        expected = row["expected"]
        if expected is not None:
            if not isinstance(expected, str):
                raise ValueError("expected должен быть именем либо null.")
            key = normalize_name(expected).casefold()
            if key not in names:
                raise ValueError(f"Проверочный профиль отсутствует в базе: {expected}")
            expected = names[key]
        cases.append((path, expected))
    return cases


def evaluate(store, engine, cases):
    people = {person.id: person for person in store.people()}
    samples = store.samples()
    training_paths = {sample.path.resolve() for sample in samples}
    training_hashes = {sample.content_hash for sample in samples}
    known, encodings, skipped = [], [], []
    for sample in samples:
        try:
            if sample.path.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError("Слишком большое изображение базы")
            digest = file_hash(sample.path)
            encoding = sample.encoding
            if encoding is None or sample.content_hash != digest or sample.model_version != engine.model_version:
                encoding = engine.encode_image_file(sample.path)
            known.append(people[sample.person_id])
            encodings.append(encoding)
            training_hashes.add(digest)
        except (OSError, ValueError) as error:
            skipped.append({"sample_id": sample.id, "error": str(error)})
    available_names = {person.name for person in known}
    if any(expected is not None and expected not in available_names for _, expected in cases):
        raise ValueError("Для одного из проверочных профилей нет пригодных образцов в базе.")
    # Prevent direct resubstitution. A new capture session must still be arranged
    # by the user: hashes cannot detect every edited copy of a registration photo.
    for path, _ in cases:
        if path.resolve() in training_paths:
            raise ValueError("Проверочные фото не должны быть образцами регистрации.")
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_IMAGE_BYTES:
            if file_hash(path) in training_hashes:
                raise ValueError("Проверочное фото является копией образца регистрации.")
    engine.set_known_faces(known, encodings)
    counts = dict(known_total=0, known_correct=0, known_wrong_person=0, known_rejected=0,
                  unknown_total=0, unknown_false_accept=0, no_face=0, multiple_faces=0, errors=0)
    results = []
    for path, expected in cases:
        counts["known_total" if expected is not None else "unknown_total"] += 1
        started = time.perf_counter()
        row = {"image": str(path), "expected": expected}
        try:
            faces = engine.recognize(engine.read_image(path))
            accepted = [face.name for face in faces if face.status == "known"]
            if not faces:
                counts["no_face"] += 1
            if len(faces) > 1:
                counts["multiple_faces"] += 1
            row.update(status=faces[0].status if len(faces) == 1 else "no_face" if not faces else "multiple_faces",
                       predicted=accepted, distances=[face.distance for face in faces])
            if expected is None:
                counts["unknown_false_accept"] += int(bool(accepted))
            elif len(faces) != 1:
                counts["known_rejected"] += 1
            elif accepted == [expected]:
                counts["known_correct"] += 1
            elif accepted:
                counts["known_wrong_person"] += 1
            else:
                counts["known_rejected"] += 1
        except (OSError, ValueError) as error:
            counts["errors"] += 1
            row.update(status="error", error=str(error))
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        results.append(row)
    return {"counts": counts, "index_samples": len(encodings), "skipped_index_samples": skipped,
            "cases": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=AppConfig.data_dir)
    parser.add_argument("--threshold", type=float, default=AppConfig.match_threshold)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = replace(AppConfig(), data_dir=args.data_dir, match_threshold=args.threshold)
    store = FaceStore(config.data_dir, read_only=True)
    if args.output.resolve().is_relative_to(store.root):
        parser.error("Отчёт нужно сохранить вне каталога данных.")
    if args.output.exists():
        parser.error("Файл отчёта уже существует. Выберите новое имя.")
    cases = load_cases(args.manifest, store.people())
    engine = ModelProcess(config)
    try:
        engine.set_cancel_event(threading.Event())
        report = evaluate(store, engine, cases)
    finally:
        engine.close()
    report.update(threshold=config.match_threshold, ambiguity_margin=config.ambiguity_margin,
                  model_version=engine.model_version)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 1 if report["counts"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
