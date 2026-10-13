"""Application workflows: import, enrollment, index preparation, and live recognition."""
from __future__ import annotations

import logging
import time

from .recognition import InvalidSample
from .engine_process import ModelProcess
from .models import RecognitionResult
from .storage import MAX_IMAGE_BYTES, file_hash, legacy_images, normalize_name


class FaceService:
    def __init__(self, store, config, engine_factory=ModelProcess):
        self.store = store
        self.config = config
        self.engine_factory = engine_factory
        self._engine = None

    def engine(self, context):
        context.check_cancelled()
        if self._engine is None or getattr(self._engine, "closed", False):
            context.status("Загрузка модели распознавания…")
            self._engine = self.engine_factory(self.config)
        if hasattr(self._engine, "set_cancel_event"):
            self._engine.set_cancel_event(context.cancel)
        context.check_cancelled()
        return self._engine

    def close(self):
        if self._engine is not None and hasattr(self._engine, "close"):
            self._engine.close()

    @property
    def model_closed(self):
        return self._engine is None or getattr(self._engine, "closed", True)

    def prepare_index(self, context):
        engine = self.engine(context)
        people = {person.id: person for person in self.store.people()}
        samples = self.store.samples()
        known_people, encodings = [], []
        skipped = 0
        for number, sample in enumerate(samples, 1):
            context.check_cancelled()
            context.status(f"Подготовка базы: {number}/{len(samples)}", number / len(samples))
            try:
                if sample.path.stat().st_size > MAX_IMAGE_BYTES:
                    raise InvalidSample("Изображение превышает лимит размера.")
                content_hash = file_hash(sample.path)
                encoding = sample.encoding
                if (encoding is None or sample.model_version != engine.model_version
                        or sample.content_hash != content_hash):
                    encoding = engine.encode_image_file(sample.path)
                    context.check_cancelled()
                    self.store.update_encoding(sample.id, content_hash, encoding, engine.model_version)
                encodings.append(encoding)
                known_people.append(people[sample.person_id])
            except (OSError, ValueError) as error:
                skipped += 1
                logging.warning("Skipping stored sample %s: %s", sample.id, error)
                # Invalidate a previously accepted image if it no longer decodes safely.
                self.store.update_encoding(sample.id, sample.content_hash, None, None)
        engine.set_known_faces(known_people, encodings)
        return len(encodings), skipped

    def import_legacy(self, context):
        images = legacy_images(self.config.legacy_dir)
        if not images:
            return "В старой базе нет фотографий для импорта. Папки test и pig исключены."
        engine = self.engine(context)
        added = skipped = existing = 0
        for number, (name, path) in enumerate(images, 1):
            context.check_cancelled()
            context.status(f"Импорт старой базы: {number}/{len(images)}", number / len(images))
            try:
                if path.stat().st_size > MAX_IMAGE_BYTES:
                    raise InvalidSample("Изображение превышает лимит размера.")
                source_key = str(path.resolve()) + ":" + file_hash(path)
                if self.store.has_source(source_key):
                    existing += 1
                    continue
                encoding, image_bytes = engine.prepare_sample_file(path)
                context.check_cancelled()
                person = self.store.get_or_create_person(name)
                saved = self.store.add_sample(person.id, image_bytes, encoding,
                                             engine.model_version, source_key)
                added += int(saved)
                existing += int(not saved)
            except (OSError, ValueError) as error:
                skipped += 1
                logging.warning("Skipping legacy image %s: %s", path, error)
        return f"Импорт: добавлено {added}, уже есть {existing}, отклонено {skipped}. Исходные файлы сохранены."

    def import_photos(self, context, name, paths):
        name = normalize_name(name)
        if len(paths) > self.config.max_samples_per_import:
            raise ValueError(f"Выберите не более {self.config.max_samples_per_import} фотографий за один импорт.")
        engine = self.engine(context)
        added = rejected = existing = 0
        last_error = ""
        for number, path in enumerate(paths, 1):
            context.check_cancelled()
            context.status(f"Импорт фото: {number}/{len(paths)}", number / len(paths))
            try:
                encoding, image_bytes = engine.prepare_sample_file(path)
                context.check_cancelled()
                person = self.store.get_or_create_person(name)
                saved = self.store.add_sample(person.id, image_bytes, encoding, engine.model_version)
                added += int(saved)
                existing += int(not saved)
            except (OSError, ValueError) as error:
                rejected += 1
                last_error = str(error)
        if not added and not existing:
            raise ValueError(f"Подходящих фото нет. Отклонено: {rejected}. {last_error}")
        return f"Фото: добавлено {added}, уже есть {existing}, отклонено {rejected}. Исходные файлы сохранены."

    def collect_samples(self, context, name, source):
        person = self.store.get_or_create_person(name)
        engine = self.engine(context)
        added = rejected = duplicates = 0
        deadline = None
        last_sample_time = 0.0
        context.status("Открытие источника. В кадре должен быть только выбранный человек.")
        with context.capture(source, self.config) as capture:
            iterator = (capture.live_frames(context.cancel) if isinstance(source, int)
                        else capture.file_frames(context.cancel))
            for frame in iterator:
                context.check_cancelled()
                context.preview(frame)
                if isinstance(source, int):
                    # Start the enrollment clock after the first frame, not model loading.
                    if deadline is None:
                        deadline = time.monotonic() + self.config.enrollment_seconds
                    if time.monotonic() >= deadline:
                        break
                    if frame.captured_at - last_sample_time < self.config.sample_interval:
                        continue
                last_sample_time = frame.captured_at
                progress = (1 - (deadline - time.monotonic()) / self.config.enrollment_seconds
                            if isinstance(source, int) else None)
                try:
                    encoding, image_bytes = engine.prepare_sample(frame.image)
                    context.check_cancelled()
                    saved = self.store.add_sample(person.id, image_bytes,
                                                 encoding, engine.model_version)
                    added += int(saved)
                    duplicates += int(not saved)
                    context.status(f"Сохранено: {added}. Отклонено: {rejected}. Смотрите в камеру.", progress)
                except InvalidSample as error:
                    rejected += 1
                    context.status(str(error), progress)
                if added + duplicates >= self.config.max_samples_per_import:
                    break
        context.check_cancelled()
        if added + duplicates == 0:
            raise ValueError(f"Не сохранено ни одного образца. Отклонено кадров: {rejected}. "
                             "Нужно одно достаточно крупное и чёткое лицо при хорошем освещении.")
        return f"Готово: добавлено {added}, повторов {duplicates}, отклонено {rejected}."

    def recognize_camera(self, context):
        accepted, skipped = self.prepare_index(context)
        if not accepted:
            raise ValueError("В базе нет подходящих образцов. Добавьте человека или импортируйте фотографии.")
        engine = self.engine(context)
        context.status(f"Открытие камеры. Образцов: {accepted}; отклонено: {skipped}.")
        with context.capture(self.config.camera_index, self.config) as capture:
            for frame in capture.live_frames(context.cancel):
                context.check_cancelled()
                started = time.monotonic()
                matches = engine.recognize(frame.image)
                context.check_cancelled()
                labels = []
                for match in matches:
                    if match.status == "known":
                        labels.append(f"{match.name} (расстояние {match.distance:.3f})")
                    elif match.status == "ambiguous":
                        labels.append("Неоднозначное совпадение")
                    else:
                        labels.append("Неизвестный человек")
                elapsed = time.monotonic() - started
                message = "; ".join(labels) if labels else "Лицо не обнаружено"
                result = RecognitionResult(frame.sequence, frame.captured_at,
                                           (frame.image.shape[1], frame.image.shape[0]), tuple(matches))
                context.status(f"{message} · обработка {elapsed * 1000:.0f} мс",
                               recognized_at=frame.captured_at, recognition=result)
                context.cancel.wait(max(0, self.config.recognition_interval - elapsed))
