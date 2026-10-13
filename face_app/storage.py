"""Private local images and validated JSON embeddings in transactional SQLite.

Names never become paths. Legacy photos can be imported, but pickle is never read.
Each operation opens its own connection so GUI and worker threads do not share one.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unicodedata
import uuid

from .models import Person, Sample, validate_encoding


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(name):
    name = unicodedata.normalize("NFKC", name).strip()
    if not name or len(name) > 80 or any(unicodedata.category(c).startswith("C") for c in name):
        raise ValueError("Введите имя от 1 до 80 символов без управляющих знаков.")
    return name


class FaceStore:
    def __init__(self, root, *, read_only=False):
        self.read_only = read_only
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise ValueError("Каталог данных не должен быть символической ссылкой.")
        if not read_only:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = self.root.resolve()
        self.images_dir = self.root / "profiles"
        if self.images_dir.is_symlink():
            raise ValueError("Каталог изображений не должен быть ссылкой.")
        if not read_only:
            self.images_dir.mkdir(mode=0o700, exist_ok=True)
        self.database = self.root / "faces.sqlite3"
        if self.database.is_symlink():
            raise ValueError("Файл базы не должен быть символической ссылкой.")
        if read_only:
            with self.connection() as db:
                if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise ValueError("Для проверки нужна существующая база версии 1.")
            return
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Неизвестная версия базы. Необходима миграция.")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    name_key TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS samples (
                    id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL REFERENCES people(id),
                    filename TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL,
                    encoding_json TEXT,
                    model_version TEXT,
                    source_key TEXT UNIQUE,
                    UNIQUE(person_id, content_hash)
                );
                PRAGMA user_version = 1;
            """)
        os.chmod(self.root, 0o700)
        os.chmod(self.images_dir, 0o700)
        os.chmod(self.database, 0o600)

    @contextmanager
    def connection(self):
        db = (sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=5)
              if self.read_only else sqlite3.connect(self.database, timeout=5))
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def people(self):
        with self.connection() as db:
            rows = db.execute("""
                SELECT p.id, p.name, COUNT(s.id) AS sample_count
                FROM people p LEFT JOIN samples s ON p.id = s.person_id
                GROUP BY p.id ORDER BY p.name_key
            """).fetchall()
        return [Person(row["id"], row["name"], row["sample_count"]) for row in rows]

    def get_or_create_person(self, name):
        name = normalize_name(name)
        key = name.casefold()
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO people VALUES (?, ?, ?)",
                       (uuid.uuid4().hex, name, key))
            row = db.execute("SELECT id, name FROM people WHERE name_key = ?", (key,)).fetchone()
        return Person(row["id"], row["name"])

    def _image_path(self, person_id, filename):
        if len(person_id) != 32 or any(c not in "0123456789abcdef" for c in person_id):
            raise ValueError("Некорректный идентификатор профиля.")
        if Path(filename).name != filename or Path(filename).suffix not in IMAGE_SUFFIXES:
            raise ValueError("Некорректное имя изображения в базе.")
        path = self.images_dir / person_id / filename
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("Изображения базы не должны быть ссылками.")
        if not path.resolve().is_relative_to(self.images_dir):
            raise ValueError("Изображение находится за пределами базы.")
        return path

    def has_source(self, source_key):
        with self.connection() as db:
            return db.execute("SELECT 1 FROM samples WHERE source_key = ?", (source_key,)).fetchone() is not None

    def add_sample(self, person_id, image_bytes, encoding, model_version, source_key=None):
        encoding = validate_encoding(encoding)
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("Изображение пустое или превышает 20 МиБ.")
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        sample_id = uuid.uuid4().hex
        filename = sample_id + ".jpg"
        destination = self._image_path(person_id, filename)
        with self.connection() as db:
            if db.execute("SELECT 1 FROM people WHERE id = ?", (person_id,)).fetchone() is None:
                raise ValueError("Профиль не найден.")
            duplicate = db.execute("SELECT 1 FROM samples WHERE person_id = ? AND content_hash = ?",
                                   (person_id, content_hash)).fetchone()
            if duplicate or (source_key and self.has_source(source_key)):
                return False
            destination.parent.mkdir(mode=0o700, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                    temporary = Path(output.name)
                    output.write(image_bytes)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, destination)
                db.execute("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (sample_id, person_id, filename, content_hash,
                            json.dumps(encoding, allow_nan=False), model_version, source_key))
            except BaseException:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                raise
        return True

    def samples(self, person_id=None):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM samples " +
                              ("WHERE person_id = ? " if person_id is not None else "") +
                              "ORDER BY id", (person_id,) if person_id is not None else ()).fetchall()
        result = []
        for row in rows:
            try:
                encoding = validate_encoding(json.loads(row["encoding_json"])) if row["encoding_json"] else None
            except (ValueError, TypeError, OverflowError):
                encoding = None  # A damaged cache is rebuilt from its checked image.
            result.append(Sample(row["id"], row["person_id"],
                                 self._image_path(row["person_id"], row["filename"]),
                                 row["content_hash"], encoding, row["model_version"]))
        return result

    def rename_person(self, person_id, name):
        name = normalize_name(name)
        with self.connection() as db:
            try:
                changed = db.execute("UPDATE people SET name = ?, name_key = ? WHERE id = ?",
                                     (name, name.casefold(), person_id)).rowcount
            except sqlite3.IntegrityError as error:
                raise ValueError("Профиль с таким именем уже существует.") from error
            if not changed:
                raise ValueError("Профиль не найден.")

    def delete_sample(self, sample_id):
        self._delete(sample_id=sample_id)

    def delete_person(self, person_id):
        self._delete(person_id=person_id)

    def _delete(self, *, sample_id=None, person_id=None):
        """Stage photos before committing; restore them if the transaction fails.

        A crash can leave a private .deleting-* folder. It is deliberately not
        erased on startup: it may contain photos needed for manual recovery.
        """
        staging = Path(tempfile.mkdtemp(prefix=".deleting-", dir=self.root))
        moved = []
        try:
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                if person_id is not None:
                    if not db.execute("SELECT 1 FROM people WHERE id = ?", (person_id,)).fetchone():
                        raise ValueError("Профиль не найден.")
                    rows = db.execute("SELECT * FROM samples WHERE person_id = ?", (person_id,)).fetchall()
                else:
                    rows = db.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchall()
                    if not rows:
                        raise ValueError("Образец не найден.")
                # Validate every path before moving anything.
                paths = [self._image_path(row["person_id"], row["filename"]) for row in rows]
                if any(path.exists() and not path.is_file() for path in paths):
                    raise ValueError("Вместо фотографии найден каталог или специальный файл. Удаление отменено.")
                for path in paths:
                    if path.exists():
                        destination = staging / path.name
                        os.replace(path, destination)
                        moved.append((path, destination))
                if person_id is not None:
                    db.execute("DELETE FROM samples WHERE person_id = ?", (person_id,))
                    db.execute("DELETE FROM people WHERE id = ?", (person_id,))
                else:
                    db.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
        except BaseException:
            for original, temporary in reversed(moved):
                os.replace(temporary, original)
            raise
        else:
            for _, temporary in moved:
                temporary.unlink()
        finally:
            if not any(staging.iterdir()):
                staging.rmdir()

    def update_encoding(self, sample_id, content_hash, encoding, model_version):
        encoded = json.dumps(validate_encoding(encoding), allow_nan=False) if encoding is not None else None
        with self.connection() as db:
            db.execute("UPDATE samples SET content_hash = ?, encoding_json = ?, model_version = ? WHERE id = ?",
                       (content_hash, encoded, model_version, sample_id))


def legacy_images(root):
    """Only direct photos in person folders; exclude test data and demo assets."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        return []
    result = []
    for folder in sorted(root.iterdir()):
        if (not folder.is_dir() or folder.is_symlink() or folder.name.startswith(".")
                or folder.name.casefold() in {"test", "tests", "pig"}):
            continue
        for image in sorted(folder.iterdir()):
            if (image.is_file() and not image.is_symlink()
                    and image.suffix.lower() in IMAGE_SUFFIXES):
                result.append((folder.name, image))
    return result
