"""Consistent database snapshot and only its referenced images in a private ZIP."""
from datetime import datetime, timezone
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile

from .storage import MAX_IMAGE_BYTES


def create_backup(store, destination, context):
    destination = Path(destination).absolute()
    if destination.is_symlink() or destination.exists():
        raise ValueError("Выберите новое имя: существующие резервные копии не перезаписываются.")
    if destination.resolve().is_relative_to(store.root):
        raise ValueError("Сохраните копию вне каталога данных приложения.")
    context.check_cancelled()
    with tempfile.TemporaryDirectory(prefix=".backup-", dir=destination.parent) as work:
        work = Path(work)
        snapshot, archive = work / "faces.sqlite3", work / "backup.zip"
        with store.connection() as locked:
            # Prevent writes/deletions while taking the snapshot and copying photos.
            # A separate reader connection lets SQLite.backup finish under this lock.
            locked.execute("BEGIN IMMEDIATE")
            with closing(sqlite3.connect(store.database)) as source, closing(sqlite3.connect(snapshot)) as target:
                def check_backup(status, remaining, total):
                    context.check_cancelled()
                source.backup(target, pages=128, progress=check_backup)
            rows = locked.execute("SELECT person_id, filename, content_hash FROM samples ORDER BY id").fetchall()
            people_count = locked.execute("SELECT COUNT(*) FROM people").fetchone()[0]
            manifest = {"format": "face-app-backup-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                        "people": people_count, "samples": len(rows), "images": {}}
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
                output.write(snapshot, "data/faces.sqlite3")
                for number, row in enumerate(rows, 1):
                    context.check_cancelled()
                    path = store._image_path(row["person_id"], row["filename"])
                    if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
                        raise ValueError("Образец отсутствует или слишком велик. Копия не создана.")
                    name = "data/" + path.relative_to(store.root).as_posix()
                    digest = hashlib.sha256()
                    with path.open("rb") as image, output.open(name, "w") as entry:
                        for chunk in iter(lambda: image.read(65536), b""):
                            context.check_cancelled()
                            digest.update(chunk)
                            entry.write(chunk)
                    if digest.hexdigest() != row["content_hash"]:
                        raise ValueError("Образец изменился на диске. Обновите базу распознаванием и повторите копирование.")
                    manifest["images"][name] = digest.hexdigest()
                    context.status(f"Резервная копия: {number}/{len(rows)}", number / len(rows))
                output.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        context.check_cancelled()
        os.chmod(archive, 0o600)
        # Atomic publication with no overwrite, even if another writer wins a race.
        os.link(archive, destination)
    return f"Копия сохранена: {destination}. Профилей: {people_count}, образцов: {len(rows)}."
