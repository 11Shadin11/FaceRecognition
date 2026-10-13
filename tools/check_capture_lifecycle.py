"""Repeat OpenCV file capture, EOF and cancellation without a physical camera."""
import argparse
import gc
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=100)
    args = parser.parse_args(argv)
    if not 1 <= args.cycles <= 1000:
        parser.error("--cycles должен быть от 1 до 1000")
    import cv2
    import numpy as np
    from face_app.capture import CaptureStream
    from face_app.config import AppConfig

    records = []
    with tempfile.TemporaryDirectory(prefix="face-capture-check-") as directory:
        path = Path(directory) / "synthetic.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
        if not writer.isOpened():
            raise RuntimeError("Не удалось создать тестовое видео MJPG.")
        try:
            frame = np.full((240, 320, 3), 100, dtype=np.uint8)
            for _ in range(480):
                writer.write(frame)
        finally:
            writer.release()
        for number in range(1, args.cycles + 1):
            before = time.monotonic()
            stream = CaptureStream(str(path), AppConfig())
            count = 0
            try:
                stream.start()
                for frame in stream.file_frames(threading.Event()):
                    count += 1
                    if number % 2:
                        break  # Cancel while the bounded producer can still run.
            finally:
                stream.close()
            assert count >= 1 and not stream.process.is_alive() and stream.closed
            active = multiprocessing.active_children()
            assert not active, [process.pid for process in active]
            gc.collect()
            descriptor_dir = Path("/proc/self/fd") if Path("/proc/self/fd").exists() else Path("/dev/fd")
            records.append({"cycle": number, "frames": count, "seconds": time.monotonic() - before,
                            "file_descriptors": len(list(descriptor_dir.iterdir())),
                            "threads": len(threading.enumerate())})
    first, last = records[0], records[-1]
    assert last["file_descriptors"] <= first["file_descriptors"] + 2, (first, last)
    assert last["threads"] <= first["threads"] + 1, (first, last)
    print(json.dumps({"cycles": args.cycles, "cancelled": (args.cycles + 1) // 2,
                      "eof": args.cycles // 2, "first": first, "last": last,
                      "children_left": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
