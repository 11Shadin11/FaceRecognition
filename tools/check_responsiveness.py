"""Repeat real model inference and Tk preview without accessing a camera.

This measures resource use, not recognition accuracy. Data lives in a temporary
store. An optional photo is read only; it never becomes an enrollment sample.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))] if values else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 5 <= args.seconds <= 3600:
        parser.error("--seconds должен быть от 5 до 3600")
    import numpy as np
    import tkinter as tk
    from face_app.capture import CapturedFrame
    from face_app.config import AppConfig
    from face_app.models import RecognitionResult
    from face_app.ui import FaceApplication

    with tempfile.TemporaryDirectory(prefix="face-ui-check-") as directory:
        window = tk.Tk()
        window.withdraw()
        app = FaceApplication(window, AppConfig(data_dir=Path(directory) / "data"))
        window.update_idletasks()
        started = None
        intervals, processing, resources = [], [], []
        previous_tick = None
        warmup_deadline = time.monotonic() + 60
        next_resources = 0
        completed = 0

        def workload(context):
            nonlocal started, completed
            engine = app.service.engine(context)
            frame = engine.read_image(args.image) if args.image else np.full((480, 640, 3), 100, dtype=np.uint8)
            engine.set_known_faces([], [])
            next_inference = 0
            sequence = 0
            while not context.cancel.is_set():
                now = time.monotonic()
                sequence += 1
                context.preview(CapturedFrame(sequence, now, frame))
                if now >= next_inference:
                    before = time.perf_counter()
                    faces = engine.recognize(frame)
                    processing.append((time.perf_counter() - before) * 1000)
                    result = RecognitionResult(sequence, now, (frame.shape[1], frame.shape[0]), tuple(faces))
                    context.status("Проверка ресурсов без камеры", recognized_at=now, recognition=result)
                    if started is None:
                        started = time.monotonic()
                    completed += 1
                    next_inference = now + app.config.recognition_interval
                context.cancel.wait(1 / 30)

        failure = None

        def tick():
            nonlocal previous_tick, next_resources, failure
            now = time.monotonic()
            if started is not None:
                if previous_tick is not None:
                    intervals.append((now - previous_tick) * 1000)
                previous_tick = now
                if now >= next_resources and app.service._engine is not None:
                    pids = [str(os.getpid()), str(app.service._engine.process.pid)]
                    output = subprocess.run(["ps", "-o", "pid=,rss=,%cpu=", "-p", ",".join(pids)],
                                            capture_output=True, text=True, timeout=2)
                    if output.returncode == 0:
                        rows = [line.split() for line in output.stdout.splitlines()]
                        resources.append({"elapsed": round(now - started, 1),
                                          "rss_mib": sum(int(row[1]) for row in rows) / 1024,
                                          "cpu_percent": sum(float(row[2]) for row in rows)})
                    next_resources = now + 10
                if now - started >= args.seconds:
                    app.close()
                    return
            elif now > warmup_deadline or app.last_outcome == "error":
                failure = app.status_text.get() or "Таймаут загрузки модели"
                app.close()
                return
            if not app.closing:
                window.after(20, tick)

        app._start(workload, activity="recognition")
        window.after(20, tick)
        window.mainloop()
        report = {"kind": "offline-tk-model-resource-check", "requested_seconds": args.seconds,
                  "completed_inferences": completed, "image_supplied": args.image is not None,
                  "tk_timer_p95_ms": percentile(intervals, .95), "tk_timer_max_ms": max(intervals, default=0),
                  "inference_p95_ms": percentile(processing, .95), "inference_mean_ms": sum(processing) / max(1, len(processing)),
                  "processes_closed": app.service.model_closed and not app.runner.busy,
                  "resources": resources, "error": failure}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "resources"}, ensure_ascii=False), flush=True)
        return 1 if failure or not report["processes_closed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
