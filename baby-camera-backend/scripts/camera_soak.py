"""Bounded, local camera/detector soak test. Never constructs alert/event services.

Run from repository root: .venv/Scripts/python.exe -m scripts.camera_soak
Touch data/camera_soak/STOP to stop. Re-run with the same output to resume the
original deadline; missed observation time is not counted as healthy uptime.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
import time

# Low-rate CPU inference should sleep between jobs rather than spin OpenMP
# workers. Must be set before importing numerical libraries; explicit operator
# settings take precedence. Validate here before promoting to service startup.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import cv2
import numpy as np

from backend.camera_stream import CameraStream
from backend.config import Settings
from backend.detector.detector_manager import DetectorManager
from backend.frame_reader import FrameReader


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=72)
    parser.add_argument("--sample-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path("data/camera_soak"))
    args = parser.parse_args()
    if args.hours <= 0 or args.sample_seconds <= 0:
        parser.error("durations must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # OS lock is released on crash. Avoid duplicate RTSP sessions on resume.
    lease = (output / "run.lock").open("a+b")
    lease.seek(0)
    if os.name == "nt":
        import msvcrt
        if (output / "run.lock").stat().st_size == 0:
            lease.write(b"0")
            lease.flush()
        lease.seek(0)
        msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        now = time.time()
        manifest = {"started_at": now, "deadline_at": now + args.hours * 3600,
                    "duration_hours": args.hours}
        atomic_json(manifest_path, manifest)
    if time.time() >= manifest["deadline_at"] or (output / "STOP").exists():
        return
    handler = RotatingFileHandler(output / "runner.log", maxBytes=2_000_000,
                                  backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings()
    manifest["camera_ids"] = [c.id for c in settings.cameras]
    atomic_json(manifest_path, manifest)
    revision_files = sorted(Path("backend").rglob("*.py")) + [Path(__file__)]
    revision = hashlib.sha256(b"".join(p.read_bytes() for p in revision_files)).hexdigest()
    runtimes = []
    results = {}
    result_lock = threading.Lock()
    previous = {}
    try:
        import psutil
        process = psutil.Process()
        process.cpu_percent()
    except ImportError:
        process = None
    last_sample = time.monotonic()
    stop_reason = "failed"
    try:
        for config in settings.cameras:
            camera = CameraStream(config.source)
            manager = DetectorManager.from_settings(settings, camera_id=config.id)
            reader = FrameReader(camera, settings.frame_interval_sec)

            def result_callback(frame, result, metadata, camera_id=config.id):
                with result_lock:
                    results.setdefault(camera_id, {})[result.detector_name] = {
                        "at": time.time(), **asdict(result)}

            manager.start_workers(result_callback, max_queue_size=1)
            reader.register_callback(manager.submit_frame)
            runtimes.append((config, camera, manager, reader))
            reader.start()
        while time.time() < manifest["deadline_at"] and not (output / "STOP").exists():
            now = time.monotonic()
            elapsed = max(0.001, now - last_sample)
            record = {"at": time.time(), "utc": datetime.now(timezone.utc).isoformat(),
                      "pid": os.getpid(), "revision": revision, "process_cpu_sec": time.process_time(),
                      "runtime": {"omp_wait_policy": os.environ.get("OMP_WAIT_POLICY")},
                      "status": "running", "deadline_at": manifest["deadline_at"], "cameras": []}
            if process is not None:
                record["resources"] = {"rss_bytes": process.memory_info().rss,
                    "cpu_percent": process.cpu_percent(), "threads": process.num_threads()}
            for config, camera, manager, reader in runtimes:
                state = camera.status()
                old = previous.get(config.id, {})
                frame = camera.get_latest_frame()
                image_metrics = None
                if frame is not None:
                    gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
                    digest = hashlib.sha256(frame.tobytes()).hexdigest()
                    image_metrics = {"shape": list(frame.shape), "brightness": float(gray.mean()),
                                     "contrast": float(gray.std()), "sha256": digest,
                                     "same_as_previous_sample": digest == old.get("digest"),
                                     "mean_pixel_change": (float(np.abs(gray.astype(float) - old["gray"]).mean())
                                                           if "gray" in old else None)}
                    # Only two latest local previews; no accumulating video archive.
                    cv2.imwrite(str(output / f"camera-{len(record['cameras']) + 1}-latest.jpg"), frame)
                    old.update(digest=digest, gray=gray)
                state["capture_fps"] = ((state["frames_read"] - old["frames"]) / elapsed
                                        if "frames" in old else None)
                old["frames"] = state["frames_read"]
                previous[config.id] = old
                with result_lock:
                    latest_results = dict(results.get(config.id, {}))
                record["cameras"].append({"camera_id": config.id, "capture": state,
                    "reader_alive": bool(reader._thread and reader._thread.is_alive()),
                    "image": image_metrics, "workers": manager.worker_status(),
                    "results": latest_results})
            with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            atomic_json(output / "latest.json", record)
            last_sample = now
            remaining = min(args.sample_seconds, manifest["deadline_at"] - time.time())
            until = time.monotonic() + max(0, remaining)
            while time.monotonic() < until and not (output / "STOP").exists():
                time.sleep(min(1, max(0, until - time.monotonic())))
        stop_reason = "stopped" if (output / "STOP").exists() else "completed"
    finally:
        atomic_json(output / "exit.json", {"at": time.time(), "status": stop_reason,
                                            "pid": os.getpid()})
        for _, camera, manager, reader in runtimes:
            reader.stop()
            manager.stop_workers()
            camera.release()
        lease.close()


if __name__ == "__main__":
    main()
