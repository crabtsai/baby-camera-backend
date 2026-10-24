"""Bounded local image collection; no alerts, automatic labels, or training.

Run as python -m scripts.collect_calibration --seconds 120 --interval 5.
Touch the session's STOP file to finish early. Every session has a new directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid

import cv2

from backend.camera_stream import CameraStream
from backend.config import Settings
from backend.frame_reader import FrameReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--interval', type=float, default=5)
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.seconds, args.interval)) or not (0 < args.seconds <= 3600 and 1 <= args.interval <= 300):
        parser.error('seconds must be in (0, 3600]; interval in [1, 300]')
    output = Path('data/calibration') / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    settings = Settings()
    deadline = time.monotonic() + args.seconds
    lock = threading.Lock()
    stop = threading.Event()
    counts = {c.id: 0 for c in settings.cameras}
    errors = {c.id: 0 for c in settings.cameras}
    runtimes = []
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'seconds': args.seconds,
                'interval': args.interval, 'camera_ids': list(counts), 'labels': 'unreviewed',
                'training_performed': False, 'status': 'running'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(str(output.resolve()), flush=True)
    manifest['status'] = 'failed'
    try:
        for index, config in enumerate(settings.cameras):
            folder = output / f'camera-{index + 1}'
            folder.mkdir()
            camera = CameraStream(config.source)
            reader = FrameReader(camera, args.interval)

            def save(frame, camera_id=config.id, folder=folder):
                if stop.is_set() or time.monotonic() >= deadline or (output / 'STOP').exists():
                    return
                stamp = datetime.now(timezone.utc)
                name = stamp.strftime('%Y%m%dT%H%M%S-%fZ') + '.jpg'
                try:
                    if not cv2.imwrite(str(folder / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise OSError('JPEG write failed')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    row = {'camera_id': camera_id, 'captured_at': stamp.isoformat(),
                           'image': f'{folder.name}/{name}', 'shape': list(frame.shape),
                           'sha256': hashlib.sha256((folder / name).read_bytes()).hexdigest(),
                           'brightness': float(gray.mean()),
                           'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                           'review': {'occupancy': 'unreviewed', 'face_visibility': 'unreviewed',
                                      'visible_spit_up': 'unreviewed', 'choking': 'not_assessable_from_still'}}
                    with lock:
                        with (output / 'samples.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        counts[camera_id] += 1
                except Exception:
                    with lock:
                        errors[camera_id] += 1
                    raise

            reader.register_callback(save)
            runtimes.append((camera, reader))
            reader.start()
        while time.monotonic() < deadline and not (output / 'STOP').exists():
            time.sleep(min(.25, max(0, deadline - time.monotonic())))
        manifest['status'] = 'stopped' if (output / 'STOP').exists() else 'completed'
    finally:
        stop.set()
        for camera, reader in runtimes:
            reader.stop()
            camera.release()
        manifest.update(counts=counts, write_errors=errors, ended_at=datetime.now(timezone.utc).isoformat())
        if not all(counts.values()) or any(errors.values()):
            manifest['status'] = 'incomplete'
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
