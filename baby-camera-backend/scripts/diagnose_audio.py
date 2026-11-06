"""Explicit, local-only camera recording and five-second cry diagnostics."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
import wave

import numpy as np

from backend.audio_monitor import AudioClassifier, CryPolicy
from backend.config import Settings, _get, _get_float


def capture(camera, ffmpeg, folder, seconds):
    # Do not print commands or decoder errors: they may contain credentials.
    target = folder / f'camera-{camera[0]}.wav'
    try:
        result = subprocess.run(
            [ffmpeg, '-nostdin', '-loglevel', 'error', '-rtsp_transport', 'tcp',
             '-timeout', '8000000', '-i', camera[1].source, '-t', str(seconds),
             '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
             '-c:a', 'pcm_s16le', '-n', str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=seconds + 30,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        if result.returncode:
            return {'camera_id': camera[1].id, 'error': 'capture_failed'}
        return {'camera_id': camera[1].id, 'file': str(target)}
    except (OSError, subprocess.TimeoutExpired):
        return {'camera_id': camera[1].id, 'error': 'capture_failed_or_timed_out'}


def analyze(path, classifier, cry_threshold=.5):
    with wave.open(str(path), 'rb') as recording:
        samples = np.frombuffer(recording.readframes(recording.getnframes()), dtype='<i2').astype(np.float32) / 32768
    policy = CryPolicy(cry_threshold=cry_threshold)
    windows = []
    for start in range(0, len(samples) - 79999, 80000):
        chunk = samples[start:start + 80000]
        began = time.monotonic()
        scores = classifier(chunk)
        elapsed = time.monotonic() - began
        windows.append({
            'start_seconds': start / 16000,
            'rms': float(np.sqrt(np.mean(chunk ** 2))),
            'peak': float(np.max(np.abs(chunk))),
            'cry_score': scores['Baby cry, infant cry'],
            'scream_score': scores['Screaming'],
            'top_classes': sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5],
            'inference_seconds': elapsed,
            'policy_event_offline': policy.observe(scores, start / 16000 + 5),
        })
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--camera', help='Record only this configured camera ID')
    args = parser.parse_args()
    if not 5 <= args.seconds <= 120:
        parser.error('--seconds must be between 5 and 120')
    ffmpeg = _get('audio', 'ffmpeg_path', '') or shutil.which('ffmpeg')
    if not ffmpeg:
        parser.error('ffmpeg unavailable')
    cameras = list(enumerate(Settings().cameras, 1))
    if args.camera:
        cameras = [(index, camera) for index, camera in cameras if camera.id == args.camera]
    if not cameras:
        parser.error('No matching configured cameras')
    folder = Path('data/recordings') / datetime.now().strftime('audio-diagnostic-%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True)
    print(f'Recording {len(cameras)} cameras for {args.seconds}s into {folder}', flush=True)
    with ThreadPoolExecutor(max_workers=max(1, len(cameras))) as pool:
        results = list(pool.map(lambda camera: capture(camera, ffmpeg, folder, args.seconds), cameras))
    threshold = _get_float('audio', 'cry_threshold', .5)
    threshold = threshold if np.isfinite(threshold) and .25 <= threshold <= 1 else .5
    report = {'recordings': results, 'outbound_notifications': False, 'cry_threshold': threshold}
    report_path = folder / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if any('file' in result for result in results):
        print('Recording complete. Loading local classifier.', flush=True)
        classifier = AudioClassifier(_get('audio', 'model_path', 'models/audio-ast'))
        for result in results:
            if 'file' in result:
                result['windows'] = analyze(result['file'], classifier, threshold)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Report: {report_path}', flush=True)
    return int(any('error' in result for result in results))


if __name__ == '__main__':
    raise SystemExit(main())
