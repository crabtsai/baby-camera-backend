"""Local AudioSet classification. No raw audio retained or sent off-device."""
from __future__ import annotations

import logging
import math
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from backend.config import _get, _get_bool
from backend.schemas import EventRecord

log = logging.getLogger(__name__)


class CryPolicy:
    """At least 4s of cry-classified windows; one high-score scream window.

    Silence must last 10s before rearming; disconnected time is never silence.
    """
    def __init__(self, window_seconds=5):
        self.window_seconds = window_seconds
        self.last_at = None
        self.count = 0
        self.quiet = 0
        self.latched = False
        self.last_alert = -float('inf')

    def reset_gap(self):
        self.last_at = None
        self.count = self.quiet = 0

    def observe(self, scores, at):
        if self.last_at is not None and not 0 < at - self.last_at <= self.window_seconds + 1.5:
            self.reset_gap()
        self.last_at = at
        cry = scores.get('Baby cry, infant cry', 0) >= .5
        scream = scores.get('Screaming', 0) >= .7
        if cry or scream:
            self.quiet = 0
            self.count += 1
        else:
            self.count = 0
            self.quiet += 1
            if self.quiet >= math.ceil(10 / self.window_seconds):
                self.latched = False
        if not self.latched and at - self.last_alert >= 60 and (scream or (cry and self.count >= math.ceil(4 / self.window_seconds))):
            self.latched = True
            self.last_alert = at
            return 'scream_warning' if scream else 'cry_warning'
        return None


class AudioClassifier:
    def __init__(self, path):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.torch = torch
        self.extractor = ASTFeatureExtractor.from_pretrained(path, local_files_only=True)
        self.model = ASTForAudioClassification.from_pretrained(path, local_files_only=True).eval()
        self.labels = self.model.config.id2label
        if not {'Baby cry, infant cry', 'Screaming'}.issubset(set(self.labels.values())):
            raise ValueError('Required audio classes missing')

    def __call__(self, samples):
        features = self.extractor(samples, sampling_rate=16000, return_tensors='pt')
        with self.torch.inference_mode():
            scores = self.model(**features).logits[0].sigmoid().cpu().tolist()
        return {self.labels[i]: float(v) for i, v in enumerate(scores)}


class AudioMonitor:
    def __init__(self, settings, database):
        self.settings, self.database = settings, database
        self.enabled = _get_bool('audio', 'enabled', False)
        self.ffmpeg = _get('audio', 'ffmpeg_path', '') or shutil.which('ffmpeg')
        self.model_path = _get('audio', 'model_path', 'models/audio-ast')
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.queues = {c.id: queue.Queue(maxsize=1) for c in settings.cameras}
        self.policies = {c.id: CryPolicy() for c in settings.cameras}
        self.states = {c.id: {'status': 'starting', 'last_result_at': None} for c in settings.cameras}
        self.processes = {}
        self.threads = []
        self.error = None
        self.last_alert = None

    def start(self):
        if self.enabled:
            t = threading.Thread(target=self._infer, daemon=True, name='audio-inference')
            self.threads.append(t)
            t.start()

    def _infer(self):
        try:
            if not self.ffmpeg or not Path(self.ffmpeg).is_file():
                raise ValueError('ffmpeg unavailable')
            classifier = AudioClassifier(self.model_path)
            for camera in self.settings.cameras:
                t = threading.Thread(target=self._capture, args=(camera,), daemon=True, name=f'audio-{camera.id}')
                self.threads.append(t)
                t.start()
        except Exception as exc:
            self.error = type(exc).__name__ + ': audio model/decoder unavailable'
            log.exception('Audio initialization failed')
            return
        while not self.stop_event.is_set():
            did_work = False
            for camera_id, q in self.queues.items():
                try:
                    captured, samples = q.get_nowait()
                except queue.Empty:
                    continue
                did_work = True
                policy = self.policies[camera_id]
                if time.monotonic() - captured > 4:
                    policy.reset_gap()
                    continue
                try:
                    scores = classifier(samples)
                    if time.monotonic() - captured > 4:
                        raise TimeoutError('Audio inference too slow')
                    event = policy.observe(scores, captured)
                    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    with self.lock:
                        self.states[camera_id] = {'status': 'listening', 'last_result_at': time.time(),
                            'cry_score': scores['Baby cry, infant cry'], 'scream_score': scores['Screaming'],
                            'top_classes': top, 'episode_active': policy.latched}
                    if event:
                        record = self.database.insert_event(EventRecord(camera_id, event,
                            '疑似嬰兒哭聲，請查看' if event == 'cry_warning' else '疑似尖叫聲，請查看',
                            scores['Baby cry, infant cry' if event == 'cry_warning' else 'Screaming'],
                            'audio_classifier', extra={'window_seconds': 5, 'source_identity_verified': False,
                                'score_kind': 'model_score_not_accuracy', 'notify_line': False}))
                        with self.lock:
                            self.last_alert = {'id': record.id, 'camera_id': camera_id, 'message': record.message,
                                               'at': time.time(), 'event_type': event}
                except Exception:
                    policy.reset_gap()
                    with self.lock:
                        self.states[camera_id] = {'status': 'error', 'last_result_at': None}
                    log.exception('Audio classification failed')
            if not did_work:
                self.stop_event.wait(.05)

    def _capture(self, camera):
        while not self.stop_event.is_set():
            process = None
            try:
                # Never log the command/source: RTSP URLs contain credentials.
                process = subprocess.Popen([self.ffmpeg, '-nostdin', '-loglevel', 'error',
                    '-rtsp_transport', 'tcp', '-timeout', '8000000', '-i', camera.source,
                    '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1'],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                with self.lock:
                    self.processes[camera.id] = process
                while not self.stop_event.is_set():
                    raw = process.stdout.read(160000)  # exactly five seconds, non-overlapping
                    if len(raw) != 160000:
                        raise OSError('Audio stream interrupted')
                    packet = (time.monotonic(), np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.)
                    q = self.queues[camera.id]
                    try:
                        q.put_nowait(packet)
                    except queue.Full:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                        q.put_nowait(packet)
            except Exception:
                with self.lock:
                    self.states[camera.id] = {'status': 'reconnecting', 'last_result_at': None}
            finally:
                if process is not None:
                    process.kill()
                    process.wait()
                    process.stdout.close()
                with self.lock:
                    self.processes.pop(camera.id, None)
            self.stop_event.wait(3)

    def status(self):
        with self.lock:
            cameras = {k: dict(v) for k, v in self.states.items()}
            for state in cameras.values():
                if state['status'] == 'listening' and time.time() - (state['last_result_at'] or 0) > 12:
                    state['status'] = 'stale'
            return {'enabled': self.enabled, 'error': self.error, 'cameras': cameras,
                    'last_alert': dict(self.last_alert) if self.last_alert else None,
                    'notification': 'browser_only', 'raw_audio_saved': False, 'window_seconds': 5}

    def stop(self):
        self.stop_event.set()
        with self.lock:
            for p in self.processes.values():
                p.kill()
        for t in self.threads:
            t.join(timeout=3)
