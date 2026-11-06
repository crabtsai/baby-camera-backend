"""Keep a user-labelled cry reference and evaluate it locally, without training or alerts."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

from backend.audio_monitor import AudioClassifier, CryPolicy
from backend.config import _get
from scripts.diagnose_audio import analyze


def summarize(windows, duration, threshold=.5):
    scores = [window['cry_score'] for window in windows]
    return {
        'duration_seconds': duration,
        'full_windows': len(windows),
        'unscored_tail_seconds': max(0, duration - len(windows) * 5),
        'windows_above_cry_threshold': sum(score >= threshold for score in scores),
        'minimum_cry_score': min(scores) if scores else None,
        'maximum_cry_score': max(scores) if scores else None,
        'offline_policy_events': sum(bool(window['policy_event_offline']) for window in windows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path, help='User-labelled positive cry audio')
    parser.add_argument('--cry-threshold', type=float, default=.5)
    parser.add_argument('--comparison', type=Path, action='append', default=[],
                        help='Additional audio; labels are unverified, not assumed negative')
    args = parser.parse_args()
    CryPolicy(cry_threshold=args.cry_threshold)
    sources = [args.reference, *args.comparison]
    if any(not path.is_file() for path in sources):
        parser.error('An input file is missing')
    ffmpeg = _get('audio', 'ffmpeg_path', '') or shutil.which('ffmpeg')
    if not ffmpeg:
        parser.error('ffmpeg unavailable')
    folder = Path('data/calibration/audio-references') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True)
    report = {
        'purpose': 'local_reference_evaluation',
        'model_path': _get('audio', 'model_path', 'models/audio-ast'),
        'weights_changed': False, 'threshold_changed': False,
        'cry_threshold': args.cry_threshold, 'scream_threshold': .7,
        'independent_validation': False,
        'label_scope': 'File label only; individual windows have not been annotated.',
        'files': [],
    }
    report_path = folder / 'report.json'
    for index, source in enumerate(sources):
        target = folder / f'{index:02d}-source{source.suffix}'
        shutil.copy2(source, target)
        decoded = folder / f'{index:02d}-16khz.wav'
        subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(target),
                        '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
                        '-c:a', 'pcm_s16le', '-n', str(decoded)],
                       check=True, timeout=120, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        report['files'].append({
            'role': 'user_labelled_cry' if index == 0 else 'unverified_comparison',
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'source_copy': target.name, 'decoded': decoded.name,
        })
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Local references saved: {folder}', flush=True)
    classifier = AudioClassifier(report['model_path'])
    for entry in report['files']:
        decoded = folder / entry['decoded']
        with wave.open(str(decoded), 'rb') as recording:
            duration = recording.getnframes() / recording.getframerate()
        entry['windows'] = analyze(decoded, classifier, args.cry_threshold)
        entry['summary'] = summarize(entry['windows'], duration, args.cry_threshold)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(entry['role'], json.dumps(entry['summary']), flush=True)
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
