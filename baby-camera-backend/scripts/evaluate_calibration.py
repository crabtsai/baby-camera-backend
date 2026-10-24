"""Offline threshold sweep on a collected session; predictions are not labels."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    models = {'pose': YOLO('backend/detector/model/yolo11n-pose.pt'),
              'face': YOLO('models/yolov11n_face_detection.pt')}
    rows = []
    for path in sorted(args.session.glob('camera-*/*.jpg')):
        frame = cv2.imread(str(path))
        row = {'image': path.relative_to(args.session).as_posix(), 'models': {}}
        for name, model in models.items():
            result = model.predict(frame, conf=.1, verbose=False, device='cpu')[0]
            row['models'][name] = [{'label': result.names[int(b.cls.item())],
                                    'confidence': float(b.conf.item()), 'box': b.xyxy[0].tolist()}
                                   for b in result.boxes]
        rows.append(row)
    (args.session / 'predictions.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    summary = {}
    for camera in sorted({r['image'].split('/')[0] for r in rows}):
        samples = [r for r in rows if r['image'].startswith(camera + '/')]
        summary[camera] = {'samples': len(samples), 'detected_frames': {
            model: {str(t): sum(any(b['confidence'] >= t for b in r['models'][model]) for r in samples)
                    for t in [.1, .2, .25, .3, .4, .5]} for model in models}}
    (args.session / 'evaluation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
