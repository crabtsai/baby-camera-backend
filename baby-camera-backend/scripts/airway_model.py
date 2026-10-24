"""Local-only airway dataset, training and held-out evaluation tools.

Run with python -m scripts.airway_model --help. Never enables a model automatically.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import yaml

from backend.detector.airway_model import AIRWAY_LABELS, check_model_contract

SPLITS = ("train", "val", "test")
STATES = ("clear", "abnormal", "unknown")
FIELDS = ("image", "session", "assessment")


def initialize(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for kind in ("images", "labels"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
    manifest = root / "samples.csv"
    if not manifest.exists():
        manifest.write_text(",".join(FIELDS) + "\n", encoding="utf-8")
    config = root / "dataset.yaml"
    if not config.exists():
        config.write_text(yaml.safe_dump({
            "path": str(root.resolve()),
            **{split: f"images/{split}" for split in SPLITS},
            "names": dict(enumerate(AIRWAY_LABELS)),
        }, sort_keys=False), encoding="utf-8")
    return {"dataset": str(config), "status": "needs_human_annotations"}


def validate(root: Path):
    """Strict layout, explicit negative labels, session and pixel leakage checks."""
    root = root.resolve()
    errors, records = [], []
    counts = {split: Counter() for split in SPLITS}
    try:
        config = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
        expected = {"path": str(root), **{s: f"images/{s}" for s in SPLITS},
                    "names": dict(enumerate(AIRWAY_LABELS))}
        if config != expected:
            errors.append("dataset.yaml must match the generated local layout and eight class IDs")
        with (root / "samples.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(FIELDS):
                errors.append("samples.csv header must be image,session,assessment")
            rows = list(reader)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {"valid": False, "errors": [str(exc)], "counts": counts}, []
    seen, sessions, hashes, label_paths = set(), {}, {}, set()
    for index, row in enumerate(rows, 2):
        name = row.get("image") or ""
        path = (root / name).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            errors.append(f"row {index}: image outside dataset")
            continue
        parts = relative.parts
        if len(parts) != 3 or parts[0] != "images" or parts[1] not in SPLITS:
            errors.append(f"row {index}: use images/<train|val|test>/<filename>")
            continue
        split = parts[1]
        if path in seen:
            errors.append(f"duplicate image: {name}")
        seen.add(path)
        session = (row.get("session") or "").strip()
        if not session or row.get("assessment") not in STATES:
            errors.append(f"row {index}: session and human assessment are required")
        if session in sessions and sessions[session] != split:
            errors.append(f"session leakage across splits: {session}")
        sessions[session] = split
        image = cv2.imread(str(path))
        if image is None:
            errors.append(f"unreadable image: {name}")
            continue
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if digest in hashes:
            errors.append(f"duplicate decoded image: {name} and {hashes[digest]}")
        hashes[digest] = name
        label_path = root / "labels" / split / (path.stem + ".txt")
        if label_path in label_paths:
            errors.append(f"multiple images share one label file: {label_path.name}")
        label_paths.add(label_path)
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            errors.append(f"missing reviewed label file: {label_path.name}")
            continue
        for line_number, line in enumerate(lines, 1):
            try:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError()
                cls = int(fields[0])
                x, y, w, h = map(float, fields[1:])
                if not (0 <= cls < len(AIRWAY_LABELS) and
                        all(math.isfinite(v) for v in (x, y, w, h)) and
                        0 < w <= 1 and 0 < h <= 1 and
                        0 <= x-w/2 <= x+w/2 <= 1 and 0 <= y-h/2 <= y+h/2 <= 1):
                    raise ValueError()
                counts[split][AIRWAY_LABELS[cls]] += 1
            except ValueError:
                errors.append(f"invalid YOLO box: {label_path.name}:{line_number}")
        records.append({**row, "path": path, "split": split})
    actual = {p.resolve() for s in SPLITS for p in (root / "images" / s).glob("*") if p.is_file()}
    if actual != seen:
        errors.append("image files and samples.csv differ; every image needs human review")
    for split in SPLITS:
        missing = set(AIRWAY_LABELS) - counts[split].keys()
        if missing:
            errors.append(f"{split} lacks annotations: {', '.join(sorted(missing))}")
        states = {r["assessment"] for r in records if r["split"] == split}
        if set(STATES) - states:
            errors.append(f"{split} needs clear, abnormal and unknown examples")
    return {"valid": not errors, "errors": errors, "counts": counts,
            "images": dict(Counter(r["split"] for r in records)),
            "note": "Format checks do not establish annotation correctness or sufficient sample size."}, records


def load_model(path: Path):
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise ValueError("Provide an existing trusted local .pt file; no automatic download")
    from ultralytics import YOLO
    return YOLO(str(path.resolve()))


def inspect(path: Path):
    model = load_model(path)
    try:
        check_model_contract(model)
        error = None
    except ValueError as exc:
        error = str(exc)
    return {"model": str(path), "task": model.task, "classes": model.names,
            "compatible": error is None, "error": error,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "note": "Class compatibility is not an accuracy or deployment approval."}


def evaluate(path: Path, records, output: Path):
    from backend.detector.face_cover import FaceCoverDetector
    detector = FaceCoverDetector(model_path=str(path.resolve()))
    if detector.predict_fn is None:
        raise ValueError(detector.disabled_reason)
    confusion = {truth: {prediction: 0 for prediction in STATES} for truth in STATES}
    predictions = []
    for record in records:
        if record["split"] != "test":
            continue
        result = detector.detect(cv2.imread(str(record["path"])))
        predicted = result.extra["assessment"]
        confusion[record["assessment"]][predicted] += 1
        predictions.append({"image": record["image"], "expected": record["assessment"],
                            "predicted": predicted, "message": result.message})
    positives = sum(confusion["abnormal"].values())
    negatives = sum(sum(confusion[s].values()) for s in ("clear", "unknown"))
    report = {"model": inspect(path), "confusion": confusion, "predictions": predictions,
              "missed_obstruction_rate": 1-confusion["abnormal"]["abnormal"]/positives if positives else None,
              "false_alert_rate": sum(confusion[s]["abnormal"] for s in ("clear", "unknown"))/negatives if negatives else None,
              "unknown_rate": sum(confusion[s]["unknown"] for s in STATES)/len(predictions) if predictions else None,
              "deployment_ready": False,
              "limitations": "Still-frame detector evaluation only; requires camera/night-mode and continuous pipeline/event-delay validation."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "validate", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, default=Path("data/airway"))
        if name in ("train", "evaluate"):
            command.add_argument("--model", type=Path, required=True)
        if name == "train":
            command.add_argument("--epochs", type=int, default=100)
            command.add_argument("--device", default="cpu")
            command.add_argument("--batch", type=int, default=4)
        if name == "evaluate":
            command.add_argument("--output", type=Path, default=Path("data/airway/evaluation.json"))
    command = commands.add_parser("inspect")
    command.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            report = initialize(args.data)
        elif args.command == "inspect":
            report = inspect(args.model)
        else:
            report, records = validate(args.data)
            if not report["valid"]:
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 1
            if args.command == "train":
                if args.epochs < 1 or args.batch < 1:
                    raise ValueError("epochs and batch must be positive")
                model = load_model(args.model)
                if model.task != "detect":
                    raise ValueError("Training requires a detection base model")
                # Explicit local data only; no remote tracking or automatic deployment.
                model.train(data=str((args.data / "dataset.yaml").resolve()),
                            epochs=args.epochs, device=args.device, batch=args.batch,
                            imgsz=640, workers=0, seed=42, project="runs/airway",
                            name="train", exist_ok=False, amp=False, plots=False)
                best = Path(model.trainer.best)
                report = {"weights": str(best), "model": inspect(best), "deployment_ready": False,
                          "next_step": "Run evaluate on held-out test sessions and review errors."}
            elif args.command == "evaluate":
                report = evaluate(args.model, records, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report.get("compatible") is False else 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
