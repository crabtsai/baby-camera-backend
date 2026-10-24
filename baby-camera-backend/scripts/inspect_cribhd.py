"""Verify the public CribHD archive and extract data only into a new directory."""
import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import yaml

SOURCE = "https://coe.northeastern.edu/Research/AClab/CribHD/CribHD.zip"


def inspect_archive(archive: Path, destination: Path):
    if destination.exists():
        raise ValueError("Use a new output directory; existing files will not be overwritten")
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"source": SOURCE, "reference": "https://github.com/ostadabbas/CribNet",
              "bytes": archive.stat().st_size,
              "sha256": digest,
              "files": {}, "datasets": [], "weights": [],
              "ready_for_airway_training": False}
    allowed = {".jpg", ".jpeg", ".png", ".txt", ".yaml", ".yml", ".json", ".md"}
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if sum(item.file_size for item in entries) > 5 * 1024**3:
            raise ValueError("Archive exceeds 5 GiB extraction budget")
        if source.testzip() is not None:
            raise ValueError("Archive CRC check failed")
        for item in entries:
            name = PurePosixPath(item.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or any(":" in p for p in name.parts):
                raise ValueError("Unsafe archive path")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive contains a symbolic link")
        destination.mkdir(parents=True)
        counts = Counter()
        for item in entries:
            name = PurePosixPath(item.filename.replace("\\", "/"))
            if item.is_dir() or "__MACOSX" in name.parts:
                continue
            counts[name.suffix.lower()] += 1
            if name.suffix.lower() in {".pt", ".pth", ".onnx"}:
                report["weights"].append(str(name))
            if name.suffix.lower() not in allowed:
                continue
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if name.suffix.lower() in {".yaml", ".yml"}:
                config = yaml.safe_load(target.read_text(encoding="utf-8-sig"))
                if isinstance(config, dict) and "names" in config:
                    report["datasets"].append({"file": str(name), "classes": config["names"]})
        report["files"] = dict(counts)
    (destination / "archive-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/airway/public/cribhd"))
    args = parser.parse_args()
    print(json.dumps(inspect_archive(args.archive, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
