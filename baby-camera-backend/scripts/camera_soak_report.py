"""Summarize actual observations, explicitly preserving gaps and startup failures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time


def summarize(path):
    stats = defaultdict(lambda: {"samples": 0, "fresh_samples": 0, "fps": [],
        "ages": [], "identical_samples": 0, "unavailable_model_samples": 0,
        "worker_error_samples": 0})
    timestamps, rss, cpu = [], [], []
    last = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # A runner may currently be appending its final line.
            last = record
            timestamps.append(record["at"])
            resources = record.get("resources", {})
            if "rss_bytes" in resources:
                rss.append(resources["rss_bytes"])
                cpu.append(resources["cpu_percent"])
            for camera in record["cameras"]:
                item = stats[camera["camera_id"]]
                capture = camera["capture"]
                item["samples"] += 1
                item["fresh_samples"] += bool(capture["frame_fresh"])
                if capture["capture_fps"] is not None:
                    item["fps"].append(capture["capture_fps"])
                if capture["frame_age_sec"] is not None:
                    item["ages"].append(capture["frame_age_sec"])
                item["identical_samples"] += bool((camera.get("image") or {}).get("same_as_previous_sample"))
                item["unavailable_model_samples"] += any(not w.get("operational", True) for w in camera["workers"])
                item["worker_error_samples"] += any(w["last_error"] for w in camera["workers"])
                item["latest_capture"] = capture
                item["latest_workers"] = camera["workers"]
    for item in stats.values():
        fps, ages = item.pop("fps"), item.pop("ages")
        item["median_fps"] = statistics.median(fps) if fps else None
        item["max_frame_age_sec"] = max(ages) if ages else None
        item["fresh_sample_ratio"] = item["fresh_samples"] / item["samples"]
    return {"samples": len(timestamps), "observation_span_sec": timestamps[-1] - timestamps[0] if timestamps else 0,
        "max_sample_gap_sec": max((b-a for a,b in zip(timestamps,timestamps[1:])), default=0),
        "latest_sample_age_sec": time.time() - timestamps[-1] if timestamps else None,
        "latest_pid": last["pid"] if last else None,
        "peak_rss_bytes": max(rss, default=None), "median_process_cpu_percent": statistics.median(cpu) if cpu else None,
        "cameras": dict(stats), "note": "Sample availability is not continuous uptime; inspect gaps and per-revision records. CPU percent uses one logical core = 100%."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/camera_soak"))
    args = parser.parse_args()
    report = summarize(args.output / "samples.jsonl")
    print(json.dumps(report, ensure_ascii=True, indent=2))
