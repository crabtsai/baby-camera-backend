from pathlib import Path

for path in ["data/snapshots", "data/recordings", "logs"]:
    Path(path).mkdir(parents=True, exist_ok=True)

print("Project folders created.")
