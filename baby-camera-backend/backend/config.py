from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _to_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _to_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Baby Camera Backend")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = _to_int("APP_PORT", 8000)

    camera_id: str = os.getenv("CAMERA_ID", "baby-cam-01")
    camera_source: str = os.getenv("CAMERA_SOURCE", "0")

    frame_interval_sec: float = _to_float("FRAME_INTERVAL_SEC", 1.0)
    stream_jpeg_quality: int = _to_int("STREAM_JPEG_QUALITY", 80)

    database_path: Path = Path(os.getenv("DATABASE_PATH", "data/baby_camera.db"))
    snapshot_dir: Path = Path(os.getenv("SNAPSHOT_DIR", "data/snapshots"))
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))

    enable_dark_detector: bool = _to_bool(os.getenv("ENABLE_DARK_DETECTOR"), True)
    dark_brightness_threshold: float = _to_float("DARK_BRIGHTNESS_THRESHOLD", 30)
    dark_duration_sec: float = _to_float("DARK_DURATION_SEC", 10)
    dark_cooldown_sec: float = _to_float("DARK_COOLDOWN_SEC", 300)

    enable_motion_detector: bool = _to_bool(os.getenv("ENABLE_MOTION_DETECTOR"), True)
    motion_diff_threshold: float = _to_float("MOTION_DIFF_THRESHOLD", 8)
    motion_low_ratio_threshold: float = _to_float("MOTION_LOW_RATIO_THRESHOLD", 0.003)
    motion_duration_sec: float = _to_float("MOTION_DURATION_SEC", 60)
    motion_cooldown_sec: float = _to_float("MOTION_COOLDOWN_SEC", 600)

    enable_baby_detector: bool = _to_bool(os.getenv("ENABLE_BABY_DETECTOR"), False)
    enable_face_cover_detector: bool = _to_bool(os.getenv("ENABLE_FACE_COVER_DETECTOR"), False)

    alert_webhook_url: str = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    # 送到 Firebase Cloud Function 的驗證密鑰，會放在 x-alert-key header。
    alert_ingest_key: str = os.getenv("ALERT_INGEST_KEY", "").strip()
    alert_timeout_sec: float = _to_float("ALERT_TIMEOUT_SEC", 5)
    alert_dry_run: bool = _to_bool(os.getenv("ALERT_DRY_RUN"), False)

    def ensure_dirs(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
