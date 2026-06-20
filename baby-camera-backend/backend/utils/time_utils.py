from __future__ import annotations

from datetime import datetime


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
