from __future__ import annotations

from datetime import datetime


def now_text() -> str:
    # 給 log 或畫面顯示用的人類可讀時間。
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_for_filename() -> str:
    # 給檔名用，包含微秒避免同一秒多張截圖撞名。
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
