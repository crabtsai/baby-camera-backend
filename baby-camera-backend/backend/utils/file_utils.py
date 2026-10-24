from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    # 小工具：確保指定資料夾存在，並回傳 Path 方便後續串接檔名。
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
