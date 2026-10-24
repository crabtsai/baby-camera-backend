from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from backend.config import Settings


def setup_logging(settings: Settings) -> None:
    # 統一設定 console + 檔案 log；後端啟動時只需呼叫一次。
    settings.ensure_dirs()
    log_file = settings.log_dir / "app.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # 清掉既有 handler，避免 uvicorn reload 後重複輸出同一行 log。
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_file,
        # 單一 log 檔最多 5MB，保留 3 份舊檔，避免長時間執行把硬碟寫滿。
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
