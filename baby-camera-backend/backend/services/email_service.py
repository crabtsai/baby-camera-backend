from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EmailService:
    # 預留 Email 通知。

    def send_email(self, subject: str, body: str) -> None:
        logger.info("EmailService is not implemented yet. subject=%s", subject)
