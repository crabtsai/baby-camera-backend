from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI

from backend.alert_service import AlertService
from backend.audio_monitor import AudioMonitor
from backend.api.config_api import router as config_router
from backend.api.detector_lab_api import router as detector_lab_router
from backend.api.event_api import router as event_router
from backend.api.health_api import router as health_router
from backend.api.stream_api import router as stream_router
from backend.camera_manager import CameraManager
from backend.config import get_settings
from backend.database import Database
from backend.logger import setup_logging
from backend.services.snapshot_service import SnapshotService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if getattr(app.state, "local_monitor_only", False):
        settings = replace(settings, alert_dry_run=True, alert_include_snapshot_image=False)
    setup_logging(settings)
    logger.info("Starting %s", settings.app_name)

    database = Database(settings.database_path)
    snapshot_service = SnapshotService(settings.snapshot_dir)
    alert_service = AlertService(settings)
    camera_manager = CameraManager(
        settings=settings,
        database=database,
        snapshot_service=snapshot_service,
        alert_service=alert_service,
    )
    camera_manager.start()
    default_runtime = camera_manager.default_runtime()

    app.state.settings = settings
    app.state.camera_manager = camera_manager
    # 以下幾個 state 保留給舊版 API 或測試使用，代表預設相機。
    app.state.camera = default_runtime.camera
    app.state.database = database
    app.state.snapshot_service = snapshot_service
    app.state.alert_service = alert_service
    app.state.detector_manager = default_runtime.detector_manager
    app.state.event_manager = default_runtime.event_manager
    app.state.frame_reader = default_runtime.frame_reader
    app.state.audio_monitor = AudioMonitor(settings, database)
    app.state.audio_monitor.start()

    try:
        yield
    finally:
        logger.info("Stopping %s", settings.app_name)
        app.state.audio_monitor.stop()
        camera_manager.stop()


app = FastAPI(
    title="Baby Camera Backend",
    description="Modular Python backend for a smart baby camera system.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(stream_router)
app.include_router(event_router)
app.include_router(config_router)
app.include_router(detector_lab_router)
