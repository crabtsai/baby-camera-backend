from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.alert_service import AlertService
from backend.api.config_api import router as config_router
from backend.api.event_api import router as event_router
from backend.api.health_api import router as health_router
from backend.api.stream_api import router as stream_router
from backend.camera_stream import CameraStream
from backend.config import get_settings
from backend.database import Database
from backend.detector.detector_manager import DetectorManager
from backend.event_manager import EventManager
from backend.frame_reader import FrameReader
from backend.logger import setup_logging
from backend.services.snapshot_service import SnapshotService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    logger.info("Starting %s", settings.app_name)

    camera = CameraStream(settings.camera_source)
    database = Database(settings.database_path)
    snapshot_service = SnapshotService(settings.snapshot_dir)
    alert_service = AlertService(settings)
    detector_manager = DetectorManager.from_settings(settings)
    event_manager = EventManager(
        settings=settings,
        database=database,
        snapshot_service=snapshot_service,
        alert_service=alert_service,
    )

    frame_reader = FrameReader(camera, interval_sec=settings.frame_interval_sec)

    def detect_callback(frame):
        results = detector_manager.run_all(frame)
        event_manager.handle_results(frame, results)

    frame_reader.register_callback(detect_callback)
    frame_reader.start()

    app.state.settings = settings
    app.state.camera = camera
    app.state.database = database
    app.state.snapshot_service = snapshot_service
    app.state.alert_service = alert_service
    app.state.detector_manager = detector_manager
    app.state.event_manager = event_manager
    app.state.frame_reader = frame_reader

    try:
        yield
    finally:
        logger.info("Stopping %s", settings.app_name)
        frame_reader.stop()
        camera.release()


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
