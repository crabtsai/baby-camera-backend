from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.alert_service import AlertService
from backend.camera_stream import CameraStream
from backend.config import CameraConfig, Settings
from backend.database import Database
from backend.detector.detector_manager import DetectorManager
from backend.event_manager import EventManager
from backend.frame_reader import FrameReader
from backend.services.snapshot_service import SnapshotService


@dataclass
class CameraRuntime:
    # 單台相機的完整執行管線。
    # 每台相機都擁有自己的 detector 狀態與告警狀態，避免不同畫面互相污染。
    config: CameraConfig
    camera: CameraStream
    detector_manager: DetectorManager
    event_manager: EventManager
    frame_reader: FrameReader

    @property
    def camera_id(self) -> str:
        return self.config.id

    def start(self) -> None:
        # detector worker 先啟動，再啟動抽幀；避免第一張 frame 進來時沒人處理。
        def handle_detector_result(frame, result, metadata):
            self.event_manager.handle_result(frame, result)

        self.detector_manager.start_workers(handle_detector_result, max_queue_size=1)

        def submit_frame(frame):
            self.detector_manager.submit_frame(
                frame,
                metadata={"camera_id": self.camera_id, "source": "webcam"},
            )

        self.frame_reader.register_callback(submit_frame)
        self.frame_reader.start()

    def stop(self) -> None:
        # 停止順序反過來：先停止抽幀，再停止 detector worker，最後釋放 camera。
        self.frame_reader.stop()
        self.detector_manager.stop_workers()
        self.camera.release()

    def status(self) -> dict[str, Any]:
        camera_status = self.camera.status()
        occupancy = self.detector_manager.occupancy_status()
        if not camera_status.get("frame_fresh", False):
            occupancy.update(state="unknown", reason="攝影機畫面已過期或中斷")
        return {
            "camera_id": self.camera_id,
            "source": self.config.display_source,
            "camera": camera_status,
            "occupancy": occupancy,
            "detectors": self.detector_manager.names(),
            "detector_workers": self.detector_manager.worker_status(),
            "airway_alert": self.event_manager.local_airway_alert(),
            "presence_snapshot": self.event_manager.presence_snapshot,
            "reference_snapshot": self.event_manager.reference_snapshot,
        }


class CameraManager:
    # 多相機管理器：統一保存所有 CameraRuntime，API 可依 camera_id 取用。
    def __init__(
        self,
        settings: Settings,
        database: Database,
        snapshot_service: SnapshotService,
        alert_service: AlertService,
    ):
        self.settings = settings
        self.runtimes: dict[str, CameraRuntime] = {}

        for camera_config in settings.cameras:
            camera = CameraStream(camera_config.source)
            self.runtimes[camera_config.id] = CameraRuntime(
                config=camera_config,
                camera=camera,
                detector_manager=DetectorManager.from_settings(settings, camera_id=camera_config.id),
                event_manager=EventManager(
                    settings=settings,
                    database=database,
                    snapshot_service=snapshot_service,
                    alert_service=alert_service,
                    camera_id=camera_config.id,
                ),
                frame_reader=FrameReader(
                    camera,
                    interval_sec=settings.frame_interval_sec,
                ),
            )

    @property
    def default_camera_id(self) -> str:
        # 保留舊版 /stream、/camera 不帶 camera_id 時的預設相機。
        if self.settings.camera_id in self.runtimes:
            return self.settings.camera_id
        return next(iter(self.runtimes))

    def start(self) -> None:
        for runtime in self.runtimes.values():
            runtime.start()

    def stop(self) -> None:
        for runtime in self.runtimes.values():
            runtime.stop()

    def get(self, camera_id: str | None = None) -> CameraRuntime | None:
        return self.runtimes.get(camera_id or self.default_camera_id)

    def list_status(self) -> list[dict[str, Any]]:
        return [runtime.status() for runtime in self.runtimes.values()]

    def default_runtime(self) -> CameraRuntime:
        runtime = self.get(self.default_camera_id)
        if runtime is None:
            raise RuntimeError("找不到預設相機")
        return runtime
