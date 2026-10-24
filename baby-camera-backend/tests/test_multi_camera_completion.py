from pathlib import Path

import numpy as np

from backend.config import Settings, mask_camera_source
from backend.database import Database
from backend.schemas import EventRecord
from backend.services.line_service import LineService
from backend.services.snapshot_service import SnapshotService


def test_database_filters_events_by_camera_id(tmp_path):
    # 多相機事件共用同一個 SQLite，查詢時應可依 camera_id 過濾。
    database = Database(tmp_path / "events.db")
    database.insert_event(
        EventRecord(
            camera_id="cam-1",
            event_type="room_too_dark",
            detector_name="dark_detector",
            message="cam 1",
            confidence=0.9,
        )
    )
    database.insert_event(
        EventRecord(
            camera_id="cam-2",
            event_type="room_too_dark",
            detector_name="dark_detector",
            message="cam 2",
            confidence=0.9,
        )
    )

    events = database.list_events(camera_id="cam-2")

    assert len(events) == 1
    assert events[0]["camera_id"] == "cam-2"


def test_snapshot_service_saves_under_camera_folder(tmp_path):
    # 截圖依 camera_id 分資料夾，回查告警圖片時不會混在一起。
    service = SnapshotService(tmp_path)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    event = EventRecord(
        camera_id="cam-1",
        event_type="room_too_dark",
        detector_name="dark_detector",
        message="dark",
        confidence=0.8,
    )

    path = service.save_snapshot(frame, event, camera_id="cam-1")

    assert path is not None
    assert Path(path).parent.name == "cam-1"
    assert Path(path).exists()


def test_line_service_payload_includes_snapshot_image(tmp_path):
    # Firebase payload 會帶事件截圖 base64，Cloud Function 可再轉成 LINE 圖片訊息。
    image_path = tmp_path / "snapshot.jpg"
    image_path.write_bytes(b"fake-jpeg")
    service = LineService(
        Settings(
            alert_include_snapshot_image=True,
            alert_snapshot_max_bytes=1000,
        )
    )
    event = EventRecord(
        id=7,
        camera_id="cam-1",
        event_type="room_too_dark",
        detector_name="dark_detector",
        message="dark",
        confidence=0.8,
        snapshot_path=str(image_path),
    )

    payload = service._build_payload(event, level="warning", notify_line=True)

    assert payload["cameraId"] == "cam-1"
    assert payload["snapshotImage"]["included"] is True
    assert payload["snapshotImage"]["mimeType"] == "image/jpeg"
    assert payload["snapshotImage"]["base64"] == "ZmFrZS1qcGVn"


def test_camera_source_masks_rtsp_password():
    # RTSP URL 可能會包含攝影機帳密，對外狀態或前端頁面不可直接顯示密碼。
    source = "rtsp://baby-user:secret-pass@192.168.1.50:554/stream1"

    masked = mask_camera_source(source)

    assert masked == "rtsp://baby-user:***@192.168.1.50:554/stream1"
    assert "secret-pass" not in masked
