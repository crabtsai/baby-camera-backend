from __future__ import annotations

import configparser
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit


CONFIG_PATH = Path("config.ini")


def _load_config() -> configparser.ConfigParser:
    # 統一從 config.ini 讀設定；不再讀取 .env，避免來源混在一起。
    # 關閉 interpolation，避免 RTSP 密碼含有 % 時被 configparser 誤判成插值語法。
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(CONFIG_PATH, encoding="utf-8")
    return parser


_CONFIG = _load_config()


def _get(section: str, key: str, default: str) -> str:
    # 設定檔缺 section 或 key 時回傳預設值，讓開發階段少一個設定也能啟動。
    if not _CONFIG.has_section(section):
        return default
    value = _CONFIG.get(section, key, fallback=default)
    return value.strip()


def _get_bool(section: str, key: str, default: bool = False) -> bool:
    # 支援常見布林寫法，config.ini 可以寫 true/yes/1/on。
    value = _get(section, key, str(default))
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _get_float(section: str, key: str, default: float) -> float:
    try:
        return float(_get(section, key, str(default)))
    except ValueError:
        return default


def _get_int(section: str, key: str, default: int) -> int:
    try:
        return int(float(_get(section, key, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class CameraConfig:
    # 單台攝影機設定。id 會寫入事件與告警 payload，用來辨識是哪台相機。
    id: str
    source: str
    source_type: str = "opencv"

    @property
    def display_source(self) -> str:
        # 顯示給前端或 log 看的來源需要遮掉 RTSP 帳密，避免分享畫面時洩漏。
        return mask_camera_source(self.source)


def mask_camera_source(source: str | int) -> str:
    # RTSP/MJPEG URL 可能包含 user:password，狀態頁只顯示 user:***@host。
    text = str(source)
    parsed = urlsplit(text)
    if not parsed.scheme or "@" not in parsed.netloc:
        return text

    userinfo, hostinfo = parsed.netloc.rsplit("@", 1)
    username = userinfo.split(":", 1)[0]
    masked_netloc = f"{username}:***@{hostinfo}" if username else f"***@{hostinfo}"
    return urlunsplit((parsed.scheme, masked_netloc, parsed.path, parsed.query, parsed.fragment))


def _build_tapo_rtsp_source(section: str) -> str:
    # Tapo RTSP 常用格式：rtsp://帳號:密碼@IP:554/stream1 或 /stream2。
    # stream1 通常為主碼流，畫質較高；stream2 通常為子碼流，延遲與流量較低。
    host = _get(section, "rtsp_host", "").strip()
    username = _get(section, "rtsp_username", "").strip()
    password = _get(section, "rtsp_password", "")
    port = _get_int(section, "rtsp_port", 554)
    stream = _get(section, "rtsp_stream", "stream1").strip().lstrip("/") or "stream1"

    if not host:
        # 沒填 IP 時保留 source 作為後備，讓設定未完成時仍可用一般 OpenCV source 啟動。
        return _get(section, "source", "0")

    credentials = ""
    if username or password:
        credentials = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    return f"rtsp://{credentials}{host}:{port}/{stream}"


def _load_camera_config(section: str, camera_id: str) -> CameraConfig:
    # source_type 可選：
    # - opencv：直接把 source 交給 OpenCV，支援 USB index、rtsp、mjpeg、影片檔。
    # - tapo_rtsp：用 Tapo 欄位自動組成 RTSP URL。
    source_type = _get(section, "source_type", "opencv").lower()
    source = _build_tapo_rtsp_source(section) if source_type == "tapo_rtsp" else _get(section, "source", "0")
    return CameraConfig(id=camera_id, source=source, source_type=source_type)


def _get_list(section: str, key: str) -> list[str]:
    # config.ini 內用逗號分隔多個值，例如 ids = baby-cam-01,baby-cam-02。
    value = _get(section, key, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_camera_configs() -> tuple[CameraConfig, ...]:
    # 多相機設定格式：
    # [cameras]
    # ids = baby-cam-01,baby-cam-02
    #
    # [camera.baby-cam-01]
    # source = 0
    #
    # 沒有 [cameras] 時，維持舊版單相機 [camera] 設定。
    camera_ids = _get_list("cameras", "ids")
    if len(camera_ids) != len(set(camera_ids)):
        duplicates = sorted({camera_id for camera_id in camera_ids if camera_ids.count(camera_id) > 1})
        raise ValueError(f"config.ini [cameras] ids 不可重複：{', '.join(duplicates)}")
    if not camera_ids:
        return (
            _load_camera_config("camera", _get("camera", "id", "baby-cam-01")),
        )

    configs: list[CameraConfig] = []
    for camera_id in camera_ids:
        section = f"camera.{camera_id}"
        configs.append(_load_camera_config(section, camera_id))
    return tuple(configs)


@dataclass(frozen=True)
class Settings:
    # 所有設定集中在這裡，其他模組只依賴 Settings，不直接讀 config.ini。
    app_name: str = _get("app", "name", "Baby Camera Backend")
    app_host: str = _get("app", "host", "0.0.0.0")
    app_port: int = _get_int("app", "port", 8000)

    camera_id: str = _get("camera", "id", "baby-cam-01")
    camera_source: str = _get("camera", "source", "0")
    cameras: tuple[CameraConfig, ...] = _load_camera_configs()

    frame_interval_sec: float = _get_float("stream", "frame_interval_sec", 1.0)
    stream_jpeg_quality: int = _get_int("stream", "jpeg_quality", 80)

    database_path: Path = Path(_get("paths", "database_path", "data/baby_camera.db"))
    snapshot_dir: Path = Path(_get("paths", "snapshot_dir", "data/snapshots"))
    log_dir: Path = Path(_get("paths", "log_dir", "logs"))

    enable_dark_detector: bool = _get_bool("detectors.dark", "enabled", True)
    enable_safety_detector: bool = _get_bool("detectors.safety", "enabled", False)
    safety_notify_line: bool = _get_bool("detectors.safety", "notify_line", False)
    dark_brightness_threshold: float = _get_float("detectors.dark", "brightness_threshold", 30)
    dark_duration_sec: float = _get_float("detectors.dark", "duration_sec", 10)
    dark_cooldown_sec: float = _get_float("detectors.dark", "cooldown_sec", 300)

    enable_motion_detector: bool = _get_bool("detectors.motion", "enabled", True)
    motion_diff_threshold: float = _get_float("detectors.motion", "diff_threshold", 8)
    motion_low_ratio_threshold: float = _get_float("detectors.motion", "low_ratio_threshold", 0.003)
    motion_duration_sec: float = _get_float("detectors.motion", "duration_sec", 60)
    motion_cooldown_sec: float = _get_float("detectors.motion", "cooldown_sec", 600)

    enable_baby_detector: bool = _get_bool("detectors.baby", "enabled", False)
    baby_yolo_model_path: str = _get("detectors.baby", "yolo_model_path", "")
    baby_confidence_threshold: float = _get_float("detectors.baby", "confidence_threshold", 0.4)
    baby_face_labels: str = _get("detectors.baby", "face_labels", "baby_face,face")

    enable_prone_sleep_detector: bool = _get_bool("detectors.prone_sleep", "enabled", False)
    prone_sleep_yolo_model_path: str = _get("detectors.prone_sleep", "yolo_model_path", "")
    prone_sleep_keypoint_confidence_threshold: float = _get_float(
        "detectors.prone_sleep", "keypoint_confidence_threshold", 0.35
    )
    prone_sleep_min_body_keypoints: int = _get_int(
        "detectors.prone_sleep", "min_body_keypoints", 3
    )
    prone_sleep_max_visible_face_keypoints: int = _get_int(
        "detectors.prone_sleep", "max_visible_face_keypoints", 1
    )

    enable_face_cover_detector: bool = _get_bool("detectors.face_cover", "enabled", False)
    enable_face_parts_detector: bool = _get_bool("detectors.face_parts", "enabled", False)
    face_parts_model_path: str = _get("detectors.face_parts", "model_path", "models/face_landmarker.task")
    face_parts_confidence: float = _get_float("detectors.face_parts", "confidence", 0.5)
    face_cover_yolo_model_path: str = _get("detectors.face_cover", "yolo_model_path", "")
    face_cover_face_confidence_threshold: float = _get_float(
        "detectors.face_cover",
        "face_confidence_threshold",
        0.4,
    )
    face_cover_part_confidence_threshold: float = _get_float(
        "detectors.face_cover",
        "part_confidence_threshold",
        0.25,
    )
    face_cover_require_nose: bool = _get_bool("detectors.face_cover", "require_nose", True)
    face_cover_require_mouth: bool = _get_bool("detectors.face_cover", "require_mouth", True)

    alert_webhook_url: str = _get("alert", "webhook_url", "")
    # 送到 Firebase Cloud Function 的驗證密鑰，會放在 x-alert-key header。
    alert_ingest_key: str = _get("alert", "ingest_key", "")
    alert_timeout_sec: float = _get_float("alert", "timeout_sec", 5)
    alert_dry_run: bool = _get_bool("alert", "dry_run", False)
    alert_include_snapshot_image: bool = _get_bool("alert", "include_snapshot_image", True)
    alert_snapshot_max_bytes: int = _get_int("alert", "snapshot_max_bytes", 900_000)

    def ensure_dirs(self) -> None:
        # 啟動時先建立資料夾，避免寫 DB、截圖或 log 時才失敗。
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _camera_detector_section(camera_id: str, detector: str) -> str:
        return f"camera.{camera_id}.detectors.{detector}"

    def camera_detector_bool(
        self,
        camera_id: str,
        detector: str,
        key: str,
        default: bool,
    ) -> bool:
        # 允許每台相機覆寫 detector 開關，例如 [camera.baby-cam-02.detectors.motion]。
        return _get_bool(self._camera_detector_section(camera_id, detector), key, default)

    def camera_detector_float(
        self,
        camera_id: str,
        detector: str,
        key: str,
        default: float,
    ) -> float:
        # 允許每台相機覆寫 detector 數值門檻。
        return _get_float(self._camera_detector_section(camera_id, detector), key, default)

    def camera_detector_text(
        self,
        camera_id: str,
        detector: str,
        key: str,
        default: str,
    ) -> str:
        # 允許每台相機覆寫模型路徑或 class label 設定。
        return _get(self._camera_detector_section(camera_id, detector), key, default)


@lru_cache
def get_settings() -> Settings:
    # FastAPI 生命週期內共用同一份設定；修改 config.ini 後需重啟才會生效。
    settings = Settings()
    settings.ensure_dirs()
    return settings
