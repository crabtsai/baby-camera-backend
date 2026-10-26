from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.config import Settings
from backend.detector.baby_detector import BabyDetector
from backend.detector.base_detector import BaseDetector
from backend.detector.dark_detector import DarkDetector
from backend.detector.face_cover import FaceCoverDetector
from backend.detector.baby_face_parts import BabyFacePartsDetector
from backend.detector.motion_detector import MotionDetector
from backend.detector.prone_sleep_detector import ProneSleepDetector
from backend.schemas import DetectionResult

logger = logging.getLogger(__name__)
DetectorResultCallback = Callable[[np.ndarray, DetectionResult, dict[str, Any]], Any]


@dataclass
class DetectorJob:
    # worker queue 裡保存的不只是影像，也保存 frame index / timestamp 等來源資訊。
    # WebCam 即時偵測可以不帶 metadata；Detector Lab 影片模擬會用 metadata 回填結果。
    frame: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectorWorker:
    # 每個 detector 一個 worker thread，讓 YOLO、motion、dark 可以各自處理自己的 frame。
    # 佇列最多保留 10 張最新 frame；如果 detector 來不及算，就丟掉最舊 frame。
    # 這樣可以兼顧短暫緩衝與即時性，避免一直處理很久以前的畫面。
    def __init__(
        self,
        detector: BaseDetector,
        result_callback: DetectorResultCallback,
        max_queue_size: int = 10,
    ):
        self.detector = detector
        self.result_callback = result_callback
        self.frame_queue: queue.Queue[DetectorJob] = queue.Queue(maxsize=max(1, max_queue_size))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._loop,
            name=f"detector-{detector.name}",
            daemon=True,
        )
        self.frames_submitted = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.frames_failed = 0
        self.last_processing_sec: float | None = None
        self.last_result_at: float | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        if self.thread.is_alive():
            return
        logger.info("Starting detector worker: %s", self.detector.name)
        self.stop_event.clear()
        self.thread.start()

    def stop(self) -> None:
        logger.info("Stopping detector worker: %s", self.detector.name)
        self.stop_event.set()
        self.thread.join(timeout=3)

    def submit_frame(self, frame: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        # 每個 detector 拿到自己的 frame copy，避免不同 thread 同時讀寫同一份影像資料。
        job = DetectorJob(frame=frame.copy(), metadata=dict(metadata or {}))
        self.frames_submitted += 1

        try:
            self.frame_queue.put_nowait(job)
            return
        except queue.Full:
            # queue 滿了代表 detector 算太慢；丟掉最舊 frame，保留較新的畫面。
            self.frames_dropped += 1

        try:
            self.frame_queue.get_nowait()
            self.frame_queue.task_done()
        except queue.Empty:
            pass

        try:
            self.frame_queue.put_nowait(job)
        except queue.Full:
            # 極端競態下仍滿佇列時略過本張，下一輪 FrameReader 會再送最新 frame。
            self.frames_dropped += 1

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                job = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                started = time.perf_counter()
                results = self.detector.detect_many(job.frame)
                self.last_error = None
                for result in results:
                    self.result_callback(job.frame, result, job.metadata)
                self.frames_processed += 1
                self.last_result_at = time.time()
            except Exception as exc:
                self.frames_failed += 1
                self.last_error = str(exc)
                logger.exception("Detector worker failed: %s", self.detector.name)
            finally:
                self.last_processing_sec = time.perf_counter() - started
                self.frame_queue.task_done()

        self.detector.close()

    def status(self) -> dict[str, Any]:
        # 提供 /health 查看各 detector worker 是否還活著，以及是否有掉 frame。
        return {
            "name": self.detector.name,
            "alive": self.thread.is_alive(),
            "operational": getattr(self.detector, "operational", True) and not (hasattr(self.detector, "predict_fn") and self.detector.predict_fn is None),
            "max_queue_size": self.frame_queue.maxsize,
            "queued_frames": self.frame_queue.qsize(),
            "frames_submitted": self.frames_submitted,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "frames_failed": self.frames_failed,
            "last_processing_sec": self.last_processing_sec,
            "last_result_at": self.last_result_at,
            "last_error": self.last_error,
            "capabilities": getattr(self.detector, "capabilities", {}),
            "assessments": getattr(self.detector, "last_results", []),
        }


class DetectorManager:
    # 統一管理所有 detector。
    # Detector Lab 仍可用 run_all 同步測試；WebCam 即時偵測則使用 worker thread 平行處理。
    def __init__(self, detectors: list[BaseDetector]):
        self.detectors = detectors
        self.workers: list[DetectorWorker] = []
        logger.info("Enabled detectors: %s", [d.name for d in detectors])

    @classmethod
    def from_settings(cls, settings: Settings, camera_id: str | None = None) -> "DetectorManager":
        # 依照 config.ini 開關建立 detector；未啟用的 detector 不會被加入執行清單。
        detectors: list[BaseDetector] = []
        camera = camera_id or settings.camera_id

        enable_dark = settings.camera_detector_bool(
            camera,
            "dark",
            "enabled",
            settings.enable_dark_detector,
        )
        if enable_dark:
            # 畫面亮度偵測：不依賴模型，成本低，適合常駐。
            detectors.append(
                DarkDetector(
                    settings.camera_detector_float(
                        camera,
                        "dark",
                        "brightness_threshold",
                        settings.dark_brightness_threshold,
                    )
                )
            )

        enable_motion = settings.camera_detector_bool(
            camera,
            "motion",
            "enabled",
            settings.enable_motion_detector,
        )
        if enable_motion:
            # 動作偵測：比較前後 frame 的差異，用來找長時間低動作。
            detectors.append(
                MotionDetector(
                    diff_threshold=settings.camera_detector_float(
                        camera,
                        "motion",
                        "diff_threshold",
                        settings.motion_diff_threshold,
                    ),
                    low_ratio_threshold=settings.camera_detector_float(
                        camera,
                        "motion",
                        "low_ratio_threshold",
                        settings.motion_low_ratio_threshold,
                    ),
                )
            )

        enable_baby = settings.camera_detector_bool(
            camera,
            "baby",
            "enabled",
            settings.enable_baby_detector,
        )
        if enable_baby:
            # 寶寶臉部偵測：使用 YOLO 模型判斷畫面中是否有 baby face / face。
            baby_face_labels = settings.camera_detector_text(
                camera,
                "baby",
                "face_labels",
                settings.baby_face_labels,
            )
            detectors.append(
                BabyDetector(
                    model_path=settings.camera_detector_text(
                        camera,
                        "baby",
                        "yolo_model_path",
                        settings.baby_yolo_model_path,
                    ),
                    confidence_threshold=settings.camera_detector_float(
                        camera,
                        "baby",
                        "confidence_threshold",
                        settings.baby_confidence_threshold,
                    ),
                    face_labels={
                        label.strip()
                        for label in baby_face_labels.split(",")
                        if label.strip()
                    },
                )
            )

        enable_prone_sleep = settings.camera_detector_bool(
            camera, "prone_sleep", "enabled", settings.enable_prone_sleep_detector
        )
        if enable_prone_sleep:
            detectors.append(
                ProneSleepDetector(
                    model_path=settings.camera_detector_text(
                        camera, "prone_sleep", "yolo_model_path", settings.prone_sleep_yolo_model_path
                    ),
                    keypoint_confidence_threshold=settings.camera_detector_float(
                        camera,
                        "prone_sleep",
                        "keypoint_confidence_threshold",
                        settings.prone_sleep_keypoint_confidence_threshold,
                    ),
                    min_body_keypoints=int(
                        settings.camera_detector_float(
                            camera,
                            "prone_sleep",
                            "min_body_keypoints",
                            float(settings.prone_sleep_min_body_keypoints),
                        )
                    ),
                    max_visible_face_keypoints=int(
                        settings.camera_detector_float(
                            camera,
                            "prone_sleep",
                            "max_visible_face_keypoints",
                            float(settings.prone_sleep_max_visible_face_keypoints),
                        )
                    ),
                )
            )

        if settings.camera_detector_bool(camera, "face_parts", "enabled", settings.enable_face_parts_detector):
            detectors.append(BabyFacePartsDetector.from_settings(settings, camera))

        enable_face_cover = settings.camera_detector_bool(
            camera,
            "face_cover",
            "enabled",
            settings.enable_face_cover_detector,
        )
        if enable_face_cover:
            # 臉部遮蔽偵測：需要能辨識 face / nose / mouth 的 YOLO 模型。
            detectors.append(
                FaceCoverDetector(
                    model_path=settings.camera_detector_text(
                        camera,
                        "face_cover",
                        "yolo_model_path",
                        settings.face_cover_yolo_model_path,
                    ),
                    face_confidence_threshold=settings.camera_detector_float(
                        camera,
                        "face_cover",
                        "face_confidence_threshold",
                        settings.face_cover_face_confidence_threshold,
                    ),
                    part_confidence_threshold=settings.camera_detector_float(
                        camera,
                        "face_cover",
                        "part_confidence_threshold",
                        settings.face_cover_part_confidence_threshold,
                    ),
                    require_nose=settings.camera_detector_bool(
                        camera,
                        "face_cover",
                        "require_nose",
                        settings.face_cover_require_nose,
                    ),
                    require_mouth=settings.camera_detector_bool(
                        camera,
                        "face_cover",
                        "require_mouth",
                        settings.face_cover_require_mouth,
                    ),
                )
            )

        if settings.camera_detector_bool(camera, "safety", "enabled", settings.enable_safety_detector):
            from backend.detector.safety_pipeline import SafetyPipeline
            safety = SafetyPipeline.from_settings(settings, camera)
            detectors = safety.attach_dependents(detectors) + [safety]

        return cls(detectors)

    def run_all(self, frame: np.ndarray) -> list[DetectionResult]:
        # 單一 detector 失敗時只記錄錯誤，不中斷其他 detector 與後端服務。
        results: list[DetectionResult] = []
        for detector in self.detectors:
            try:
                results.extend(detector.detect_many(frame))
            except Exception:
                logger.exception("Detector failed: %s", detector.name)
        return results

    def start_workers(self, result_callback: DetectorResultCallback, max_queue_size: int = 10) -> None:
        # WebCam 即時模式：每個 detector 建立自己的 thread 與 frame queue。
        if self.workers:
            return
        self.workers = [
            DetectorWorker(detector, result_callback=result_callback, max_queue_size=max_queue_size)
            for detector in self.detectors
        ]
        for worker in self.workers:
            worker.start()

    def submit_frame(self, frame: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        # FrameReader 每抽到一張 frame，就廣播給所有 detector worker。
        # 沒有 worker 時保留 no-op，避免沒有啟用 detector 時出錯。
        for worker in self.workers:
            worker.submit_frame(frame, metadata=metadata)

    def wait_until_idle(self, timeout_sec: float = 30) -> bool:
        # Detector Lab 影片模擬用：等待所有 worker 把已提交或被丟棄的 frame 處理完。
        stop_at = time.monotonic() + timeout_sec
        while time.monotonic() < stop_at:
            if all(worker.frame_queue.unfinished_tasks == 0 for worker in self.workers):
                return True
            time.sleep(0.05)
        return False

    def stop_workers(self) -> None:
        for worker in self.workers:
            worker.stop()
        self.workers = []

    def names(self) -> list[str]:
        return [name for detector in self.detectors
                for name in [detector.name, *(d.name for d in getattr(detector, "dependents", []))]]

    def occupancy_status(self) -> dict[str, Any]:
        for detector in self.detectors:
            if hasattr(detector, "occupancy"):
                return detector.occupancy.status()
        return {"state": "unknown", "reason": "在床判斷未啟用", "identity_verified": False}

    def worker_status(self) -> list[dict[str, Any]]:
        return [worker.status() for worker in self.workers]
