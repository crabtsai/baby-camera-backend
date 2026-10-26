from __future__ import annotations

import base64
import threading
import time
from dataclasses import replace
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend.detector.baby_detector import BabyDetector
from backend.detector.dark_detector import DarkDetector
from backend.detector.detector_manager import DetectorManager
from backend.detector.face_cover import FaceCoverDetector
from backend.detector.baby_face_parts import BabyFacePartsDetector
from backend.detector.motion_detector import MotionDetector
from backend.detector.safety_pipeline import SafetyPipeline, parse_region
from backend.schemas import DetectionResult

router = APIRouter()


class DetectorLabParams(BaseModel):
    safety_sleep_region: str = ""
    # 調參頁送進來的 detector 參數，之後新增 detector 時可以往這裡擴充。
    dark_brightness_threshold: float = Field(default=30, ge=0, le=255)
    motion_diff_threshold: float = Field(default=8, ge=0, le=255)
    motion_low_ratio_threshold: float = Field(default=0.003, ge=0, le=1)
    baby_confidence_threshold: float = Field(default=0.4, ge=0, le=1)
    baby_face_labels: str = "baby_face,face"
    face_cover_face_confidence_threshold: float = Field(default=0.4, ge=0, le=1)
    face_cover_part_confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    face_cover_require_nose: bool = True
    face_cover_require_mouth: bool = True
    webcam_frame_interval_sec: float = Field(default=1.0, ge=0, le=5)


class DetectorLabFrame(BaseModel):
    # 單張測試 frame，image_data 是瀏覽器 canvas 產生的 data URL。
    image_data: str
    timestamp_sec: float | None = None


class DetectorLabRequest(BaseModel):
    # 前端一次送多張 frame，後端依序跑 detector，保留影片時間點方便對照。
    detector: str = "all"
    source_type: str = "image"
    camera_id: str | None = None
    params: DetectorLabParams = Field(default_factory=DetectorLabParams)
    frames: list[DetectorLabFrame]


def _decode_frame(image_data: str) -> np.ndarray:
    # 支援 data:image/jpeg;base64,... 或純 base64 字串。
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    raw = base64.b64decode(image_data)
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("無法解析影像資料")
    return frame


def _result_to_dict(result: DetectionResult) -> dict[str, Any]:
    # dataclass 轉成前端容易渲染的 JSON。
    return {
        "detector_name": result.detector_name,
        "event_type": result.event_type,
        "is_abnormal": result.is_abnormal,
        "confidence": result.confidence,
        "message": result.message,
        "extra": result.extra,
    }


def _split_labels(text: str) -> set[str]:
    # 使用逗號分隔 YOLO class 名稱，例如 baby_face,face。
    return {label.strip() for label in text.split(",") if label.strip()}


def _build_detectors(
    detector: str,
    params: DetectorLabParams,
    settings: Any | None = None,
    camera_id: str | None = None,
):
    # 每次測試都建立新的 detector，避免 motion 前一張 frame 狀態被上次測試污染。
    if settings is not None and camera_id:
        baby_model_path = settings.camera_detector_text(
            camera_id,
            "baby",
            "yolo_model_path",
            settings.baby_yolo_model_path,
        )
        face_cover_model_path = settings.camera_detector_text(
            camera_id,
            "face_cover",
            "yolo_model_path",
            settings.face_cover_yolo_model_path,
        )
    else:
        baby_model_path = getattr(settings, "baby_yolo_model_path", "") if settings is not None else ""
        face_cover_model_path = getattr(settings, "face_cover_yolo_model_path", "") if settings is not None else ""

    available = {
        "baby_face_parts_detector": BabyFacePartsDetector.from_settings(settings, camera_id),
        "dark_detector": DarkDetector(params.dark_brightness_threshold),
        "motion_detector": MotionDetector(
            diff_threshold=params.motion_diff_threshold,
            low_ratio_threshold=params.motion_low_ratio_threshold,
            include_debug_images=True,
        ),
        "baby_detector": BabyDetector(
            model_path=baby_model_path,
            confidence_threshold=params.baby_confidence_threshold,
            face_labels=_split_labels(params.baby_face_labels),
        ),
        "face_cover_detector": FaceCoverDetector(
            model_path=face_cover_model_path,
            face_confidence_threshold=params.face_cover_face_confidence_threshold,
            part_confidence_threshold=params.face_cover_part_confidence_threshold,
            require_nose=params.face_cover_require_nose,
            require_mouth=params.face_cover_require_mouth,
        ),
    }

    if detector in {"all", "safety_pipeline"}:
        safety = SafetyPipeline.from_settings(settings, camera_id or settings.camera_id) if settings else SafetyPipeline()
        if params.safety_sleep_region.strip():
            safety.options = replace(safety.options, region=parse_region(params.safety_sleep_region))
            if safety.scene_fn is not None:
                safety.scene_fn.options = safety.options
            safety.capabilities["calibration"] = {"available": True, "reason": "測試區域"}
        available["safety_pipeline"] = safety

    if detector == "all":
        return safety.attach_dependents([d for d in available.values() if d is not safety]) + [safety]

    selected = available.get(detector)
    if selected is None:
        raise ValueError(f"未知 detector：{detector}")
    return [selected]


def _build_summary(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    # 彙整每個 detector 的異常次數與平均信心值，方便快速比較參數效果。
    summary: dict[str, dict[str, Any]] = {}
    for frame_result in frame_results:
        for detection in frame_result["detections"]:
            name = detection["detector_name"]
            if name == "safety_pipeline":
                name += ":" + detection["event_type"]
            item = summary.setdefault(name, {"total": 0, "abnormal": 0, "unknown": 0, "confidence_sum": 0.0})
            item["total"] += 1
            item["confidence_sum"] += float(detection["confidence"] or 0)
            if detection["is_abnormal"]:
                item["abnormal"] += 1
            if detection.get("extra", {}).get("evaluable") is False:
                item["unknown"] += 1

    for item in summary.values():
        total = max(1, item["total"])
        item["avg_confidence"] = round(item["confidence_sum"] / total, 4)
        del item["confidence_sum"]

    return summary


def _run_sync_lab(detectors, frames: list[DetectorLabFrame]) -> dict[str, Any]:
    # 圖片或單純調參模式：同一個 request 內依序跑 detector，結果最容易對照。
    frame_results: list[dict[str, Any]] = []

    for index, frame_payload in enumerate(frames):
        frame = _decode_frame(frame_payload.image_data)
        detections = [_result_to_dict(result) for detector in detectors for result in detector.detect_many(frame)]
        frame_results.append(
            {
                "index": index,
                "timestamp_sec": frame_payload.timestamp_sec,
                "detections": detections,
            }
        )

    return {
        "ok": True,
        "mode": "sync",
        "frames": frame_results,
        "summary": _build_summary(frame_results),
        "worker_status": [],
    }


def _run_webcam_simulation_lab(
    detectors,
    frames: list[DetectorLabFrame],
    frame_interval_sec: float,
) -> dict[str, Any]:
    # 影片測試模式：模擬實際 WebCam 進入 DetectorManager worker queue 的流程。
    # 每個 detector 自己排隊、自己計算；queue 滿了會丟最舊 frame，和即時模式一致。
    manager = DetectorManager(detectors)
    lock = threading.Lock()
    frame_results: list[dict[str, Any]] = [
        {
            "index": index,
            "timestamp_sec": frame_payload.timestamp_sec,
            "detections": [],
        }
        for index, frame_payload in enumerate(frames)
    ]

    def collect_result(frame: np.ndarray, result: DetectionResult, metadata: dict[str, Any]):
        index = int(metadata.get("index", -1))
        if index < 0 or index >= len(frame_results):
            return
        with lock:
            frame_results[index]["detections"].append(_result_to_dict(result))

    manager.start_workers(collect_result)
    try:
        for index, frame_payload in enumerate(frames):
            frame = _decode_frame(frame_payload.image_data)
            manager.submit_frame(
                frame,
                metadata={
                    "index": index,
                    "timestamp_sec": frame_payload.timestamp_sec,
                    "source": "detector_lab_video",
                },
            )
            # 模擬 WebCam 實際送 frame 的節奏；設成 0 可快速壓力測試 queue 丟幀行為。
            if frame_interval_sec > 0 and index < len(frames) - 1:
                time.sleep(frame_interval_sec)

        completed = manager.wait_until_idle(timeout_sec=60)
        worker_status = manager.worker_status()
    finally:
        manager.stop_workers()

    for frame_result in frame_results:
        # 多執行緒完成順序不固定，排序後前端顯示較穩定。
        frame_result["detections"].sort(key=lambda item: item["detector_name"])

    return {
        "ok": True,
        "mode": "webcam_simulation",
        "completed": completed,
        "frame_interval_sec": frame_interval_sec,
        "frames": frame_results,
        "summary": _build_summary(frame_results),
        "worker_status": worker_status,
    }


@router.post("/detector-lab/run")
def run_detector_lab(payload: DetectorLabRequest, request: Request):
    # Detector Lab 測試入口：不寫資料庫、不發告警，只回傳每張 frame 的判斷結果。
    try:
        detectors = _build_detectors(
            payload.detector, payload.params, request.app.state.settings, camera_id=payload.camera_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if payload.source_type == "video":
        return _run_webcam_simulation_lab(
            detectors,
            payload.frames,
            frame_interval_sec=payload.params.webcam_frame_interval_sec,
        )
    try:
        return _run_sync_lab(detectors, payload.frames)
    finally:
        for detector in detectors:
            detector.close()


@router.get("/detector-lab", response_class=HTMLResponse)
def detector_lab_page():
    # Detector Lab 是調參入口：上傳圖片/影片、調整參數、立即看每個 detector 結果。
    return """
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Detector Lab 調參窗口</title>
  <style>
    :root { --bg:#f5f7fb; --panel:#fff; --text:#17202a; --muted:#667085; --line:#d9e0ea; --accent:#2563eb; --ok:#0f8f5f; --bad:#c0392b; --warn:#b26a00; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:16px 22px; border-bottom:1px solid var(--line); background:var(--panel); }
    h1 { margin:0; font-size:20px; letter-spacing:0; }
    main { display:grid; grid-template-columns:340px minmax(0,1fr); gap:16px; max-width:1360px; margin:0 auto; padding:16px; }
    aside, section { background:var(--panel); border:1px solid var(--line); border-radius:8px; }
    aside { padding:16px; align-self:start; }
    .field { display:grid; gap:7px; margin-bottom:16px; }
    label { color:var(--muted); font-size:13px; font-weight:700; }
    select, input[type="file"], input[type="number"], input[type="text"] { width:100%; min-height:36px; border:1px solid var(--line); border-radius:6px; padding:6px 9px; font:inherit; background:#fff; }
    input[type="range"] { width:100%; }
    .range-line { display:grid; grid-template-columns:1fr 82px; gap:8px; align-items:center; }
    button { min-height:38px; border:1px solid var(--accent); background:var(--accent); color:#fff; border-radius:6px; padding:0 14px; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .workspace { display:grid; grid-template-rows:auto 1fr; overflow:hidden; }
    .preview { min-height:300px; background:#111827; display:grid; place-items:center; border-bottom:1px solid var(--line); }
    .preview img, .preview video { max-width:100%; max-height:52vh; object-fit:contain; }
    .preview-stack { position:relative; display:grid; place-items:center; max-width:100%; }
    .preview-stack canvas { display:block; max-width:100%; max-height:52vh; border:1px solid rgba(255,255,255,.18); }
    .results { padding:14px; overflow-x:auto; }
    .status { color:var(--muted); font-size:14px; margin-bottom:12px; }
    .summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin-bottom:14px; }
    .summary-item { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcff; }
    .summary-item strong { display:block; margin-bottom:6px; overflow-wrap:anywhere; }
    .process-list { display:grid; gap:8px; margin-bottom:14px; }
    .process-item { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; font-size:13px; }
    .process-title { display:flex; justify-content:space-between; gap:12px; margin-bottom:6px; font-weight:800; }
    .process-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:6px 10px; color:var(--muted); }
    .process-grid span { color:var(--text); font-weight:700; }
    .process-images { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin-top:10px; }
    .process-image { border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#111827; }
    .process-image img { display:block; width:100%; height:150px; object-fit:contain; background:#111827; }
    .process-image div { padding:6px 8px; color:#e5e7eb; font-size:12px; background:#1f2937; }
    table { width:100%; border-collapse:collapse; font-size:14px; }
    th, td { border-bottom:1px solid var(--line); padding:9px 8px; text-align:left; vertical-align:top; }
    th { color:var(--muted); font-size:12px; white-space:nowrap; }
    .pill { display:inline-flex; align-items:center; min-height:24px; padding:2px 8px; border-radius:999px; border:1px solid currentColor; font-size:12px; font-weight:800; }
    .ok { color:var(--ok); } .bad { color:var(--bad); } .warn { color:var(--warn); }
    code { white-space:pre-wrap; overflow-wrap:anywhere; font-size:12px; }
    canvas { display:none; }
    @media (max-width:900px) { main { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Detector Lab 調參窗口</h1>
    <button id="run" type="button" disabled>執行偵測</button>
  </header>
  <main>
    <aside>
      <div class="field">
        <label for="file">測試圖片或影片</label>
        <input id="file" type="file" accept="image/*,video/*">
      </div>
      <div class="field">
        <label for="camera-id">套用相機設定</label>
        <select id="camera-id"></select>
      </div>
      <div class="field">
        <label for="detector">偵測項目</label>
        <select id="detector">
          <option value="all">全部 detector</option>
          <option value="dark_detector">畫面太暗</option>
          <option value="motion_detector">寶寶動作 / 低動作</option>
          <option value="baby_detector">寶寶臉部</option>
          <option value="face_cover_detector">臉部遮蔽</option>
          <option value="baby_face_parts_detector">寶寶口鼻定位（MediaPipe 離線）</option>
          <option value="safety_pipeline">俯拍：品質／異物／頭部遮擋／邊界／睡姿</option>
        </select>
      </div>
      <div class="field">
        <label for="safety-region">床墊區域：左、上、右、下（0～1）</label>
        <input id="safety-region" type="text" placeholder="例如 0.1,0.1,0.9,0.9；空白使用相機設定">
        <small>這裡只套用本次測試。正式監測請設定各相機的 sleep_region；相機移動後需重新校正。</small>
      </div>
      <div class="field">
        <label for="dark">畫面太暗門檻</label>
        <div class="range-line"><input id="dark" type="range" min="0" max="255" step="1" value="30"><input id="dark-value" type="number" min="0" max="255" step="1" value="30"></div>
      </div>
      <div class="field">
        <label for="motion-diff">動作差異門檻</label>
        <div class="range-line"><input id="motion-diff" type="range" min="0" max="80" step="1" value="8"><input id="motion-diff-value" type="number" min="0" max="80" step="1" value="8"></div>
      </div>
      <div class="field">
        <label for="motion-low">低動作比例門檻</label>
        <div class="range-line"><input id="motion-low" type="range" min="0" max="0.05" step="0.0005" value="0.003"><input id="motion-low-value" type="number" min="0" max="0.05" step="0.0005" value="0.003"></div>
      </div>
      <div class="field">
        <label for="baby-threshold">寶寶臉部信心門檻</label>
        <div class="range-line"><input id="baby-threshold" type="range" min="0" max="1" step="0.01" value="0.4"><input id="baby-threshold-value" type="number" min="0" max="1" step="0.01" value="0.4"></div>
      </div>
      <div class="field">
        <label for="baby-labels">寶寶臉部 class 名稱</label>
        <input id="baby-labels" type="text" value="baby_face,face">
      </div>
      <div class="field">
        <label for="face-threshold">遮蔽臉部信心門檻</label>
        <div class="range-line"><input id="face-threshold" type="range" min="0" max="1" step="0.01" value="0.4"><input id="face-threshold-value" type="number" min="0" max="1" step="0.01" value="0.4"></div>
      </div>
      <div class="field">
        <label for="part-threshold">遮蔽口鼻信心門檻</label>
        <div class="range-line"><input id="part-threshold" type="range" min="0" max="1" step="0.01" value="0.25"><input id="part-threshold-value" type="number" min="0" max="1" step="0.01" value="0.25"></div>
      </div>
      <div class="field">
        <label><input id="require-nose" type="checkbox" checked> 要求偵測到鼻子</label>
        <label><input id="require-mouth" type="checkbox" checked> 要求偵測到嘴巴</label>
      </div>
      <div class="field">
        <label for="max-frames">影片最多抽幀數</label>
        <input id="max-frames" type="number" min="2" max="80" step="1" value="24">
      </div>
      <div class="field">
        <label for="webcam-interval">WebCam 模擬送幀間隔秒數</label>
        <input id="webcam-interval" type="number" min="0" max="5" step="0.1" value="1.0">
      </div>
      <div class="field">
        <label>
          <input id="annotate-video" type="checkbox">
          繪製偵測結果並另存影片
        </label>
      </div>
    </aside>
    <section class="workspace">
      <div class="preview" id="preview"><span class="pill warn">尚未載入檔案</span></div>
      <div class="results">
        <div id="status" class="status">選擇圖片或影片後即可開始調參。</div>
        <div id="summary" class="summary"></div>
        <div id="process" class="process-list"></div>
        <table>
          <thead><tr><th>Frame</th><th>時間</th><th>Detector</th><th>狀態</th><th>信心值</th><th>訊息</th><th>細節</th></tr></thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </section>
  </main>
  <canvas id="canvas"></canvas>
  <script>
    const fileInput = document.getElementById("file");
    const preview = document.getElementById("preview");
    const runButton = document.getElementById("run");
    const statusText = document.getElementById("status");
    const summary = document.getElementById("summary");
    const process = document.getElementById("process");
    const rows = document.getElementById("rows");
    const canvas = document.getElementById("canvas");
    const detector = document.getElementById("detector");
    const cameraId = document.getElementById("camera-id");
    const maxFrames = document.getElementById("max-frames");
    const annotateVideo = document.getElementById("annotate-video");
    const linkedInputs = [["dark","dark-value"],["motion-diff","motion-diff-value"],["motion-low","motion-low-value"],["baby-threshold","baby-threshold-value"],["face-threshold","face-threshold-value"],["part-threshold","part-threshold-value"]];
    let currentFile = null;
    let objectUrl = null;
    let annotatedUrl = null;

    for (const [rangeId, numberId] of linkedInputs) {
      const range = document.getElementById(rangeId);
      const number = document.getElementById(numberId);
      // range 與 number 共用同一個參數，方便拖拉或精準輸入。
      range.addEventListener("input", () => number.value = range.value);
      number.addEventListener("input", () => range.value = number.value);
    }

    function clearObjectUrl() {
      // 釋放瀏覽器暫存 URL，避免反覆載入影片時占用記憶體。
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = null;
      if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
      annotatedUrl = null;
    }

    function setStatus(text) {
      statusText.textContent = text;
    }

    async function loadCameraOptions() {
      // 從 /config 讀取相機列表，Detector Lab 可指定要套用哪台相機的模型路徑與設定。
      try {
        const response = await fetch("/config", { cache: "no-store" });
        const data = await response.json();
        const cameras = data.cameras || [];
        cameraId.innerHTML = "";
        for (const item of cameras) {
          const option = document.createElement("option");
          option.value = item.id;
          option.textContent = `${item.id} (${item.source})`;
          cameraId.appendChild(option);
        }
      } catch (error) {
        cameraId.innerHTML = '<option value="">預設相機</option>';
      }
    }

    fileInput.addEventListener("change", () => {
      // 載入檔案後先做前端預覽；真正偵測要等按下「執行偵測」。
      currentFile = fileInput.files?.[0] || null;
      summary.innerHTML = "";
      process.innerHTML = "";
      rows.innerHTML = "";
      clearObjectUrl();
      if (!currentFile) {
        runButton.disabled = true;
        preview.innerHTML = '<span class="pill warn">尚未載入檔案</span>';
        setStatus("選擇圖片或影片後即可開始調參。");
        return;
      }
      objectUrl = URL.createObjectURL(currentFile);
      runButton.disabled = false;
      if (currentFile.type.startsWith("video/")) {
        preview.innerHTML = `<video src="${objectUrl}" controls muted></video>`;
        setStatus("影片已載入，可調整參數後執行偵測。");
      } else {
        preview.innerHTML = `<img src="${objectUrl}" alt="測試圖片">`;
        setStatus("圖片已載入，可調整參數後執行偵測。");
      }
    });

    function getParams() {
      // 收集左側調參值，送到 /detector-lab/run 建立一次性的 detector。
      return {
        dark_brightness_threshold: Number(document.getElementById("dark-value").value),
        motion_diff_threshold: Number(document.getElementById("motion-diff-value").value),
        motion_low_ratio_threshold: Number(document.getElementById("motion-low-value").value),
        baby_confidence_threshold: Number(document.getElementById("baby-threshold-value").value),
        baby_face_labels: document.getElementById("baby-labels").value,
        safety_sleep_region: document.getElementById("safety-region").value,
        face_cover_face_confidence_threshold: Number(document.getElementById("face-threshold-value").value),
        face_cover_part_confidence_threshold: Number(document.getElementById("part-threshold-value").value),
        face_cover_require_nose: document.getElementById("require-nose").checked,
        face_cover_require_mouth: document.getElementById("require-mouth").checked,
        webcam_frame_interval_sec: Number(document.getElementById("webcam-interval").value),
      };
    }

    function drawElementToDataUrl(element, width, height) {
      // 將圖片或影片目前畫面畫到 canvas，再轉成 base64 給後端 OpenCV 解碼。
      // 限制最大寬度可以降低傳輸量與 YOLO 測試成本。
      const maxWidth = 960;
      const scale = Math.min(1, maxWidth / Math.max(1, width));
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      canvas.getContext("2d").drawImage(element, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.86);
    }

    async function imageToFrame(file) {
      // 圖片只需要轉成一張 frame。
      const image = new Image();
      image.src = URL.createObjectURL(file);
      await image.decode();
      const imageData = drawElementToDataUrl(image, image.naturalWidth, image.naturalHeight);
      URL.revokeObjectURL(image.src);
      return [{ image_data: imageData, timestamp_sec: 0, width: canvas.width, height: canvas.height }];
    }

    function waitForSeek(video, time) {
      // 影片抽幀需要先 seek 到指定秒數，等瀏覽器完成後才能擷取畫面。
      return new Promise((resolve, reject) => {
        const timeout = window.setTimeout(() => reject(new Error("影片 seek 逾時")), 5000);
        video.onseeked = () => { window.clearTimeout(timeout); resolve(); };
        video.currentTime = time;
      });
    }

    async function videoToFrames(file) {
      // 將影片平均抽成多張 frame，方便快速測不同參數對整段影片的效果。
      const video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.src = URL.createObjectURL(file);
      await new Promise((resolve, reject) => {
        video.onloadedmetadata = resolve;
        video.onerror = () => reject(new Error("影片載入失敗"));
      });
      const count = Math.max(2, Math.min(80, Number(maxFrames.value) || 24));
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : count;
      const frames = [];
      for (let i = 0; i < count; i += 1) {
        const timestamp = count === 1 ? 0 : (duration * i) / (count - 1);
        await waitForSeek(video, Math.min(duration, timestamp));
        const imageData = drawElementToDataUrl(video, video.videoWidth, video.videoHeight);
        frames.push({ image_data: imageData, timestamp_sec: Number(timestamp.toFixed(3)), width: canvas.width, height: canvas.height });
      }
      URL.revokeObjectURL(video.src);
      return frames;
    }

    function loadImage(dataUrl) {
      // 標註預覽需要先把 base64 frame 載回 Image，再畫到 canvas。
      return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = dataUrl;
      });
    }

    function collectDetections(frameResult) {
      // detector 的框資訊放在 extra.detections，這裡攤平成繪圖用格式。
      const items = [];
      for (const detection of frameResult.detections || []) {
        const extra = detection.extra || {};
        for (const item of detection.extra?.detections || []) {
          items.push({
            ...item,
            detectorName: detection.detector_name,
            abnormal: detection.is_abnormal,
            sourceWidth: extra.frame_width,
            sourceHeight: extra.frame_height,
          });
        }
      }
      return items;
    }

    function drawLabel(context, text, x, y, color) {
      // 在偵測框上方畫 class 名稱與信心值。
      context.font = "14px system-ui, sans-serif";
      const metrics = context.measureText(text);
      const width = metrics.width + 12;
      const top = Math.max(4, y - 24);
      context.fillStyle = "rgba(17, 24, 39, 0.84)";
      context.fillRect(x, top, width, 22);
      context.fillStyle = color;
      context.fillText(text, x + 6, top + 16);
    }

    function normalizeDetectionBox(context, detection) {
      // YOLO 通常回傳像素座標，但這裡也容忍 0~1 正規化座標或不同 frame 尺寸。
      if (!Array.isArray(detection.box) || detection.box.length < 4) return;
      let [x1, y1, x2, y2] = detection.box.map((value) => Number(value));
      if ([x1, y1, x2, y2].some((value) => !Number.isFinite(value))) return null;

      const canvasWidth = context.canvas.width;
      const canvasHeight = context.canvas.height;
      const maxCoord = Math.max(Math.abs(x1), Math.abs(y1), Math.abs(x2), Math.abs(y2));
      if (maxCoord <= 1.5) {
        // 有些模型或測試資料會給 normalized xyxy，需放大到 canvas 尺寸。
        x1 *= canvasWidth;
        x2 *= canvasWidth;
        y1 *= canvasHeight;
        y2 *= canvasHeight;
      } else if (detection.sourceWidth && detection.sourceHeight) {
        // 若 detector 計算尺寸和目前輸出 canvas 不同，依比例縮放回輸出影片。
        x1 *= canvasWidth / Number(detection.sourceWidth);
        x2 *= canvasWidth / Number(detection.sourceWidth);
        y1 *= canvasHeight / Number(detection.sourceHeight);
        y2 *= canvasHeight / Number(detection.sourceHeight);
      }

      const left = Math.min(x1, x2);
      const right = Math.max(x1, x2);
      const top = Math.min(y1, y2);
      const bottom = Math.max(y1, y2);
      const clampedLeft = Math.max(0, Math.min(canvasWidth - 1, left));
      const clampedTop = Math.max(0, Math.min(canvasHeight - 1, top));
      const clampedRight = Math.max(0, Math.min(canvasWidth - 1, right));
      const clampedBottom = Math.max(0, Math.min(canvasHeight - 1, bottom));
      const width = Math.max(1, clampedRight - clampedLeft);
      const height = Math.max(1, clampedBottom - clampedTop);
      return { x: clampedLeft, y: clampedTop, width, height };
    }

    function drawDetectionBox(context, detection) {
      // 依 YOLO box 座標畫框；異常用紅色，正常用綠色。
      const box = normalizeDetectionBox(context, detection);
      if (!box) return false;
      const color = detection.estimated ? "#f59e0b" : detection.abnormal ? "#ff4d4f" : "#22c55e";
      context.strokeStyle = color;
      context.lineWidth = 3;
      context.strokeRect(box.x, box.y, box.width, box.height);
      const confidence = Number(detection.confidence || 0).toFixed(2);
      const label = detection.label || detection.detectorName || "detection";
      drawLabel(context, detection.estimated ? `${label}（估計位置）` : `${label} ${confidence}`, box.x, box.y, color);
      return true;
    }

    async function drawAnnotatedFrameToCanvas(frame, frameResult, targetCanvas) {
      // 將原始 frame、偵測摘要、YOLO 框畫在同一張 canvas 上。
      const image = await loadImage(frame.image_data);
      targetCanvas.width = frame.width;
      targetCanvas.height = frame.height;
      const context = targetCanvas.getContext("2d");
      context.drawImage(image, 0, 0, targetCanvas.width, targetCanvas.height);
      const detections = collectDetections(frameResult);
      let drawnBoxCount = 0;
      for (const detection of detections) {
        if (drawDetectionBox(context, detection)) drawnBoxCount += 1;
      }
      for (const result of frameResult.detections || []) {
        if (result.detector_name !== "safety_pipeline") continue;
        const extra = result.extra || {};
        const boxes = [[extra.sleep_region, "床墊", "#38bdf8"], [extra.body, "身體", "#a78bfa"],
          [extra.head, "頭部", "#fbbf24"], ...(extra.objects || []).map(o => [o.box, o.label, "#fb7185"])];
        for (const [box, label, color] of boxes) {
          if (!box) continue;
          const [x1,y1,x2,y2] = box;
          context.strokeStyle = color;
          context.lineWidth = 2;
          context.strokeRect(x1*targetCanvas.width, y1*targetCanvas.height, (x2-x1)*targetCanvas.width, (y2-y1)*targetCanvas.height);
          drawLabel(context, label, x1*targetCanvas.width, y1*targetCanvas.height, color);
        }
      }
      drawFrameHeader(context, frameResult, drawnBoxCount, detections.length);
    }

    async function showAnnotatedImage(frames, data) {
      // 圖片測試完成後，直接在預覽區顯示畫框結果。
      const displayCanvas = document.createElement("canvas");
      await drawAnnotatedFrameToCanvas(frames[0], (data.frames || [])[0] || { detections: [] }, displayCanvas);
      preview.innerHTML = '<div class="preview-stack"></div>';
      preview.querySelector(".preview-stack").appendChild(displayCanvas);
    }

    async function showAnnotatedVideo(blob) {
      // 影片測試完成後，預覽區改播標註後的 WebM。
      if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
      annotatedUrl = URL.createObjectURL(blob);
      preview.innerHTML = `<video src="${annotatedUrl}" controls autoplay muted loop></video>`;
    }

    function drawFrameHeader(context, frameResult, drawnBoxCount = 0, rawBoxCount = 0) {
      // 在輸出影片左上角畫本 frame 的整體偵測狀態。
      const abnormal = (frameResult.detections || []).some((item) => item.is_abnormal);
      const unknown = (frameResult.detections || []).some((item) => item.extra?.evaluable === false);
      const title = abnormal ? "偵測結果：異常" : unknown ? "偵測結果：部分無法判讀" : "偵測結果：正常";
      const details = (frameResult.detections || [])
        .map((item) => `${item.detector_name} ${Number(item.confidence || 0).toFixed(2)}`)
        .join(" / ");
      const boxText = rawBoxCount > 0
        ? `可繪製框：${drawnBoxCount} / 原始框：${rawBoxCount}`
        : "本幀沒有可繪製的偵測框";
      context.fillStyle = "rgba(17, 24, 39, 0.78)";
      context.fillRect(0, 0, context.canvas.width, 74);
      context.fillStyle = abnormal ? "#ff7875" : unknown ? "#fbbf24" : "#86efac";
      context.font = "18px system-ui, sans-serif";
      context.fillText(title, 14, 23);
      context.fillStyle = "#e5e7eb";
      context.font = "13px system-ui, sans-serif";
      context.fillText(details.slice(0, 120), 14, 44);
      context.fillStyle = rawBoxCount > 0 && drawnBoxCount === 0 ? "#facc15" : "#cbd5e1";
      context.fillText(boxText, 14, 64);
    }

    async function buildAnnotatedVideo(frames, data) {
      // 使用 canvas.captureStream + MediaRecorder 將標註後 frame 錄成 WebM。
      if (!window.MediaRecorder) {
        throw new Error("目前瀏覽器不支援 MediaRecorder，無法另存標註影片");
      }
      if (!frames.length) return null;

      canvas.width = frames[0].width;
      canvas.height = frames[0].height;
      const stream = canvas.captureStream(6);
      const options = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? { mimeType: "video/webm;codecs=vp9" }
        : { mimeType: "video/webm" };
      const recorder = new MediaRecorder(stream, options);
      const chunks = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };

      const finished = new Promise((resolve) => {
        recorder.onstop = resolve;
      });

      recorder.start();
      for (let index = 0; index < frames.length; index += 1) {
        const frame = frames[index];
        const frameResult = (data.frames || [])[index] || { detections: [] };
        await drawAnnotatedFrameToCanvas(frame, frameResult, canvas);
        await new Promise((resolve) => setTimeout(resolve, 170));
      }
      recorder.stop();
      await finished;

      return new Blob(chunks, { type: "video/webm" });
    }

    function downloadBlob(blob) {
      // 使用瀏覽器下載機制另存標註影片。
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `detector-lab-annotated-${Date.now()}.webm`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    function renderSummary(data) {
      // 顯示每個 detector 在本次測試中的異常次數與平均信心值。
      summary.innerHTML = "";
      if (data.mode === "webcam_simulation") {
        const modeDiv = document.createElement("div");
        modeDiv.className = "summary-item";
        const workers = (data.worker_status || [])
          .map((item) => `${item.name}: 處理 ${item.frames_processed} / 丟棄 ${item.frames_dropped}`)
          .join("<br>");
        modeDiv.innerHTML = `<strong>WebCam 模擬模式</strong><div>送幀間隔：${data.frame_interval_sec ?? "-"} 秒</div><div>Detector worker queue：每個 detector 最多保留 10 張</div><div>${workers}</div>`;
        summary.appendChild(modeDiv);
      }
      for (const [name, item] of Object.entries(data.summary || {})) {
        const div = document.createElement("div");
        div.className = "summary-item";
        div.innerHTML = `<strong>${name}</strong><div>異常：${item.abnormal} / ${item.total}</div><div>無法判讀：${item.unknown || 0}</div><div>平均信心值：${name.startsWith("safety_pipeline:") ? "規則判斷，不提供機率" : item.avg_confidence}</div>`;
        summary.appendChild(div);
      }
    }

    function formatNumber(value, digits = 6) {
      // 將後端數值統一格式化，避免 null/NaN 讓畫面顯示不清楚。
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    }

    function renderProcess(data) {
      // 專門顯示 motion detector 的影像處理過程：
      // 背景參考、目前前景、差異圖、二值化動作遮罩。
      process.innerHTML = "";
      const items = [];
      for (const frame of data.frames || []) {
        for (const detection of frame.detections || []) {
          if (detection.detector_name !== "motion_detector") continue;
          items.push({ frame, detection });
        }
      }

      if (!items.length) return;

      for (const { frame, detection } of items) {
        const extra = detection.extra || {};
        const debugImages = extra.debug_images || {};
        const div = document.createElement("div");
        div.className = "process-item";
        div.innerHTML = `
          <div class="process-title">
            <span>Frame ${frame.index} / ${frame.timestamp_sec ?? "-"} 秒</span>
            <span class="${detection.is_abnormal ? "bad" : "ok"}">${detection.is_abnormal ? "低動作" : "正常"}</span>
          </div>
          <div class="process-grid">
            <div>步驟：<span>${extra.step || "-"}</span></div>
            <div>動作像素：<span>${extra.motion_pixels ?? "-"}</span></div>
            <div>總像素：<span>${extra.total_pixels ?? "-"}</span></div>
            <div>motion_ratio：<span>${formatNumber(extra.motion_ratio)}</span></div>
            <div>變化百分比：<span>${formatNumber(extra.changed_pixel_percent, 4)}%</span></div>
            <div>低動作門檻：<span>${formatNumber(extra.low_ratio_threshold)}</span></div>
            <div>差異門檻：<span>${extra.diff_threshold ?? "-"}</span></div>
            <div>信心值：<span>${formatNumber(detection.confidence, 4)}</span></div>
          </div>
          <div style="margin-top:6px;color:var(--muted);">
            比較方式：absdiff(目前前景灰階圖, 上一張背景參考灰階圖)，再套用 diff_threshold 產生動作遮罩。
          </div>
          <div style="margin-top:6px;color:var(--muted);">${extra.decision || detection.message}</div>
          <div class="process-images">
            ${debugImages.background_reference ? `<div class="process-image"><img src="${debugImages.background_reference}" alt="背景參考圖"><div>1. 背景參考：上一張 frame</div></div>` : ""}
            ${debugImages.gray_blur ? `<div class="process-image"><img src="${debugImages.gray_blur}" alt="目前前景圖"><div>2. 目前前景：這一張 frame</div></div>` : ""}
            ${debugImages.frame_diff ? `<div class="process-image"><img src="${debugImages.frame_diff}" alt="前景背景差異圖"><div>3. 前景 - 背景差異</div></div>` : ""}
            ${debugImages.threshold_mask ? `<div class="process-image"><img src="${debugImages.threshold_mask}" alt="二值化動作遮罩"><div>4. 差異超過門檻的動作遮罩</div></div>` : ""}
          </div>
        `;
        process.appendChild(div);
      }
    }

    function sanitizeExtraForTable(extra) {
      // 表格細節只顯示可讀的數值與判斷資料，不直接印出 base64 圖片。
      // debug_images 裡面是 data:image/jpeg;base64,...，字串非常長，會讓畫面很難閱讀。
      const clean = { ...(extra || {}) };
      if (clean.debug_images) {
        clean.debug_images = Object.fromEntries(
          Object.keys(clean.debug_images).map((key) => [key, "[已在上方影像處理過程顯示]"])
        );
      }
      return clean;
    }

    function renderRows(data) {
      // 顯示每張 frame 的完整 detector 回傳內容，方便對照參數與結果。
      rows.innerHTML = "";
      for (const frame of data.frames || []) {
        for (const detection of frame.detections || []) {
          const readableExtra = sanitizeExtraForTable(detection.extra);
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td>${frame.index}</td>
            <td>${frame.timestamp_sec ?? "-"}</td>
            <td>${detection.detector_name}<br>${detection.event_type}</td>
            <td><span class="pill ${detection.extra?.evaluable === false ? "warn" : detection.is_abnormal ? "bad" : "ok"}">${detection.extra?.evaluable === false ? "無法判讀" : detection.is_abnormal ? "異常" : "正常"}</span></td>
            <td>${detection.extra?.confidence_kind === "rule_based_not_probability" ? "—" : Number(detection.confidence || 0).toFixed(4)}</td>
            <td>${detection.message}</td>
            <td><code>${JSON.stringify(readableExtra, null, 2)}</code></td>`;
          rows.appendChild(tr);
        }
      }
    }

    runButton.addEventListener("click", async () => {
      // 主流程：抽 frame -> 呼叫後端 detector -> 渲染摘要/表格/處理圖 -> 顯示標註結果。
      if (!currentFile) return;
      runButton.disabled = true;
      summary.innerHTML = "";
      process.innerHTML = "";
      rows.innerHTML = "";
      setStatus("正在抽取影像並執行偵測...");
      try {
        const frames = currentFile.type.startsWith("video/") ? await videoToFrames(currentFile) : await imageToFrame(currentFile);
        const isVideo = currentFile.type.startsWith("video/");
        const intervalSec = Number(document.getElementById("webcam-interval").value);
        setStatus(isVideo
          ? `已抽取 ${frames.length} 張 frame，正在以 WebCam 模擬節奏送入 detector queue（${intervalSec} 秒/張）...`
          : `已抽取 ${frames.length} 張 frame，正在送到後端分析...`);
        const response = await fetch("/detector-lab/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            detector: detector.value,
            source_type: currentFile.type.startsWith("video/") ? "video" : "image",
            camera_id: cameraId.value || null,
            params: getParams(),
            frames,
          }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        renderSummary(data);
        renderProcess(data);
        renderRows(data);
        if (isVideo) {
          setStatus("正在繪製偵測結果並產生預覽影片...");
          const blob = await buildAnnotatedVideo(frames, data);
          if (blob) {
            await showAnnotatedVideo(blob);
            if (annotateVideo.checked) downloadBlob(blob);
          }
          setStatus(annotateVideo.checked ? `完成：分析 ${frames.length} 張 frame，標註影片已顯示並下載。` : `完成：分析 ${frames.length} 張 frame，標註影片已顯示。`);
        } else if (frames.length) {
          await showAnnotatedImage(frames, data);
          setStatus(`完成：分析 ${frames.length} 張 frame，標註圖片已顯示。`);
        } else {
          setStatus(`完成：分析 ${frames.length} 張 frame。`);
        }
      } catch (error) {
        setStatus(`偵測失敗：${error}`);
      } finally {
        runButton.disabled = false;
      }
    });

    loadCameraOptions();
  </script>
</body>
</html>
"""
