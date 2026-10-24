from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from backend.utils.image_utils import encode_jpeg

router = APIRouter()


def _get_runtime(request: Request, camera_id: str | None = None):
    runtime = request.app.state.camera_manager.get(camera_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail=f"找不到相機：{camera_id}")
    return runtime


def mjpeg_generator(request: Request, camera_id: str | None = None):
    # MJPEG 串流產生器：持續輸出 JPEG frame，瀏覽器 <img> 可直接播放。
    runtime = _get_runtime(request, camera_id)
    camera = runtime.camera
    settings = request.app.state.settings

    while True:
        # 只有背景 reader 讀攝影機，HTTP 用戶端僅消費有效的新畫面。
        frame = camera.get_latest_frame()

        if frame is None:
            time.sleep(0.2)
            continue

        try:
            jpg = encode_jpeg(frame, quality=settings.stream_jpeg_quality)
        except Exception:
            time.sleep(0.2)
            continue

        # MJPEG 格式：瀏覽器會把每個 JPEG frame 當成連續影像播放。
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        time.sleep(0.03)


@router.get("/stream")
def stream(request: Request, camera_id: str | None = Query(default=None)):
    # 即時畫面入口；只負責顯示畫面，不代表 AI 偵測頻率。
    # AI 偵測頻率由 FrameReader + config.ini 的 frame_interval_sec 控制。
    _get_runtime(request, camera_id)
    return StreamingResponse(
        mjpeg_generator(request, camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/stream/{camera_id}")
def stream_by_camera(camera_id: str, request: Request):
    # 多相機串流入口，例如 /stream/baby-cam-02。
    _get_runtime(request, camera_id)
    return StreamingResponse(
        mjpeg_generator(request, camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/camera", response_class=HTMLResponse)
def camera_page():
    # 這個頁面提供人工確認用：左側看即時畫面，右側看攝影機讀取狀態。
    return """
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>寶寶攝影機即時畫面</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d9e0ea;
      --ok: #0f8f5f;
      --bad: #c0392b;
      --warn: #b26a00;
      --accent: #2563eb;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      padding: 18px;
      max-width: 1280px;
      margin: 0 auto;
    }

    .stream-wrap {
      min-height: 360px;
      background: #111827;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      display: grid;
      place-items: center;
    }

    .stream-wrap img {
      display: block;
      width: 100%;
      height: auto;
      max-height: calc(100vh - 130px);
      object-fit: contain;
      background: #111827;
    }

    aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      align-self: start;
    }

    .status-row {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }

    .status-row:last-child {
      border-bottom: 0;
    }

    .label {
      color: var(--muted);
      white-space: nowrap;
    }

    .value {
      text-align: right;
      overflow-wrap: anywhere;
      font-weight: 600;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
      border: 1px solid currentColor;
    }

    .ok {
      color: var(--ok);
    }

    .bad {
      color: var(--bad);
    }

    .warn {
      color: var(--warn);
    }

    button {
      min-height: 36px;
      border: 1px solid var(--accent);
      background: var(--accent);
      color: #fff;
      border-radius: 6px;
      padding: 0 14px;
      font-weight: 700;
      cursor: pointer;
    }

    .toolbar {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    select {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }

    @media (max-width: 820px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }

      main {
        grid-template-columns: 1fr;
        padding: 12px;
      }

      .stream-wrap {
        min-height: 220px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>寶寶攝影機即時畫面</h1>
    <div class="toolbar">
      <select id="camera-select" aria-label="選擇相機"></select>
      <label><input id="auto-follow" type="checkbox" checked>自動切換候選床位</label>
      <button id="reload-stream" type="button">重新連線</button>
      <button id="enable-audio-alarm" type="button">啟用聲音警報（播放測試音）</button>
    </div>
  </header>
  <main>
    <section class="stream-wrap" aria-label="即時串流">
      <img id="stream" src="/stream" alt="攝影機即時畫面">
    </section>
    <aside aria-label="攝影機狀態">
      <div class="status-row">
        <span class="label">哭聲／尖叫</span>
        <span id="audio-status" class="value">等待音訊狀態</span>
      </div>
      <div id="audio-alert" role="alert" style="color:#b42318;font-weight:bold"></div>
      <div class="status-row">
        <span class="label">口鼻／奶嘴判讀</span>
        <span id="airway-status" class="value">等待狀態；未確認不等於安全</span>
      </div>
      <div class="status-row">
        <span class="label">目前床位判斷</span>
        <span id="occupancy" class="value">等待影像</span>
      </div>
      <div class="status-row">
        <span class="label">各床狀態</span>
        <span id="all-occupancy" class="value">等待影像</span>
      </div>
      <div class="status-row">
        <span class="label">連線狀態</span>
        <span class="value"><span id="opened" class="pill warn">讀取中</span></span>
      </div>
      <div class="status-row">
        <span class="label">來源</span>
        <span id="source" class="value">-</span>
      </div>
      <div class="status-row">
        <span class="label">最後讀取</span>
        <span id="last-read" class="value">-</span>
      </div>
      <div class="status-row">
        <span class="label">已有畫面</span>
        <span id="has-frame" class="value">-</span>
      </div>
      <div class="status-row">
        <span class="label">讀取張數</span>
        <span id="frames-read" class="value">-</span>
      </div>
      <div class="status-row">
        <span class="label">錯誤</span>
        <span id="last-error" class="value">-</span>
      </div>
    </aside>
  </main>
  <script>
    const stream = document.getElementById("stream");
    const cameraSelect = document.getElementById("camera-select");
    const opened = document.getElementById("opened");
    const source = document.getElementById("source");
    const lastRead = document.getElementById("last-read");
    const hasFrame = document.getElementById("has-frame");
    const framesRead = document.getElementById("frames-read");
    const lastError = document.getElementById("last-error");
    const reloadButton = document.getElementById("reload-stream");
    const autoFollow = document.getElementById("auto-follow");
    const occupancyText = document.getElementById("occupancy");
    const allOccupancyText = document.getElementById("all-occupancy");
    const occupancyLabels = {present: "人物候選", absent: "未偵測到人物", unknown: "無法確認"};
    let selectedCameraId = "";
    let alarmContext = null;
    let lastAudioAlertId = null;
    function beep() {
      if (!alarmContext || alarmContext.state !== "running") return;
      for (let i = 0; i < 3; i++) {
        const oscillator = alarmContext.createOscillator();
        const gain = alarmContext.createGain();
        oscillator.frequency.value = 880;
        gain.gain.value = 0.12;
        oscillator.connect(gain); gain.connect(alarmContext.destination);
        oscillator.start(alarmContext.currentTime + i * .4);
        oscillator.stop(alarmContext.currentTime + i * .4 + .2);
      }
    }
    document.getElementById("enable-audio-alarm").addEventListener("click", async () => {
      try {
        if (!alarmContext) alarmContext = new (window.AudioContext || window.webkitAudioContext)();
        await alarmContext.resume();
        beep();
        document.getElementById("enable-audio-alarm").textContent = "聲音警報已啟用／測試音";
      } catch (_) {
        document.getElementById("enable-audio-alarm").textContent = "聲音啟用失敗，請檢查瀏覽器權限";
      }
    });

    function yesNo(value) {
      // 將布林值轉成中文顯示，讓狀態面板比較直覺。
      return value ? "是" : "否";
    }

    function formatTime(timestamp) {
      // 後端 status 回傳 Unix timestamp，前端轉成台灣常見日期格式。
      if (!timestamp) return "-";
      return new Date(timestamp * 1000).toLocaleString("zh-TW", {
        hour12: false,
      });
    }

    function setPill(isOpened) {
      // 依攝影機是否開啟切換狀態膠囊的文字與顏色。
      opened.textContent = isOpened ? "已連線" : "未連線";
      opened.className = `pill ${isOpened ? "ok" : "bad"}`;
    }

    async function refreshStatus() {
      // 每秒呼叫 /health，更新右側攝影機狀態，不需要重整整個頁面。
      try {
        const response = await fetch("/health", { cache: "no-store" });
        const data = await response.json();
        const cameras = data.cameras || [];
        const suggestion = data.occupancy_summary?.suggested_camera_id;
        if (autoFollow.checked && suggestion && suggestion !== selectedCameraId) {
          selectedCameraId = suggestion;
          stream.src = streamUrl();
        }
        if (!selectedCameraId) {
          selectedCameraId = data.default_camera_id || data.camera_id || "";
        }
        renderCameraOptions(cameras, selectedCameraId);
        const current = cameras.find((item) => item.camera_id === selectedCameraId) || cameras[0] || {};
        const camera = current.camera || data.camera || {};
        const audio = data.audio || {};
        const audioState = audio.cameras?.[selectedCameraId] || {};
        const audioLabels = {starting: "音訊模型載入中", listening: "正在辨識哭聲／尖叫", reconnecting: "音訊中斷，重連中", stale: "音訊結果過期", error: "音訊推論失敗"};
        document.getElementById("audio-status").textContent = !audio.enabled ? "未啟用" : audio.error
          ? "音訊模型或解碼器不可用" : (audioLabels[audioState.status] || "無法判讀");
        const alert = audio.last_alert;
        const recent = alert && Date.now() / 1000 - alert.at < 60;
        document.getElementById("audio-alert").textContent = recent
          ? `${alert.camera_id}：${alert.message}（${new Date(alert.at * 1000).toLocaleTimeString()}）` : "";
        if (alert && alert.id !== lastAudioAlertId) {
          lastAudioAlertId = alert.id;
          if (recent) beep();
        }
        const airway = (current.detector_workers || []).flatMap(w => w.assessments || [])
          .find(r => r.event_type === "face_cover_warning");
        document.getElementById("airway-status").textContent = airway
          ? `${airway.assessment === "abnormal" ? "疑似遮擋：" : airway.assessment === "clear" ? "本張判讀：" : "無法判讀："}${airway.message}`
          : "尚未啟用專用模型，無法判讀口鼻、棉被與奶嘴";
        occupancyText.textContent = current.occupancy?.reason || "在床判斷未啟用";
        allOccupancyText.textContent = cameras.map(item =>
          `${item.camera_id}：${occupancyLabels[item.occupancy?.state] || "無法確認"}`).join("；");

        setPill(Boolean(camera.opened));
        source.textContent = camera.source ?? "-";
        lastRead.textContent = camera.last_read_ok ? "成功" : "失敗";
        hasFrame.textContent = yesNo(camera.has_latest_frame);
        framesRead.textContent = camera.frames_read ?? 0;
        lastError.textContent = camera.last_error || "-";
      } catch (error) {
        document.getElementById("audio-status").textContent = "連線失敗，哭聲警報無法更新";
        document.getElementById("airway-status").textContent = "狀態已失效，無法判讀口鼻";
        occupancyText.textContent = "狀態已失效，無法確認";
        allOccupancyText.textContent = "狀態已失效，無法確認";
        opened.textContent = "狀態讀取失敗";
        opened.className = "pill bad";
        lastError.textContent = String(error);
      }
    }

    function renderCameraOptions(cameras, currentCameraId) {
      const existing = Array.from(cameraSelect.options).map((option) => option.value).join(",");
      const incoming = cameras.map((item) => item.camera_id).join(",");
      if (existing === incoming) {
        cameraSelect.value = currentCameraId;
        return;
      }
      cameraSelect.innerHTML = "";
      for (const item of cameras) {
        const option = document.createElement("option");
        option.value = item.camera_id;
        option.textContent = `${item.camera_id} (${item.source})`;
        cameraSelect.appendChild(option);
      }
      cameraSelect.value = currentCameraId;
    }

    function streamUrl() {
      const ts = Date.now();
      return selectedCameraId
        ? `/stream?camera_id=${encodeURIComponent(selectedCameraId)}&ts=${ts}`
        : `/stream?ts=${ts}`;
    }

    cameraSelect.addEventListener("change", () => {
      autoFollow.checked = false;
      selectedCameraId = cameraSelect.value;
      stream.src = streamUrl();
      refreshStatus();
    });

    autoFollow.addEventListener("change", refreshStatus);

    reloadButton.addEventListener("click", () => {
      // 加上 timestamp 避免瀏覽器快取舊的 /stream 連線。
      stream.src = streamUrl();
      refreshStatus();
    });

    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""
