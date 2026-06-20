from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from backend.utils.image_utils import encode_jpeg

router = APIRouter()


def mjpeg_generator(request: Request):
    camera = request.app.state.camera
    settings = request.app.state.settings

    while True:
        # 優先使用背景抽幀已讀到的最新畫面，沒有畫面時才主動讀取攝影機。
        frame = camera.get_latest_frame()
        if frame is None:
            frame = camera.get_frame()

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
def stream(request: Request):
    return StreamingResponse(
        mjpeg_generator(request),
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
    <button id="reload-stream" type="button">重新連線</button>
  </header>
  <main>
    <section class="stream-wrap" aria-label="即時串流">
      <img id="stream" src="/stream" alt="攝影機即時畫面">
    </section>
    <aside aria-label="攝影機狀態">
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
    const opened = document.getElementById("opened");
    const source = document.getElementById("source");
    const lastRead = document.getElementById("last-read");
    const hasFrame = document.getElementById("has-frame");
    const framesRead = document.getElementById("frames-read");
    const lastError = document.getElementById("last-error");
    const reloadButton = document.getElementById("reload-stream");

    function yesNo(value) {
      return value ? "是" : "否";
    }

    function formatTime(timestamp) {
      if (!timestamp) return "-";
      return new Date(timestamp * 1000).toLocaleString("zh-TW", {
        hour12: false,
      });
    }

    function setPill(isOpened) {
      opened.textContent = isOpened ? "已連線" : "未連線";
      opened.className = `pill ${isOpened ? "ok" : "bad"}`;
    }

    async function refreshStatus() {
      try {
        const response = await fetch("/health", { cache: "no-store" });
        const data = await response.json();
        const camera = data.camera || {};

        setPill(Boolean(camera.opened));
        source.textContent = camera.source ?? "-";
        lastRead.textContent = camera.last_read_ok ? "成功" : "失敗";
        hasFrame.textContent = yesNo(camera.has_latest_frame);
        framesRead.textContent = camera.frames_read ?? 0;
        lastError.textContent = camera.last_error || "-";
      } catch (error) {
        opened.textContent = "狀態讀取失敗";
        opened.className = "pill bad";
        lastError.textContent = String(error);
      }
    }

    reloadButton.addEventListener("click", () => {
      stream.src = `/stream?ts=${Date.now()}`;
      refreshStatus();
    });

    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""
