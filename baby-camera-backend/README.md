# Baby Camera Backend 寶寶攝影機 Python 後端

這是一個給「智慧寶寶攝影機」使用的 Python 後端專案骨架。
設計目標是：**先能跑，再慢慢擴充 AI 偵測能力**。

目前第一版包含：

- FastAPI 後端服務
- OpenCV 攝影機讀取
- MJPEG 即時串流
- 定時抽幀分析
- 模組化 Detector 偵測器架構
- 畫面太暗偵測
- 長時間低動作提醒
- YOLO Pose 趴睡風險偵測
- 異常事件管理
- 冷卻時間，避免 LINE 洗版
- SQLite 事件紀錄
- 異常截圖儲存
- Webhook 告警，預留接 Firebase Cloud Function / LINE Messaging API

---

## 1. 專案架構

```text
baby-camera-backend/
├── backend/
│   ├── main.py                    # FastAPI 主入口，只負責啟動 API / 背景任務
│   ├── config.py                  # 系統設定，讀取 config.ini
│   ├── camera_stream.py           # 攝影機串流來源管理
│   ├── frame_reader.py            # 抽幀模組，固定時間取 frame 給偵測器
│   ├── event_manager.py           # 異常事件判斷、去抖動、冷卻時間
│   ├── alert_service.py           # Webhook / LINE / Email 告警服務入口
│   ├── database.py                # SQLite 資料庫操作
│   ├── schemas.py                 # 共用資料格式
│   ├── logger.py                  # logging 設定
│   │
│   ├── detector/
│   │   ├── base_detector.py       # 所有偵測器的共同介面
│   │   ├── dark_detector.py       # 畫面太暗
│   │   ├── motion_detector.py     # 長時間低動作
│   │   ├── baby_detector.py       # 寶寶位置偵測，第一版預留
│   │   ├── face_cover.py          # 臉部遮蔽偵測，第一版預留
│   │   ├── prone_sleep_detector.py # YOLO Pose 趴睡風險偵測
│   │   └── detector_manager.py    # 統一管理所有 detector
│   │
│   ├── services/
│   │   ├── line_service.py        # 預留 LINE 直接推播
│   │   ├── email_service.py       # 預留 Email 通知
│   │   └── snapshot_service.py    # 異常截圖儲存
│   │
│   ├── api/
│   │   ├── health_api.py          # /health 系統狀態
│   │   ├── stream_api.py          # /stream 即時畫面
│   │   ├── event_api.py           # /events 查詢異常事件
│   │   └── config_api.py          # /config 查詢設定
│   │
│   └── utils/
│       ├── image_utils.py         # OpenCV 共用工具
│       ├── time_utils.py          # 時間工具
│       └── file_utils.py          # 檔案處理
│
├── data/
│   ├── snapshots/                 # 異常截圖
│   ├── recordings/                # 未來事件錄影可放這裡
│   └── baby_camera.db             # SQLite 資料庫，執行後自動產生
│
├── logs/
│   └── app.log                    # 系統 log，執行後自動產生
│
├── tests/
├── config.ini.example             # 設定範例
├── requirements.txt
├── run.py                         # 本機啟動入口
└── README.md
```

---

## 2. 系統資料流程

```text
攝影機畫面
   ↓
camera_stream.py 取得即時影像
   ↓
frame_reader.py 定時抽幀
   ↓
detector_manager.py 執行所有偵測器
   ↓
event_manager.py 判斷異常是否成立
   ↓
database.py 儲存事件
   ↓
snapshot_service.py 儲存異常截圖
   ↓
alert_service.py 發送 webhook 告警
   ↓
Firebase Cloud Function / LINE Messaging API
```

這個設計的核心原則是：

> Detector 只負責判斷「這張圖有沒有異常」，EventManager 才負責判斷「要不要真的告警」。

這樣才不會因為一兩張畫面異常就狂發 LINE。

---

## 3. 安裝方式

### 建立虛擬環境

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

### 安裝套件

```bash
pip install -r requirements.txt
```

### 建立設定檔

Windows：

```bash
copy .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

---

## 4. 啟動後端

方式一：

```bash
python run.py
```

方式二：

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

啟動後可以打開：

```text
http://localhost:8000/health
```

如果要看即時畫面：

```text
http://localhost:8000/stream
```

多相機時可指定相機：

```text
http://localhost:8000/stream?camera_id=baby-cam-02
http://localhost:8000/stream/baby-cam-02
```

攝影機即時畫面頁面：
```text
http://localhost:8000/camera
```

Detector 調參窗口：
```text
http://localhost:8000/detector-lab
```

## 設定檔：config.ini

目前後端統一讀取專案根目錄的 `config.ini`，不再讀取 `.env`。

常用設定位置：

```ini
[camera]
id = baby-cam-01
source_type = opencv
source = 0

; Tapo RTSP 單相機範例：
; source_type = tapo_rtsp
; rtsp_host = 192.168.1.50
; rtsp_port = 554
; rtsp_username = your_tapo_camera_user
; rtsp_password = your_tapo_camera_password
; rtsp_stream = stream1

; 多相機時啟用：
; [cameras]
; ids = baby-cam-01,baby-cam-02
;
; [camera.baby-cam-01]
; source_type = opencv
; source = 0
;
; [camera.baby-cam-02]
; source_type = tapo_rtsp
; rtsp_host = 192.168.1.50
; rtsp_port = 554
; rtsp_username = your_tapo_camera_user
; rtsp_password = your_tapo_camera_password
; rtsp_stream = stream1
;
; 每台相機可覆寫 detector 門檻：
; [camera.baby-cam-02.detectors.dark]
; brightness_threshold = 45
;
; [camera.baby-cam-02.detectors.motion]
; low_ratio_threshold = 0.005

[detectors.baby]
enabled = true
yolo_model_path = backend/detector/model/yolo11n-pose.pt
confidence_threshold = 0.4
face_labels = baby_face,face

[alert]
webhook_url = https://your-cloud-function-url
ingest_key = your-alert-key
timeout_sec = 10
dry_run = false
include_snapshot_image = true
snapshot_max_bytes = 900000
```

修改 `config.ini` 後要重啟後端才會生效。

寶寶臉部偵測目前採用 YOLO 判斷：

- YOLO 偵測到 `baby_face` 或 `face` 時，視為寶寶在畫面中
- 沒有偵測到寶寶臉部時，產生 `baby_position_warning`
- 如果使用一般 pose 模型只輸出 `person`，請把 `BABY_FACE_LABELS` 改成 `person`，或改用有訓練 baby face / face class 的模型

```ini
[detectors.baby]
enabled = true
yolo_model_path = backend/detector/model/yolo11n-pose.pt
confidence_threshold = 0.4
face_labels = baby_face,face
```

臉部遮蔽偵測目前採用 YOLO 口鼻可見性判斷：

- YOLO 有偵測到 `face`，但沒有偵測到 `nose` 或 `mouth` 時，判斷為可能遮蔽
- 未設定 `[detectors.face_cover] yolo_model_path` 時，偵測器會安全停用，不會影響後端啟動
- 未來訓練 YOLO 時，建議 class 至少包含 `face`、`nose`、`mouth`

```ini
[detectors.face_cover]
enabled = true
yolo_model_path = models/face_cover_yolo.pt
face_confidence_threshold = 0.4
part_confidence_threshold = 0.25
require_nose = true
require_mouth = true
```

查詢事件：

```text
http://localhost:8000/events
```

手動測試告警：

```text
http://localhost:8000/test-alert
```

API 文件：

```text
http://localhost:8000/docs
```

---

## 5. .env 重要設定說明

### 攝影機來源

```env
CAMERA_SOURCE=0
```

常見設定：

```env
CAMERA_SOURCE=0
CAMERA_SOURCE=rtsp://192.168.1.10:8554/baby
CAMERA_SOURCE=http://192.168.1.10:8080/video
CAMERA_SOURCE=data/videos/test.mp4
```

---

### 畫面太暗偵測

```env
ENABLE_DARK_DETECTOR=true
DARK_BRIGHTNESS_THRESHOLD=30
DARK_DURATION_SEC=10
DARK_COOLDOWN_SEC=300
```

意思是：

- 平均亮度低於 30 時視為太暗
- 連續太暗 10 秒才成立事件
- 告警後 300 秒內不重複發送同類告警

---

### 長時間低動作提醒

```env
ENABLE_MOTION_DETECTOR=true
MOTION_DIFF_THRESHOLD=8
MOTION_LOW_RATIO_THRESHOLD=0.003
MOTION_DURATION_SEC=60
MOTION_COOLDOWN_SEC=600
```

意思是：

- 使用前後 frame 差異判斷動作量
- 連續低動作 60 秒才提醒
- 告警後 600 秒內不重複提醒

注意：

> 這只是低動作提醒，不等於呼吸偵測，也不能當成醫療級監測。

---

### 告警 Webhook

```env
ALERT_WEBHOOK_URL=https://your-cloud-function-url
```

建議流程：

```text
Python 後端
   ↓
呼叫 Firebase Cloud Function URL
   ↓
Cloud Function 寫入 Firestore
   ↓
LINE Messaging API 推播
```

這樣 Python 端不用直接保存 LINE Token，比較安全，也比較好維護。

---

## 6. API 說明

### GET `/health`

確認系統狀態。

範例：

```json
{
  "status": "ok",
  "camera_id": "baby-cam-01",
  "default_camera_id": "baby-cam-01",
  "camera": {
    "source": "0",
    "opened": true,
    "last_read_ok": true,
    "has_latest_frame": true
  },
  "detectors": ["dark_detector", "motion_detector"],
  "cameras": [
    {
      "camera_id": "baby-cam-01",
      "source": "0",
      "camera": {
        "opened": true,
        "frames_read": 12
      },
      "detectors": ["dark_detector", "motion_detector"],
      "detector_workers": []
    }
  ]
}
```

---

### GET `/stream`

MJPEG 即時影像串流。

手機或電腦瀏覽器打開：

```text
http://localhost:8000/stream
```

多相機指定：

```text
http://localhost:8000/stream?camera_id=baby-cam-02
http://localhost:8000/stream/baby-cam-02
```

---

### GET `/events`

取得最近異常事件。

參數：

```text
/events?limit=50
/events?camera_id=baby-cam-01&limit=50
```

---

### POST `/test-alert`

手動送出測試告警。

多相機指定測試：

```text
/test-alert?camera_id=baby-cam-02
```

用途：

- 測試 Firebase Cloud Function URL 是否正常
- 測試 LINE 推播是否正常
- 不需要真的等到畫面異常

---

### GET `/config`

查看目前主要設定。

---

## 7. 如何新增新的偵測器

例如你要新增「安全區域偵測」：

新增檔案：

```text
backend/detector/safe_zone_detector.py
```

內容範例：

```python
from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult

class SafeZoneDetector(BaseDetector):
    name = "safe_zone_detector"

    def detect(self, frame):
        # TODO: 在這裡放你的 OpenCV / AI 判斷邏輯
        return DetectionResult(
            detector_name=self.name,
            event_type="safe_zone_warning",
            is_abnormal=False,
            confidence=0.0,
            message="寶寶仍在安全區域內",
            extra={}
        )
```

然後到：

```text
backend/detector/detector_manager.py
```

把它加入：

```python
detectors.append(SafeZoneDetector())
```

主程式 `main.py` 不需要大改。

---

## 8. 第一版測試順序

建議依照這個順序測：

```text
1. python run.py 啟動後端
2. 打開 /health 確認後端活著
3. 打開 /stream 確認看得到攝影機
4. 用手遮住攝影機，測試 dark_detector
5. 等待事件成立，確認 data/baby_camera.db 是否有紀錄
6. 確認 data/snapshots 是否有異常截圖
7. 設定 ALERT_WEBHOOK_URL，測試 /test-alert
8. 接上 Firebase Cloud Function + LINE Messaging API
```

---

## 9. 目前版本定位

這一版的定位是：

> 寶寶攝影機 Python 後端 MVP 骨架。

先把完整流程打通：

```text
看得到畫面 → 抽幀 → 偵測 → 事件成立 → 存資料庫 → 存截圖 → 發告警
```

後續再慢慢加上：

- YOLO 寶寶偵測
- 臉部遮蔽偵測
- 安全區域偵測
- WebRTC 串流
- 事件回放
- 前端 Dashboard
- 手機 App

---

## 10. 重要提醒

這個專案是學習與輔助提醒用途，不是醫療設備，也不能取代成人照護。

尤其是：

- 長時間低動作不等於呼吸停止
- 臉部遮蔽偵測需要大量測試
- AI 判斷一定會有誤判與漏判
- 寶寶睡眠安全仍應以實際照護與安全睡眠環境為主

## LineService / Firebase Cloud Function Status: OK

Python backend now sends LINE-bound alert events through Firebase Cloud Function, matching the payload shape used by `D:/baby-camera-alert/python/push_firebase.py`.

Config:

- `ALERT_WEBHOOK_URL`: Firebase Cloud Function URL
- `ALERT_INGEST_KEY`: value sent as `x-alert-key`
- `ALERT_DRY_RUN=true`: log only, do not send the HTTP request

Alert policy table lives in `backend/event_manager.py` as `alert_policies`:

- `notify_line=True`: record the event and send Firebase / LINE
- `notify_line=False`: record only, no LINE message
- `alert_after_count`: send LINE only after the event reaches the threshold
- `max_alerts_per_day`: limit LINE messages for noisy event types

Default policies:

- `room_too_dark`: warning, at most one LINE message per day
- `low_motion_warning`: info, record only
- `baby_position_warning`: critical, LINE after 2 occurrences
- `face_cover_warning`: critical, LINE immediately
- `test_alert`: critical, manual test alert

## Current Progress / 目前完成狀態

目前已完成智慧寶寶攝影機後端 MVP 的主要骨架，包含攝影機讀取、即時串流、Detector 架構、事件成立判斷、LINE 告警流程，以及 Detector Lab 調參工具。

### 1. 攝影機與串流

- `camera_stream.py` 已可用 OpenCV 讀取 USB camera、RTSP、MJPEG 或影片來源
- 支援多相機，每台相機都有獨立 CameraStream、FrameReader、DetectorManager worker 與 EventManager 狀態
- `/stream` 提供預設相機 MJPEG 即時串流，`/stream?camera_id=<id>` 可指定相機
- `/camera` 提供網頁版即時畫面與攝影機狀態，頁面可切換相機
- `/health` 可查看所有相機是否開啟、是否讀到 frame、最後錯誤訊息與 detector worker 狀態
- `/events?camera_id=<id>` 可依相機查詢事件
- 告警截圖會依相機分資料夾保存，例如 `data/snapshots/baby-cam-01/...`
- Firebase 告警 payload 可附上 `snapshotImage` base64，供 Cloud Function 轉成 LINE 圖片訊息

### 2. 設定檔改為 config.ini

目前所有設定統一改用 `config.ini`，不再讀取 `.env`。

常用設定：

```ini
[detectors.baby]
enabled = true
yolo_model_path = backend/detector/model/yolo11n-pose.pt
confidence_threshold = 0.4
face_labels = baby_face,face

[alert]
webhook_url = https://ingestalert-qffjkcl3ea-de.a.run.app
ingest_key = your-alert-key
timeout_sec = 10
dry_run = false
```

修改 `config.ini` 後需要重啟後端。

### 3. 寶寶臉部辨識

今天已先針對「寶寶臉部辨識」補齊第一版 YOLO 接入。

目前 `baby_detector.py` 的邏輯：

```text
載入 YOLO 模型
→ 對 frame 做推論
→ 若偵測到 baby_face / face，代表寶寶臉部存在
→ 若沒有偵測到，產生 baby_position_warning
```

目前模型路徑設定在：

```ini
[detectors.baby]
yolo_model_path = backend/detector/model/yolo11n-pose.pt
face_labels = baby_face,face
```

注意：目前使用的 `yolo11n-pose.pt` 通常是 pose/person 模型，可能輸出 `person` 而不是 `face`。如果先用這顆模型測「畫面中是否有人」，可以把 `face_labels` 改成：

```ini
face_labels = person
```

若要真正做 baby face 偵測，後續建議使用或訓練 class 包含 `baby_face` / `face` 的 YOLO 模型。

### 4. Detector Lab 調參窗口

已新增：

```text
http://localhost:8000/detector-lab
```

目前支援：

- 上傳圖片或影片
- 選擇 detector
- 調整暗度、動作、寶寶臉部、臉部遮蔽相關門檻
- 影片抽 frame 後送後端偵測
- 顯示每張 frame 的結果、信心值、細節
- 將 YOLO 偵測框與結果直接畫在預覽畫面上
- 可另存標註後的 `.webm` 影片

### 5. 臉部遮蔽偵測

`face_cover_detector.py` 已先補好架構，未來訓練好 YOLO 後可使用：

```text
有 face，但沒有 nose 或 mouth
→ 判斷可能遮蔽臉部
```

目前可透過 `config.ini` 設定模型路徑與信心門檻。

### 6. LINE 告警

目前 Python 後端會呼叫 Firebase Cloud Function，再由 Firebase 寫入 Firestore 並發 LINE。

事件告警策略集中在 `event_manager.py` 的 `alert_policies`：

- 可設定只記錄不發 LINE
- 可設定達指定次數才發 LINE
- 可設定一天最多發幾次
- `/test-alert` 可測試 Firebase / LINE 流程

### 7. 趴睡風險偵測

已加入 `prone_sleep_detector.py`，使用 COCO Pose 的鼻、眼、耳、肩與髖關鍵點可見度判斷風險。只有身體姿勢足以判讀、但大部分臉部關鍵點不可見時才會回報；狀態需持續 10 秒才建立事件並通知。

這是單鏡頭姿勢風險提示，不是醫療判斷，也無法只靠單張 RGB 畫面證明寶寶一定趴睡。建議使用固定俯視角度並以實際環境資料調整門檻。

```ini
[detectors.prone_sleep]
enabled = false
yolo_model_path = backend/detector/model/yolo11n-pose.pt
keypoint_confidence_threshold = 0.35
min_body_keypoints = 3
max_visible_face_keypoints = 1
```

### 8. 目前測試狀態

目前後端測試已通過：

```text
pytest -> 25 passed
```

已涵蓋：

- dark detector
- motion detector
- baby detector
- face cover detector
- prone sleep detector
- event alert policy

### 俯拍安全檢測（新增）

新增 `SafetyPipeline`，共用定位結果判斷畫面品質、睡眠區異物、頭部遮擋、邊界與睡姿。
支援 Detector Lab 測試與正常／異常／無法判讀狀態；新事件預設只記錄。
每台相機會獨立判斷人物候選，空床或未知狀態時暫停人物相關告警；即時頁可自動切換候選床位。
正式使用前須逐台設定床墊範圍，異物及睡姿分類另需對應模型。
詳見 [俯拍管線設定與限制](docs/top-down-safety.md)。


## Windows 雙擊啟動

完成本機環境與 `config.ini` 設定後，雙擊 `start_windows.bat`：

1. 使用專案內的 `.venv`，後端已運行時直接沿用。
2. 未運行時，在背景啟動本機監看服務，等待就緒後開啟預設瀏覽器的 `/camera`。
3. 在頁面按「啟用聲音警報」，保持頁面開啟且電腦未靜音。重新整理或重開頁面後需再次啟用。

啟動失敗時視窗會保留錯誤訊息；後端啟動輸出位於 `logs/launcher.log`。
關閉網頁不會停止背景後端。此啟動方式維持本機監看模式，不會自動發送 LINE。

新電腦需先建立 `.venv`、安裝 `requirements.txt`，並複製 `config.ini.example` 為 `config.ini` 後填入本機設定。
哭聲辨識還需 `requirements-audio.txt`、FFmpeg 與本機音訊模型，詳見 `docs/audio-monitor.md`。
GitHub 不包含帳密、本機模型、截圖、錄影或事件資料庫；請另外在本機配置。
