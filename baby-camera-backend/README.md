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
│   ├── config.py                  # 系統設定，讀取 .env
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
├── .env.example                   # 設定範例
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

攝影機即時畫面頁面：
```text
http://localhost:8000/camera
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
  "camera": {
    "source": "0",
    "opened": true,
    "last_read_ok": true,
    "has_latest_frame": true
  },
  "detectors": ["dark_detector", "motion_detector"]
}
```

---

### GET `/stream`

MJPEG 即時影像串流。

手機或電腦瀏覽器打開：

```text
http://localhost:8000/stream
```

---

### GET `/events`

取得最近異常事件。

參數：

```text
/events?limit=50
```

---

### POST `/test-alert`

手動送出測試告警。

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
- 趴睡偵測
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
