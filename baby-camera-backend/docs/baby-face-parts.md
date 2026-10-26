# MediaPipe 離線口鼻定位

口鼻定位已由 Roboflow 改成 MediaPipe Face Landmarker。無 API key、無 HTTP 推論、無 Docker；套件與模型在本機備妥後，使用 CPU 離線執行。

## 模型與安裝

- [官方 Python 文件](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- [官方模型下載](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- [核心 Face Mesh 模型卡／Apache 2.0 授權](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)

本機模型：`models/face_landmarker.task`。本次下載 SHA-256：`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。權重不提交 Git；新環境必須另外下載，服務啟動不會自動上網抓檔。

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
curl.exe -L --fail -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
.venv/Scripts/python.exe -m scripts.check_face_parts
```

套件固定 `mediapipe==1.1.1`、`opencv-contrib-python==4.10.0.84`；OpenCV 與專案原版本一致，避免自動升級到 5.x。Windows 更新 OpenCV 前須先停止占用它的監看服務。

## 設定

在本機 `config.ini` 設定：

```ini
[detectors.face_parts]
enabled = true
model_path = models/face_landmarker.task
confidence = 0.5
```

各相機可用 `[camera.<camera_id>.detectors.face_parts]` 覆寫（實際 section 命名遵循 config.ini.example）。舊 api_key、api_url、allow_cloud、timeout_sec 不再讀取。`/config` 顯示 `model_id=mediapipe-face-landmarker` 及 `offline=true`。

每台相機使用自己的 MediaPipe instance，採 IMAGE 模式逐張判讀，不沿用舊畫面的追蹤結果。最多偵測兩張臉，遇到多臉就回報無法定位。監看主管線仍使用原本的人物與畫面品質閘門。

## 結果解讀

模型輸出 478 個估計關鍵點，程式將它們整理成臉、鼻部與嘴唇的估計框及部分關鍵點；Detector Lab 用橘框及「估計位置」標示，不顯示假的可見度機率。`/camera` 的「寶寶口鼻位置」顯示當前定位訊息。

- `localization_status=located`：單一臉部的估計位置有效。
- `localization_status=unknown`：無臉、多臉、超出畫面、幾何無效、缺模型或推論失敗。
- 為相容舊介面保留 `face_parts_visibility` 事件名稱，但 `visibility_confirmed=false`、`assessment=unknown`、`evaluable=false`；定位不代表口鼻實際露出或呼吸正常。
- 此模組永不產生異常警報。被子遮擋仍由獨立 `FaceCoverDetector` 負責；它仍缺專用模型，不能宣稱已解決。

合成黑色影像測試的 `inference_completed=true` 代表模型載入與推論成功；此影像沒有臉，回報 unknown 正常。這不是嬰兒、側臉或夜視準確率驗證。模型對遮擋、臉部尺寸、角度和光線敏感；關鍵點可能是推估而非直接觀測。

## 本機驗證（2026-10-26）

124 項測試通過，包含真實模型 CPU 推論與 Python socket 連線禁止測試；`pip check` 通過。監看排程已重啟，兩台攝影機均有新鮮影像並載入口鼻模組。另各取一張即時畫面在記憶體內執行 MediaPipe，兩次推論成功，但當下均未找到單一臉部，故回報 unknown，尚不能當成現場口鼻定位準確率驗證。沒有上傳或新增保存攝影機畫面。
