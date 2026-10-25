# Roboflow 寶寶口鼻位置偵測

模型來源：[sho / baby face parts](https://universe.roboflow.com/sho-4iv4r/baby-face-parts)，固定模型 ID `baby-face-parts/2`。官方頁面列為 Roboflow 3.0 Object Detection (Fast)，類別為 HEAD、EYE、NOSE、MOUTH、BABY。資料頁標示 CC BY 4.0，使用時應保留作者 sho 與來源連結；模型部署另依 Roboflow 模型授權。

## 判讀範圍

新增獨立的 `baby_face_parts_detector`，輸出 `face_parts_visibility`，不取代八類 `FaceCoverDetector`。只有單一頭部內的鼻與嘴都被偵測到，才回報本張可見；缺少口鼻、多人、模型失敗時為 unknown，不會建立遮擋警報。此模型沒有 blanket、covered_face 或 pacifier 類別，不能辨識被子遮擋或判斷奶嘴是否安全。

頁面公開的 mAP@50 71.5%、precision 77.8%、recall 63.4% 是原資料集指標，並非目前攝影機現場準確度。實際使用仍需驗證日夜、距離及角度。

## 連接模型

Roboflow Universe 模型需要 API key。此模型不能直接把網站連結填入 `yolo_model_path`。目前接線使用 Roboflow Inference HTTP API，依[官方本機部署文件](https://docs.roboflow.com/deployment/self-hosted/self-hosted)準備 Inference Server（支援 Authorization header 的 1.5.0 或更新版）。本機尚未安裝 Docker／推論服務，因此這是待完成的環境步驟。

官方流程是在已具備 Docker 的環境安裝 `inference-cli` 並執行 `inference server start`。建議使用獨立環境，不更動監看程式依賴；只將服務綁定本機。第一次載入 Universe 模型需要連網取得模型與驗證 API key。

在不受 Git 追蹤的 `config.ini` 加入：

```ini
[detectors.face_parts]
enabled = true
api_url = http://127.0.0.1:9001
api_key = YOUR_ROBOFLOW_API_KEY
allow_cloud = false
confidence = 0.4
timeout_sec = 10
```

API key 請直接填入本機設定，不貼在聊天、程式碼或 GitHub。`/config` 只顯示是否設定，不回傳 key。先用合成黑色影像檢查連線（不擷取攝影機）：

```powershell
.venv/Scripts/python.exe -m scripts.check_face_parts
```

service_responded=true 只證明連線與回應格式正確，黑色影像回報 unknown 為正常結果。確認服務可用後重新啟動監看服務，`/camera` 的「寶寶口鼻位置」會顯示結果，Detector Lab 可選「寶寶口鼻」檢查框。啟用 safety pipeline 時沿用人物與畫面品質閘門，無人物候選不會推論。

預設只接受 loopback 服務，停用環境 proxy、HTTP redirect 與每次請求的 model monitoring。若使用者明確同意將寶寶影像送至 Roboflow 雲端，才可設定 `api_url=https://serverless.roboflow.com` 並將 `allow_cloud=true`。本次未授權也未執行雲端影像傳送。LINE 通知設定保持獨立。

## 驗證狀態

已測試 HEAD／NOSE／MOUTH 框轉換、單一頭部對應、低信心值、多人與漏檢、連線失敗清除舊結果、API key 不進入狀態／錯誤訊息、HTTP payload 與禁止自動 redirect。這些為模擬回應與程式測試；尚未取得 API key，因此未完成真實模型下載與攝影機現場推論。
