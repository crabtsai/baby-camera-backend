# 口鼻遮擋與奶嘴：實作及尚缺資源

目前只完成判讀規則、模型契約及頁面狀態。**尚無符合規格的模型，口鼻遮擋告警未啟用。** 本機只有 face detector 與 pose detector，不能拿其鼻部關鍵點或臉部框冒充遮擋證據。

## 判讀

- 只有同一張臉內的部位可參與判斷；多人、缺位置、推論失敗皆為 unknown。
- 明確的 `blanket_on_nose` / `blanket_on_mouth` / `covered_face` 證據優先，奶嘴不能消除此提醒。
- `pacifier_in_mouth` 專指正在口中使用的奶嘴；床上奶嘴、靠近臉的奶嘴不能標為此類。鼻部仍須可見，奶嘴框與鼻部重疊時為 unknown。
- 未看見嘴巴且沒有口中奶嘴證據、或未看見鼻子，都回傳 unknown，不將模型漏檢直接當成棉被遮擋。
- 棉被框與口鼻框相交只足以回報 unknown，無法證明實際遮擋；框可能包含背景。
- clear 只描述該張模型的可見性證據，不是呼吸／吞嚥正常或安全的證明。
- require_nose / require_mouth 舊設定目前保留相容介面，但不允許略過上述必要可見性檢查。

`extra.evaluable=false` 會重設既有事件持續計時。若未來有合格模型並啟用，疑似遮擋連續達到現有 face_cover_warning 的 5 秒條件才記錄事件（冷卻 180 秒）。目前背景入口仍是本機紀錄模式，不會發送 LINE。

## 本機聲音提醒

`/camera` 按「啟用聲音警報」後會播放測試音。通過上述持續時間與冷卻條件的口鼻遮擋事件，會透過 `/health` 的各相機 `airway_alert` 顯示並播放三聲提示音；所有相機均會提醒，不受目前選取畫面影響。同一事件每個頁面只播放一次，新事件保留 60 秒，服務重啟不重播歷史資料。此通道獨立於 LINE 開關。

聲音由開啟頁面的電腦／手機播放，需保持頁面開啟、瀏覽器音訊允許且音量開啟；背景頁面或裝置休眠可能延遲提醒。警報代表近期發生事件，不代表目前仍在遮擋。攝影機本體喇叭尚未整合，需先確認型號與控制介面。模型缺失時不會產生口鼻遮擋事件，也不會因此響鈴。

## 模型契約

本機 Ultralytics YOLO detect 權重須包含以下全部類別，缺類別／模型錯誤則停用並顯示 unknown，不在啟動時下載權重：

```text
face
nose
mouth
pacifier_in_mouth
blanket
blanket_on_nose
blanket_on_mouth
covered_face
```

nose / mouth 必須標註可見部分，不能依人臉幾何猜出被遮住的位置。兩種 blanket_on_* 框為可確認的棉被遮擋口鼻區域；covered_face 框為確認遭覆蓋的頭臉區域。單張無法確認時應由連續影像及人工覆核確認或排除，不製造假真值。

訓練需要實際拍攝視角、日／夜模式、頭部朝向、奶嘴形狀、衣物、照顧者與空床的人工標註；按拍攝時段切分訓練與驗證，避免相鄰影格洩漏。先報告各類漏報、誤報、unknown 比例與事件延遲，再決定是否啟用。

既有寶寶照片與使用者提供的「正在吃奶嘴」僅是觀測背景，尚不是逐部位標註，也沒有用於模型訓練。遮擋樣本應以假娃娃與布料測試，不能遮住真實寶寶來製造資料。

## 尚存限制

主管線仍以人物候選為閘門：完全覆蓋導致人體模型漏檢時，本 detector 不執行；即使加入模型仍可能漏報。完全覆蓋時的 covered_face 分支只在已有有效人物候選的情況下適用。未具備經驗證的頭部追蹤與遮擋資料前，不能宣稱解決此限制。

`/camera` 已顯示口鼻判讀狀態，包括未啟用、unknown 與資料過期。服務 health=ok 只代表服務健康，不代表口鼻遮擋功能可用。

## 本機模型工作流程

使用專案 `.venv`，下列指令不會更改線上設定或自動啟用模型：

```powershell
.venv/Scripts/python.exe -m scripts.airway_model init
.venv/Scripts/python.exe -m scripts.airway_model inspect --model models/yolov11n_face_detection.pt
.venv/Scripts/python.exe -m scripts.airway_model validate
```

`inspect` 顯示 task、實際類別、SHA-256 及缺少的類別；現有人臉模型只有 `face`，因此檢查不通過屬於預期結果。僅載入可信任的本機 `.pt`。

`init` 建立 `data/airway/images/{train,val,test}`、對應 `labels`、`dataset.yaml` 與 `samples.csv`，不覆寫既有檔案。每張影像須有同名 YOLO 標註文字檔；人工確認的負樣本使用空檔，不將遺漏標註當作負樣本。類別 ID 依上方八類順序，從 0 開始。每列為 `class_id x_center y_center width height`，座標正規化到 0–1。

`samples.csv` 範例（影像與標註須自行準備，範例不是訓練資料）：

```csv
image,session,assessment
images/train/day01_001.jpg,camera1-day01,clear
images/val/night02_001.jpg,camera1-night02,abnormal
images/test/day03_001.jpg,camera2-day03,unknown
```

assessment 是人工覆核的單張判讀：clear / abnormal / unknown。相同拍攝場次與其衍生影像必須留在同一 split。檢查器拒絕跨 split 場次、重複解碼影像、缺標註、越界／非有限框及缺少類別／判讀狀態；格式通過不代表樣本數足夠或人工標註正確。相似影格仍須人工去重。

資料完成後，以可信任的本機 detection 模型為起點：

```powershell
.venv/Scripts/python.exe -m scripts.airway_model train --model models/yolov11n_face_detection.pt --epochs 100 --device cpu
.venv/Scripts/python.exe -m scripts.airway_model evaluate --model runs/airway/train/weights/best.pt
```

訓練輸出目錄可能自動加序號，以指令實際回傳的 weights 路徑為準。測試使用獨立 test 場次及實際 `FaceCoverDetector` 規則，產生逐張預測、三態混淆矩陣、漏報率（unknown 也計入遮擋漏報）、誤報率與 unknown 比例。這不是端到端告警驗證，仍需測試人物閘門、日夜模式與連續事件延遲。報告不會自動標記可上線。

資料、公開下載、報告及訓練產物均留在忽略追蹤的 `data/airway`、`runs/airway`，不推送照片或權重。資料格式依 [Ultralytics 官方文件](https://docs.ultralytics.com/datasets/detect/)。

## 公開資料來源檢查（2026-10-24）

- [CribNet / CribHD](https://github.com/ostadabbas/CribNet)：官方提供[資料 ZIP](https://coe.northeastern.edu/Research/AClab/CribHD/CribHD.zip)，含被子、玩具與假娃娃模擬場景。研究的被子遮擋判定結合分割與姿勢點，並非本專案八類模型；不可直接將 blanket 標籤改名為 blanket_on_nose。README 將程式限於非商業用途；資料再利用應另檢查隨附授權。
- [Roboflow Baby](https://universe.roboflow.com/oscar6207-gmail-com/baby-trill)：頁面列 CC BY 4.0，包含 face_all_covered / face_mouth_covered / face_pacifier 等標籤，但無已訓練模型，下載要求登入。第 1 版含增強影像且沒有 test split，需要檢查標註語意並按來源重新分組。不能把整臉狀態框直接當成精確口鼻部位框。

下載 CribHD 後可檢查 CRC、計算 SHA-256、僅解壓資料檔並列出類別：

```powershell
.venv/Scripts/python.exe -m scripts.inspect_cribhd data/airway/downloads/CribHD.zip
```

此步只準備研究資料，不會下載或假造一個可用的口鼻警報模型。

本次實際下載檢查結果：ZIP 為 640,664,878 bytes，CRC 通過，SHA-256 為 `c771c76755ebc39f090ad50d83a10c1e5cc0e8b979c21c75fcafc57f319e4f9f`。解壓位置 `data/airway/public/cribhd`，詳細清單為該目錄的 `archive-report.json`。檔案內沒有模型權重；被子 YOLO 資料實際為 train 449、valid 40、test 10 張，只有 `blanket` 一類；模擬場景另有 120 張。全包 4,617 個 JPG 包含多種標註格式的重複影像，不可當成 4,617 個獨立樣本。尚缺口鼻、奶嘴與精確遮擋標註，不能直接用此包訓練完整八類模型。

## 現場啟用測試

依使用者要求，已將本機 `[detectors.face_cover] enabled=true` 並重啟排程。模型路徑仍未設定，因此模組會回報 unknown／模型不可用，不會產生遮擋警報。`/health` 預期為 degraded，表示這個必要模型缺失；其他相機擷取、臉部與動作偵測繼續執行。這是接線測試，不是遮擋辨識能力測試。
