# 哭聲／尖叫提醒

兩台 RTSP 攝影機均已確認提供 8 kHz、單聲道 pcm_alaw 音訊。FFmpeg 以 TCP 獨立讀音訊、轉為 16 kHz PCM；影像與音訊管線分開。原始音訊只留在記憶體，不持續錄音、不上傳外部服務。

本機模型：[MIT AST AudioSet](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)，採 sigmoid 多標籤分數。下載版本記錄於 `models/audio-ast/source.json`；執行時只讀本機檔案。模型分數不是準確率或臨床機率。

每 5 秒非重疊音訊窗判讀一次：Baby cry, infant cry 分數至少 0.5，或 Screaming 分數至少 0.7，即產生本機事件與網頁提示。這是「5 秒片段被分類為哭聲」，不代表能逐秒證明連續哭滿 5 秒。尖叫同樣需等待此音訊窗完成。擷取、相機緩衝及推論會增加延遲，沒有即時性保證。

同一段哭聲只提醒一次；需連續兩個未達門檻的音訊窗（約 10 秒）才能重新進入提醒狀態，且每台相機兩次警報至少間隔 60 秒。中斷不算安靜、不累積哭聲時間。過期或處理過慢的資料不觸發告警。音訊斷線自動重連，API 顯示 reconnecting/stale/error，不能當成安靜。

音訊不受人物定位閘門限制，避免人體漏檢讓哭聲被跳過；相機 ID 代表收音來源，不能證明是哪張床的嬰兒在哭。兩台可能聽到同一聲音，也可能因電視、成人尖叫或其他嬰兒誤報。

## 網頁警報

`/camera` 顯示哭聲狀態及最近 60 秒的紅色提醒。按「啟用聲音警報（播放測試音）」後，可播放三聲短提示音。切換相機不會停掉另一台的提醒。頁面重新載入後須重新啟用聲音；關閉頁面、裝置靜音或瀏覽器暫停背景分頁時不保證能聽到。

預設僅啟用網頁與 SQLite 事件。明確設定 `[audio] notify_line=true` 後，哭聲／尖叫可透過既有 Firebase 入口傳送 LINE 文字與觸發相機的新畫面；這個選項在 `start_windows.bat` 與背景排程也生效，但不開啟其他偵測器的外部通知。需要有效 webhook，且 `[alert] dry_run=false`。音訊附圖會啟用，不受其他偵測器的本機模式影響；圖片仍受上傳大小限制。沒有新畫面時傳文字並說明未附圖，不使用過期圖片。

網路發送使用獨立、有容量限制的佇列，不阻塞音訊推論；同段哭聲仍只提醒一次且每台至少間隔 60 秒。發送失敗或佇列滿會顯示於 `/health` 的 `audio.last_notification`，目前不自動重送。`gateway_accepted` 只代表 Firebase 接受，LINE 由雲端非同步發送，不代表手機已送達。`audio.notification=browser_and_line` 表示音訊 LINE 路徑已啟用。背景排程會自動載入 `[audio] enabled=true`；哭聲功能不能推論嗆奶或呼吸狀態。

既有 Firebase 另有同相機、同事件類型 300 秒的 LINE 冷卻限制，因此本機事件成立不代表每次都會有新的 LINE 訊息。手機是否跳出提示音或橫幅，也取決於 LINE 通知與勿擾設定。

## 安裝與設定

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-audio.txt
```

模型需要 config.json、preprocessor_config.json、model.safetensors，放入 models/audio-ast。設定：

```ini
[audio]
enabled = true
model_path = models/audio-ast
ffmpeg_path = <本機 ffmpeg.exe 絕對路徑>
```

哭聲預設門檻為 0.5，可在 `[audio]` 設定 `cry_threshold`（允許 0.25–1，無效值退回 0.5）；尖叫門檻與音訊窗保持不變。變更需驗證並重啟監控，`/health` 的 `audio.cry_threshold` 顯示實際生效門檻。

## 使用個人哭聲參考校正

```powershell
.venv\Scripts\python.exe -m scripts.evaluate_cry_reference "<本機哭聲檔.mp3>" --cry-threshold 0.25 --comparison "<對照音訊.wav>"
```

工具將音檔副本、SHA-256、16 kHz 單聲道音訊及逐窗報告保存在 Git 忽略的 `data/calibration/audio-references/`。參考檔由使用者標示為哭聲，但不假定每一秒都只有哭聲；對照檔標籤仍需人工確認。完整 5 秒窗才計分，報告另列尾段未評分秒數。此工具不修改模型權重、設定或發送警報。

2026-11-06 的本機個人參考校正：50.91 秒參考檔共 10 個完整窗，門檻 0.5 時 4 窗達標，0.25 時 6 窗達標。兩段先前攝影機錄音及一段狗叫樣本合計 11 窗，在兩種門檻下皆未達標。這是同一份參考資料上的門檻校正，不是獨立準確率測試，也不是模型重新訓練；未達標的窗仍可能漏報。單一正樣本及少量對照不能證明泛化能力或排除其他說話、電視聲的誤報。

## 驗證與限制

需要保留現場音訊排查漏報時，可明確執行一次本機錄音診斷：

```powershell
.venv\Scripts\python.exe -m scripts.diagnose_audio --seconds 30
```

會同時錄下各攝影機音訊，保存在已被 Git 忽略的 `data/recordings/audio-diagnostic-時間/`，並產生 `report.json`，列出每個完整 5 秒窗的音量、哭聲／尖叫分數、前五項分類及推論時間。離線策略事件僅供診斷，不發警報、不寫事件資料庫、不上傳。可錄 5–120 秒；正常監控仍不保存原始音訊。離線策略由全新狀態開始，與正在運作中的警報冷卻狀態可能不同。

94 項測試通過，涵蓋持續時間、重複抑制、安靜重設、冷卻及斷線。另用[ESC-50 標註樣本](https://github.com/karolpiczak/ESC-50)作本機 smoke test：嬰兒哭聲 1-187207-A-20.wav 的 2 秒切片漏檢，5 秒完整音訊分數約 0.532，超過初始 0.5 門檻。這也是改採 5 秒窗的依據，屬開發校正，不是獨立驗證準確率。靜音與狗叫負樣本另做檢查。

尚未完成你家實際哭聲、尖叫、說話、電視、距離與夜間環境的標註驗證，因此仍有誤報／漏報；不刻意讓寶寶哭來測試。ESC-50 檔案只供本機測試，保留資料來源與其授權限制（主要為 CC BY-NC）；不打包發布為本專案資產。
