# 哭聲／尖叫提醒

兩台 RTSP 攝影機均已確認提供 8 kHz、單聲道 pcm_alaw 音訊。FFmpeg 以 TCP 獨立讀音訊、轉為 16 kHz PCM；影像與音訊管線分開。原始音訊只留在記憶體，不持續錄音、不上傳外部服務。

本機模型：[MIT AST AudioSet](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)，採 sigmoid 多標籤分數。下載版本記錄於 `models/audio-ast/source.json`；執行時只讀本機檔案。模型分數不是準確率或臨床機率。

每 5 秒非重疊音訊窗判讀一次：Baby cry, infant cry 分數至少 0.5，或 Screaming 分數至少 0.7，即產生本機事件與網頁提示。這是「5 秒片段被分類為哭聲」，不代表能逐秒證明連續哭滿 5 秒。尖叫同樣需等待此音訊窗完成。擷取、相機緩衝及推論會增加延遲，沒有即時性保證。

同一段哭聲只提醒一次；需連續兩個未達門檻的音訊窗（約 10 秒）才能重新進入提醒狀態，且每台相機兩次警報至少間隔 60 秒。中斷不算安靜、不累積哭聲時間。過期或處理過慢的資料不觸發告警。音訊斷線自動重連，API 顯示 reconnecting/stale/error，不能當成安靜。

音訊不受人物定位閘門限制，避免人體漏檢讓哭聲被跳過；相機 ID 代表收音來源，不能證明是哪張床的嬰兒在哭。兩台可能聽到同一聲音，也可能因電視、成人尖叫或其他嬰兒誤報。

## 網頁警報

`/camera` 顯示哭聲狀態及最近 60 秒的紅色提醒。按「啟用聲音警報（播放測試音）」後，可播放三聲短提示音。切換相機不會停掉另一台的提醒。頁面重新載入後須重新啟用聲音；關閉頁面、裝置靜音或瀏覽器暫停背景分頁時不保證能聽到。

這次僅啟用網頁與 SQLite 事件，不發送 LINE。背景排程會自動載入 `[audio] enabled=true`。口鼻遮擋模型仍缺失，與哭聲功能分開，不能用哭聲推論嗆奶或呼吸狀態。

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

目前初始閾值與音訊窗固定於 `backend/audio_monitor.py`；變更需重跑驗證與重啟監控。

## 驗證與限制

94 項測試通過，涵蓋持續時間、重複抑制、安靜重設、冷卻及斷線。另用[ESC-50 標註樣本](https://github.com/karolpiczak/ESC-50)作本機 smoke test：嬰兒哭聲 1-187207-A-20.wav 的 2 秒切片漏檢，5 秒完整音訊分數約 0.532，超過初始 0.5 門檻。這也是改採 5 秒窗的依據，屬開發校正，不是獨立驗證準確率。靜音與狗叫負樣本另做檢查。

尚未完成你家實際哭聲、尖叫、說話、電視、距離與夜間環境的標註驗證，因此仍有誤報／漏報；不刻意讓寶寶哭來測試。ESC-50 檔案只供本機測試，保留資料來源與其授權限制（主要為 CC BY-NC）；不打包發布為本專案資產。
