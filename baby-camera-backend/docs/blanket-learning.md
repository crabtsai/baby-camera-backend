# YOLO 棉被學習流程

開啟 `/blanket-learning` 可在本機標註照片；`/camera` 顯示收集數量及試驗模型狀態。模型預測與標註資料都不會上傳。

## 已實作

- 每次確認單一人物重新出現，收集一張照片；持續出現期間預設每 60 秒補拍。這是人物候選，不能驗證寶寶身分。
- 畫面品質不足、多人、未確認在床時不收集。每台攝影機最多 1000 張待整理照片，達上限停止收集並顯示訊息，不自動刪除照片。
- 照片、拍攝時段及待標註狀態保存在 `data/airway/blanket/pending/<camera>/`，不進 Git。重啟或確認人物離開後使用新拍攝時段 ID。
- 當同一影格有棉被分割結果，自動儲存 `.proposal.json` 建議輪廓；建議不等於訓練答案。沒有檢出棉被也不能自動標成「無棉被」。
- 標註頁可畫多個棉被輪廓、清除錯誤建議，或明確確認沒有棉被。確認後產生 `.review.json`，不立即更換模型。
- 匯出器只接受人工確認標註；同一時段不能同時分到訓練、驗證及測試集。應另留不同拍攝時段、日夜及棉被樣式測試，不能只用相鄰畫面測試。
- 即時模型支援 YOLO 分割輪廓與頭部區域交集，避免只用矩形框把棉被空白角落算成遮擋。

## 首輪訓練結果（2026-10-26）

使用已下載的 CribHD-B 公開棉被輪廓標註，保留原始分割：449 張訓練、40 張驗證、10 張測試。以官方 YOLO11n-seg 預訓練權重轉移學習，CPU、320 像素、5 epochs、凍結前 10 層、seed 17。這是確認流程可運作的初始候選模型，不是完成實際鏡位驗證的模型。

公開測試集分割 mAP50 = 0.7042、mAP50–95 = 0.5398、recall = 0.5331。這些指標不代表口鼻遮擋偵測率；測試集僅 10 張，且未驗證目前攝影機的夜間、棉被、奶嘴或轉頭情境。首張本機收集照片仍未成功檢出棉被，因此需要確認本機輪廓後再訓練，不能宣稱已學會這條棉被。

候選權重保存在 `models/blanket_candidate.pt`；本機目前使用 `objects_shadow=true`，可觀察輸出但不由候選棉被模型觸發警報。既有臉部參考變化與追蹤失效提醒繼續運作。

## 設定

```ini
[camera.bed-camera.detectors.blanket_learning]
enabled = true
interval_sec = 60
max_images = 1000

[camera.bed-camera.detectors.safety]
object_model_path = models/blanket_candidate.pt
hazardous_labels = blanket
object_imgsz = 320
objects_shadow = true
```

設定相機 ID 必須對應自己的床位。攝影機移位後要重新校準床墊區域。只設定 blanket 代表這個模型不辨識枕頭、玩具等其他類別。

## 後續訓練指令

先在標註頁確認照片輪廓，再匯出新版本，不會覆寫原資料集。PowerShell 範例：

```powershell
.venv/Scripts/python.exe -m scripts.blanket_model export-reviewed --base data/airway/blanket/public-v1 --output data/airway/blanket/local-v2
.venv/Scripts/python.exe -m scripts.blanket_model train --data data/airway/blanket/local-v2/dataset.yaml --weights models/blanket_candidate.pt --epochs 20 --name blanket-local-v2
```

訓練結束會在該 run 寫入 `evaluation.json`，不自動覆蓋目前模型或變更警報設定。新的攝影機照片還需確認標註；不能無條件把模型預測反覆加入訓練。

若要對已有照片重新產生建議輪廓：

```powershell
.venv/Scripts/python.exe -m scripts.blanket_model propose --weights models/blanket_candidate.pt
```

指令只處理尚無建議的照片，不覆寫人工確認結果。模型訓練須與監測共享 CPU，執行時應留意 `/health` 的推論延遲與畫面新鮮度。

參考：[Ultralytics 分割訓練及輸出格式](https://docs.ultralytics.com/tasks/segment/)、[CribNet / CribHD](https://github.com/ostadabbas/CribNet)。

開發測試依賴：`pip install -r requirements-dev.txt`，再執行 `python -m pytest -q`。
