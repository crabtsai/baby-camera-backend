# 2026-10-16 實拍校正

15:01–15:03（台灣時間）完成一次 120 秒、每 5 秒抽樣的本機收集。
Session：`data/calibration/20261016-150131-769e4f`。
第 1 台 24 張、第 2 台 23 張，共 47 張 640×360 JPEG，寫入錯誤 0。
人工檢視初始畫面：第 1 床無寶寶、第 2 床有寶寶側頭及包覆身體。
未逐張完成真值標註；以下是模型檢出張數，不是準確率或安全偵測靈敏度。

## 門檻比較

| 模型／門檻 | 第 1 台（24 張） | 第 2 台（23 張） |
| --- | ---: | ---: |
| Pose 0.5（原值） | 0 | 18 |
| Pose 0.4（候選） | 0 | 23 |
| 臉部 0.4（原值） | 0 | 17 |
| 臉部 0.25（候選） | 0 | 22 |
| 臉部 0.1 | 4 | 23 |

0.1 在空床出現疑似誤檢，因此未採用。最低信心的三張第 2 台畫面已畫框檢視；框落在頭臉附近，但側臉與耳朵框不能證明口鼻可見。
以正式 BabyDetector 回放，0.25 在第 2 台檢出 22/23 張、第 1 台 0/24 張。

已在本機 `config.ini` 加入第 2 台的暫定覆寫：

```ini
[camera.baby-cam-02.detectors.baby]
confidence_threshold = 0.25

[camera.baby-cam-02.detectors.safety]
confidence = 0.4
```

第 1 台維持原設定。Safety 的 confidence 同時作用於定位及關鍵點門檻，後續仍須驗證頭部框；本次未宣稱頭部定位改善。設定須下次啟動服務才載入。本次只啟動限時擷取及離線評估，未啟動告警服務、未發訊息，收集已結束。

## 遮臉、溢奶與嗆奶的實際狀態

- face_cover 目前停用、無口鼻模型；異物／睡姿模型及床墊校正也缺少。降低臉部門檻不等於完成遮臉偵測。
- 已在 samples.jsonl 加入 face_visibility 與 visible_spit_up 人工標註欄位，預設 unreviewed。模型輸出另存 predictions.json，不當作真值。
- 未訓練或更新權重；這 2 分鐘同場景照片不足以訓練並驗證遮臉／溢奶偵測。後續需要不同時段、光線、姿勢及自然發生的事件，按錄製時段分開訓練與驗證，避免相鄰照片洩漏。
- 嗆奶欄位固定 not_assessable_from_still；照片無法確認吞嚥、呼吸道阻塞或吸入。尚未新增嗆奶告警，不用白色物體或低動作當作診斷。
- 遮臉情境可用假娃娃與布料製作測試資料，不應遮住真實寶寶或刻意引發溢奶／嗆奶。溢奶樣本僅收集自然出現且照護優先後的資料。
- 如需溢奶研究，需先確認口部可見及足夠解析度，再收集連續短片及人工確認的事件時間；目前側頭視角與 640×360 畫面不足以驗證此能力。

## 重跑

```powershell
.venv\Scripts\python.exe -m scripts.collect_calibration --seconds 120 --interval 5
.venv\Scripts\python.exe -m scripts.evaluate_calibration data/calibration/<session>
```

收集每次建立新目錄，最長允許 1 小時，可建立該 session 的 `STOP` 檔提前停止。
照片、manifest、人工標註欄位及評估均位於 Git 忽略的 `data/calibration/`，未上傳外部服務。
68 項既有 pytest 測試通過；兩台實機收集及正式臉部 detector 回放成功。這些驗證不代表臨床有效性。

安全睡眠參考：[NICHD 仰睡與溢奶說明](https://safetosleep.nichd.nih.gov/reduce-risk/back-sleeping)。避免因擔心溢奶而自行改成趴睡；監視影像不能取代現場照護。
