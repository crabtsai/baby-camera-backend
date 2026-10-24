# Windows 持續監控

安裝／啟動：在專案根目錄執行 `./scripts/install_monitor_task.ps1`。
工作名稱 `BabyCameraMonitor`，使用本專案 `.venv`、固定工作目錄及 `scripts.run_monitor`；不使用開發用 reload。

- 一般權限：目前使用者登入 Windows 後自動啟動，登出會中斷監控。
- 系統管理員執行安裝腳本：使用目前帳號 S4U、開機觸發，不儲存密碼。需在目標電腦驗證此模式；目前安裝的是登入觸發。
- 異常退出後每分鐘重試，最多 999 次；不限制執行時間、同一排程不重複啟動。
- 執行期間防止閒置睡眠；不改全機電源配置。手動睡眠、關機、登出、斷電仍會中斷監控。
- 相機中斷由 CameraStream 自動重連。程序仍活著但卡住不會觸發排程的退出重試；請用 `/health` 檢查。

每秒約抽一張作偵測，兩台相機獨立判斷人物候選。連續符合在床候選條件才執行動作／臉部偵測；空床或不能判讀時停止人物相關事件累積。尚未校正床墊區域，當前人物候選範圍是整張影像，不等於已辨識嬰兒身分。品質檢查持續執行。

此背景入口啟用 local_monitor_only，僅本機事件與截圖，不發送 LINE／Webhook，亦不更動原 config.ini 的告警設定。此模式的設定會反映在執行中 `/config`。需要對外告警時須另行明確啟用並驗證。

口鼻遮擋、溢奶、嗆奶尚未有可用且驗證過的模型；服務啟動成功不表示這些功能已完成。

查看：[即時畫面](http://127.0.0.1:8000/camera)、[健康狀態](http://127.0.0.1:8000/health)。日誌在 `logs/app.log`，自動輪替；事件及異常截圖在 `data`，目前沒有自動清理政策，需注意磁碟空間。

```powershell
Get-ScheduledTask -TaskName BabyCameraMonitor
Get-ScheduledTaskInfo -TaskName BabyCameraMonitor
Stop-ScheduledTask -TaskName BabyCameraMonitor
Start-ScheduledTask -TaskName BabyCameraMonitor
# 禁止下次登入自動啟動（不會停止目前正在執行的工作）：
Disable-ScheduledTask -TaskName BabyCameraMonitor
```
