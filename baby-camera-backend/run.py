import uvicorn
from backend.config import get_settings

if __name__ == "__main__":
    # 本機開發啟動入口；正式部署時也可以直接用 uvicorn backend.main:app。
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        # reload=True 方便開發時改檔自動重啟，正式環境建議關閉。
        reload=True,
    )
