"""Single-process unattended monitor; no reload, outbound alerts opt-in."""
from pathlib import Path
import ctypes
import os

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    from backend.config import get_settings
    from backend.main import app
    import uvicorn

    # Scheduling local monitoring does not opt in to external messaging.
    app.state.local_monitor_only = True
    settings = get_settings()
    if os.name == 'nt':
        # Prevent idle sleep only while this process runs; never change power plans.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        uvicorn.run(app, host=settings.app_host, port=settings.app_port,
                    access_log=False, reload=False)
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
