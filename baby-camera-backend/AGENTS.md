# Project workflow

- The user requests that completed project updates be committed and pushed to the configured GitHub origin. After appropriate verification, push updates without asking for the same permission again. Report any push failure honestly; never force-push or discard unrelated changes.
- Never commit config.ini, .env, credentials, camera captures, recordings, databases, runtime logs, or downloaded model weights. Use placeholders in examples.
- The double-click entry point is start_windows.bat. It uses the local .venv and scripts.open_monitor, reuses a running server, and opens /camera.
- Preserve the local-only notification behavior of scripts.run_monitor unless the user explicitly requests changing automatic outbound notifications.
