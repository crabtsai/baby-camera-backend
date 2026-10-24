"""Double-click launcher: reuse the monitor or start it, then open its page."""
from __future__ import annotations

import argparse
import configparser
import json
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def is_monitor(url):
    try:
        with urllib.request.urlopen(url + '/health', timeout=2) as response:
            data = json.load(response)
        return isinstance(data, dict) and 'cameras' in data and 'audio' in data
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser', action='store_true', help='Check/start without opening a tab')
    args = parser.parse_args()
    if not (ROOT / 'config.ini').is_file():
        print('Missing config.ini. Copy config.ini.example and configure your cameras first.')
        return 1
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / 'config.ini', encoding='utf-8-sig')
    host = config.get('app', 'host', fallback='0.0.0.0')
    host = '127.0.0.1' if host in ('0.0.0.0', '') else '::1' if host == '::' else host
    port = config.getint('app', 'port', fallback=8000)
    url_host = '[' + host + ']' if ':' in host else host
    url = f'http://{url_host}:{port}'
    if not is_monitor(url):
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f'Port {port} is busy, but the monitor is not ready. Try again shortly.')
                return 1
        except OSError:
            pass
        log_dir = ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        with (log_dir / 'launcher.log').open('ab') as log:
            process = subprocess.Popen(
                [sys.executable, '-m', 'scripts.run_monitor'], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
        print('Starting monitor...')
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if is_monitor(url):
                break
            if process.poll() is not None:
                print('Monitor exited. Check logs/launcher.log.')
                return 1
            time.sleep(1)
        else:
            print('Monitor is still starting. Check logs/launcher.log, then try again.')
            return 1
    print(f'Monitor ready: {url}/camera')
    print('Enable the sound alarm on the page. Keep the page open and volume on.')
    if not args.no_browser and not webbrowser.open(url + '/camera'):
        print('Could not open your browser. Open the URL above manually.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
