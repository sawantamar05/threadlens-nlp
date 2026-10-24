"""Loopback-only launcher; only terminates the subprocesses it creates."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time

import httpx

from threadlens.config import ROOT, settings


def check_port(port: int) -> None:
    if not 1024 <= port <= 65535:
        raise ValueError("Choose an unprivileged port between 1024 and 65535.")
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is unavailable. Use --api-port / --ui-port; no existing process was stopped."
            ) from exc


def serve(api_port: int = 8000, ui_port: int = 8501, demo: bool = False) -> None:
    """Run in foreground; Ctrl+C stops precisely these two child processes."""
    if api_port == ui_port:
        raise ValueError("API and UI need different ports.")
    check_port(api_port)
    check_port(ui_port)
    env = os.environ.copy()
    root = settings.local / "demo" if demo else settings.root
    if demo and not (root / "artifacts/model_registry.json").exists():
        from threadlens.demo import prepare_demo

        prepare_demo(root)
    env["THREADLENS_ROOT"] = str(root)
    env["THREADLENS_API_URL"] = f"http://127.0.0.1:{api_port}"
    env["PYTHONUNBUFFERED"] = "1"
    logs = root / ".local/logs"
    logs.mkdir(parents=True, exist_ok=True)
    children: list[subprocess.Popen] = []
    streams = []
    previous = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupted)
        commands = [
            [
                sys.executable,
                "-m",
                "uvicorn",
                "threadlens.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
            ],
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "ui/app.py"),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(ui_port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
        ]
        for name, command in zip(("api", "ui"), commands, strict=True):
            stream = (logs / f"{name}.log").open("a")
            streams.append(stream)
            children.append(
                subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
            )
            if name == "api":
                for _ in range(120):
                    if children[0].poll() is not None:
                        raise RuntimeError(f"API exited. Inspect {logs / 'api.log'}")
                    try:
                        response = httpx.get(f"http://127.0.0.1:{api_port}/health", timeout=1)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError(f"API startup timed out; inspect {logs / 'api.log'}")
                ready = httpx.get(f"http://127.0.0.1:{api_port}/ready", timeout=120)
                print(f"Model readiness ({ready.status_code}): {ready.text}", flush=True)
        print(
            f"ThreadLens {'SYNTHETIC DEMO' if demo else 'LOCAL'}\nAPI: http://127.0.0.1:{api_port}\nDocs: http://127.0.0.1:{api_port}/docs\nUI: http://127.0.0.1:{ui_port}\nLogs: {logs}\nCtrl+C stops both services.",
            flush=True,
        )
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        raise RuntimeError(f"A service exited. Inspect logs at {logs}")
    except KeyboardInterrupt:
        print("Stopping ThreadLens child processes…", flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for stream in streams:
            stream.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
