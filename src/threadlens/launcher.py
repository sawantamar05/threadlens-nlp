"""Loopback-only launcher; only terminates the subprocesses it creates."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

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


def _process_identity(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _state_path(api_port: int, ui_port: int) -> Path:
    return settings.local / f"launcher-{api_port}-{ui_port}.json"


def stop(api_port: int = 8000, ui_port: int = 8501) -> None:
    """Signal only the recorded launcher, checking start time and command against PID reuse."""
    path = _state_path(api_port, ui_port)
    if not path.is_file():
        raise RuntimeError(
            "No recorded ThreadLens launcher for these ports. Use the same ports/root as serve."
        )
    state = json.loads(path.read_text())
    pid = state.get("pid", 0)
    identity = state.get("identity", "")
    if type(pid) is not int or pid <= 1 or not identity or _process_identity(pid) != identity:
        raise RuntimeError(
            "Launcher record is stale or does not match the process; no signal was sent."
        )
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not path.exists() or _process_identity(pid) != identity:
            print("Stopped the recorded ThreadLens launcher and its child services.")
            return
        time.sleep(0.1)
    raise RuntimeError(
        "Launcher has not finished shutdown; inspect its logs. No other process was signaled."
    )


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
    state_path = _state_path(api_port, ui_port)

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
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"pid": os.getpid(), "identity": _process_identity(os.getpid())})
        )
        print(
            f"ThreadLens {'SYNTHETIC DEMO' if demo else 'LOCAL'}\nAPI: http://127.0.0.1:{api_port}\nDocs: http://127.0.0.1:{api_port}/docs\nUI: http://127.0.0.1:{ui_port}\nLogs: {logs}\nCtrl+C or threadlens stop --api-port {api_port} --ui-port {ui_port} stops both services.",
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
        try:
            if (
                state_path.is_file()
                and json.loads(state_path.read_text()).get("pid") == os.getpid()
            ):
                state_path.unlink()
        except (OSError, ValueError):
            pass
        for sig, handler in previous.items():
            signal.signal(sig, handler)
