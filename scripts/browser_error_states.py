"""Exercise actual missing API and missing-artifact UI states in isolated processes."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from threadlens.launcher import check_port

ROOT = Path(__file__).resolve().parents[1]


def wait_http(url: str, child: subprocess.Popen) -> None:
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError("Fixture service exited; inspect .local/browser-fixtures logs")
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Fixture did not become available at {url}")


def main() -> None:
    local = ROOT / ".local" / "browser-fixtures"
    local.mkdir(parents=True, exist_ok=True)
    empty_root = Path(tempfile.mkdtemp(prefix="missing-model-", dir=local))
    children = []
    logs = []
    try:
        for port in (8031, 8531, 8532, 65500):
            check_port(port)
        env = os.environ.copy()
        env["THREADLENS_ROOT"] = str(empty_root)
        log = (local / "missing-model.log").open("w")
        logs.append(log)
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "threadlens.cli",
                    "serve",
                    "--api-port",
                    "8031",
                    "--ui-port",
                    "8531",
                ],
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
        wait_http("http://127.0.0.1:8031/health", children[-1])
        ready = httpx.get("http://127.0.0.1:8031/ready", timeout=5)
        assert ready.status_code == 503, "Empty model fixture must report actual 503 readiness"
        wait_http("http://127.0.0.1:8531/_stcore/health", children[-1])
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/browser_smoke.py"),
                "--url",
                "http://127.0.0.1:8531",
                "--prefix",
                "missing-model",
                "--missing-model",
            ],
            check=True,
            cwd=ROOT,
        )
        env["THREADLENS_API_URL"] = "http://127.0.0.1:65500"
        log = (local / "missing-api.log").open("w")
        logs.append(log)
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(ROOT / "ui/app.py"),
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    "8532",
                    "--server.headless",
                    "true",
                ],
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
        wait_http("http://127.0.0.1:8532/_stcore/health", children[-1])
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/browser_smoke.py"),
                "--url",
                "http://127.0.0.1:8532",
                "--prefix",
                "missing-api",
                "--unavailable",
            ],
            check=True,
            cwd=ROOT,
        )
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        for log in logs:
            log.close()
        print("Stopped only fixture processes created by this script.", flush=True)


if __name__ == "__main__":
    main()
