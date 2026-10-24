#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
.venv/bin/python -m pip install uv==0.13.5
.venv/bin/uv pip sync requirements.lock
.venv/bin/uv pip install --no-deps -e .
.venv/bin/threadlens doctor
