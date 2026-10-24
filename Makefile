.PHONY: setup check test lint run demo doctor collect experiments
setup:
	python3.11 -m venv .venv
	.venv/bin/python -m pip install uv
	.venv/bin/uv pip sync requirements.lock
	.venv/bin/uv pip install --no-deps -e .
check: lint test
lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
test:
	.venv/bin/pytest -q
run:
	.venv/bin/threadlens serve
demo:
	.venv/bin/threadlens demo
	.venv/bin/threadlens serve --demo
doctor:
	.venv/bin/threadlens doctor
collect:
	.venv/bin/threadlens collect
experiments:
	.venv/bin/threadlens experiments
