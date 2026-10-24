"""Local developer acceptance: isolated demo and safe process/port boundaries."""

import json
from pathlib import Path

import pytest

from threadlens.cli import parser
from threadlens.config import Settings
from threadlens.demo import prepare_demo
from threadlens.launcher import check_port, serve
from threadlens.models.inference import ModelManager


def test_demo_reload_and_isolation(tmp_path: Path):
    root = tmp_path / "isolated-demo"
    registry = prepare_demo(root)
    assert registry["models"][0]["provenance"] == "synthetic_demo"
    manager = ModelManager(Settings(root=root))
    prediction = manager.predict(["Could you explain how to profile this program?"])[0]
    assert prediction["model_version"] == "synthetic-demo-v1"
    assert set(prediction["scores"]) == {"analysis", "request", "commentary"}
    assert any("Synthetic" in warning for warning in prediction["warnings"])
    assert (root / ".local/labeled.jsonl").is_file()
    assert not (tmp_path / "artifacts").exists()


def test_demo_does_not_replace_real_registry(tmp_path: Path):
    folder = tmp_path / "artifacts"
    folder.mkdir()
    original = {"models": [{"provenance": "weak_label"}], "active_model": "real"}
    (folder / "model_registry.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="non-demo"):
        prepare_demo(tmp_path)
    assert json.loads((folder / "model_registry.json").read_text()) == original


def test_launcher_refuses_conflicts_without_stopping_processes():
    for port in (-1, 80, 65536):
        with pytest.raises(ValueError):
            check_port(port)
    with pytest.raises(ValueError, match="different ports"):
        serve(api_port=8099, ui_port=8099)


def test_experiment_cli_supports_required_commit_boundary():
    parsed = parser().parse_args(["experiments", "--stage", "improved", "--device", "cpu"])
    assert parsed.stage == "improved"
    assert parsed.device == "cpu"
