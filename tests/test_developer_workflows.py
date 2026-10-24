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


def test_prepare_review_cli_imports_without_replacing_drafts(tmp_path, monkeypatch, capsys):
    from threadlens import cli
    from threadlens.storage.annotations import AnnotationStore

    config = Settings(root=tmp_path)
    config.local.mkdir()
    source = config.local / "labeled.jsonl"
    record = {
        "comment_id": "synthetic-1",
        "text": "Could you explain this test?",
        "label": "request",
        "provenance": "synthetic_demo",
    }
    source.write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(cli, "settings", config)
    monkeypatch.setattr("sys.argv", ["threadlens", "prepare-review"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["inserted"] == 1
    record["label"] = "commentary"
    source.write_text(json.dumps(record) + "\n")
    cli.main()
    assert json.loads(capsys.readouterr().out)["inserted"] == 0
    assert (
        AnnotationStore(config.database).history("synthetic-1", blind=False)["draft"]["label"]
        == "request"
    )


def test_stop_refuses_reused_pid_without_signaling(tmp_path, monkeypatch):
    from threadlens import launcher

    monkeypatch.setattr(launcher, "settings", Settings(root=tmp_path))
    path = launcher._state_path(8000, 8501)
    path.parent.mkdir()
    path.write_text(json.dumps({"pid": 12345, "identity": "old start time and command"}))
    monkeypatch.setattr(launcher, "_process_identity", lambda pid: "different unrelated process")
    signals = []
    monkeypatch.setattr(launcher.os, "kill", lambda *args: signals.append(args))
    with pytest.raises(RuntimeError, match="no signal was sent"):
        launcher.stop()
    assert signals == []


def test_stop_targets_matching_launcher_only(tmp_path, monkeypatch, capsys):
    from threadlens import launcher

    monkeypatch.setattr(launcher, "settings", Settings(root=tmp_path))
    path = launcher._state_path(8000, 8501)
    path.parent.mkdir()
    path.write_text(json.dumps({"pid": 12345, "identity": "fixed identity"}))
    monkeypatch.setattr(launcher, "_process_identity", lambda pid: "fixed identity")
    signals = []

    def terminate(pid, sig):
        signals.append((pid, sig))
        path.unlink()

    monkeypatch.setattr(launcher.os, "kill", terminate)
    launcher.stop()
    assert signals == [(12345, launcher.signal.SIGTERM)]
    assert "Stopped" in capsys.readouterr().out
