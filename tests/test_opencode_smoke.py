"""Offline evidence checks and tracked CLI sessions with no real model or key."""

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mlflow import MlflowClient
from pydantic import SecretStr

from decomp_judge import opencode_smoke as agent
from decomp_judge.devai_smoke import TrackingSettings


def transcript(workspace: Path, *, status: str = "completed", path: str = "src/model.py") -> str:
    return "\n".join(
        json.dumps(event)
        for event in [
            {
                "type": "tool_use",
                "part": {
                    "tool": "read",
                    "state": {
                        "status": status,
                        "input": {"filePath": str(workspace / path)},
                        "output": "1: from sklearn.svm import SVR",
                    },
                },
            },
            {"type": "text", "part": {"text": "Прочитан src/model.py:1, используется SVR."}},
        ]
    )


@pytest.mark.parametrize(
    "status,path,expected",
    [
        ("completed", "src/model.py", True),
        ("error", "src/model.py", False),
        ("completed", "src/train.py", False),
    ],
)
def test_only_completed_target_read_counts(
    tmp_path: Path, status: str, path: str, expected: bool
) -> None:
    summary = agent.inspect_events(transcript(tmp_path, status=status, path=path), tmp_path)
    assert summary["target_read_verified"] is expected


def test_text_claim_alone_does_not_verify_read(tmp_path: Path) -> None:
    raw = json.dumps({"type": "text", "part": {"text": "I read the file"}})
    assert agent.inspect_events(raw, tmp_path)["target_read_verified"] is False


@pytest.mark.parametrize("mode", ["success", "no-read", "timeout"])
def test_tracked_session_preserves_evidence_and_redacts_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    prepared = tmp_path / "prepared"
    workspace = prepared / "workspace"
    (workspace / "src").mkdir(parents=True)
    content = b"from sklearn.svm import SVR\n"
    (workspace / "src/model.py").write_bytes(content)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "source_commit": "a" * 40,
                "task": "39_Test",
                "file_sha256": {"src/model.py": sha256(content).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(agent, "committed_version", lambda: ("b" * 40, "test/agent"))
    monkeypatch.setattr(
        agent,
        "Credentials",
        lambda: SimpleNamespace(litellm_api_key=SecretStr("offline-key-placeholder")),
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        commands.append(command)
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, stdout="1.19.4\n")
        assert kwargs["env"]["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
        assert kwargs["env"]["LITELLM_API_KEY"] == "offline-key-placeholder"
        assert kwargs["env"]["OPENCODE_CONFIG_CONTENT"]
        if mode == "timeout":
            raise subprocess.TimeoutExpired(
                command, 1, output=b"partial", stderr=b"offline-key-placeholder"
            )
        raw = (
            transcript(workspace)
            if mode == "success"
            else json.dumps({"type": "text", "part": {"text": "I read it"}})
        )
        return subprocess.CompletedProcess(
            command, 0, stdout=raw.encode(), stderr=b"offline-key-placeholder"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    tracking = TrackingSettings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}",
        mlflow_experiment_name="agent-offline-test",
    )
    output = tmp_path / "run"

    def execute() -> str:
        return agent.run(
            tmp_path / "opencode.exe",
            root / "configs/opencode-judge.json",
            root / "configs/opencode-file-smoke.txt",
            prepared,
            output,
            tracking,
            agent.RuntimeSettings(),
        )

    if mode == "success":
        run_id = execute()
    else:
        with pytest.raises(RuntimeError):
            execute()
        run_id = json.loads((output / "run.json").read_text(encoding="utf-8"))["run_id"]
    assert len(commands) == 2
    saved = "\n".join(p.read_text(encoding="utf-8") for p in output.iterdir() if p.is_file())
    assert "offline-key-placeholder" not in saved
    assert "[REDACTED]" in (output / "stderr.txt").read_text(encoding="utf-8")
    assert (output / "events.jsonl").exists()
    client = MlflowClient(tracking_uri=tracking.mlflow_tracking_uri)
    record = client.get_run(run_id)
    assert record.info.status == ("FINISHED" if mode == "success" else "FAILED")
    assert record.data.params["opencode_version"] == "1.19.4"
    assert record.data.tags["commit"] == "b" * 40
    if mode == "success":
        assert record.data.metrics["target_read_verified"] == 1
        assert record.data.metrics["workspace_unchanged"] == 1
    else:
        assert (output / "error.json").exists()
