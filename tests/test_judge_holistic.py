"""Offline checks for one-request judging with the agent's evidence contract."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from mlflow import MlflowClient
from pydantic import SecretStr

from decomp_judge import judge_holistic as judge
from decomp_judge.devai_smoke import TrackingSettings


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    app = tmp_path / "app"
    app.mkdir()
    (app / "app.js").write_text("SOURCE_MARKER\nsecond line\n", encoding="utf-8")
    spec = tmp_path / "spec.txt"
    spec.write_text("Task text", encoding="utf-8")
    req = tmp_path / "req.json"
    req.write_text(
        json.dumps([{"requirement_id": "R1", "criteria": "Display text", "satisfied": False}]),
        encoding="utf-8",
    )
    return app, spec, req


def test_prompt_includes_sources_numbered_and_excludes_labels(
    inputs: tuple[Path, Path, Path],
) -> None:
    app, spec, req = inputs
    (app / "secret.env").write_text("ENV_MARKER\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    messages, manifest, requirements = judge.build_messages(
        spec, req, app, root / "configs/judge-holistic.md", root / "configs/judge-agent.md"
    )
    assert list(manifest) == ["app.js"]
    assert len(requirements) == 1
    assert "1: SOURCE_MARKER\n2: second line" in messages[1]["content"]
    assert '"satisfied"' not in messages[1]["content"]
    assert "ENV_MARKER" not in messages[1]["content"]
    assert '"NA"' in messages[0]["content"]


def test_dry_run_needs_no_key(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, spec, req = inputs
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_holistic",
            "--spec",
            str(spec),
            "--requirements",
            str(req),
            "--app",
            str(app),
            "--instruction",
            str(root / "configs/judge-holistic.md"),
            "--policy",
            str(root / "configs/judge-agent.md"),
        ],
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("No key or model call during dry run")

    monkeypatch.setattr(judge, "Credentials", forbidden)
    monkeypatch.setattr(judge, "call_model", forbidden)
    judge.main()


@pytest.mark.parametrize("mode", ["valid", "bad-evidence", "request-error"])
def test_tracked_single_call(
    inputs: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    app, spec, req = inputs
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(judge, "committed_version", lambda: ("b" * 40, "test/holistic"))
    monkeypatch.setattr(
        judge, "Credentials", lambda: SimpleNamespace(litellm_api_key=SecretStr("fake-key"))
    )
    calls = []
    raw = json.dumps(
        {
            "verdicts": [
                {
                    "requirement_id": "R1",
                    "verdict": "PASS",
                    "evidence": [
                        {
                            "file": "missing.js" if mode == "bad-evidence" else "app.js",
                            "line_start": 1,
                            "line_end": 2,
                        }
                    ],
                    "reason": "Evidence",
                }
            ]
        }
    )

    def fake_call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(1)
        if mode == "request-error":
            raise ValueError("Rejected fake-key")
        return "Explanation\n```json\n" + raw + "\n```", {
            "tokens": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            "finish_reason": "stop",
        }

    monkeypatch.setattr(judge, "call_model", fake_call)
    tracking = TrackingSettings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'db.sqlite'}",
        mlflow_experiment_name="holistic-offline",
    )
    output = tmp_path / "output"
    args = (
        root / "opencode.json",
        root / "configs/judge-holistic.md",
        root / "configs/judge-agent.md",
        spec,
        req,
        app,
        output,
        judge.HolisticSettings(),
        tracking,
    )
    if mode == "valid":
        run_id = judge.run(*args)
    else:
        with pytest.raises(RuntimeError):
            judge.run(*args)
        run_id = json.loads((output / "run.json").read_text())["run_id"]
    assert calls == [1]
    record = MlflowClient(tracking_uri=tracking.mlflow_tracking_uri).get_run(run_id)
    assert record.info.status == ("FINISHED" if mode == "valid" else "FAILED")
    assert record.data.params["temperature"] == "0"
    assert record.data.metrics["workspace_unchanged"] == 1
    assert (output / "verdicts.json").exists() is (mode == "valid")
    assert "fake-key" not in "".join(p.read_text(encoding="utf-8") for p in output.iterdir())
    if mode == "valid":
        assert record.data.metrics["surrounding_text_removed"] == 1
        assert record.data.metrics["total_tokens"] == 120


def test_adapter_temperature_and_no_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "opencode.json").read_text())

    def completion(**kwargs: Any) -> SimpleNamespace:
        assert kwargs["temperature"] == 0 and kwargs["num_retries"] == 0
        assert kwargs["model"] == "iairlab/qwen3.8-27b"
        assert kwargs["api_base"] == "https://api.duckduck.cloud/v1"
        assert kwargs["custom_llm_provider"] == "openai"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
            usage=None,
        )

    fake = ModuleType("litellm")
    monkeypatch.setattr(fake, "completion", completion, raising=False)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    assert judge.call_model(config, [], "fake-key", judge.HolisticSettings())[0] == "{}"
