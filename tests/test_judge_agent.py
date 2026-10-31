"""Generic judge protocol and tracked OpenCode runner; no real key or API."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mlflow import MlflowClient
from pydantic import SecretStr

from decomp_judge import judge_agent as agent
from decomp_judge.devai_smoke import TrackingSettings
from decomp_judge.judge_validator import Requirement, normalize_response, validate_judgment


def reply(file: str = "src/app.txt", start: int = 1, end: int = 2) -> str:
    return json.dumps(
        {
            "verdicts": [
                {
                    "requirement_id": "R1.1",
                    "verdict": "PASS",
                    "evidence": [{"file": file, "line_start": start, "line_end": end}],
                    "reason": "Подтверждено кодом",
                }
            ]
        }
    )


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    app = tmp_path / "app"
    (app / "src").mkdir(parents=True)
    (app / "src/app.txt").write_text("PRIVATE_SOURCE_MARKER\nsecond line\n", encoding="utf-8")
    spec = tmp_path / "spec.txt"
    spec.write_text("Specification", encoding="utf-8")
    requirements = tmp_path / "requirements.json"
    requirements.write_text(
        json.dumps([{"requirement_id": "R1.1", "criteria": "Display text"}]), encoding="utf-8"
    )
    return app, spec, requirements


@pytest.mark.parametrize(
    "file,start,end",
    [
        ("missing.txt", 1, 1),
        ("../outside.txt", 1, 1),
        ("/etc/passwd", 1, 1),
        ("C:/file.txt", 1, 1),
        ("src\\app.txt", 1, 1),
        ("src/app.txt", 0, 1),
        ("src/app.txt", 2, 1),
        ("src/app.txt", 1, 3),
    ],
)
def test_invalid_evidence_rejected(
    inputs: tuple[Path, Path, Path], file: str, start: int, end: int
) -> None:
    with pytest.raises(ValueError):
        validate_judgment(
            reply(file, start, end), [Requirement(requirement_id="R1.1", criteria="x")], inputs[0]
        )


def test_ids_preserve_type_and_missing_pass_evidence_rejected(
    inputs: tuple[Path, Path, Path],
) -> None:
    data = json.loads(reply())
    data["verdicts"][0]["requirement_id"] = 1
    with pytest.raises(ValueError):
        validate_judgment(
            json.dumps(data), [Requirement(requirement_id="1", criteria="x")], inputs[0]
        )
    data = json.loads(reply())
    data["verdicts"][0]["evidence"] = []
    with pytest.raises(ValueError):
        validate_judgment(
            json.dumps(data), [Requirement(requirement_id="R1.1", criteria="x")], inputs[0]
        )
    data["verdicts"][0]["verdict"] = "NA"
    assert validate_judgment(
        json.dumps(data), [Requirement(requirement_id="R1.1", criteria="x")], inputs[0]
    )


def test_tokens_count_once_and_final_step_only() -> None:
    events = [
        {"type": "step_start"},
        {"type": "text", "part": {"text": "intermediate"}},
        {
            "type": "step_finish",
            "part": {"tokens": {"input": 10, "output": 2, "reasoning": 3, "total": 15}},
        },
        {"type": "step_start"},
        {"type": "text", "part": {"text": reply()}},
        {
            "type": "step_finish",
            "part": {"tokens": {"input": 20, "output": 4, "reasoning": 1, "total": 25}},
        },
    ]
    result = agent.collect_events("\n".join(json.dumps(e) for e in events))
    assert result["raw_response"] == reply()
    assert result["tokens"]["totals"] == {"input": 30, "output": 6, "reasoning": 4, "total": 40}


def test_extra_null_fields_in_evidence_are_not_silently_removed(
    inputs: tuple[Path, Path, Path],
) -> None:
    data = json.loads(reply())
    data["verdicts"][0]["evidence"][0]["extra_metadata"] = None
    with pytest.raises(ValueError, match="extra_metadata"):
        validate_judgment(
            json.dumps(data), [Requirement(requirement_id="R1.1", criteria="x")], inputs[0]
        )


@pytest.mark.parametrize("opening,newline", [("```json", "\n"), ("```", "\n"), ("```json", "\r\n")])
def test_single_outer_fence_preserves_json_content(opening: str, newline: str) -> None:
    body = "  " + reply() + "  "
    raw = "\n" + opening + newline + body + newline + "```\n"
    normalized, removed = normalize_response(raw)
    assert removed is True
    assert normalized == body


@pytest.mark.parametrize(
    "raw",
    [
        reply(),
        "",
        "not JSON",
        "```python\n{}\n```",
        "```json\n{}",
        "Explanation\n```json\n{}\n```",
        "```json\n{}\n```\nExplanation",
        "```json\n{}\n```\n```json\n{}\n```",
    ],
)
def test_other_formats_are_not_repaired(raw: str) -> None:
    assert normalize_response(raw) == (raw, False)


def test_wrapped_invalid_schema_still_fails(inputs: tuple[Path, Path, Path]) -> None:
    data = json.loads(reply())
    data["verdicts"][0]["evidence"][0]["unexpected"] = None
    raw = "```json\n" + json.dumps(data) + "\n```"
    normalized, removed = normalize_response(raw)
    assert removed
    with pytest.raises(ValueError, match="unexpected"):
        validate_judgment(normalized, [Requirement(requirement_id="R1.1", criteria="x")], inputs[0])


def test_dry_run_cannot_read_key_or_call_opencode(
    inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app, spec, requirements = inputs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "judge_agent",
            "--app",
            str(app),
            "--spec",
            str(spec),
            "--requirements",
            str(requirements),
        ],
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Dry run must not read keys or run OpenCode")

    monkeypatch.setattr(agent, "Credentials", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    agent.main()
    assert "DRY RUN" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["valid", "wrapped", "bad-evidence", "timeout", "mutated"])
def test_tracked_run(
    inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    app, spec, requirements = inputs
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent, "committed_version", lambda: ("a" * 40, "test/judge"))
    monkeypatch.setattr(
        agent, "Credentials", lambda: SimpleNamespace(litellm_api_key=SecretStr("fake-secret"))
    )

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, stdout=b"1.19.4\n")
        effective = json.loads(kwargs["env"]["OPENCODE_CONFIG_CONTENT"])
        assert effective["agent"]["requirements-judge"]["temperature"] == 0
        assert effective["agent"]["requirements-judge"]["steps"] == 4
        assert kwargs["cwd"] == app.resolve()
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert "PRIVATE_SOURCE_MARKER" not in command[-1]
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, 10, output=b"partial", stderr=b"fake-secret")
        if mode == "mutated":
            (app / "unexpected.txt").write_text("changed", encoding="utf-8")
        raw = reply("missing.txt") if mode == "bad-evidence" else reply()
        if mode == "wrapped":
            raw = "```json\n" + raw + "\n```"
        events = [
            {"type": "step_start"},
            {"type": "tool_use", "part": {"tool": "read", "state": {"status": "completed"}}},
            {"type": "text", "part": {"text": raw}},
            {
                "type": "step_finish",
                "part": {"tokens": {"input": 10, "output": 20, "reasoning": 0, "total": 30}},
            },
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(json.dumps(e) for e in events).encode(),
            stderr=b"fake-secret",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    tracking = TrackingSettings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_experiment_name="judge-offline",
    )
    output = tmp_path / "output"
    arguments = (
        tmp_path / "opencode.exe",
        root / "opencode.json",
        root / "configs/judge-agent.md",
        spec,
        requirements,
        app,
        output,
        agent.AgentSettings(max_steps=4),
        tracking,
    )
    if mode in {"valid", "wrapped"}:
        run_id = agent.run(*arguments)
    else:
        with pytest.raises(RuntimeError):
            agent.run(*arguments)
        run_id = json.loads((output / "run.json").read_text())["run_id"]
    record = MlflowClient(tracking_uri=tracking.mlflow_tracking_uri).get_run(run_id)
    assert record.info.status == ("FINISHED" if mode in {"valid", "wrapped"} else "FAILED")
    assert record.data.params["temperature"] == "0"
    assert (output / "timing.json").exists()
    assert (output / "tokens.json").exists()
    assert (output / "trace.json").exists()
    if mode not in {"valid", "wrapped"}:
        assert (output / "error.json").exists()
    assert "fake-secret" not in "".join(
        p.read_text(encoding="utf-8") for p in output.iterdir() if p.is_file()
    )
    if mode in {"valid", "wrapped"}:
        assert record.data.metrics["tokens_total"] == 30
        assert record.data.metrics["workspace_unchanged"] == 1
        assert (output / "verdicts.json").exists()
        assert record.data.metrics["markdown_fence_removed"] == int(mode == "wrapped")
        assert record.data.metrics["format_violation"] == int(mode == "wrapped")
        if mode == "wrapped":
            assert (output / "raw-response.txt").read_text(encoding="utf-8").startswith("```json")
            assert (output / "normalized-response.txt").read_text(encoding="utf-8") == reply()
    elif mode == "bad-evidence":
        assert (output / "raw-response.txt").read_text(encoding="utf-8") == reply("missing.txt")
        assert not (output / "verdicts.json").exists()
