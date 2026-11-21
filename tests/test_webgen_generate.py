"""Test isolated task input, persistent results and safe resume without a model."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import pytest
from pydantic import SecretStr

from decomp_judge import webgen_generate as generation


def settings() -> generation.GenerationConfig:
    return generation.GenerationConfig(
        model="iairlab/qwen3.8-27b",
        api_base="https://api.example.test/v1",
        opencode_version="1.19.4",
        instruction="instruction.md",
    )


def test_generation_limits_can_be_disabled_or_set_explicitly() -> None:
    default = settings()
    config = generation.build_config(default, "Implement")
    assert default.timeout_seconds is None
    assert "steps" not in config["agent"]["webgen-builder"]
    assert config["plugin"][0].endswith("/configs/webgen-no-output-cap.mjs")
    bounded = default.model_copy(update={"max_steps": 10, "timeout_seconds": 60})
    assert generation.build_config(bounded, "Implement")["agent"]["webgen-builder"]["steps"] == 10


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"id": "../bad", "instruction": "Build"}],
        [{"id": "000001", "instruction": "  "}],
        [{"id": "000001", "instruction": "Build"}] * 2,
    ],
)
def test_invalid_tasks_are_rejected(tmp_path: Path, rows: list[dict[str, Any]]) -> None:
    source = tmp_path / "test.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        generation.load_tasks(source)


def test_only_specification_and_id_are_loaded(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "000001",
                "instruction": "Build an app",
                "requirements": "HIDDEN REQUIREMENT",
                "satisfied": True,
            }
        )
    )
    tasks = generation.load_tasks(source)
    assert tasks[0].model_dump() == {"id": "000001", "instruction": "Build an app"}


def test_exclusive_batch_lock_releases_after_error(tmp_path: Path) -> None:
    lock = tmp_path / "lock"
    with pytest.raises(RuntimeError), generation.batch_lock(lock):
        with pytest.raises(OSError), generation.batch_lock(lock):
            pytest.fail("A second batch must not acquire the lock")
        raise RuntimeError("Worker interrupted")
    with generation.batch_lock(lock):
        pass


def test_dependencies_removed_but_source_kept(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.js").write_text("source")
    (tmp_path / "node_modules/pkg").mkdir(parents=True)
    (tmp_path / "node_modules/pkg/index.js").write_text("dependency")
    (tmp_path / "package-lock.json").write_text("{}")
    generation.clean_dependencies(tmp_path)
    assert not (tmp_path / "node_modules").exists()
    assert set(generation.app_files(tmp_path)) == {"src/main.js", "package-lock.json"}


@pytest.mark.parametrize("mode", ["success", "empty", "api-error"])
def test_task_status_and_provenance_are_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    dataset = tmp_path / "test.jsonl"
    dataset.write_text('{"id":"000001","instruction":"Build an app"}\n')
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
    mlflow.set_experiment("generation-tests")
    output = tmp_path / "output"
    output.mkdir()

    def fake_process(
        command: list[str],
        app: Path,
        env: dict[str, str],
        timeout: int | None,
    ) -> tuple[str, str, int]:
        assert "HIDDEN REQUIREMENT" not in command[-1]
        assert "--pure" not in command  # Pure mode disables the output-limit plugin.
        assert timeout is None
        assert env["LITELLM_API_KEY"] == "test-key"
        if mode == "success":
            (app / "index.html").write_text("<h1>App</h1>")
        text = (
            json.dumps({"type": "error", "message": "test-key failed"})
            if mode == "api-error"
            else ""
        )
        return text, "", 0

    monkeypatch.setattr(generation, "run_process", fake_process)
    task = generation.load_tasks(dataset)[0]
    result = generation.run_task(
        task,
        0,
        output,
        settings(),
        generation.build_config(settings(), "Implement"),
        "a" * 40,
        42,
        "test-key",
        dataset_sha256="b" * 64,
    )
    assert result["status"] == ("GENERATED" if mode == "success" else "FAILED")
    stored = json.loads((output / "task-000001/run.json").read_text())
    assert stored["run_id"] == result["run_id"]
    assert stored["commit"] == "a" * 40
    assert stored["requirements_evaluated"] is False
    assert "test-key" not in (output / "task-000001/raw-output.jsonl").read_text()
    record = mlflow.MlflowClient().get_run(result["run_id"])
    assert record.info.status == ("FINISHED" if mode == "success" else "FAILED")
    counts = generation.update_summary(
        output, [task, generation.GenerationTask(id="000002", instruction="B")]
    )
    assert counts["PENDING"] == 1


def test_dry_run_does_not_read_key_or_call_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "test.jsonl"
    dataset.write_text('{"id":"000001","instruction":"Build"}\n')
    monkeypatch.setattr(
        "sys.argv", ["generate", "--input", str(dataset), "--output", str(tmp_path / "out")]
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Dry run must not read credentials")

    monkeypatch.setattr(generation, "Credentials", forbidden)
    generation.main()
    assert "DRY RUN: 1 specifications" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_resume_skips_generated_apps_and_refuses_changed_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "test.jsonl"
    original = "\n".join(
        json.dumps({"id": f"{i:06d}", "instruction": f"Build {i}"}) for i in (1, 2)
    )
    dataset.write_text(original)
    output = tmp_path / "batch"
    monkeypatch.setenv("GENERATION_CONTAINER", "1")
    monkeypatch.setattr(
        generation, "Credentials", lambda: SimpleNamespace(litellm_api_key=SecretStr("test-key"))
    )
    monkeypatch.setattr(generation, "source_version", lambda: "a" * 40)
    real_output = subprocess.check_output

    def version(command: list[str], **kwargs: Any) -> Any:
        return (
            "1.19.4\n" if command == ["opencode", "--version"] else real_output(command, **kwargs)
        )

    monkeypatch.setattr(subprocess, "check_output", version)
    calls = []

    def fake_process(
        command: list[str], app: Path, env: dict[str, str], timeout: int | None
    ) -> tuple[str, str, int]:
        calls.append(app)
        (app / "index.html").write_text("<h1>App</h1>")
        return "", "", 0

    monkeypatch.setattr(generation, "run_process", fake_process)
    arguments = ["generate", "--input", str(dataset), "--output", str(output), "--execute"]
    monkeypatch.setattr("sys.argv", arguments + ["--limit", "1"])
    generation.main()
    assert len(calls) == 1
    dataset.write_text(original + "\n" + json.dumps({"id": "000003", "instruction": "Extra"}))
    monkeypatch.setattr("sys.argv", arguments + ["--resume"])
    with pytest.raises(SystemExit):
        generation.main()
    assert len(calls) == 1
    dataset.write_text(original)
    generation.main()
    generation.main()
    assert len(calls) == 2
    (output / "task-000001/app/index.html").write_text("changed")
    with pytest.raises(ValueError, match="Generated source changed"):
        generation.main()
    assert len(calls) == 2
