"""Offline checks for the strict requirement verdict protocol."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from mlflow import MlflowClient
from pydantic import SecretStr

from decomp_judge import devai_smoke
from decomp_judge.devai_smoke import load_messages, parse_judgment


def response(ids: list[int]) -> str:
    return json.dumps(
        {"verdicts": [{"requirement_id": i, "verdict": "PASS", "reason": "Evidence"} for i in ids]}
    )


def test_unordered_ids_are_matched_by_id() -> None:
    result = parse_judgment(response([1, 0]), [0, 1])
    assert [v.requirement_id for v in result.verdicts] == [1, 0]


@pytest.mark.parametrize("ids", [[], [0], [0, 0], [0, 1, 2], [0, 9]])
def test_missing_duplicate_and_extra_ids_are_rejected(ids: list[int]) -> None:
    with pytest.raises(ValueError):
        parse_judgment(response(ids), [0, 1])


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        '```json\n{"verdicts": []}\n```',
        '{"verdicts":[{"requirement_id":true,"verdict":"PASS","reason":"x"}]}',
        '{"verdicts":[{"requirement_id":0,"verdict":"MAYBE","reason":"x"}]}',
        '{"verdicts":[{"requirement_id":0,"verdict":"PASS","reason":" "}]}',
    ],
)
def test_invalid_json_schema_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_judgment(raw, [0])


def test_message_roles_and_unicode(tmp_path: Path) -> None:
    path = tmp_path / "messages.json"
    messages = [{"role": "system", "content": "Judge"}, {"role": "user", "content": "ТЗ"}]
    path.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    assert load_messages(path) == messages
    path.write_text(json.dumps(messages[::-1]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_messages(path)


def test_default_cli_cannot_call_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    messages = tmp_path / "messages.json"
    messages.write_text(
        json.dumps([{"role": "system", "content": "Judge"}, {"role": "user", "content": "Task"}]),
        encoding="utf-8",
    )
    config = Path(__file__).resolve().parents[1] / "configs" / "devai-smoke.yaml"
    monkeypatch.setattr(
        sys, "argv", ["devai_smoke", "--config", str(config), "--messages", str(messages)]
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Dry run must not read credentials or call the model")

    monkeypatch.setattr(devai_smoke, "request_once", forbidden)
    monkeypatch.setattr(devai_smoke, "Credentials", forbidden)
    monkeypatch.setattr(devai_smoke, "run_tracked", forbidden)
    devai_smoke.main()
    assert "DRY RUN" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["valid", "invalid", "request-error", "provider-error"])
def test_mlflow_keeps_inputs_and_failed_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = devai_smoke.TrackingSettings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}",
        mlflow_experiment_name="devai-offline-test",
        seed=7,
    )
    config = Path(__file__).resolve().parents[1] / "configs" / "devai-smoke.yaml"
    messages = tmp_path / "messages.json"
    message_text = json.dumps(
        [{"role": "system", "content": "Judge"}, {"role": "user", "content": "ТЗ"}],
        ensure_ascii=False,
    )
    messages.write_text(message_text, encoding="utf-8")
    raw = response(list(range(7))) if mode == "valid" else "not JSON"
    calls: list[int] = []

    def fake_request(
        config: devai_smoke.RequestConfig, messages: list[dict[str, str]], seed: int
    ) -> str:
        calls.append(seed)
        if mode == "request-error":
            raise RuntimeError("provider details must not be stored")
        if mode == "provider-error":
            raise devai_smoke.ModelRequestError("BadRequestError", "Test parameter rejected", 400)
        return raw

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Offline tracking test must not read credentials")

    monkeypatch.setattr(devai_smoke, "request_once", fake_request)
    monkeypatch.setattr(devai_smoke, "Credentials", forbidden)
    output = tmp_path / "output"

    def execute() -> str:
        return devai_smoke.run_tracked(
            config, messages, output, settings, commit="offline-test", branch="test/devai"
        )

    if mode == "valid":
        run_id = execute()
    else:
        expected_error = ValueError if mode == "invalid" else RuntimeError
        with pytest.raises(expected_error):
            execute()
        run_id = json.loads((output / "run.json").read_text(encoding="utf-8"))["run_id"]
    assert calls == [7]

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    run = client.get_run(run_id)
    assert run.info.status == ("FINISHED" if mode == "valid" else "FAILED")
    assert run.data.tags["commit"] == "offline-test"
    assert run.data.tags["issue"] == "4"
    assert run.data.params["seed"] == "7"
    saved_messages = Path(client.download_artifacts(run_id, "inputs/messages.json"))
    assert saved_messages.read_text(encoding="utf-8") == message_text
    saved_config = Path(client.download_artifacts(run_id, "inputs/config.yaml"))
    assert saved_config.read_text(encoding="utf-8") == config.read_text(encoding="utf-8")
    if mode in {"request-error", "provider-error"}:
        expected_kind = "RuntimeError" if mode == "request-error" else "BadRequestError"
        assert run.data.tags["request_error_type"] == expected_kind
        assert not (output / "raw-response.txt").exists()
        assert "provider details" not in str(run.data)
        if mode == "provider-error":
            saved_error = Path(client.download_artifacts(run_id, "outputs/request-error.json"))
            diagnostic = json.loads(saved_error.read_text(encoding="utf-8"))
            assert diagnostic["http_status"] == 400
            assert diagnostic == json.loads(
                (output / "request-error.json").read_text(encoding="utf-8")
            )
    else:
        assert run.data.metrics["response_valid"] == int(mode == "valid")
        saved_raw = Path(client.download_artifacts(run_id, "outputs/raw-response.txt"))
        assert saved_raw.read_text(encoding="utf-8") == raw
        assert (output / "raw-response.txt").read_text(encoding="utf-8") == raw
    assert (output / "verdicts.json").exists() == (mode == "valid")
    with pytest.raises(FileExistsError):
        execute()
    assert calls == [7]


@pytest.mark.parametrize("dirty", [False, True])
def test_real_execution_requires_a_clean_checkout(
    monkeypatch: pytest.MonkeyPatch, dirty: bool
) -> None:
    commands: list[list[str]] = []

    def fake_git(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1] == "status":
            text = "?? uncommitted.py" if dirty else ""
        elif command[1] == "rev-parse":
            text = "a" * 40
        else:
            text = "docs/devai-data-smoke-test"
        return subprocess.CompletedProcess(command, 0, stdout=text, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_git)
    if dirty:
        with pytest.raises(ValueError, match="Commit"):
            devai_smoke.committed_version()
        assert len(commands) == 1
    else:
        assert devai_smoke.committed_version() == ("a" * 40, "docs/devai-data-smoke-test")
        assert len(commands) == 3


def test_provider_diagnostics_remove_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "devai-smoke.yaml"
    import yaml

    config = devai_smoke.RequestConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    placeholder = "offline-placeholder"

    class BadRequestError(RuntimeError):
        status_code = 400

    def reject(**kwargs: object) -> None:
        raise BadRequestError(f"Test rejection: key={placeholder}, Bearer another-placeholder")

    fake = ModuleType("litellm")
    monkeypatch.setattr(fake, "completion", reject, raising=False)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setattr(
        devai_smoke,
        "Credentials",
        lambda: SimpleNamespace(litellm_api_key=SecretStr(placeholder)),
    )
    with pytest.raises(devai_smoke.ModelRequestError) as caught:
        devai_smoke.request_once(config, [{"role": "user", "content": "test"}], 42)
    diagnostic = caught.value.diagnostic
    assert diagnostic["http_status"] == 400
    assert diagnostic["error_type"] == "BadRequestError"
    assert placeholder not in str(diagnostic)
    assert "another-placeholder" not in str(diagnostic)
    assert "Test rejection" in str(diagnostic)
