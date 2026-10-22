"""Validate the DevAI prompt and make one explicitly requested model call."""

import argparse
import json
import os
import re
import subprocess
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal

import mlflow
import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from decomp_judge.config import Settings


class RequestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    api_base: str
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    expected_ids: list[StrictInt]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    task: str
    issue: int = Field(gt=0)
    pr: str
    custom_llm_provider: str | None = None

    @field_validator("expected_ids")
    @classmethod
    def unique_ids(cls, value: list[int]) -> list[int]:
        if not value or len(set(value)) != len(value) or any(i < 0 for i in value):
            raise ValueError("Expected IDs must be nonempty, unique and nonnegative")
        return value


class Credentials(BaseSettings):
    # The key stays in process memory and is never serialized with the configuration.
    model_config = SettingsConfigDict(env_file=None, extra="ignore")
    litellm_api_key: SecretStr | None = None


class TrackingSettings(Settings):
    # Tracking uses environment variables without loading a credentials file.
    model_config = SettingsConfigDict(env_file=None, extra="ignore")


class RuntimeOverrides(BaseSettings):
    """Cluster-specific values that do not require editing committed YAML."""

    model_config = SettingsConfigDict(env_prefix="DEVAI_", env_file=None, extra="ignore")
    api_base: str | None = None
    model: str | None = None
    custom_llm_provider: str | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    pr: str | None = None


def effective_config(text: str) -> RequestConfig:
    values = RequestConfig.model_validate(yaml.safe_load(text)).model_dump()
    values.update(RuntimeOverrides().model_dump(exclude_none=True))
    return RequestConfig.model_validate(values)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: StrictInt
    verdict: Literal["PASS", "FAIL"]
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def one_line(cls, value: str) -> str:
        if not value.strip() or "\n" in value or "\r" in value:
            raise ValueError("The reason must be one nonempty line")
        return value


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[Verdict]


class ModelRequestError(RuntimeError):
    """Provider diagnostics with the known API key removed before tracking."""

    def __init__(self, kind: str, detail: str, status: int | None) -> None:
        super().__init__("Model request failed")
        self.diagnostic = {"error_type": kind, "detail": detail, "http_status": status}


def load_messages(path: Path) -> list[dict[str, str]]:
    return parse_messages(path.read_text(encoding="utf-8"))


def parse_messages(raw: str) -> list[dict[str, str]]:
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != 2:
        raise ValueError("Expected exactly one system message and one user message")
    messages = [Message.model_validate(item) for item in data]
    if [m.role for m in messages] != ["system", "user"]:
        raise ValueError("Message order must be system, user")
    return [m.model_dump() for m in messages]


def parse_judgment(raw: str, expected_ids: list[int]) -> Judgment:
    judgment = Judgment.model_validate_json(raw)
    ids = [v.requirement_id for v in judgment.verdicts]
    if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
        raise ValueError("Response must contain every expected ID exactly once")
    return judgment


def request_once(config: RequestConfig, messages: list[dict[str, str]], seed: int) -> str:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    credentials = Credentials()
    if credentials.litellm_api_key is None:
        raise ModelRequestError("ValueError", "Set LITELLM_API_KEY before execution", None)
    api_key = credentials.litellm_api_key.get_secret_value()
    try:
        import litellm

        litellm.telemetry = False
        response = litellm.completion(
            model=config.model,
            api_base=config.api_base,
            api_key=api_key,
            messages=messages,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
            num_retries=0,
            caching=False,
            seed=seed,
            custom_llm_provider=config.custom_llm_provider,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            choice = response.choices[0]
            message = choice.message
            usage = getattr(response, "usage", None)
            reasoning = getattr(message, "reasoning_content", None)
            counters = {
                name: value
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                if type(value := getattr(usage, name, None)) is int
            }
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            if type(reasoning_tokens) is int:
                counters["reasoning_tokens"] = reasoning_tokens
            summary = {
                "finish_reason": getattr(choice, "finish_reason", None),
                "usage": counters,
                "message_fields": sorted(vars(message)),
                "content_type": type(content).__name__,
                "reasoning_content_chars": len(reasoning) if isinstance(reasoning, str) else 0,
                "has_tool_calls": bool(getattr(message, "tool_calls", None)),
                "has_refusal": bool(getattr(message, "refusal", None)),
            }
            raise ValueError(
                "The model returned no text in choices[0].message.content; "
                + json.dumps(summary, ensure_ascii=False)
            )
        return content
    except Exception as exc:
        detail = str(exc)
        if api_key:
            detail = detail.replace(api_key, "[REDACTED]")
        detail = re.sub(r"(?i)Bearer\s+[^\s'\"<>]+", "Bearer [REDACTED]", detail)
        status = getattr(exc, "status_code", None)
        raise ModelRequestError(
            type(exc).__name__, detail[:2000], status if type(status) is int else None
        ) from None


def committed_version() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("status", "--porcelain"):
        raise ValueError("Commit the project changes before a real model run")
    return git("rev-parse", "HEAD"), git("branch", "--show-current")


def run_tracked(
    config_path: Path,
    messages_path: Path,
    output: Path,
    settings: TrackingSettings,
    *,
    commit: str,
    branch: str,
) -> str:
    # Keep one snapshot so saved inputs are exactly the inputs used for the call.
    config_text = config_path.read_text(encoding="utf-8")
    messages_text = messages_path.read_text(encoding="utf-8")
    config = effective_config(config_text)
    messages = parse_messages(messages_text)
    output.mkdir(parents=True, exist_ok=False)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"{branch}-devai-smoke") as run:
        run_id = str(run.info.run_id)
        mlflow.set_tags(
            {
                "stage": "baseline",
                "dataset": "DevAI",
                "model_family": "holistic-llm",
                "pr": config.pr,
                "issue": str(config.issue),
                "commit": commit,
                "source_commit": config.source_commit,
                "task": config.task,
                "mlflow.note.content": (
                    "Single static-text judgment of the published OpenHands workspace. "
                    "Human labels are excluded from the prompt. This run checks the "
                    "pipeline, not benchmark accuracy. Binary file contents are not supplied."
                ),
            }
        )
        mlflow.log_params(
            {
                "model": config.model,
                "api_base": config.api_base,
                "max_tokens": config.max_tokens,
                "timeout_seconds": config.timeout_seconds,
                "seed": settings.seed,
                "num_retries": 0,
                "expected_ids": json.dumps(config.expected_ids),
                "messages_sha256": sha256(messages_text.encode("utf-8")).hexdigest(),
            }
        )
        mlflow.log_text(config_text, "inputs/config.yaml")
        mlflow.log_text(messages_text, "inputs/messages.json")
        mlflow.log_dict(config.model_dump(), "inputs/effective-config.json")
        (output / "effective-config.json").write_text(
            config.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        provenance = {
            "run_id": run_id,
            "commit": commit,
            "seed": settings.seed,
            "source_commit": config.source_commit,
            "task": config.task,
            "messages_sha256": sha256(messages_text.encode("utf-8")).hexdigest(),
            "config_path": str(config_path),
            "model": config.model,
        }
        (output / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        mlflow.log_dict(provenance, "run.json")
        started = perf_counter()
        try:
            raw = request_once(config, messages, settings.seed)
        except Exception as exc:
            if isinstance(exc, ModelRequestError):
                diagnostic = exc.diagnostic
                mlflow.log_dict(diagnostic, "outputs/request-error.json")
                (output / "request-error.json").write_text(
                    json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                mlflow.set_tag("request_error_type", str(diagnostic["error_type"]))
            else:
                mlflow.set_tag("request_error_type", type(exc).__name__)
            raise RuntimeError(
                f"Model request failed; see MLflow run {run_id} and {output}"
            ) from None
        finally:
            mlflow.log_metric("request_seconds", perf_counter() - started)
        (output / "raw-response.txt").write_text(raw, encoding="utf-8")
        mlflow.log_text(raw, "outputs/raw-response.txt")
        try:
            judgment = parse_judgment(raw, config.expected_ids)
        except ValueError:
            mlflow.log_metric("response_valid", 0)
            raise ValueError(f"Invalid judgment JSON; raw response saved in run {run_id}") from None
        text = judgment.model_dump_json(indent=2) + "\n"
        (output / "verdicts.json").write_text(text, encoding="utf-8")
        mlflow.log_text(text, "outputs/verdicts.json")
        mlflow.log_metrics({"response_valid": 1, "verdict_count": len(judgment.verdicts)})
        return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = effective_config(args.config.read_text(encoding="utf-8"))
    messages = load_messages(args.messages)
    if not args.execute:
        print(f"DRY RUN: {len(messages)} messages; expected IDs: {config.expected_ids}")
        print(f"Model: {config.model}; endpoint: {config.api_base}")
        print("No key was read; no model call was made.")
        return
    if args.output is None:
        parser.error("--output is required with --execute")
    commit, branch = committed_version()
    run_id = run_tracked(
        args.config,
        args.messages,
        args.output,
        TrackingSettings(),
        commit=commit,
        branch=branch,
    )
    print(f"Response saved and validated. MLflow run: {run_id}")


if __name__ == "__main__":
    main()
