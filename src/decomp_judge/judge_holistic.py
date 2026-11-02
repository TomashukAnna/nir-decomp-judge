"""Evaluate the same application inputs in one LiteLLM request, with shared validation."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from decomp_judge.devai_smoke import Credentials, TrackingSettings, committed_version
from decomp_judge.judge_agent import snapshot
from decomp_judge.judge_validator import (
    Requirement,
    load_requirements,
    normalize_response,
    validate_judgment,
)
from decomp_judge.opencode_smoke import redact


class HolisticSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HOLISTIC_", env_file=None, extra="ignore")
    max_tokens: int = Field(default=16384, gt=0)
    timeout_seconds: float = Field(default=600, gt=0)
    max_prompt_bytes: int = Field(default=1_000_000, gt=0)
    dataset: str = "application"
    task: str = "unspecified"
    pr: str = "not-opened"


def build_messages(
    spec: Path,
    requirements: Path,
    app: Path,
    instruction: Path,
    policy: Path,
) -> tuple[list[dict[str, str]], dict[str, str], list[Requirement]]:
    text = spec.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Specification is empty")
    req = load_requirements(requirements)
    before = snapshot(app)
    shared = policy.read_text(encoding="utf-8")
    if "# Оценка\n" not in shared or "# Ответ\n" not in shared:
        raise ValueError("Shared verdict policy and response schema are missing")
    shared = "# Оценка\n" + shared.split("# Оценка\n", 1)[1]
    # Iteration advice belongs only to the agent; verdict/evidence rules stay identical.
    shared = "\n".join(
        line for line in shared.splitlines() if not line.startswith("Заверши работу итоговым JSON")
    )
    system = instruction.read_text(encoding="utf-8") + "\n\n" + shared
    parts = [
        "SPECIFICATION\n" + text,
        "REQUIREMENTS\n" + json.dumps([r.model_dump() for r in req], ensure_ascii=False),
        "FILE TREE\n" + "\n".join(before),
    ]
    for name in before:
        path = app / name
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            parts.append(f"UNREADABLE BINARY FILE {name}; content not supplied")
            continue
        numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(content.splitlines(), 1))
        parts.append(f"BEGIN FILE {name}\n{numbered}\nEND FILE {name}")
    if snapshot(app) != before:
        raise ValueError("Application changed while building prompt")
    return (
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        before,
        req,
    )


def call_model(
    config: dict[str, Any], messages: list[dict[str, str]], key: str, settings: HolisticSettings
) -> tuple[str, dict[str, Any]]:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    litellm.telemetry = False
    provider, model = config["model"].split("/", 1)
    response = litellm.completion(
        model=model,
        custom_llm_provider="openai",
        api_base=config["provider"][provider]["options"]["baseURL"],
        api_key=key,
        messages=messages,
        temperature=0,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_seconds,
        num_retries=0,
        caching=False,
    )
    usage = getattr(response, "usage", None)
    counters = {
        name: value
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if type(value := getattr(usage, name, None)) is int
    }
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None)
    if type(reasoning) is int:
        counters["reasoning_tokens"] = reasoning
    choice = response.choices[0]
    return choice.message.content or "", {
        "tokens": counters,
        "finish_reason": getattr(choice, "finish_reason", None),
    }


def run(
    config_path: Path,
    instruction: Path,
    policy: Path,
    spec: Path,
    requirements: Path,
    app: Path,
    output: Path,
    settings: HolisticSettings,
    tracking: TrackingSettings,
) -> str:
    messages, before, requirement_snapshot = build_messages(
        spec, requirements, app, instruction, policy
    )
    messages_text = json.dumps(messages, ensure_ascii=False, indent=2)
    if len(messages_text.encode("utf-8")) > settings.max_prompt_bytes:
        raise ValueError("Prompt exceeds configured limit; no files were silently truncated")
    if output.resolve().is_relative_to(app.resolve()):
        raise ValueError("Output must be outside application")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provider = config["model"].split("/", 1)[0]
    if config["provider"][provider]["options"]["apiKey"] != "{env:LITELLM_API_KEY}":
        raise ValueError("Key must come from environment")
    commit, branch = committed_version()
    credential = Credentials().litellm_api_key
    if credential is None or not credential.get_secret_value().strip():
        raise ValueError("Set LITELLM_API_KEY")
    key = credential.get_secret_value()
    output.mkdir(parents=True, exist_ok=False)
    mlflow.set_tracking_uri(tracking.mlflow_tracking_uri)
    mlflow.set_experiment(tracking.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"{branch}-holistic-judge") as active:
        run_id = str(active.info.run_id)

        def save(name: str, text: str) -> None:
            safe = redact(text, key)
            (output / name).write_text(safe, encoding="utf-8")
            mlflow.log_text(safe, name)

        mlflow.set_tags(
            {
                "stage": "baseline",
                "dataset": settings.dataset,
                "model_family": "holistic-llm",
                "pr": settings.pr,
                "task": settings.task,
                "commit": commit,
                "input_mode": "specification_and_requirements",
                "mlflow.note.content": "One static-text request; no tools or execution. "
                "Same verdict policy and validator as agent; seed recorded but not applied.",
            }
        )
        mlflow.log_params(
            {
                "model": config["model"],
                "temperature": 0,
                "max_tokens": settings.max_tokens,
                "timeout_seconds": settings.timeout_seconds,
                "seed": tracking.seed,
                "seed_applied_to_model": False,
                "num_retries": 0,
            }
        )
        provenance = {
            "run_id": run_id,
            "commit": commit,
            "config_path": str(config_path),
            "model": config["model"],
            "temperature": 0,
            "seed": tracking.seed,
            "seed_applied_to_model": False,
            "specification_available": True,
            "task": settings.task,
            "dataset": settings.dataset,
            "max_tokens": settings.max_tokens,
            "input_mode": "specification_and_requirements",
        }
        save("run.json", json.dumps(provenance, indent=2))
        save("config.json", json.dumps(config, ensure_ascii=False, indent=2))
        save("messages.json", messages_text)
        save("manifest.json", json.dumps(before, indent=2))
        valid = False
        started = perf_counter()
        try:
            raw, usage = call_model(config, messages, key, settings)
            save("raw-response.txt", raw)
            save("tokens.json", json.dumps(usage, indent=2))
            mlflow.log_metrics(usage["tokens"])
            normalized, fence, prose = normalize_response(raw)
            save("normalized-response.txt", normalized)
            try:
                json.loads(raw)
                raw_json_valid = True
            except ValueError:
                raw_json_valid = False
            formatting = {
                "markdown_fence_removed": fence,
                "surrounding_text_removed": prose,
                "raw_json_valid": raw_json_valid,
                "format_violation": not raw_json_valid,
            }
            save("format.json", json.dumps(formatting))
            mlflow.log_metrics({k: int(v) for k, v in formatting.items()})
            result = validate_judgment(normalized, requirement_snapshot, app)
            save("verdicts.json", result.model_dump_json(indent=2))
            mlflow.log_metric("verdict_count", len(result.verdicts))
            valid = True
        except Exception as exc:
            save("error.json", json.dumps({"error_type": type(exc).__name__, "detail": str(exc)}))
            raise RuntimeError(f"Holistic judge failed; run {run_id}; see {output}") from None
        finally:
            seconds = perf_counter() - started
            try:
                unchanged = snapshot(app) == before
            except (OSError, ValueError):
                unchanged = False
            save(
                "timing.json",
                json.dumps(
                    {
                        "session_seconds": seconds,
                        "workspace_unchanged": unchanged,
                        "response_valid": valid,
                    }
                ),
            )
            mlflow.log_metrics(
                {
                    "session_seconds": seconds,
                    "workspace_unchanged": int(unchanged),
                    "response_valid": int(valid),
                }
            )
            if not unchanged:
                raise RuntimeError(f"Application changed; run {run_id}")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("opencode.json"))
    parser.add_argument("--instruction", type=Path, default=Path("configs/judge-holistic.md"))
    parser.add_argument("--policy", type=Path, default=Path("configs/judge-agent.md"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        messages, files, _ = build_messages(
            args.spec, args.requirements, args.app, args.instruction, args.policy
        )
        print(f"DRY RUN: {len(files)} files, {len(messages)} messages; no key or model call.")
        return
    if args.output is None:
        parser.error("--output required with --execute")
    run_id = run(
        args.config,
        args.instruction,
        args.policy,
        args.spec,
        args.requirements,
        args.app,
        args.output,
        HolisticSettings(),
        TrackingSettings(),
    )
    print(f"Judgment validated. MLflow run: {run_id}; output: {args.output}")


if __name__ == "__main__":
    main()
