"""Run a read-only OpenCode requirements judge with tracked inputs and evidence."""

import argparse
import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from decomp_judge.devai_smoke import Credentials, TrackingSettings, committed_version
from decomp_judge.judge_validator import load_requirements, normalize_response, validate_judgment
from decomp_judge.opencode_smoke import redact


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JUDGE_", env_file=None, extra="ignore")
    max_steps: int = Field(default=8, gt=0)
    timeout_seconds: float = Field(default=600, gt=0)
    expected_version: str = "1.19.4"
    dataset: str = "application"
    task: str = "unspecified"
    pr: str = "not-opened"


def snapshot(app: Path) -> dict[str, str]:
    if not app.is_dir():
        raise ValueError("Application directory does not exist")
    files = {}
    for path in sorted(app.rglob("*")):
        relative = path.relative_to(app)
        if ".git" in relative.parts:
            continue
        if path.name == ".env" or path.name.startswith(".env."):
            continue
        if path.is_symlink():
            raise ValueError("Application symlinks are not supported")
        if path.is_file():
            files[relative.as_posix()] = sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError("Application directory is empty")
    return files


def collect_events(raw: str) -> dict[str, Any]:
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    tools: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    texts: list[str] = []
    errors: list[dict[str, Any]] = []
    counters = ("input", "output", "reasoning", "total")
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Event must be an object")
        part = event.get("part", {})
        if not isinstance(part, dict):
            raise ValueError("Event part must be an object")
        kind = event.get("type")
        if kind == "step_start":
            texts = []
        elif kind == "text":
            if not isinstance(part.get("text"), str):
                raise ValueError("Text event has no text")
            texts.append(part["text"])
        elif kind == "tool_use":
            tools.append(part)
        elif kind == "step_finish":
            tokens = part.get("tokens")
            if isinstance(tokens, dict):
                usage.append(tokens)
        elif kind == "error":
            errors.append(event)
    totals = {
        name: sum(step[name] for step in usage)
        for name in counters
        if usage and all(type(step.get(name)) is int and step[name] >= 0 for step in usage)
    }
    return {
        "event_count": len(events),
        "tools": tools,
        "errors": errors,
        "raw_response": "".join(texts),
        "tokens": {
            "steps": usage,
            "totals": totals,
            "scope": "Reported step_finish events only; not total API billing",
        },
    }


def run(
    executable: Path,
    config_path: Path,
    instruction_path: Path,
    spec_path: Path,
    requirements_path: Path,
    app: Path,
    output: Path,
    settings: AgentSettings,
    tracking: TrackingSettings,
) -> str:
    requirements = load_requirements(requirements_path)
    spec = spec_path.read_text(encoding="utf-8")
    if not spec.strip():
        raise ValueError("Specification must not be empty")
    app = app.resolve()
    before = snapshot(app)
    if output.resolve().is_relative_to(app):
        raise ValueError("Output directory must be outside the application")
    instruction = instruction_path.read_text(encoding="utf-8")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["default_agent"] = "requirements-judge"
    config["permission"] = {
        "*": "deny",
        "read": {"*": "allow", "*.env": "deny", "*.env.*": "deny"},
        "glob": "allow",
        "grep": "allow",
        "external_directory": "deny",
    }
    config["agent"] = {
        "requirements-judge": {
            "mode": "primary",
            "temperature": 0,
            "steps": settings.max_steps,
            "prompt": instruction,
        }
    }
    if config["provider"]["lab"]["options"]["apiKey"] != "{env:LITELLM_API_KEY}":
        raise ValueError("Provider key must reference LITELLM_API_KEY")
    prompt = (
        "SPECIFICATION\n"
        + spec
        + "\n\nREQUIREMENTS\n"
        + json.dumps([r.model_dump() for r in requirements], ensure_ascii=False)
        + "\n\nInspect the current application directory using file tools. "
        "Return the judgment JSON required by your instruction."
    )
    commit, branch = committed_version()
    credential = Credentials().litellm_api_key
    if credential is None or not credential.get_secret_value().strip():
        raise ValueError("Set LITELLM_API_KEY before execution")
    key = credential.get_secret_value()
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENCODE_")}
    env.update(LITELLM_API_KEY=key, OPENCODE_CONFIG_CONTENT=json.dumps(config))
    for name in ("CONFIG", "DATA", "CACHE", "STATE"):
        env[f"XDG_{name}_HOME"] = str((output / "runtime" / name.lower()).resolve())
    for flag in (
        "PROJECT_CONFIG",
        "AUTOUPDATE",
        "MODELS_FETCH",
        "CLAUDE_CODE",
        "EXTERNAL_SKILLS",
        "DEFAULT_PLUGINS",
        "LSP_DOWNLOAD",
    ):
        env[f"OPENCODE_DISABLE_{flag}"] = "1"
    executable = executable.resolve()
    mlflow.set_tracking_uri(tracking.mlflow_tracking_uri)
    mlflow.set_experiment(tracking.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"{branch}-agent-judge") as active:
        run_id = str(active.info.run_id)

        def save(name: str, value: str) -> None:
            safe = redact(value, key)
            (output / name).write_text(safe, encoding="utf-8")
            mlflow.log_text(safe, name)

        mlflow.set_tags(
            {
                "stage": "baseline",
                "dataset": settings.dataset,
                "model_family": "opencode-agent",
                "pr": settings.pr,
                "commit": commit,
                "task": settings.task,
                "mlflow.note.content": "Static read-only judge. Seed recorded, not forced "
                "through the model adapter. Temperature zero does not ensure determinism.",
            }
        )
        mlflow.log_params(
            {
                "model": config["model"],
                "temperature": 0,
                "max_steps": settings.max_steps,
                "seed": tracking.seed,
                "seed_applied_to_model": False,
                "timeout_seconds": settings.timeout_seconds,
                "config_path": str(config_path),
            }
        )
        provenance = {
            "run_id": run_id,
            "commit": commit,
            "config_path": str(config_path),
            "seed": tracking.seed,
            "seed_applied_to_model": False,
            "task": settings.task,
            "dataset": settings.dataset,
            "model": config["model"],
            "temperature": 0,
            "max_steps": settings.max_steps,
        }
        save("run.json", json.dumps(provenance, indent=2))
        save("config.json", json.dumps(config, ensure_ascii=False, indent=2))
        save("prompt.txt", prompt)
        save("manifest.json", json.dumps(before, indent=2))
        save("raw-output.jsonl", "")
        save("stderr.txt", "")
        save("raw-response.txt", "")
        save("trace.json", "[]")
        save("tokens.json", json.dumps({"steps": [], "totals": {}, "available": False}))
        started = perf_counter()
        valid = False
        try:
            version = (
                subprocess.run(
                    [str(executable), "--version"],
                    env=env,
                    capture_output=True,
                    check=True,
                    timeout=30,
                    stdin=subprocess.DEVNULL,
                )
                .stdout.decode()
                .strip()
            )
            mlflow.log_param("opencode_version", version)
            if version != settings.expected_version:
                raise ValueError(f"Expected OpenCode {settings.expected_version}, got {version}")
            process = subprocess.run(
                [
                    str(executable),
                    "--pure",
                    "run",
                    "--format",
                    "json",
                    "--agent",
                    "requirements-judge",
                    "--dir",
                    str(app),
                    prompt,
                ],
                cwd=app,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=settings.timeout_seconds,
            )
            stdout = process.stdout.decode("utf-8", errors="replace")
            save("raw-output.jsonl", stdout)
            save("stderr.txt", process.stderr.decode("utf-8", errors="replace"))
            collected = collect_events(stdout)
            save("trace.json", json.dumps(collected["tools"], ensure_ascii=False, indent=2))
            save("raw-response.txt", collected["raw_response"])
            normalized, fence_removed = normalize_response(collected["raw_response"])
            save("normalized-response.txt", normalized)
            try:
                json.loads(collected["raw_response"])
                raw_json_valid = True
            except ValueError:
                raw_json_valid = False
            save(
                "format.json",
                json.dumps(
                    {
                        "markdown_fence_removed": fence_removed,
                        "raw_json_valid": raw_json_valid,
                        "format_violation": not raw_json_valid,
                    }
                ),
            )
            mlflow.log_metrics(
                {
                    "markdown_fence_removed": int(fence_removed),
                    "raw_json_valid": int(raw_json_valid),
                    "format_violation": int(not raw_json_valid),
                }
            )
            save("tokens.json", json.dumps(collected["tokens"], indent=2))
            mlflow.log_metrics({f"tokens_{k}": v for k, v in collected["tokens"]["totals"].items()})
            if process.returncode or collected["errors"]:
                raise ValueError("OpenCode returned an error; inspect raw output")
            completed = [
                t
                for t in collected["tools"]
                if t.get("tool") in {"read", "glob", "grep"}
                and t.get("state", {}).get("status") == "completed"
            ]
            if not completed:
                raise ValueError("No completed file-reading or search tool calls")
            result = validate_judgment(normalized, requirements, app)
            save("verdicts.json", result.model_dump_json(indent=2))
            mlflow.log_metric("verdict_count", len(result.verdicts))
            valid = True
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or b"").decode("utf-8", errors="replace")
            save("raw-output.jsonl", partial)
            save("stderr.txt", (exc.stderr or b"").decode("utf-8", errors="replace"))
            try:
                partial_events = collect_events(partial)
            except (ValueError, TypeError):
                pass
            else:
                save("trace.json", json.dumps(partial_events["tools"], ensure_ascii=False))
                save("tokens.json", json.dumps(partial_events["tokens"]))
                save("raw-response.txt", partial_events["raw_response"])
            save("error.json", json.dumps({"error_type": "TimeoutExpired"}))
            raise RuntimeError(f"Agent timed out; run {run_id}, output {output}") from None
        except Exception as exc:
            save("error.json", json.dumps({"error_type": type(exc).__name__, "detail": str(exc)}))
            raise RuntimeError(f"Agent failed; run {run_id}, output {output}") from None
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
                save("error.json", json.dumps({"error_type": "ApplicationChanged"}))
                raise RuntimeError(f"Application changed; run {run_id}")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--config", type=Path, default=Path("opencode.json"))
    parser.add_argument("--instruction", type=Path, default=Path("configs/judge-agent.md"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    requirements = load_requirements(args.requirements)
    if not args.spec.read_text(encoding="utf-8").strip():
        parser.error("Specification is empty")
    files = snapshot(args.app)
    if not args.execute:
        print(
            f"DRY RUN: {len(requirements)} requirements, {len(files)} files; no key or model call."
        )
        return
    if args.executable is None or args.output is None:
        parser.error("--executable and --output are required with --execute")
    settings = AgentSettings()
    if args.max_steps is not None:
        settings = AgentSettings.model_validate(
            {**settings.model_dump(), "max_steps": args.max_steps}
        )
    run_id = run(
        args.executable,
        args.config,
        args.instruction,
        args.spec,
        args.requirements,
        args.app,
        args.output,
        settings,
        TrackingSettings(),
    )
    print(f"Judgment validated. MLflow run: {run_id}; output: {args.output}")


if __name__ == "__main__":
    main()
