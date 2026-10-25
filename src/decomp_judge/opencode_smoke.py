"""Run one tracked OpenCode session and verify a completed file-reading tool event."""

import argparse
import json
import os
import re
import subprocess
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from decomp_judge.devai_smoke import Credentials, TrackingSettings, committed_version


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=None, extra="ignore")
    timeout_seconds: float = Field(default=600, gt=0)
    expected_version: str = "1.19.4"


def redact(text: str, key: str) -> str:
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"(?i)Bearer\s+[^\s'\"<>]+", "Bearer [REDACTED]", text)


def inspect_events(text: str, workspace: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("OpenCode events must be JSON objects")
    reads: list[dict[str, Any]] = []
    replies: list[str] = []
    target = (workspace / "src/model.py").resolve()
    for event in events:
        part = event.get("part", {})
        if event.get("type") == "text" and isinstance(part.get("text"), str):
            replies.append(part["text"])
        if event.get("type") != "tool_use" or part.get("tool") != "read":
            continue
        state = part.get("state", {})
        if state.get("status") != "completed":
            continue
        raw_path = state.get("input", {}).get("filePath")
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path)
        path = path if path.is_absolute() else workspace / path
        if path.resolve() == target and isinstance(state.get("output"), str):
            reads.append({"path": "src/model.py", "output_chars": len(state["output"])})
    return {
        "event_count": len(events),
        "target_read_verified": any(r["output_chars"] > 0 for r in reads),
        "completed_reads": reads,
        "final_response": "\n".join(replies),
        "errors": [e for e in events if e.get("type") == "error"],
    }


def run(
    executable: Path,
    config_path: Path,
    prompt_path: Path,
    prepared: Path,
    output: Path,
    tracking: TrackingSettings,
    runtime: RuntimeSettings,
) -> str:
    commit, branch = committed_version()
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    system_path = config_path.parent / "opencode-judge-system.txt"
    system = system_path.read_text(encoding="utf-8").strip()
    config["agent"]["requirements-judge"]["prompt"] = system
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    workspace = (prepared / "workspace").resolve()
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))

    def intact() -> bool:
        actual = {
            p.relative_to(workspace).as_posix()
            for p in workspace.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(workspace).parts
        }
        return actual == set(manifest["file_sha256"]) and all(
            (workspace / name).is_file()
            and sha256((workspace / name).read_bytes()).hexdigest() == digest
            for name, digest in manifest["file_sha256"].items()
        )

    if not intact():
        raise ValueError("Workspace differs from the pinned preparation manifest")
    credentials = Credentials()
    if credentials.litellm_api_key is None:
        raise ValueError("Set LITELLM_API_KEY before execution")
    key = credentials.litellm_api_key.get_secret_value()
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENCODE_")}
    for name in ("CONFIG", "DATA", "CACHE", "STATE"):
        env[f"XDG_{name}_HOME"] = str((output / "runtime" / name.lower()).resolve())
    env.update(
        LITELLM_API_KEY=key,
        OPENCODE_CONFIG_CONTENT=json.dumps(config),
        OPENCODE_DISABLE_PROJECT_CONFIG="1",
        OPENCODE_DISABLE_AUTOUPDATE="1",
        OPENCODE_DISABLE_MODELS_FETCH="1",
        OPENCODE_DISABLE_CLAUDE_CODE="1",
        OPENCODE_DISABLE_EXTERNAL_SKILLS="1",
        OPENCODE_DISABLE_DEFAULT_PLUGINS="1",
        OPENCODE_DISABLE_LSP_DOWNLOAD="1",
    )
    executable = executable.resolve()
    version = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=30,
    ).stdout.strip()
    if version != runtime.expected_version:
        raise ValueError(f"Expected OpenCode {runtime.expected_version}, got {version}")
    mlflow.set_tracking_uri(tracking.mlflow_tracking_uri)
    mlflow.set_experiment(tracking.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"{branch}-opencode-file-smoke") as active:
        run_id = str(active.info.run_id)
        mlflow.set_tags(
            {
                "stage": "baseline",
                "dataset": "DevAI",
                "model_family": "tool-agent-smoke",
                "pr": "not-opened",
                "commit": commit,
                "source_commit": manifest["source_commit"],
                "task": manifest["task"],
                "mlflow.note.content": (
                    "Technical file-reading session, not requirement evaluation. "
                    "No human labels. Seed recorded but not forced through the OpenCode adapter."
                ),
            }
        )
        mlflow.log_params(
            {
                "seed": tracking.seed,
                "seed_applied_to_model": False,
                "model": config["model"],
                "opencode_version": version,
                "timeout_seconds": runtime.timeout_seconds,
                "prompt_sha256": sha256(prompt.encode()).hexdigest(),
            }
        )
        provenance = {
            "run_id": run_id,
            "commit": commit,
            "source_commit": manifest["source_commit"],
            "seed": tracking.seed,
            "seed_applied_to_model": False,
            "opencode_version": version,
            "config_path": str(config_path),
            "model": config["model"],
            "task": manifest["task"],
        }

        def save(name: str, text: str) -> None:
            safe = redact(text, key)
            (output / name).write_text(safe, encoding="utf-8")
            mlflow.log_text(safe, name)

        save("run.json", json.dumps(provenance, indent=2))
        save("config.json", config_text)
        save("system.txt", system)
        save("task.txt", prompt)
        save("manifest.json", json.dumps(manifest, indent=2))
        started = perf_counter()
        command = [
            str(executable),
            "--pure",
            "run",
            "--format",
            "json",
            "--agent",
            "requirements-judge",
            "--dir",
            str(workspace),
            prompt,
        ]
        try:
            process = subprocess.run(
                command, capture_output=True, env=env, timeout=runtime.timeout_seconds
            )
            stdout = process.stdout.decode("utf-8", errors="replace")
            stderr = process.stderr.decode("utf-8", errors="replace")
            save("events.jsonl", stdout)
            save("stderr.txt", stderr)
            summary = inspect_events(stdout, workspace)
            summary["return_code"] = process.returncode
            summary["workspace_unchanged"] = intact()
            save("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
            save("final-response.txt", summary["final_response"])
            mlflow.log_metrics(
                {
                    "target_read_verified": int(summary["target_read_verified"]),
                    "event_count": summary["event_count"],
                    "workspace_unchanged": int(summary["workspace_unchanged"]),
                }
            )
            if (
                process.returncode
                or summary["errors"]
                or not summary["target_read_verified"]
                or not summary["final_response"].strip()
                or not summary["workspace_unchanged"]
            ):
                raise ValueError("File-reading smoke test failed; inspect saved events and summary")
        except subprocess.TimeoutExpired as exc:
            save("events.jsonl", (exc.stdout or b"").decode("utf-8", errors="replace"))
            save("stderr.txt", (exc.stderr or b"").decode("utf-8", errors="replace"))
            save("error.json", json.dumps({"type": "TimeoutExpired"}))
            raise RuntimeError(f"OpenCode timed out; MLflow run {run_id}") from None
        except Exception as exc:
            save("error.json", json.dumps({"type": type(exc).__name__, "detail": str(exc)}))
            raise RuntimeError(f"OpenCode failed; see {output}; MLflow run {run_id}") from None
        finally:
            mlflow.log_metric("session_seconds", perf_counter() - started)
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/opencode-judge.json"))
    parser.add_argument("--prompt", type=Path, default=Path("configs/opencode-file-smoke.txt"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("DRY RUN: no key read, no OpenCode session and no model call.")
        return
    run_id = run(
        args.executable,
        args.config,
        args.prompt,
        args.prepared,
        args.output,
        TrackingSettings(),
        RuntimeSettings(),
    )
    print(f"File read verified. MLflow run: {run_id}; output: {args.output}")


if __name__ == "__main__":
    main()
