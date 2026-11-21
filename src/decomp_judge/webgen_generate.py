"""Generate one application per WebGen specification with a resumable OpenCode batch."""

import argparse
import csv
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import mlflow
import yaml
from pydantic import BaseModel, ConfigDict, Field

from decomp_judge.devai_smoke import Credentials, TrackingSettings, committed_version
from decomp_judge.judge_agent import collect_events
from decomp_judge.opencode_smoke import redact


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    api_base: str
    opencode_version: str
    temperature: int = Field(default=0, ge=0, le=0)
    max_steps: int = Field(default=50, ge=1, le=200)
    timeout_seconds: int = Field(default=1200, ge=1, le=7200)
    max_consecutive_failures: int = Field(default=3, ge=1, le=10)
    instruction: str


class GenerationTask(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    id: str = Field(pattern=r"^\d{6}$")
    instruction: str = Field(min_length=1)


def parse_tasks(raw: bytes) -> list[GenerationTask]:
    tasks = [
        GenerationTask.model_validate_json(line)
        for line in raw.decode("utf-8-sig").splitlines()
        if line.strip()
    ]
    if not tasks or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Dataset must contain tasks with unique IDs")
    if any(not task.instruction.strip() for task in tasks):
        raise ValueError("Specification cannot be blank")
    return tasks


def load_tasks(path: Path) -> list[GenerationTask]:
    return parse_tasks(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def batch_lock(path: Path) -> Iterator[None]:
    with path.open("a+b") as stream:
        if sys.platform == "win32":
            import msvcrt

            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if sys.platform == "win32":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def source_version() -> str:
    baked = Path(__file__).resolve().parents[2] / "source-commit.txt"
    version = baked.read_text().strip() if baked.exists() else committed_version()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", version):
        raise ValueError("A committed source version is required")
    return version


def build_config(settings: GenerationConfig, instruction: str) -> dict[str, Any]:
    return {
        "model": f"lab/{settings.model}",
        "small_model": f"lab/{settings.model}",
        "enabled_providers": ["lab"],
        "default_agent": "webgen-builder",
        "autoupdate": False,
        "snapshot": False,
        "provider": {
            "lab": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": settings.api_base, "apiKey": "{env:LITELLM_API_KEY}"},
                "models": {settings.model: {"name": settings.model}},
            }
        },
        "permission": {
            "*": "deny",
            "read": {"*": "allow", "*.env": "deny", "*.env.*": "deny"},
            "edit": "allow",
            "glob": "allow",
            "grep": "allow",
            "bash": "allow",
            "external_directory": "deny",
        },
        "agent": {
            "webgen-builder": {
                "mode": "primary",
                "temperature": settings.temperature,
                "steps": settings.max_steps,
                "prompt": instruction,
            }
        },
    }


def app_files(app: Path) -> dict[str, str]:
    files = {}
    excluded = {"node_modules", ".venv", ".git", "__pycache__"}
    for path in sorted(app.rglob("*")):
        relative = path.relative_to(app)
        if excluded.intersection(relative.parts):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(app.resolve()):
            raise ValueError("Generated applications must not contain symlinks")
        if path.is_file():
            if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
                raise ValueError("Generated credential files must not be included")
            files[relative.as_posix()] = sha256(path.read_bytes()).hexdigest()
    return files


def clean_dependencies(app: Path) -> None:
    # Keep source and lock files; installed dependencies can be recreated and are large.
    for path in sorted(app.rglob("*"), key=lambda p: len(p.parts)):
        if path.name not in {"node_modules", ".venv", ".git", "__pycache__"}:
            continue
        if not path.parent.resolve().is_relative_to(app.resolve()):
            raise ValueError("Dependency path escapes the application")
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def run_process(
    command: list[str], app: Path, env: dict[str, str], timeout: int
) -> tuple[str, str, int]:
    if sys.platform == "win32":
        raise ValueError("Execute generation inside the Linux container")
    with subprocess.Popen(
        command,
        cwd=app,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return (
                stdout.decode("utf-8", "replace"),
                stderr.decode("utf-8", "replace"),
                process.returncode,
            )
        finally:
            # Stop shell/server children too, including those left after OpenCode exits.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def run_task(
    task: GenerationTask,
    index: int,
    output: Path,
    settings: GenerationConfig,
    config: dict[str, Any],
    commit: str,
    seed: int,
    key: str,
    *,
    dataset_sha256: str,
) -> dict[str, Any]:
    folder = output / f"task-{task.id}"
    folder.mkdir(exist_ok=False)
    app = folder / "app"
    app.mkdir()
    (folder / "spec.txt").write_text(task.instruction + "\n", encoding="utf-8")
    prompt = "Implement this specification in the current directory:\n\n" + task.instruction
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    write_json(folder / "config.json", config)
    state: dict[str, Any] = {
        "task_index": index,
        "webgen_id": task.id,
        "status": "RUNNING",
        "commit": commit,
        "config": "configs/webgen-generation.yaml",
        "seed": seed,
        "seed_applied_to_model": False,
        "model": settings.model,
        "app_path": f"task-{task.id}/app",
        "requirements_evaluated": False,
    }
    active = mlflow.start_run(run_name=f"webgen-generate-{task.id}")
    state["run_id"] = active.info.run_id
    write_json(folder / "run.json", state)
    mlflow.set_tags(
        {
            "stage": "baseline",
            "dataset": "WebGen-Bench",
            "model_family": "qwen3.8",
            "pr": "not-opened",
            "purpose": "application-generation",
        }
    )
    mlflow.log_params(
        {
            "commit": commit,
            "seed": seed,
            "task_index": index,
            "webgen_id": task.id,
            "model": settings.model,
            "temperature": settings.temperature,
            "max_steps": settings.max_steps,
            "timeout_seconds": settings.timeout_seconds,
            "opencode_version": settings.opencode_version,
            "dataset_sha256": dataset_sha256,
        }
    )
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("OPENCODE_", "GIT_"))
    }
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    env["LITELLM_API_KEY"] = key
    for name in ("CONFIG", "DATA", "CACHE", "STATE"):
        env[f"XDG_{name}_HOME"] = str(folder / "runtime" / name.lower())
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
    started = perf_counter()
    interrupted = False
    try:
        subprocess.run(
            ["git", "init", "--quiet", str(app)], check=True, capture_output=True, env=env
        )
        stdout, stderr, code = run_process(
            [
                "opencode",
                "--pure",
                "run",
                "--format",
                "json",
                "--agent",
                "webgen-builder",
                "--dir",
                str(app),
                prompt,
            ],
            app,
            env,
            settings.timeout_seconds,
        )
        (folder / "raw-output.jsonl").write_text(redact(stdout, key), encoding="utf-8")
        (folder / "stderr.txt").write_text(redact(stderr, key), encoding="utf-8")
        events = collect_events(stdout)
        write_json(folder / "tokens.json", events["tokens"])
        mlflow.log_metrics(
            {f"tokens_{name}": count for name, count in events["tokens"]["totals"].items()}
        )
        if code or events["errors"]:
            raise ValueError("OpenCode failed; see the saved response and stderr")
        clean_dependencies(app)
        files = app_files(app)
        if not files or set(files) <= {"README.md"}:
            raise ValueError("OpenCode did not create application source files")
        write_json(folder / "files-sha256.json", files)
        state.update(status="GENERATED", file_count=len(files))
    except KeyboardInterrupt:
        interrupted = True
        state.update(status="INTERRUPTED", error_type="KeyboardInterrupt")
    except Exception as exc:
        state.update(status="FAILED", error_type=type(exc).__name__, detail=redact(str(exc), key))
    finally:
        state["seconds"] = round(perf_counter() - started, 3)
        write_json(folder / "run.json", state)
        mlflow.log_metrics(
            {
                "seconds": state["seconds"],
                "generated": int(state["status"] == "GENERATED"),
                "source_files": state.get("file_count", 0),
            }
        )
        for name in ("run.json", "config.json", "prompt.txt", "tokens.json", "files-sha256.json"):
            if (folder / name).exists():
                mlflow.log_artifact(str(folder / name))
        if (output / "generation-config.yaml").exists():
            mlflow.log_artifact(str(output / "generation-config.yaml"))
        mlflow.end_run(status="FINISHED" if state["status"] == "GENERATED" else "FAILED")
    if interrupted:
        raise KeyboardInterrupt
    return state


def update_summary(output: Path, tasks: list[GenerationTask]) -> dict[str, int]:
    rows = []
    for index, task in enumerate(tasks):
        record = output / f"task-{task.id}" / "run.json"
        state = json.loads(record.read_text()) if record.exists() else {"status": "PENDING"}
        rows.append(
            {
                "task_index": index,
                "webgen_id": task.id,
                **{
                    name: state.get(name, "")
                    for name in (
                        "status",
                        "run_id",
                        "file_count",
                        "seconds",
                        "app_path",
                        "error_type",
                    )
                },
            }
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary = output / "summary.csv.tmp"
    temporary.write_text(stream.getvalue(), encoding="utf-8-sig", newline="")
    temporary.replace(output / "summary.csv")
    counts = dict(Counter(str(row["status"]) for row in rows))
    write_json(output / "status.json", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/webgen-generation.yaml"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    raw_dataset = args.input.read_bytes()
    tasks = parse_tasks(raw_dataset)
    settings = GenerationConfig.model_validate(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    instruction = Path(settings.instruction).read_text(encoding="utf-8")
    if not args.execute:
        print(
            f"DRY RUN: {len(tasks)} specifications; model {settings.model}; no key or model call."
        )
        return
    if os.environ.get("GENERATION_CONTAINER") != "1":
        parser.error("Use the generation container")
    key = Credentials().litellm_api_key
    if key is None or not key.get_secret_value().strip():
        parser.error("Set LITELLM_API_KEY")
    version = subprocess.check_output(["opencode", "--version"], text=True).strip()
    if version != settings.opencode_version:
        parser.error(f"Unexpected OpenCode version: {version}")
    commit = source_version()
    tracking = TrackingSettings()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_digest = sha256(raw_dataset).hexdigest()
    manifest = {
        "commit": commit,
        "dataset_sha256": dataset_digest,
        "task_ids": [task.id for task in tasks],
        "seed": tracking.seed,
        "settings": settings.model_dump(),
        "instruction_sha256": sha256(instruction.encode()).hexdigest(),
    }
    with batch_lock(output / ".batch.lock"):
        target = output / "manifest.json"
        if target.exists():
            if not args.resume:
                parser.error("Batch exists; use --resume")
            if json.loads(target.read_text()) != manifest:
                parser.error("Batch inputs or settings changed")
        else:
            if any(
                path.name not in {"mlflow.db", "mlartifacts", ".batch.lock"}
                for path in output.iterdir()
            ):
                parser.error("Output must be a new batch directory")
            write_json(target, manifest)
            (output / "dataset-source.jsonl").write_bytes(raw_dataset)
            (output / "generation-config.yaml").write_text(
                args.config.read_text(encoding="utf-8"), encoding="utf-8"
            )
        mlflow.set_tracking_uri(f"sqlite:///{(output / 'mlflow.db').as_posix()}")
        client = mlflow.MlflowClient()
        experiment_name = "nir-decomp-judge/webgen-generation"
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(
                experiment_name, artifact_location=(output / "mlartifacts").as_uri()
            )
        )
        mlflow.set_experiment(experiment_id=experiment_id)
        update_summary(output, tasks)
        completed = failures = 0
        for index, task in enumerate(tasks):
            state_path = output / f"task-{task.id}" / "run.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state["status"] == "RUNNING":
                    state.update(status="INTERRUPTED", error_type="PreviousProcessStopped")
                    write_json(state_path, state)
                    mlflow.MlflowClient().set_terminated(state["run_id"], status="KILLED")
                if state["status"] == "GENERATED":
                    expected = json.loads((state_path.parent / "files-sha256.json").read_text())
                    if app_files(state_path.parent / "app") != expected:
                        raise ValueError(f"Generated source changed: {task.id}")
                continue
            if (output / f"task-{task.id}").exists():
                raise ValueError(f"Incomplete task directory needs inspection: {task.id}")
            if args.limit is not None and completed >= args.limit:
                break
            print(f"Generating {index + 1}/{len(tasks)}: {task.id}", flush=True)
            try:
                state = run_task(
                    task,
                    index,
                    output,
                    settings,
                    build_config(settings, instruction),
                    commit,
                    tracking.seed,
                    key.get_secret_value(),
                    dataset_sha256=dataset_digest,
                )
            finally:
                update_summary(output, tasks)
            completed += 1
            failures = failures + 1 if state["status"] != "GENERATED" else 0
            print(f"{task.id}: {state['status']}; MLflow {state['run_id']}", flush=True)
            if failures >= settings.max_consecutive_failures:
                print("Batch paused after consecutive failures; inspect records before resuming.")
                break
        counts = update_summary(output, tasks)
        print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
