"""Download pinned DevAI data and assemble a static-text judge prompt."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt

SOURCE_URL = "https://github.com/metauto-ai/agent-as-a-judge.git"
TEXT_SUFFIXES = {".py", ".md", ".txt"}


class PreparationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    task: str = Field(pattern=r"^[A-Za-z0-9_]+$")
    expected_ids: list[StrictInt] = Field(min_length=1)


class Requirement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    requirement_id: StrictInt = Field(ge=0)
    prerequisites: list[StrictInt]
    criteria: str = Field(min_length=1)


class Task(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    query: str = Field(min_length=1)
    requirements: list[Requirement] = Field(min_length=1)


def git(source: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True).stdout


def ensure_source(source: Path, commit: str, download: bool) -> None:
    if not source.exists():
        if not download:
            raise FileNotFoundError("Source does not exist; use --download to clone it")
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", SOURCE_URL, str(source)], check=True)
    # Reject a non-repository subdirectory that git would silently resolve to its parent.
    root = Path(git(source, "rev-parse", "--show-toplevel").decode("utf-8").strip())
    if root.resolve() != source.resolve():
        raise ValueError("--source must point to the root of the downloaded repository")
    git(source, "cat-file", "-e", f"{commit}^{{commit}}")


def workspace_files(source: Path, commit: str, prefix: str) -> dict[str, tuple[str, int]]:
    raw = git(source, "ls-tree", "-r", "-l", "-z", commit, "--", prefix + "/")
    files: dict[str, tuple[str, int]] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, object_id, size = metadata.decode("ascii").split()
        relative = PurePosixPath(raw_path.decode("utf-8")).relative_to(prefix)
        if "judge" in relative.parts:
            continue
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError(f"Unsupported project entry: {relative}")
        files[relative.as_posix()] = (object_id, int(size))
    if not files:
        raise ValueError("No published workspace found for the selected task")
    return dict(sorted(files.items()))


def assemble_prompt(
    source: Path, config: PreparationConfig, system: str
) -> tuple[list[dict[str, str]], dict[str, object]]:
    task_path = f"benchmark/devai/instances/{config.task}.json"
    task = Task.model_validate_json(git(source, "show", f"{config.source_commit}:{task_path}"))
    ids = [r.requirement_id for r in task.requirements]
    if (
        task.name != config.task
        or len(set(ids)) != len(ids)
        or len(set(config.expected_ids)) != len(config.expected_ids)
        or set(ids) != set(config.expected_ids)
    ):
        raise ValueError("Task name or requirement IDs do not match the configuration")
    if any(not set(r.prerequisites).issubset(ids) for r in task.requirements):
        raise ValueError("Requirement prerequisites reference unknown IDs")
    if not system.strip():
        raise ValueError("System prompt must not be empty")
    prefix = f"benchmark/workspaces/OpenHands/{config.task}"
    files = workspace_files(source, config.source_commit, prefix)
    included = [name for name in files if PurePosixPath(name).suffix in TEXT_SUFFIXES]
    requirements = [r.model_dump() for r in task.requirements]
    parts = [
        "TASK\n" + task.query,
        "REQUIREMENTS\n" + json.dumps(requirements, ensure_ascii=False, indent=2),
        "EVIDENCE SCOPE\nThis is a static, text-only snapshot. No program was executed. "
        "The judge/ directory is excluded because it contains evaluation artifacts. "
        "CSV data, PDF and PNG contents are not supplied; their paths and byte sizes "
        "are listed. Text files below are complete.",
        "PROJECT FILE TREE\n"
        + "\n".join(f"{name} ({size} bytes)" for name, (_, size) in files.items()),
    ]
    for name in included:
        # Git blobs avoid checkout-specific CRLF conversions and changes in local files.
        content = git(source, "cat-file", "blob", files[name][0]).decode("utf-8")
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        parts.append(f"BEGIN FILE {name}\n{content}\nEND FILE {name}")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    manifest: dict[str, object] = {
        "source_url": SOURCE_URL,
        "source_commit": config.source_commit,
        "task": config.task,
        "requirement_ids": ids,
        "included_text_files": included,
        "listed_only_files": [name for name in files if name not in included],
        "excluded_directory": "judge",
        "file_git_objects": {name: object_id for name, (object_id, _) in files.items()},
        "human_labels_included": False,
        "model_called": False,
    }
    return messages, manifest


def prepare(
    config_path: Path, system_path: Path, source: Path, output: Path, *, download: bool
) -> str:
    if output.exists():
        raise FileExistsError("Choose a fresh output directory to preserve earlier inputs")
    config = PreparationConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    system = system_path.read_text(encoding="utf-8").rstrip("\r\n")
    ensure_source(source, config.source_commit, download)
    messages, manifest = assemble_prompt(source, config, system)
    messages_text = json.dumps(messages, ensure_ascii=False, indent=2) + "\n"
    digest = sha256(messages_text.encode("utf-8")).hexdigest()
    manifest.update(messages_sha256=digest, prepared_at_utc=datetime.now(UTC).isoformat())
    output.mkdir(parents=True, exist_ok=False)
    # Explicit LF keeps the messages hash identical on Windows and Linux.
    (output / "messages.json").write_text(messages_text, encoding="utf-8", newline="\n")
    (output / "prompt.txt").write_text(
        "SYSTEM\n" + system + "\n\nUSER\n" + messages[1]["content"] + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/devai-smoke.yaml"))
    parser.add_argument("--system-prompt", type=Path, default=Path("configs/devai-system.txt"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    digest = prepare(
        args.config, args.system_prompt, args.source, args.output, download=args.download
    )
    print(f"Prepared: {args.output}; messages SHA-256: {digest}")
    print("No credentials were read and no model was called.")


if __name__ == "__main__":
    main()
