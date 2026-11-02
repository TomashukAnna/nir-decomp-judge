"""Export pinned project files into a separate OpenCode inspection workspace."""

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import yaml

from decomp_judge.prepare_devai import (
    SOURCE_URL,
    PreparationConfig,
    ensure_source,
    git,
    workspace_files,
)


def prepare(config_path: Path, source: Path, output: Path, *, download: bool = False) -> Path:
    config = PreparationConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    ensure_source(source, config.source_commit, download)
    prefix = f"benchmark/workspaces/OpenHands/{config.task}"
    files = workspace_files(source, config.source_commit, prefix)
    # Materialize only pinned blobs, never mutable working-tree files or evaluation data.
    blobs = {name: git(source, "cat-file", "blob", oid) for name, (oid, _) in files.items()}
    for name in blobs:
        parts = Path(name).parts
        if any(part in {"..", ".git", ".opencode"} for part in parts) or Path(name).is_absolute():
            raise ValueError(f"Unsupported workspace path: {name}")
        if Path(name).name in {"AGENTS.md", "CLAUDE.md", "opencode.json", "opencode.jsonc"}:
            raise ValueError(f"Project instructions must not configure the judge: {name}")
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    for name, content in blobs.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # A local Git boundary prevents discovery of the research repository's instructions.
    subprocess.run(["git", "-C", str(workspace), "init", "--quiet"], check=True)
    manifest = {
        "source_url": SOURCE_URL,
        "source_commit": config.source_commit,
        "task": config.task,
        "framework": "OpenHands",
        "excluded_directory": "judge",
        "human_labels_included": False,
        "model_called": False,
        "file_sha256": {name: sha256(blob).hexdigest() for name, blob in blobs.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/devai-smoke.yaml"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    workspace = prepare(args.config, args.source, args.output, download=args.download)
    print(f"Prepared inspection workspace: {workspace}")
    print("No key was read; no model call was made; project code was not executed.")


if __name__ == "__main__":
    main()
