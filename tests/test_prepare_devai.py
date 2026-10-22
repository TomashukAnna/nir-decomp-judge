"""Preparation tests use a tiny local Git repository, never the downloaded dataset."""

import json
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from decomp_judge import prepare_devai as preparation


def git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def example(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    git(source, "config", "user.name", "Offline Test")
    git(source, "config", "user.email", "test@example.invalid")
    git(source, "config", "core.autocrlf", "false")
    task = {
        "name": "39_Test",
        "query": "Implement a tiny project. Тест.",
        "requirements": [
            {"requirement_id": 0, "prerequisites": [], "criteria": "Load data", "satisfied": None},
            {"requirement_id": 1, "prerequisites": [0], "criteria": "Train", "satisfied": None},
        ],
    }
    prefix = "benchmark/workspaces/OpenHands/39_Test"
    files = {
        "benchmark/devai/instances/39_Test.json": json.dumps(task, ensure_ascii=False),
        prefix + "/src/train.py": "print('original source')\n",
        prefix + "/results/report.md": "Report\n",
        prefix + "/results/report.pdf": "opaque binary placeholder",
        prefix + "/gdsc_dataset.csv": "RAW_DATA_MARKER\n",
        prefix + "/judge/answer.py": "JUDGE_MARKER\n",
        "benchmark/judgment/OpenHands/human_as_a_judge/39_Test.json": "HUMAN_LABEL_MARKER",
    }
    for name, text in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "Synthetic preparation fixture")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "source_commit": git(source, "rev-parse", "HEAD"),
                "task": "39_Test",
                "expected_ids": [0, 1],
            }
        ),
        encoding="utf-8",
    )
    system = tmp_path / "system.txt"
    system.write_text("Judge each requirement.\n", encoding="utf-8")
    return source, config, system


def test_pinned_prompt_is_stable_and_has_no_labels(
    example: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source, config, system = example
    first = tmp_path / "first"
    digest = preparation.prepare(config, system, source, first, download=False)
    original = (first / "messages.json").read_bytes()
    assert sha256(original).hexdigest() == digest
    text = original.decode("utf-8")
    assert "original source" in text
    for excluded in ("HUMAN_LABEL_MARKER", "JUDGE_MARKER", "RAW_DATA_MARKER", '"satisfied":'):
        assert excluded not in text
    assert "report.pdf" in text
    assert b"\r\n" not in original
    # Worktree changes and line endings cannot change committed evidence.
    (source / "benchmark/workspaces/OpenHands/39_Test/src/train.py").write_bytes(b"MUTATED\r\n")
    second = tmp_path / "second"
    assert preparation.prepare(config, system, source, second, download=False) == digest
    assert (second / "messages.json").read_bytes() == original
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_called"] is False
    assert manifest["human_labels_included"] is False


def test_existing_output_is_preserved(example: tuple[Path, Path, Path], tmp_path: Path) -> None:
    source, config, system = example
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        preparation.prepare(config, system, source, output, download=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("ids", [[0, 0], [0], [0, 2]])
def test_mismatched_ids_are_rejected(example: tuple[Path, Path, Path], ids: list[int]) -> None:
    source, config_path, _ = example
    config = preparation.PreparationConfig.model_validate(
        {**yaml.safe_load(config_path.read_text(encoding="utf-8")), "expected_ids": ids}
    )
    with pytest.raises(ValueError, match="IDs"):
        preparation.assemble_prompt(source, config, "Judge")


def test_download_and_reuse_are_offline(
    example: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, config_path, system = example
    config = preparation.PreparationConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    destination = tmp_path / "download"
    with pytest.raises(FileNotFoundError):
        preparation.ensure_source(destination, config.source_commit, download=False)
    monkeypatch.setattr(preparation, "SOURCE_URL", str(source))
    preparation.prepare(config_path, system, destination, tmp_path / "prepared", download=True)
    assert (destination / ".git").is_dir()
    preparation.ensure_source(destination, config.source_commit, download=False)
    with pytest.raises(ValueError, match="root"):
        preparation.ensure_source(source / "benchmark", config.source_commit, download=False)
