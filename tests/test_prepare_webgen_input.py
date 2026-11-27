"""Verify the handoff input matches the pinned source and excludes evaluation fields."""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_webgen_input.py"


@pytest.mark.parametrize("crlf", [False, True])
@pytest.mark.parametrize("all_tasks", [False, True])
def test_selection_preserves_instruction_and_original_index(
    tmp_path: Path, crlf: bool, all_tasks: bool
) -> None:
    rows: list[dict[str, Any]] = [
        {"id": "000001", "instruction": "First task"},
        {"id": "000002", "instruction": "Второе ТЗ", "ui_instruct": ["Hidden case"]},
    ]
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    source = tmp_path / "source.jsonl"
    source.write_bytes(raw.replace(b"\n", b"\r\n") if crlf else raw)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"sha256_lf": sha256(raw).hexdigest()}))
    output = tmp_path / "selected"
    arguments = [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--source-config",
        str(config),
        *(["--all"] if all_tasks else ["--ids", "000002"]),
        "--output",
        str(output),
    ]
    subprocess.run(arguments, check=True, capture_output=True)
    selected = [
        json.loads(line)
        for line in (output / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    indices = [0, 1] if all_tasks else [1]
    assert selected == [
        {"id": rows[i]["id"], "instruction": rows[i]["instruction"]} for i in indices
    ]
    provenance = json.loads((output / "selection.json").read_text())
    assert provenance["tasks"] == [
        {"webgen_id": rows[i]["id"], "original_task_index": i, "selection_index": selected_index}
        for selected_index, i in enumerate(indices)
    ]
    assert provenance["selected_sha256"] == sha256((output / "test.jsonl").read_bytes()).hexdigest()
    original = (output / "test.jsonl").read_bytes()
    repeated = subprocess.run(arguments, capture_output=True)
    assert repeated.returncode != 0
    assert (output / "test.jsonl").read_bytes() == original
    config.write_text(json.dumps({"sha256_lf": "0" * 64}))
    arguments[-1] = str(tmp_path / "wrong-hash")
    wrong = subprocess.run(arguments, capture_output=True)
    assert wrong.returncode != 0
    assert not (tmp_path / "wrong-hash").exists()
