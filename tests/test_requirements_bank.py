"""Offline checks for the annotation scaffold and its data contract."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from decomp_judge.requirements_bank import (
    TASK_INDICES,
    BankTask,
    check_bank,
    document_header,
    export_bank,
    parse_requirements,
    scaffold,
    validate_task,
)


def task_data() -> dict[str, Any]:
    return {
        "task_index": 11,
        "webgen_id": "000012",
        "query": "Specification",
        "requirements": [
            {
                "requirement_id": "R1",
                "criteria": "Criterion",
                "source": "Specification",
                "kind": "explicit",
                "rule": "code",
                "parent": None,
                "prerequisites": [],
            }
        ],
    }


def test_scaffold_preserves_specs_and_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"id": f"{i + 1:06d}", "instruction": f"ТЗ {i}", "cases": "HIDDEN_MARKER"})
            for i in range(98)
        ),
        encoding="utf-8",
    )
    root = tmp_path / "bank"
    schema = root / "configs/requirements.schema.json"
    schema.parent.mkdir(parents=True)
    schema.write_text(json.dumps(BankTask.model_json_schema()), encoding="utf-8")
    scaffold(source, root)
    check_bank(root)
    assert len(list((root / "docs/requirements").glob("*.md"))) == 15
    assert len(list((root / "data/requirements").glob("*.json"))) == 15
    for index in TASK_INDICES:
        text = (root / f"data/requirements/{index}.json").read_text(encoding="utf-8")
        task = BankTask.model_validate_json(text)
        assert task.query == f"ТЗ {index}"
        assert task.requirements == []
        assert "HIDDEN_MARKER" not in text
    with pytest.raises(ValueError, match="not been filled"):
        check_bank(root, require_complete=True)
    target = root / "docs/requirements/97.md"
    target.write_text("Student edits", encoding="utf-8")
    with pytest.raises(FileExistsError):
        scaffold(source, root)
    assert target.read_text(encoding="utf-8") == "Student edits"
    with pytest.raises(ValueError, match="Markdown specification"):
        check_bank(root)


@pytest.mark.parametrize("field", list(task_data()["requirements"][0]))
def test_every_requirement_field_is_required(field: str) -> None:
    data = task_data()
    del data["requirements"][0][field]
    with pytest.raises(ValidationError):
        BankTask.model_validate(data)


@pytest.mark.parametrize(
    "patch",
    [
        {"requirement_id": 1},
        {"criteria": " "},
        {"kind": "unknown"},
        {"rule": "LLM"},
        {"prerequisites": None},
        {"extra": None},
    ],
)
def test_wrong_shapes_rejected(patch: dict[str, Any]) -> None:
    data = task_data()
    data["requirements"][0].update(patch)
    with pytest.raises(ValidationError):
        BankTask.model_validate(data)


@pytest.mark.parametrize("relation", ["parent", "prerequisites"])
@pytest.mark.parametrize("target", ["missing", "R1"])
def test_invalid_references_rejected(relation: str, target: str) -> None:
    data = task_data()
    data["requirements"][0][relation] = target if relation == "parent" else [target]
    with pytest.raises(ValueError, match="reference"):
        validate_task(BankTask.model_validate(data))


@pytest.mark.parametrize("relation", ["parent", "prerequisites"])
def test_cycles_rejected(relation: str) -> None:
    data = task_data()
    first = data["requirements"][0]
    second = {**first, "requirement_id": "R2"}
    first[relation] = "R2" if relation == "parent" else ["R2"]
    second[relation] = "R1" if relation == "parent" else ["R1"]
    data["requirements"].append(second)
    with pytest.raises(ValueError, match="Cycle"):
        validate_task(BankTask.model_validate(data))


def test_missing_source_task_leaves_no_partial_bank(tmp_path: Path) -> None:
    source = tmp_path / "short.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "bank"
    with pytest.raises(ValueError, match="all selected tasks"):
        scaffold(source, root)
    assert not root.exists()


def test_export_validates_all_tasks_before_writing_and_detects_drift(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/requirements.schema.json").write_text(
        json.dumps(BankTask.model_json_schema()), encoding="utf-8"
    )
    (tmp_path / "docs/requirements").mkdir(parents=True)
    (tmp_path / "data/requirements").mkdir(parents=True)
    row = "| R1 | Criterion `x` | “Specification” | Явное | code | — | — | Note |\n"
    for index in TASK_INDICES:
        task = BankTask(
            task_index=index, webgen_id=f"{index + 1:06d}", query="Specification", requirements=[]
        )
        (tmp_path / f"data/requirements/{index}.json").write_text(
            task.model_dump_json(), encoding="utf-8"
        )
        (tmp_path / f"docs/requirements/{index}.md").write_text(
            document_header(task) + "## Требования\n\n" + row, encoding="utf-8"
        )
    last = tmp_path / "docs/requirements/97.md"
    valid_text = last.read_text(encoding="utf-8")
    last.write_text(valid_text.replace("Явное", "Typo"), encoding="utf-8")
    before = {p: p.read_bytes() for p in (tmp_path / "data/requirements").glob("*.json")}
    with pytest.raises(ValueError, match="Unknown requirement kind"):
        export_bank(tmp_path)
    assert before == {p: p.read_bytes() for p in before}
    last.write_text(valid_text, encoding="utf-8")
    documents = {p: p.read_bytes() for p in (tmp_path / "docs/requirements").glob("*.md")}
    assert export_bank(tmp_path) == len(TASK_INDICES)
    check_bank(tmp_path, require_complete=True)
    assert documents == {p: p.read_bytes() for p in documents}
    last.write_text(valid_text.replace("Criterion", "Changed criterion"), encoding="utf-8")
    with pytest.raises(ValueError, match="requirements differ"):
        check_bank(tmp_path, require_complete=True)


def test_wrapped_row_preserves_fields() -> None:
    document = (
        "## Требования\n\n"
        "| R1 | Проверка | “source” | Неявное | code |\n"
        "— | R2, R3 | Заметка |\n\n## Статус\nГотово\n"
    )
    requirement = parse_requirements(document)[0]
    assert requirement.criteria == "Проверка"
    assert requirement.source == "“source”"
    assert requirement.kind == "implicit"
    assert requirement.parent is None
    assert requirement.prerequisites == ["R2", "R3"]


@pytest.mark.parametrize(
    "document",
    [
        "No section",
        "## Требования\n| R1 | unfinished |",
        "## Требования\n| R1 | unfinished |\n| R2 | new row |",
        "## Требования\n| R1 | x | s | Явное | code | — | — | note | extra |",
    ],
)
def test_malformed_tables_fail_instead_of_dropping_rows(document: str) -> None:
    with pytest.raises(ValueError):
        parse_requirements(document)
