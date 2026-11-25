"""Traceability and conservative offline candidate discovery."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from decomp_judge import bank_candidates as candidates
from decomp_judge.requirements_bank import (
    TASK_INDICES,
    BankRequirement,
    BankTask,
    document_header,
)


def write_task(root: Path, index: int, *, train: bool = False) -> Path:
    identifier = f"{'train_' if train else ''}{index + 1:06d}"
    requirement = BankRequirement(
        requirement_id="R1",
        criteria="Пользователь может войти в систему.",
        source="Users can log in.",
        kind="explicit",
        rule="code",
        parent=None,
        prerequisites=[],
    )
    data = {
        "task_index": index,
        "webgen_id": identifier,
        "query": "Users can log in.",
        "requirements": [requirement.model_dump()],
    }
    suffix = f"train/{identifier}" if train else str(index)
    path = root / f"data/requirements/{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    document = root / f"docs/requirements/{suffix}.md"
    document.parent.mkdir(parents=True, exist_ok=True)
    header = document_header(
        BankTask(
            task_index=index,
            webgen_id=f"{index + 1:06d}",
            query="Users can log in.",
            requirements=[requirement],
        )
    ).replace(f"{index + 1:06d}", identifier)
    document.write_text(
        header + "## Требования\n\n"
        "| ID | Формулировка | Источник | Тип | Правило | Родитель | Зависимости | Примечание |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| R1 | {requirement.criteria} | {requirement.source} | Явное | code | — | — | — |\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def annotation_root(tmp_path: Path) -> Path:
    for index in TASK_INDICES:
        write_task(tmp_path, index)
    write_task(tmp_path, 11, train=True)
    bank = tmp_path / "configs/requirement-bank.json"
    bank.parent.mkdir()
    bank.write_text('[{"source_ids": ["webgen-011-R1"]}]', encoding="utf-8")
    return tmp_path


def test_global_references_keep_old_ids_and_separate_splits(annotation_root: Path) -> None:
    rows, hashes = candidates.build_catalog(annotation_root)
    assert "webgen-011-R1" in {row.global_id for row in rows}
    assert "webgen-train-000012-R1" in {row.global_id for row in rows}
    assert len(rows) == len(TASK_INDICES) + 1
    assert len(hashes) == len(rows) * 2
    # Equal texts remain separate source requirements, rather than being silently merged.
    assert len({row.criteria for row in rows}) == 1


@pytest.mark.parametrize("problem", ["wrong_task", "missing_dependency", "duplicate_id", "md"])
def test_invalid_sources_are_rejected(annotation_root: Path, problem: str) -> None:
    path = annotation_root / "data/requirements/train/train_000012.json"
    data: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if problem == "wrong_task":
        data["task_index"] = 500
    elif problem == "missing_dependency":
        data["requirements"][0]["prerequisites"] = ["R404"]
    elif problem == "duplicate_id":
        data["requirements"].append(data["requirements"][0])
    else:
        data["requirements"][0]["criteria"] = "Different from Markdown"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        candidates.build_catalog(annotation_root)


def test_multiple_candidates_and_exclusions_are_not_merges(annotation_root: Path) -> None:
    rows, _ = candidates.build_catalog(annotation_root)
    families = [
        candidates.Family(
            id="a",
            title="A",
            criterion="A",
            parameters=["role"],
            patterns=["войти"],
            review="Review role",
        ),
        candidates.Family(
            id="b",
            title="B",
            criterion="B",
            parameters=["role"],
            patterns=["систем"],
            review="Review object",
        ),
        candidates.Family(
            id="c",
            title="C",
            criterion="C",
            parameters=["role"],
            patterns=["войти"],
            exclude_patterns=["систем"],
            review="Review",
        ),
    ]
    result = candidates.candidate_memberships(rows, families)
    assert result["a"] == result["b"]
    assert result["c"] == []
    assert all(row.rule == "code" for row in rows)


def test_neighbors_exclude_same_task_but_not_same_index_other_split(
    annotation_root: Path, tmp_path: Path
) -> None:
    rows, _ = candidates.build_catalog(annotation_root)
    rows = [row for row in rows if row.task_index in {11, 12}]
    rows = [
        row.model_copy(update={"criteria": "Пользователь может изменить свой профиль."})
        if row.task_index == 12
        else row
        for row in rows
    ]
    output = tmp_path / "analysis"
    output.mkdir()
    candidates.lexical_candidates(rows, output, {}, seed=42)
    with (output / "neighbors.csv").open(encoding="utf-8-sig") as stream:
        neighbors = list(csv.DictReader(stream))
    sources = {row.global_id: row for row in rows}
    assert neighbors
    for pair in neighbors:
        a, b = sources[pair["source_id"]], sources[pair["neighbor_id"]]
        assert (a.split, a.webgen_id) != (b.split, b.webgen_id)
        assert 0 <= float(pair["tfidf_cosine"]) <= 1
    assert any(
        pair["source_id"] == "webgen-011-R1" and pair["neighbor_id"] == "webgen-train-000012-R1"
        for pair in neighbors
    )
    assert (output / "map.html").is_file()


def test_source_parent_and_dependencies_get_global_references(annotation_root: Path) -> None:
    path = annotation_root / "data/requirements/train/train_000012.json"
    data = json.loads(path.read_text("utf-8"))
    data["requirements"].append(
        {
            **data["requirements"][0],
            "requirement_id": "R1.1",
            "parent": "R1",
            "prerequisites": ["R1"],
        }
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    document = annotation_root / "docs/requirements/train/train_000012.md"
    with document.open("a", encoding="utf-8") as stream:
        stream.write(
            "| R1.1 | Пользователь может войти в систему. | Users can log in. "
            "| Явное | code | R1 | R1 | — |\n"
        )
    rows, _ = candidates.build_catalog(annotation_root)
    child = next(row for row in rows if row.global_id == "webgen-train-000012-R1.1")
    assert child.parent_global_id == "webgen-train-000012-R1"
    assert child.prerequisite_global_ids == ["webgen-train-000012-R1"]


def test_duplicate_family_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "families.json"
    family = {
        "id": "a",
        "title": "A",
        "criterion": "A",
        "parameters": ["role"],
        "patterns": ["войти"],
        "review": "Review",
    }
    path.write_text(json.dumps([family, family]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        candidates.load_families(path)


def test_existing_results_are_never_overwritten(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        candidates.write_catalog(
            tmp_path, tmp_path, tmp_path / "families.json", seed=42, run_id="test", commit="test"
        )
