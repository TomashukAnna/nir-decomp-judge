"""Scaffold and validate the manually annotated WebGen-Bench task bank."""

import argparse
import json
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

TASK_INDICES = (11, 12, 16, 26, 28, 47, 49, 57, 58, 72, 82, 84, 92, 96, 97)


class BankRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    requirement_id: StrictStr = Field(min_length=1)
    criteria: StrictStr = Field(min_length=1)
    source: StrictStr = Field(min_length=1)
    kind: Literal["explicit", "implicit"]
    rule: Literal["code", "llm"]
    parent: StrictStr | None
    prerequisites: list[StrictStr]

    @field_validator("requirement_id", "criteria", "source", "parent")
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Text must not be blank")
        return value


class BankTask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task_index: StrictInt = Field(ge=0)
    webgen_id: StrictStr = Field(pattern=r"^\d{6}$")
    query: StrictStr = Field(min_length=1)
    requirements: list[BankRequirement]


def validate_task(task: BankTask, *, require_complete: bool = False) -> None:
    if task.task_index not in TASK_INDICES or task.webgen_id != f"{task.task_index + 1:06d}":
        raise ValueError("Task index and WebGen-Bench ID do not match the selected bank")
    if not task.query.strip():
        raise ValueError("Specification must not be blank")
    if require_complete and not task.requirements:
        raise ValueError("Requirements have not been filled in")
    ids = [r.requirement_id for r in task.requirements]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate requirement IDs")
    for requirement in task.requirements:
        refs = requirement.prerequisites
        if len(refs) != len(set(refs)):
            raise ValueError("Duplicate prerequisites")
        for reference in [*refs, *([requirement.parent] if requirement.parent else [])]:
            if reference not in ids or reference == requirement.requirement_id:
                raise ValueError("Requirement reference is missing or refers to itself")
    for relation in ("parent", "prerequisites"):
        graph = {
            r.requirement_id: ([r.parent] if r.parent else [])
            if relation == "parent"
            else r.prerequisites
            for r in task.requirements
        }
        try:
            tuple(TopologicalSorter(graph).static_order())
        except CycleError as exc:
            raise ValueError(f"Cycle in {relation}") from exc


def document_header(task: BankTask) -> str:
    return (
        f"# ТЗ № {task.task_index} — WebGen-Bench {task.webgen_id}\n\n"
        "## Исходное ТЗ\n\n" + task.query + "\n\n"
    )


def parse_requirements(document: str) -> list[BankRequirement]:
    if "## Требования\n" not in document:
        raise ValueError("Markdown requirements section is missing")
    section = document.split("## Требования\n", 1)[1].split("\n## ", 1)[0]
    requirements = []
    pending = ""
    for line in section.splitlines():
        line = line.strip()
        if not line:
            continue
        if pending:
            if line.startswith("|"):
                raise ValueError("Incomplete Markdown table row")
            line = pending + " " + line
        elif not line.startswith("|"):
            raise ValueError("Unexpected text in requirements table")
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] == "ID" or all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) < 8 or not line.endswith("|"):
            pending = line
            continue
        pending = ""
        if len(cells) != 8:
            raise ValueError("Each Markdown requirement must contain eight columns")
        identifier, criteria, source, kind, rule, parent, dependencies, _note = cells
        if kind not in {"Явное", "Неявное"}:
            raise ValueError(f"Unknown requirement kind: {kind}")
        requirements.append(
            BankRequirement.model_validate(
                {
                    "requirement_id": identifier,
                    "criteria": criteria,
                    "source": source,
                    "kind": {"Явное": "explicit", "Неявное": "implicit"}[kind],
                    "rule": rule,
                    "parent": None if parent == "—" else parent,
                    "prerequisites": []
                    if dependencies == "—"
                    else [ref.strip() for ref in dependencies.split(",")],
                }
            )
        )
    if pending:
        raise ValueError("Incomplete Markdown table row")
    return requirements


def export_bank(root: Path) -> int:
    check_bank(root)
    outputs = {}
    for index in TASK_INDICES:
        path = root / "data/requirements" / f"{index}.json"
        task = BankTask.model_validate_json(path.read_text(encoding="utf-8"))
        document = (root / "docs/requirements" / f"{index}.md").read_text(encoding="utf-8")
        task.requirements = parse_requirements(document)
        validate_task(task, require_complete=True)
        outputs[path] = task
    # Validate the entire bank before replacing any JSON; Markdown is always read-only.
    for path, task in outputs.items():
        path.write_text(task.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return sum(len(task.requirements) for task in outputs.values())


def scaffold(source: Path, root: Path) -> None:
    tasks = []
    with source.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index not in TASK_INDICES:
                continue
            row = json.loads(line)
            # Only the specification and ID enter the bank, never acceptance cases or outputs.
            task = BankTask(
                task_index=index, webgen_id=row["id"], query=row["instruction"], requirements=[]
            )
            validate_task(task)
            tasks.append(task)
    if len(tasks) != len(TASK_INDICES):
        raise ValueError("Source does not contain all selected tasks")
    files: dict[Path, str] = {}
    for task in tasks:
        files[root / "data/requirements" / f"{task.task_index}.json"] = (
            task.model_dump_json(indent=2) + "\n"
        )
        files[root / "docs/requirements" / f"{task.task_index}.md"] = (
            document_header(task)
            + "## Требования\n\n"
            + "| ID | Формулировка | Источник в ТЗ | Явное / неявное | code / llm | "
            "Родитель | Зависимости | Спорное прочтение и выбранная трактовка |\n"
            "|---|---|---|---|---|---|---|---|\n\n"
            "## Статус\n\nКаркас. Ручной разбор требований ещё не выполнен.\n"
        )
    if any(path.exists() for path in files):
        raise FileExistsError(
            "A target file already exists; existing analyses are never overwritten"
        )
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)


def check_bank(root: Path, *, require_complete: bool = False) -> None:
    expected = {f"{index}.json" for index in TASK_INDICES}
    actual = {p.name for p in (root / "data/requirements").glob("*.json")}
    if actual != expected:
        raise ValueError(f"Bank must contain exactly the selected {len(TASK_INDICES)} JSON files")
    schema = json.loads((root / "configs/requirements.schema.json").read_text(encoding="utf-8"))
    if schema != BankTask.model_json_schema():
        raise ValueError("JSON schema differs from the validation model")
    for index in TASK_INDICES:
        path = root / "data/requirements" / f"{index}.json"
        task = BankTask.model_validate_json(path.read_text(encoding="utf-8"))
        if task.task_index != index:
            raise ValueError(f"Task index does not match filename: {path.name}")
        validate_task(task, require_complete=require_complete)
        document = (root / "docs/requirements" / f"{index}.md").read_text(encoding="utf-8")
        if not document.startswith(document_header(task)):
            raise ValueError(f"Markdown specification differs from JSON: {index}")
        if require_complete and parse_requirements(document) != task.requirements:
            raise ValueError(f"Markdown requirements differ from JSON: {index}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("scaffold")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--root", type=Path, default=Path("."))
    check = commands.add_parser("check")
    check.add_argument("--root", type=Path, default=Path("."))
    check.add_argument("--require-complete", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "scaffold":
        scaffold(args.source, args.root)
        print(f"Created {len(TASK_INDICES)} Markdown/JSON pairs; no requirements were inferred.")
    elif args.command == "export":
        count = export_bank(args.root)
        print(f"Exported {count} requirements for {len(TASK_INDICES)} tasks; Markdown unchanged.")
    else:
        check_bank(args.root, require_complete=args.require_complete)
        print(f"Requirements bank: {len(TASK_INDICES)} pairs validated.")


if __name__ == "__main__":
    main()
