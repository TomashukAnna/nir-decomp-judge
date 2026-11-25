"""Prepare a traceable catalog and manually defined search groups, entirely offline."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import plotly.express as px
from pydantic import BaseModel, ConfigDict, Field
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from decomp_judge.devai_smoke import TrackingSettings
from decomp_judge.requirements_bank import (
    TASK_INDICES,
    BankRequirement,
    BankTask,
    parse_requirements,
    validate_task,
)


class SourceTask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task_index: int = Field(ge=0)
    webgen_id: str = Field(pattern=r"^(?:train_)?\d{6}$")
    query: str = Field(min_length=1)
    requirements: list[BankRequirement] = Field(min_length=1)


class CatalogRow(BankRequirement):
    global_id: str
    split: Literal["test", "train"]
    task_index: int
    webgen_id: str
    document: str
    json_source: str
    parent_global_id: str | None
    prerequisite_global_ids: list[str]


class Family(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    criterion: str = Field(min_length=1)
    parameters: list[str] = Field(min_length=1)
    patterns: list[str] = Field(min_length=1)
    exclude_patterns: list[str] = Field(default_factory=list)
    review: str = Field(min_length=1)


def load_families(path: Path) -> list[Family]:
    families = [Family.model_validate(item) for item in json.loads(path.read_text("utf-8"))]
    if not families or len({family.id for family in families}) != len(families):
        raise ValueError("Family IDs must be unique and nonempty")
    for family in families:
        for pattern in [*family.patterns, *family.exclude_patterns]:
            re.compile(pattern)
    return families


def build_catalog(root: Path) -> tuple[list[CatalogRow], dict[str, str]]:
    """Preserve old IDs; namespace train references without editing source annotations."""
    paths = [root / f"data/requirements/{index}.json" for index in TASK_INDICES]
    paths += sorted((root / "data/requirements/train").glob("*.json"))
    if len(paths) == len(TASK_INDICES):
        raise ValueError("No train annotations found")
    rows: list[CatalogRow] = []
    hashes: dict[str, str] = {}
    seen_tasks: set[str] = set()
    for path in paths:
        task = SourceTask.model_validate_json(path.read_text("utf-8"))
        train = task.webgen_id.startswith("train_")
        expected_id = f"{'train_' if train else ''}{task.task_index + 1:06d}"
        expected_name = task.webgen_id if train else str(task.task_index)
        if task.webgen_id != expected_id or path.stem != expected_name:
            raise ValueError(f"Task identity mismatch: {path}")
        split: Literal["train", "test"] = "train" if train else "test"
        task_key = f"{split}:{task.webgen_id}"
        if task_key in seen_tasks:
            raise ValueError(f"Duplicate task: {task_key}")
        seen_tasks.add(task_key)
        if not train:
            validate_task(BankTask.model_validate(task.model_dump()), require_complete=True)
        ids = {requirement.requirement_id for requirement in task.requirements}
        if len(ids) != len(task.requirements):
            raise ValueError(f"Duplicate requirement IDs: {path}")
        for requirement in task.requirements:
            refs = [*requirement.prerequisites]
            if requirement.parent is not None:
                refs.append(requirement.parent)
            if not set(refs) <= ids or requirement.requirement_id in refs:
                raise ValueError(f"Invalid dependency: {path}: {requirement.requirement_id}")
            if len(requirement.prerequisites) != len(set(requirement.prerequisites)):
                raise ValueError(f"Duplicate prerequisites: {path}")
        for relationship in ("parent", "prerequisites"):
            graph = {
                requirement.requirement_id: ([requirement.parent] if requirement.parent else [])
                if relationship == "parent"
                else requirement.prerequisites
                for requirement in task.requirements
            }
            tuple(TopologicalSorter(graph).static_order())
        document = (
            f"docs/requirements/train/{task.webgen_id}.md"
            if train
            else f"docs/requirements/{task.task_index}.md"
        )
        document_path = root / document
        if parse_requirements(document_path.read_text("utf-8")) != task.requirements:
            raise ValueError(f"Markdown/JSON mismatch: {document_path}")
        for source_path in (path, document_path):
            hashes[source_path.relative_to(root).as_posix()] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
        prefix = (
            f"webgen-train-{task.webgen_id.removeprefix('train_')}-"
            if train
            else (f"webgen-{task.task_index:03d}-")
        )
        for requirement in task.requirements:
            rows.append(
                CatalogRow(
                    **requirement.model_dump(),
                    global_id=prefix + requirement.requirement_id,
                    split=split,
                    task_index=task.task_index,
                    webgen_id=task.webgen_id,
                    document=document,
                    json_source=path.relative_to(root).as_posix(),
                    parent_global_id=prefix + requirement.parent if requirement.parent else None,
                    prerequisite_global_ids=[prefix + ref for ref in requirement.prerequisites],
                )
            )
    if len({row.global_id for row in rows}) != len(rows):
        raise ValueError("Duplicate global references")
    return rows, hashes


def candidate_memberships(rows: list[CatalogRow], families: list[Family]) -> dict[str, list[str]]:
    """Find review candidates by explicit text rules; never assert semantic equivalence."""
    result: dict[str, list[str]] = {}
    for family in families:
        positives = [re.compile(pattern, re.IGNORECASE) for pattern in family.patterns]
        negatives = [re.compile(pattern, re.IGNORECASE) for pattern in family.exclude_patterns]
        result[family.id] = [
            row.global_id
            for row in rows
            if any(pattern.search(row.criteria) for pattern in positives)
            and not any(pattern.search(row.criteria) for pattern in negatives)
        ]
    return result


def lexical_candidates(
    rows: list[CatalogRow], output: Path, memberships: dict[str, list[str]], seed: int
) -> int:
    """Export cross-task lexical neighbors and a projection, never an automatic merge."""
    features = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(
        [row.criteria for row in rows]
    )
    largest_task = max(Counter((row.split, row.webgen_id) for row in rows).values())
    count = min(len(rows), 5 + largest_task)
    distances, indices = (
        NearestNeighbors(metric="cosine", algorithm="brute")
        .fit(features)
        .kneighbors(features, n_neighbors=count)
    )
    pair_count = 0
    with (output / "neighbors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_id", "neighbor_id", "tfidf_cosine"])
        for row, scores, positions in zip(rows, distances, indices, strict=True):
            neighbors = 0
            for distance, position in zip(scores, positions, strict=True):
                other = rows[int(position)]
                if (row.split, row.webgen_id) == (other.split, other.webgen_id):
                    continue
                similarity = float(np.clip(1 - distance, 0, 1))
                if similarity <= 0:
                    continue
                writer.writerow([row.global_id, other.global_id, f"{similarity:.6f}"])
                pair_count += 1
                neighbors += 1
                if neighbors == 5:
                    break
    projection = TruncatedSVD(n_components=2, random_state=seed).fit_transform(features)
    primary_family = {
        identifier: family_id
        for family_id, identifiers in reversed(list(memberships.items()))
        for identifier in identifiers
    }
    points = [
        {
            "x": float(point[0]),
            "y": float(point[1]),
            "ID": row.global_id,
            "Требование": row.criteria,
            "Часть": row.split,
            "Тип в разметке": row.rule,
            "Группа-кандидат": primary_family.get(row.global_id, "Пока без группы"),
        }
        for row, point in zip(rows, projection, strict=True)
    ]
    figure = px.scatter(
        points,
        x="x",
        y="y",
        color="Группа-кандидат",
        symbol="Часть",
        hover_name="ID",
        hover_data=["Требование", "Тип в разметке"],
        title="Требования: TF-IDF + SVD; цвет — ручная группа-кандидат, не кластер",
    )
    figure.update_traces(marker={"size": 6, "opacity": 0.65})
    figure.write_html(output / "map.html", include_plotlyjs=True)
    return pair_count


def write_catalog(
    root: Path, output: Path, families_path: Path, *, seed: int, run_id: str, commit: str
) -> None:
    """Export authoring materials, not an experiment or an executable checker bank."""
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    rows, hashes = build_catalog(root)
    families = load_families(families_path)
    memberships = candidate_memberships(rows, families)
    assigned = {identifier for members in memberships.values() for identifier in members}
    unassigned = [row.global_id for row in rows if row.global_id not in assigned]
    by_id = {row.global_id: row for row in rows}
    original_bank = root / "configs/requirement-bank.json"
    old_templates = json.loads(original_bank.read_text("utf-8"))
    if any(not set(template["source_ids"]) <= by_id.keys() for template in old_templates):
        raise ValueError("Existing bank references are missing from the expanded catalog")
    hashes[original_bank.relative_to(root).as_posix()] = hashlib.sha256(
        original_bank.read_bytes()
    ).hexdigest()
    groups = [
        {
            **family.model_dump(),
            "status": "candidate_not_reviewed",
            "source_ids": memberships[family.id],
            "checker_status": "not_assigned",
        }
        for family in families
    ]
    group_lookup: dict[str, list[str]] = {row.global_id: [] for row in rows}
    for family_id, members in memberships.items():
        for member in members:
            group_lookup[member].append(family_id)
    output.mkdir(parents=True)

    def dump(name: str, value: object) -> None:
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    dump("catalog.json", [row.model_dump() for row in rows])
    dump("candidates.json", {"families": groups, "unassigned_source_ids": unassigned})
    dump(
        "manifest.json",
        {
            "purpose": "Offline candidate search; not judge accuracy evaluation",
            "method": "Manual regex search groups + char_wb TF-IDF (3,5), cosine neighbors, SVD",
            "run_id": run_id,
            "seed": seed,
            "commit": commit,
            "config_path": families_path.relative_to(root).as_posix(),
            "families_sha256": hashlib.sha256(families_path.read_bytes()).hexdigest(),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": hashes,
            "tasks": len({(row.split, row.webgen_id) for row in rows}),
            "requirements": len(rows),
            "matched_requirements": len(assigned),
            "unassigned_requirements": len(unassigned),
            "multiple_candidates": sum(len(matches) > 1 for matches in group_lookup.values()),
        },
    )
    lines = [
        "# Кандидаты для расширения банка требований",
        "",
        "Это рабочая группировка для просмотра, а не готовый банк и не кластеризация эмбеддингов.",
        "Группы заданы вручную; поисковые выражения только отбирают возможные примеры.",
        "Отдельные совпадения ещё не проверены по смыслу.",
        "Один пункт может попасть в несколько групп.",
        "Исходные требования, их типы code/llm и старый банк не изменены.",
        "",
        "## Файлы",
        "",
        "- [catalog.json](catalog.json): все исходные требования с уникальными ссылками.",
        "- [candidates.json](candidates.json): группы, их параметры и полные списки исходных ID.",
        "- [review.md](review.md): формулировки внутри каждой группы и пункты без группы.",
        "- [neighbors.csv](neighbors.csv): до пяти ближайших требований из других ТЗ по TF-IDF.",
        "- [map.html](map.html): карта TF-IDF + TruncatedSVD, открывается без интернета.",
        "- [manifest.json](manifest.json): контрольные суммы и инвентаризация, не метрики судьи.",
        "",
        f"MLflow run: `{run_id}`. Seed: `{seed}`.",
        f"Код поиска: commit `{commit}`; конфигурация — `configs/requirement-families.json`.",
        "Расстояния на двумерной карте не заменяют сходство исходных векторов.",
        "Цвет показывает первую подходящую ручную группу.",
        "Все совпадения сохранены в candidates.json.",
        "",
        "## Группы",
        "",
        "| Группа | Что обобщаем | Что требуется уточнить |",
        "|---|---|---|",
    ]
    for family in families:
        lines.append(f"| {family.title} (`{family.id}`) | {family.criterion} | {family.review} |")
    lines += [
        "",
        "## Как перейти от групп к банку",
        "",
        "Просмотреть review.md: исключить ложные совпадения, разделить составные требования,",
        "выделить параметры и сохранить роли, ограничения и зависимости. Сходство действий",
        "не означает одинаковую проверку.",
        "Например, создание записи и добавление товара в корзину различаются.",
        "Пункты без группы не считаются автоматически специфичными.",
        "Часть могла не совпасть по словам.",
        "",
        "У подтверждённого шаблона должны быть условия применимости, ссылки на исходники и правило",
        "проверки. Метка code в исходной разметке не означает, что проверяющий код уже написан.",
        "Проверку цвета из первой версии банка можно сохранить.",
        "Новые checker здесь не заявляются.",
        "Неподтверждённые группы в configs/requirement-bank.json не добавляются.",
        "",
        "## Повторение",
        "",
        "Из корня проекта, в новую папку результатов:",
        "",
        "```powershell",
        "uv run python -m decomp_judge.bank_candidates `",
        "  --output results/requirement-bank-expansion-repeat",
        "```",
        "",
        "Используются локальные data/requirements/*.json и data/requirements/train/*.json",
        "и соответствующие Markdown. Нужны исходные разборы, а не вложенная рабочая копия Git.",
        "Сеть, модели, эмбеддинги и API не используются.",
        "TF-IDF считается по русским формулировкам требований.",
        "Файлы существующего запуска не перезаписываются. SEED задаёт случайность проекции.",
        "Подбор по TF-IDF основан на символах длиной 3–5 и косинусном сходстве.",
        "Высокая близость не доказывает эквивалентность.",
        "Отрицания, роли и условия могут различаться.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    review = [
        "# Просмотр групп-кандидатов",
        "",
        "Все строки ниже — кандидаты по текстовому поиску, а не подтверждённые объединения.",
        "code/llm показан из исходной разметки и не назначает правило будущему шаблону.",
        "",
    ]
    for family in families:
        review += [f"## {family.title} — {family.id}", "", family.review, ""]
        for identifier in memberships[family.id]:
            row = by_id[identifier]
            review.append(
                f"- [{identifier}](../../{row.document}): {row.criteria} ({row.kind}, {row.rule})"
            )
        if not memberships[family.id]:
            review.append("Совпадений по текущему поисковому выражению нет.")
        review.append("")
    review += ["## Пока без группы", ""]
    for identifier in unassigned:
        row = by_id[identifier]
        review.append(f"- [{identifier}](../../{row.document}): {row.criteria}")
    (output / "review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    counts = Counter(row.split for row in rows)
    pair_count = lexical_candidates(rows, output, memberships, seed)
    mlflow.log_metrics(
        {
            "requirements": len(rows),
            "tasks": len({(row.split, row.webgen_id) for row in rows}),
            "matched_candidates": len(assigned),
            "unassigned": len(unassigned),
            "neighbor_pairs": pair_count,
        }
    )
    print(f"Catalog prepared: {len(rows)} requirements, by split: {dict(counts)}")
    print(f"Candidates: {len(assigned)}; awaiting grouping: {len(unassigned)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--families", type=Path, default=Path("configs/requirement-families.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    tracking = TrackingSettings()
    seed = tracking.seed
    mlflow.set_tracking_uri(tracking.mlflow_tracking_uri)
    mlflow.set_experiment(tracking.mlflow_experiment_name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    # Do not label an uncommitted search implementation as a reproducible experiment.
    for path in (Path(__file__).resolve(), args.families.resolve()):
        relative = path.relative_to(root).as_posix()
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root)
        if committed.replace(b"\r\n", b"\n") != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"Commit the search implementation before running: {relative}")
    with mlflow.start_run(run_name=f"{branch}-bank-candidates") as run:
        mlflow.set_tags(
            {
                "stage": "baseline",
                "dataset": "WebGen-Bench annotated test + train",
                "model_family": "tf-idf",
                "pr": "not-opened",
                "mlflow.note.content": "Candidate discovery only; no automatic requirement merges.",
            }
        )
        mlflow.log_params(
            {
                "commit": commit,
                "seed": seed,
                "families": str(args.families),
                "neighbors": 5,
                "analyzer": "char_wb",
                "ngram_range": "3-5",
                "projection": "TruncatedSVD",
            }
        )
        mlflow.log_artifact(str(args.families))
        write_catalog(
            root,
            args.output.resolve(),
            args.families.resolve(),
            seed=seed,
            run_id=run.info.run_id,
            commit=commit,
        )
        mlflow.log_artifacts(str(args.output.resolve()))
        print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
