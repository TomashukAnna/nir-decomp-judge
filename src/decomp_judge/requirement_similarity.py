"""Build a traceable requirement catalog and an interactive similarity map."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from decomp_judge.devai_smoke import TrackingSettings, committed_version
from decomp_judge.requirements_bank import TASK_INDICES, BankTask, validate_task


def catalog(source: Path) -> list[dict[str, Any]]:
    rows = []
    for index in TASK_INDICES:
        path = source / f"{index}.json"
        task = BankTask.model_validate_json(path.read_text(encoding="utf-8"))
        validate_task(task, require_complete=True)
        if task.task_index != index:
            raise ValueError(f"Unexpected task index in {path}")
        prefix = f"webgen-{index:03d}-"
        for requirement in task.requirements:
            rows.append(
                {
                    **requirement.model_dump(),
                    "global_id": prefix + requirement.requirement_id,
                    "task_index": index,
                    "webgen_id": task.webgen_id,
                    "document": f"docs/requirements/{index}.md",
                    "parent_global_id": prefix + requirement.parent if requirement.parent else None,
                    "prerequisite_global_ids": [prefix + ref for ref in requirement.prerequisites],
                }
            )
    return rows


def analyze(rows: list[dict[str, Any]], output: Path, seed: int, neighbors: int) -> None:
    features = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(
        [row["criteria"] for row in rows]
    )
    similarities = cosine_similarity(features)
    projection = TruncatedSVD(n_components=2, random_state=seed).fit_transform(features)
    output.mkdir(parents=True, exist_ok=False)
    (output / "catalog.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "neighbors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["global_id", "neighbor_id", "cosine", "criteria", "neighbor_criteria"])
        for i, row in enumerate(rows):
            candidates = [
                int(j)
                for j in np.argsort(-similarities[i], kind="stable")
                if rows[j]["task_index"] != row["task_index"] and similarities[i, j] > 0
            ]
            for j in candidates[:neighbors]:
                writer.writerow(
                    [
                        row["global_id"],
                        rows[j]["global_id"],
                        f"{similarities[i, j]:.6f}",
                        row["criteria"],
                        rows[j]["criteria"],
                    ]
                )
    points = [
        {**row, "x": float(pair[0]), "y": float(pair[1])}
        for row, pair in zip(rows, projection, strict=True)
    ]
    fig = px.scatter(
        points,
        x="x",
        y="y",
        color="rule",
        symbol="kind",
        hover_name="global_id",
        hover_data=["task_index", "criteria"],
        title="WebGen-Bench: карта требований (TF-IDF)",
    )
    fig.write_html(output / "map.html", include_plotlyjs=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/requirements"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")
    commit, branch = committed_version()
    tracking = TrackingSettings()
    mlflow.set_tracking_uri(tracking.mlflow_tracking_uri)
    mlflow.set_experiment(tracking.mlflow_experiment_name)
    rows = catalog(args.source)
    with mlflow.start_run(run_name=f"{branch}-requirement-map") as run:
        tags = {
            "stage": "baseline",
            "dataset": "WebGen-Bench",
            "model_family": "tf-idf",
            "pr": "not-opened",
            "mlflow.note.content": "Cross-task lexical similarity of criteria.",
        }
        params = {
            "commit": commit,
            "seed": tracking.seed,
            "source": str(args.source),
            "neighbors": args.neighbors,
            "ngram_range": "3-5",
            "projection": "TruncatedSVD",
        }
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        analyze(rows, args.output, tracking.seed, args.neighbors)
        mlflow.log_metrics(
            {
                "requirements": len(rows),
                "tasks": len({row["task_index"] for row in rows}),
            }
        )
        (args.output / "run.json").write_text(
            json.dumps({"run_id": run.info.run_id, **params}, indent=2) + "\n", encoding="utf-8"
        )
        mlflow.log_artifacts(str(args.output))
        print(run.info.run_id)


if __name__ == "__main__":
    main()
