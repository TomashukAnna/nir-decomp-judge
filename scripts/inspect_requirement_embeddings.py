"""Audit supplied embeddings and export cross-task candidate pairs without model calls."""

import argparse
import csv
import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from decomp_judge.devai_smoke import TrackingSettings, committed_version


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected nonempty records: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")
    commit, branch = committed_version()
    if (args.output / "run.json").exists():
        parser.error("Audit already exists; choose a new output directory")
    files = {
        name: {
            "bytes": (args.input / name).stat().st_size,
            "sha256": sha256((args.input / name).read_bytes()).hexdigest(),
        }
        for name in (
            "catalog-webgen-maps.html",
            "catalog.json",
            "webgen-train-embeddings.json",
        )
    }
    catalog = load_rows(args.input / "catalog.json")
    train = load_rows(args.input / "webgen-train-embeddings.json")
    local = load_rows(Path("results/requirement-map/catalog.json"))
    stripped = [{k: v for k, v in r.items() if k not in {"embedding", "tfidf"}} for r in catalog]
    if stripped != local:
        raise ValueError("Supplied requirement metadata differs from the original catalog")
    for rows, key in ((catalog, "global_id"), (train, "id")):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {key}")
    vectors = []
    for rows in (catalog, train):
        matrix = np.asarray([r["embedding"] for r in rows], dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Invalid embeddings")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0):
            raise ValueError("Zero embedding")
        vectors.append(matrix / norms[:, None])
    if vectors[0].shape[1] != vectors[1].shape[1]:
        raise ValueError("Embedding dimensions differ")
    html = (args.input / "catalog-webgen-maps.html").read_text(encoding="utf-8-sig")
    match = re.search(r'<script[^>]*id="map-data"[^>]*>([\s\S]*?)</script>', html)
    if match is None:
        raise ValueError("Map data not found")
    maps = json.loads(match.group(1))
    map_summary = []
    for view, expected_rows in zip(maps, (catalog, train), strict=True):
        if len(view["rows"]) != len(expected_rows):
            raise ValueError("Map record count differs")
        id_key = "global_id" if view["key"] == "catalog-map" else "id"
        if [r[0] for r in view["rows"]] != [r[id_key] for r in expected_rows]:
            raise ValueError("Map IDs differ from the corresponding embedding records")
        for methods in view["views"].values():
            for points in methods.values():
                array = np.asarray(points)
                if array.shape != (len(expected_rows), 2) or not np.isfinite(array).all():
                    raise ValueError("Invalid map coordinates")
        map_summary.append(
            {
                "key": view["key"],
                "groups": view["groups"],
                "views": {name: list(methods) for name, methods in view["views"].items()},
            }
        )
    dense_scores = np.clip(vectors[0] @ vectors[0].T, -1, 1)
    sparse = DictVectorizer().fit_transform([r["tfidf"] for r in catalog])
    sparse_scores = cosine_similarity(sparse)
    args.output.mkdir(parents=True, exist_ok=True)
    tracking = TrackingSettings()
    config = {
        "commit": commit,
        "source_directory": args.input.as_posix(),
        "source": "Supervisor-supplied artifacts; original generation run ID not supplied",
        "model_label_in_html": "BGE-M3",
        "model_revision": None,
        "encoding_settings": None,
        "projection_and_clustering_settings": None,
        "seed": tracking.seed,
        "seed_used_for_audit": False,
        "neighbors": args.neighbors,
        "catalog_matches_original": True,
        "files": files,
        "maps": map_summary,
        "train_application_types": dict(Counter(r["application_type"] for r in train)),
    }
    mlflow.set_tracking_uri(tracking.mlflow_tracking_uri)
    mlflow.set_experiment(tracking.mlflow_experiment_name)
    with mlflow.start_run(run_name=f"{branch}-embedding-artifact-audit") as run:
        mlflow.set_tags(
            {
                "stage": "baseline",
                "dataset": "WebGen-Bench",
                "model_family": "BGE-M3-supplied",
                "pr": "not-opened",
                "purpose": "artifact-audit",
                "mlflow.note.content": (
                    "Audit of supplied artifacts, not a new embedding or projection run. "
                    "Cosine neighbors are candidates, not equivalence judgments."
                ),
            }
        )
        mlflow.log_params(
            {
                "commit": commit,
                "seed": tracking.seed,
                "neighbors": args.neighbors,
                "model_revision": "not-supplied",
                **{f"sha256_{name}": info["sha256"] for name, info in files.items()},
            }
        )
        metrics = {
            "requirements": len(catalog),
            "requirement_tasks": len({r["task_index"] for r in catalog}),
            "train_tasks": len(train),
            "embedding_dimensions": vectors[0].shape[1],
        }
        mlflow.log_metrics(metrics)
        config.update(run_id=run.info.run_id, metrics=metrics)
        (args.output / "run.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (args.output / "neighbors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "global_id",
                    "neighbor_id",
                    "embedding_cosine",
                    "supplied_tfidf_cosine",
                    "criteria",
                    "neighbor_criteria",
                ]
            )
            for i, row in enumerate(catalog):
                candidates = [
                    int(j)
                    for j in np.argsort(-dense_scores[i], kind="stable")
                    if catalog[j]["task_index"] != row["task_index"]
                ]
                for j in candidates[: args.neighbors]:
                    writer.writerow(
                        [
                            row["global_id"],
                            catalog[j]["global_id"],
                            f"{dense_scores[i, j]:.6f}",
                            f"{sparse_scores[i, j]:.6f}",
                            row["criteria"],
                            catalog[j]["criteria"],
                        ]
                    )
        for name in ("run.json", "neighbors.csv"):
            mlflow.log_artifact(str(args.output / name))
        mlflow.log_artifact(__file__, artifact_path="audit-code")
        print(f"Audit saved: {args.output}; MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
