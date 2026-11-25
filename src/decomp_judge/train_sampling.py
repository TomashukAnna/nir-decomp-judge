"""Cluster train specifications locally and select candidates for manual annotation."""

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
import yaml
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from decomp_judge.devai_smoke import TrackingSettings, committed_version


def functional_text(text: str) -> str:
    """Remove standalone styling instructions, retaining original text elsewhere."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    for sentence in sentences:
        styling = re.match(
            r"^(?:set|use|choose|apply|establish|configure|assign|specify|style)\b",
            sentence,
            re.IGNORECASE,
        )
        if styling and re.search(
            r"\b(?:background|colou?r|styling|style components)\b", sentence, re.IGNORECASE
        ):
            continue
        kept.append(sentence)
    result = " ".join(kept).lower()
    result = re.sub(r"\bplease (?:implement|create|develop)\b", "", result)
    result = re.sub(
        r"\b(?:the website|the application|the platform|the system) should\b", "", result
    )
    return re.sub(r"\s+", " ", result).strip()


def select_candidates(
    vectors: NDArray[np.float64],
    members: list[int],
    count: int,
    excluded: set[int],
    maximum: float,
    previously_selected: list[int],
) -> list[int]:
    """Choose a central candidate, then diverse ones; reject near duplicates globally."""
    available = [i for i in members if i not in excluded]
    selected: list[int] = []
    center = np.mean(vectors[members], axis=0)
    center /= max(float(np.linalg.norm(center)), 1e-12)
    while available and len(selected) < count:
        references = previously_selected + selected
        if references:
            similarities = vectors[available] @ vectors[references].T
            available = [
                i
                for i, score in zip(available, similarities.max(axis=1), strict=True)
                if score < maximum
            ]
        if not available:
            break
        if not selected:
            scores = vectors[available] @ center
            chosen = available[int(np.argmax(scores))]
        else:
            scores = (vectors[available] @ vectors[selected].T).max(axis=1)
            chosen = available[int(np.argmin(scores))]
        selected.append(chosen)
        available.remove(chosen)
    return selected


def run(config_path: Path) -> Path:
    commit, branch = committed_version()
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(config["dataset"])
    output = Path(config["output"])
    if output.exists():
        raise ValueError("Output already exists; configure a new directory")
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate train IDs")
    texts = [functional_text(row["instruction"]) for row in rows]
    if not all(texts):
        raise ValueError("Empty functional text")
    unique: dict[str, int] = {}
    canonical = []
    for text in texts:
        if text not in unique:
            unique[text] = len(unique)
        canonical.append(unique[text])
    documents = list(unique)
    annotated = {
        functional_text(json.loads(path.read_text(encoding="utf-8"))["query"])
        for path in Path(config["annotated_directory"]).glob("*.json")
    }
    excluded = {index for text, index in unique.items() if text in annotated}
    if not annotated:
        raise ValueError("No annotated tasks found; check annotated_directory")
    settings = TrackingSettings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name + "/train-sampling")
    with mlflow.start_run(run_name=f"{branch}-functional-tfidf") as active:
        mlflow.set_tags(
            {
                "stage": "baseline",
                "dataset": "WebGen-Bench/train",
                "model_family": "TF-IDF-SVD",
                "pr": "not-opened",
                "mlflow.note.content": (
                    "Exploratory sampling, not judge evaluation. Clusters need manual review; "
                    "lexical similarity does not establish functional equivalence."
                ),
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_params(
            {
                "commit": commit,
                "seed": settings.seed,
                "dataset_sha256": sha256(raw).hexdigest(),
                **config,
            }
        )
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.90, sublinear_tf=True
        )
        features = vectorizer.fit_transform(documents)
        svd = TruncatedSVD(
            n_components=min(config["components"], features.shape[0] - 1, features.shape[1] - 1),
            random_state=settings.seed,
        )
        vectors = np.asarray(normalize(svd.fit_transform(features)), dtype=np.float64)
        trials = []
        models = []
        for k in config["clusters"]:
            model = KMeans(n_clusters=k, n_init=10, random_state=settings.seed).fit(vectors)
            score = float(
                silhouette_score(
                    vectors,
                    model.labels_,
                    metric="cosine",
                    sample_size=min(config["silhouette_sample"], len(documents)),
                    random_state=settings.seed,
                )
            )
            trials.append(
                {
                    "k": k,
                    "silhouette_cosine": score,
                    "smallest_cluster": min(Counter(model.labels_).values()),
                }
            )
            models.append(model)
        best = max(range(len(trials)), key=lambda i: trials[i]["silhouette_cosine"])
        labels = models[best].labels_
        k = trials[best]["k"]
        selected: list[int] = []
        summaries = []
        names = vectorizer.get_feature_names_out()
        representatives = {index: texts.index(text) for text, index in unique.items()}
        output.mkdir(parents=True)
        candidates = []
        for cluster in range(k):
            members = np.flatnonzero(labels == cluster).tolist()
            picks = select_candidates(
                vectors,
                members,
                config["per_cluster"],
                excluded,
                config["max_similarity"],
                selected,
            )
            selected.extend(picks)
            weights = np.asarray(features[members].mean(axis=0)).ravel()
            terms = names[np.argsort(-weights, kind="stable")[:10]].tolist()
            original_members = [i for i, ref in enumerate(canonical) if labels[ref] == cluster]
            summaries.append(
                {
                    "cluster": cluster,
                    "rows": len(original_members),
                    "unique_texts": len(members),
                    "terms": terms,
                    "application_types": dict(
                        Counter(rows[i].get("application_type", "") for i in original_members)
                    ),
                    "selected_ids": [rows[representatives[i]]["id"] for i in picks],
                }
            )
            for rank, index in enumerate(picks):
                candidates.append(
                    {
                        **rows[representatives[index]],
                        "cluster": cluster,
                        "selection": "central" if rank == 0 else "diverse",
                        "functional_text": documents[index],
                    }
                )
        with (output / "membership.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "cluster", "canonical_id", "already_annotated"])
            for row, ref in zip(rows, canonical, strict=True):
                writer.writerow(
                    [row["id"], int(labels[ref]), rows[representatives[ref]]["id"], ref in excluded]
                )
        (output / "candidates.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates),
            encoding="utf-8",
        )
        report = [
            "# Кандидаты для расширения банка",
            "",
            f"MLflow run: `{active.info.run_id}`. Commit: `{commit}`.",
            "",
            "Выбор предварительный. Тематика может влиять на кластеры; "
            "разнообразие функций требует ручной проверки.",
            "",
        ]
        for summary in summaries:
            report.extend(
                [
                    f"## Кластер {summary['cluster']}",
                    "",
                    "Ключевые слова: " + ", ".join(summary["terms"]),
                    "",
                ]
            )
            for row in candidates:
                if row["cluster"] == summary["cluster"]:
                    report.extend(
                        [f"### {row['id']} ({row['selection']})", "", row["instruction"], ""]
                    )
        (output / "candidates.md").write_text("\n".join(report), encoding="utf-8")
        metrics = {
            "rows": len(rows),
            "unique_functional_texts": len(documents),
            "excluded_annotated": len(excluded),
            "selected": len(selected),
            "clusters": k,
            "silhouette_cosine": trials[best]["silhouette_cosine"],
            "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
        }
        mlflow.log_metrics(metrics)
        manifest = {
            "run_id": active.info.run_id,
            "commit": commit,
            "config": str(config_path),
            "seed": settings.seed,
            "dataset_sha256": sha256(raw).hexdigest(),
            "metrics": metrics,
            "trials": trials,
            "clusters": summaries,
        }
        (output / "run.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Raw specifications stay in ignored local output, not in Git or remote tracking stores.
        mlflow.log_artifact(str(output / "run.json"))
        print(
            json.dumps(
                {"output": str(output), **manifest["metrics"], "run_id": active.info.run_id},
                ensure_ascii=False,
            )
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
