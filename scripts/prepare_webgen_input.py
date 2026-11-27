"""Prepare specified WebGen tasks using only the Python standard library."""

import argparse
import json
import urllib.request
from hashlib import sha256
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--all", action="store_true", help="Include every task in source order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, help="Use an existing test.jsonl instead of downloading"
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/webgen-source.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new input directory")
    if args.ids and len(set(args.ids)) != len(args.ids):
        parser.error("Task IDs must be unique")
    config = json.loads(args.source_config.read_text(encoding="utf-8"))
    if args.source is not None:
        raw = args.source.read_bytes()
    else:
        request = urllib.request.Request(config["url"], headers={"User-Agent": "nir-decomp-judge"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    canonical = raw.replace(b"\r\n", b"\n")
    if sha256(canonical).hexdigest() != config["sha256_lf"]:
        parser.error("Source checksum differs from the pinned dataset")
    records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    by_id = {row["id"]: (index, row) for index, row in enumerate(records)}
    if len(by_id) != len(records):
        parser.error("Duplicate IDs in the dataset")
    task_ids = list(by_id) if args.all else args.ids
    missing = sorted(set(task_ids) - by_id.keys())
    if missing:
        parser.error(f"Unknown task IDs: {missing}")
    selected = []
    mapping = []
    for task_id in task_ids:
        index, row = by_id[task_id]
        if not isinstance(row["instruction"], str) or not row["instruction"].strip():
            parser.error(f"Empty instruction: {task_id}")
        selected.append({"id": task_id, "instruction": row["instruction"]})
        mapping.append(
            {"webgen_id": task_id, "original_task_index": index, "selection_index": len(mapping)}
        )
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    provenance = {
        "source": config,
        "source_sha256": sha256(raw).hexdigest(),
        "source_lf_sha256": sha256(canonical).hexdigest(),
        "selected_sha256": sha256(text.encode("utf-8")).hexdigest(),
        "tasks": mapping,
        "fields_sent_to_generator": ["id", "instruction"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "test.jsonl").write_text(text, encoding="utf-8", newline="\n")
    (args.output / "selection.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(selected)} task(s): {', '.join(task_ids)}")
    print(f"Input: {args.output / 'test.jsonl'}")


if __name__ == "__main__":
    main()
