"""Run one experiment: `uv run python -m decomp_judge --config configs/smoke.yaml`."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from decomp_judge.config import load_settings
from decomp_judge.pipeline import ExperimentConfig, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="decomp_judge", description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment config, YAML")
    args = parser.parse_args(argv)

    load_dotenv()  # MLflow and boto read the S3 keys from the environment, not from Settings
    settings = load_settings()
    cfg = ExperimentConfig.from_yaml(args.config)
    result = run_experiment(cfg, settings, config_path=args.config)
    print(f"run_id={result.run_id} mse={result.mse:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
