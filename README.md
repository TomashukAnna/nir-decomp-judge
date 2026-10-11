# Decomposition-based requirements coverage evaluation: a hybrid checklist judge and its reliability

[Русская версия](README.ru.md)

Semester research project by Anna Tomashuk.

The preliminary goal is to develop and study requirements coverage evaluation for automatically generated software. The proposed approach decomposes requirements into checkable checklist items and combines automated checks with an LLM judge. The checks, metrics and reliability evaluation protocol will be specified during the research and agreed with the supervisor.

Research question: how reliably does a hybrid checklist judge evaluate individual requirements and overall coverage?

Expected outputs: an evaluator prototype, a reproducible experimental protocol and an analysis of evaluation reliability. No research results are available yet.

## Quick start

The current smoke pipeline is a template infrastructure demonstration on synthetic data, not the requirements evaluator. Windows requires Git, Python >= 3.12, uv and GNU Make. Run commands from the repository root; Git Bash is required for the template shell hooks.

```bash
uv sync --locked
uv run pre-commit install
make check
uv lock --check
```

On Windows without symlink privileges, open skills directly from `.agents/overlay/skills/<skill>/SKILL.md`. Restart Codex after installing tools so it receives the updated PATH.

## Layout

```
src/<package>/       code that is imported and tested: config.py, pipeline.py, __main__.py
tests/               smoke test: the pipeline on a tiny input
configs/             experiment configs (YAML)
notebooks/           exploration; outputs may stay as a report of a result
docs/adr/            decision log, one file per decision
docs/meetings/       meeting records
docs/reading-log.md  papers read
data/                ignored; how to fetch the data is described below
results/             tables and figures exported from code
.github/             PR template, issue form for experiments, CODEOWNERS, CI
```

## Data

Data is not committed, open benchmarks included. WebGen-Bench is a candidate dataset. Its version, source and acquisition procedure will be documented after dataset selection. Existing local data is outside the repository in the parent folder `data WebGen-Bench`.

## Experiments

Every run is recorded in MLflow (`MLFLOW_TRACKING_URI` in `.env`, the local `mlflow.db` by default). A results row is commit, config path, seed, run id, metrics.

| commit | config | seed | run id | metrics |
|---|---|---|---|---|

## Checks

`make check` runs from three places: pre-commit on your machine, CI on every PR, the agent hook. Branch `<type>/<short-description>`, PR title `<type>: ...`; the types are feat, fix, refactor, docs, test, chore, exp. Details: [git and GitHub rules](docs/git-workflow.ru.md).

## LLM assistants

Codex assists with repository setup, code, checks and documentation. The author reviews changes and is responsible for scientific conclusions. Their instructions: [AGENTS.md](AGENTS.md); the shared lab rules are already connected as the submodule `.agents/overlay` pinned to v0.1.0 ([nir-agent-overlay](https://github.com/Industrial-AI-Research-Lab/nir-agent-overlay)); `make overlay OVERLAY_VERSION=<tag>` connects another version. You are responsible for code written with an agent.

Claude Code and Codex are both wired up: Claude reads `CLAUDE.md`, `.claude/rules/`, `.claude/skills/` and the hook in `.claude/settings.json`; Codex reads `AGENTS.md`, `.agents/skills/` and the hook in `.codex/hooks.json` (trust the project in Codex once, so its `.codex/` layer loads). The skill folders are links into the overlay, so both tools see the same skills. Any other assistant that reads `AGENTS.md` gets the rules by the pointer in its first lines.

## Guides

[Practice guides](https://github.com/Industrial-AI-Research-Lab/project-implementation-manual/blob/feat/practice-guides/nir-requirements/recommendations/README.md) of the lab manual: papers, repository and code, task tracking, agent artifacts.

## Licence

The template is MIT. Choose the licence of your own code and replace LICENSE if needed.
