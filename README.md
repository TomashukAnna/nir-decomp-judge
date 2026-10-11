# Decomposition-based requirements coverage evaluation: a hybrid checklist judge and its reliability

[Русская версия](README.ru.md)

Semester research project by Anna Tomashuk.

## Problem and goal

Multi-agent LLM systems generate web applications from textual specifications. Assessing whether an application satisfies its specification requires human evaluation or an LLM judge. A holistic evaluation using a single prompt itself requires reliability validation.

The goal is to develop a hybrid checklist judge for decomposition-based requirements coverage evaluation and study its reliability, cost and robustness to bias.

## Proposed approach

1. Extract a hierarchy of requirements from the textual specification.
2. Assign a verification rule to each requirement: deterministic or LLM-based.
3. Execute the rules against the generated web application's artifacts.
4. Attach supporting evidence to every verdict.

## Research question

Is the decomposition-based hybrid judge more reliable, less expensive and more robust to bias than a holistic single-prompt evaluation and the agent-based Agent-as-a-Judge?

The methods will be compared under the same conditions. Hypotheses will be formulated and recorded before experiments. The analysis should explain where decomposition helps and where it offers no advantage.

## Expected outputs

- A repository with the judge implementation and reproducible experiment configurations.
- An annotated requirements corpus with an assessment of annotator agreement.
- A preprint with hypotheses recorded before experiments, comparisons under the same conditions, effect sizes and confidence intervals.
- A presentation of the research results.

The project is in preparation. Research experiments have not yet been conducted; metrics and the comparison protocol remain to be specified.

## Quick start

The current smoke pipeline is a template infrastructure demonstration on synthetic data, not the requirements evaluator. Windows requires Git, Python >= 3.12, uv and GNU Make. Run commands from the repository root; Git Bash is required for the template shell hooks.

```bash
uv sync --locked
uv run pre-commit install
make check
uv lock --check
```

On Windows without symlink privileges, open skills directly from `.agents/overlay/skills/<skill>/SKILL.md`. Restart Codex after installing tools so it receives the updated PATH.

On Windows, create commits with `uv run git commit` so the Git hook can find pre-commit in `.venv`, including when the project path contains Cyrillic characters.

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
