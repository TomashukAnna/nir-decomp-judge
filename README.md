# <Project name>

[Русская версия](README.ru.md)

Semester research project (NIR) around the mas-science project. The repository was created from a [template](https://github.com/AnatolyPershinov/nir-mas-template) built on the [lab template](https://github.com/Industrial-AI-Research-Lab/nir-project-template). Replace this paragraph with the purpose of the project: the question, the data, the expected result.

## Why this template

The template makes the work run, from day one, the way it will later be reviewed and defended.
It does four things.

- **The work is visible.** A semester project is assessed by the repository, not by what is said at a meeting: pull requests show what was done and why, the commit history shows that the work went on steadily. The PR template, the experiment form and the decision log are already in `.github/` and `docs/`.
- **The result can be repeated.** Every number in the report leads to an MLflow run, and the run to a commit, a config and a seed. `uv.lock` pins the environment, and the supervisor runs the project from a clean clone with the commands in this README. Reproducibility is a requirement, and it cannot be added at the end of the semester: either it is kept from the start or it does not exist.
- **Mistakes are caught before review.** One command, `make check`, runs before every commit, in CI and in the LLM assistant's hook. Secrets, large files, failing tests and wrong branch names are stopped by a machine, so review time goes to the substance.
- **Nothing to set up.** The layout, the linter, type checks, tests, CI and the rules for LLM assistants are assembled and tested. The week that setup would take stays for research.

The template does not set the topic, the architecture or the method: that is your work. It is the same for every student of the project, so the supervisor opens any repository and knows where things are.

## First run after "Use this template"

Do this once, in the first week. Remove this section and "Why this template" in your first pull request: your README keeps the description of your project.

1. Create the repository from the template in your personal account: the **Use this template** button, name `nir-<topic>`. A public repository is recommended: only there GitHub enforces `main` protection and automatic review requests for free.
2. Clone with submodules: `git clone --recurse-submodules git@github.com:<you>/<repository>.git`. If `.agents/overlay` is empty, run `git submodule update --init`.
3. Create the branch `chore/project-setup` and rename the package: `make rename NAME=<your_package>`, then `uv lock`.
4. Set up the environment with the commands from "Quick start" and make sure `make check` passes.
5. Give the supervisor access: Settings → Collaborators → `AnatolyPershinov`, Write permission. He is already listed in `.github/CODEOWNERS`.
6. Open the first pull request `chore: set up the project`: the package rename and the project description in this README.

How to work from then on: [git and GitHub rules](docs/git-workflow.ru.md) (in Russian). Read them before the first commit.

## Quick start

```bash
cp .env.example .env          # fill in the keys; .env is ignored
uv sync                       # installs the package and the dev tools from uv.lock
uvx pre-commit install        # gitleaks, large files, make check before every commit
make check                    # ruff, mypy, pytest
make run                      # one experiment: configs/smoke.yaml -> an MLflow run
make mlflow                   # MLflow UI over the local mlflow.db
```

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

Data is not committed, open benchmarks included. Where the data lives and how to fetch it: <fill in>.

## Experiments

Every run is recorded in MLflow (`MLFLOW_TRACKING_URI` in `.env`, the local `mlflow.db` by default). A results row is commit, config path, seed, run id, metrics.

| commit | config | seed | run id | mse |
|---|---|---|---|---|

## Checks

`make check` runs from three places: pre-commit on your machine, CI on every PR, the agent hook. Branch `<type>/<short-description>`, PR title `<type>: ...`; the types are feat, fix, refactor, docs, test, chore, exp. Details: [git and GitHub rules](docs/git-workflow.ru.md).

## LLM assistants

Which agents you use and for what: <fill in>. Their instructions: [AGENTS.md](AGENTS.md); the shared lab rules are already connected as the submodule `.agents/overlay` pinned to v0.1.0 ([nir-agent-overlay](https://github.com/Industrial-AI-Research-Lab/nir-agent-overlay)); `make overlay OVERLAY_VERSION=<tag>` connects another version. You are responsible for code written with an agent.

Claude Code and Codex are both wired up: Claude reads `CLAUDE.md`, `.claude/rules/`, `.claude/skills/` and the hook in `.claude/settings.json`; Codex reads `AGENTS.md`, `.agents/skills/` and the hook in `.codex/hooks.json` (trust the project in Codex once, so its `.codex/` layer loads). The skill folders are links into the overlay, so both tools see the same skills. Any other assistant that reads `AGENTS.md` gets the rules by the pointer in its first lines.

## Guides

[Practice guides](https://github.com/Industrial-AI-Research-Lab/project-implementation-manual/blob/feat/practice-guides/nir-requirements/recommendations/README.md) of the lab manual: papers, repository and code, task tracking, agent artifacts.

## Licence

The template is MIT. Choose the licence of your own code and replace LICENSE if needed.
