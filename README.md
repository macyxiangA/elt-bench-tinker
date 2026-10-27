# elt-bench-tinker

ELT-Bench as an RLVR environment on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).
In one rollout the model reads an ELT-Bench task, makes sure the raw sources are
in the warehouse, writes SQL transformations, checks them, and calls `submit`.
The reward comes from the warehouse state at the end, graded the way ELT-Bench
grades it:

- Stage 1 (EL): every raw source table exists with the expected row count.
  If not, the reward is 0, as in the official evaluator.
- Stage 2 (T): the official evaluation SQL runs against each target model and
  ELT-Bench's own comparator (`evaluation/eva_stage2.py`) compares it with the
  ground truth. The reward is the fraction of ground-truth columns matched,
  averaged over models. `official_match` is 1 only when every model matches.

## Layout

```
elt_rl/
  task.py              # ELT-Bench task -> agent-visible files + grader-only data
  warehouse/base.py    # Warehouse interface (episode create/drop, agent SQL, grader SQL)
  warehouse/postgres.py
  warehouse/snowflake.py
  sandbox.py           # the agent's shell: one Docker container per episode
  tools.py             # bash / sql / submit
  reward.py            # stage 1 row counts + stage 2 official comparator
  env.py               # EnvGroupBuilder / RLDataset / RLDatasetBuilder
  airbyte.py           # small Airbyte API client for the official EL step
  scripted.py          # replays fixed tool calls through the real env (tests, acceptance 2)
sandbox/Dockerfile     # image for the agent's shell (psql, Python, Snowflake connector)
scripts/
  setup_airbyte.py         # registers ELT-Bench's REST API connector in Airbyte
  run_official_trains.py   # acceptance 2 on Snowflake
  check_snowflake.py       # Snowflake episode isolation smoke test
tests/                     # acceptance 1 on local Postgres
train.py                   # acceptance 3, RL with tinker-cookbook
```

Adding a warehouse means implementing `elt_rl/warehouse/base.py`. Nothing else changes.

## Isolation

- The agent's shell is a Docker container that only has a copy of the task's
  agent-visible files (`data_model.yaml`, `schemas/`, `sources/`). Ground truth,
  evaluation SQL and host credentials never enter it. With Snowflake it uses
  Docker's default bridge network. With the local Postgres test it uses the
  host network, so the Postgres admin login must require a password.
- Each rollout gets its own warehouse namespace cloned from a per-task template,
  with its own throwaway login that can reach only that namespace. Grading uses
  the admin connection after the episode ends.
  - Postgres: one database and one role per episode. The template database does
    not accept connections.
  - Snowflake: a zero-copy `CLONE` of the template database per episode, owned
    by a new role and user. The admin user needs `SYSADMIN` and `USERADMIN`.

## Setup

Needs Python >= 3.11, Docker, and an ELT-Bench checkout.

```bash
git clone -b elt_bench++ https://github.com/uiuc-kang-lab/ELT-Bench.git
git -C ELT-Bench checkout fcf3129
uv venv && uv pip install -e '.[dev,snowflake]'
docker build -t elt-sandbox sandbox
cp .env.example .env   # fill in what you need
```

## Acceptance 1: local test, no credentials

Needs a Postgres 16 you can reach as an admin.

```bash
docker run -d --name elt-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16
ELT_PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres pytest -v
```

It uses a small hand-made fixture with the schema of the official `trains`
task (`tests/fixtures/trains`) and scripted pipelines instead of a model.
Expected rewards are 1.0 for the correct pipeline, about 0.86 for a common
wrong one, 0 for doing nothing, and 0 for a correct transform on an incomplete
load.

## Acceptance 2: official `trains` task on Snowflake

Needs a Snowflake account set up with ELT-Bench's `setup/destination/setup.sql`,
an admin user with `SYSADMIN` and `USERADMIN` (`SYSADMIN` must be able to grant
usage on `AIRBYTE_WAREHOUSE`, which it can when it created the warehouse), and Airbyte (`abctl`) with
ELT-Bench's REST API source running. The data dir holds the official
`trains/sources/*.csv` and `trains/gt/trains.csv`.

```bash
set -a; source .env; set +a
python scripts/setup_airbyte.py          # once, prints the Airbyte ids for .env
python scripts/run_official_trains.py --data-dir /path/to/elt-data
```

Airbyte loads the sources into the template database once. A scripted
reference pipeline then runs through the real env on a fresh clone. The
result is graded by this reward and by ELT-Bench's unchanged `eva.py`. Both
give a full match (stage 1 success, stage 2 7/7 columns, reward 1.0).

## Acceptance 3: one RL step on Tinker

```bash
set -a; source .env; set +a   # needs TINKER_API_KEY
python train.py data_dir=/path/to/elt-data dry_run=True   # envs and prompts only
python train.py data_dir=/path/to/elt-data                # one step, Qwen3-8B, 2 rollouts
```
