# elt-bench-tinker

ELT-Bench as an RLVR environment on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).
In one rollout the model reads an ELT-Bench task, loads the raw sources into a
warehouse, writes SQL transformations, checks them, and calls `submit`. The
reward comes from executing the official ELT-Bench evaluation SQL against the
final warehouse state and comparing it with ground truth using ELT-Bench's own
comparator.

## Layout

```
elt_rl/
  task.py              # ELT-Bench task -> agent-visible files + grader-only data
  warehouse/base.py    # Warehouse interface (episode create/drop, agent SQL, grader SQL)
  warehouse/postgres.py
  sandbox.py           # per-episode workspace; local backend of cookbook's SandboxInterface
  tools.py             # bash / sql / submit
  reward.py            # execution reward via ELT-Bench evaluation/eva_stage2.py
  env.py               # EnvGroupBuilder / RLDataset / RLDatasetBuilder
tests/
  test_postgres_warehouse.py
  test_local_integration.py   # acceptance 1: correct=1, wrong<1, empty=0
```

Adding a warehouse means implementing `elt_rl/warehouse/base.py`; nothing else changes.

## Local test (no credentials)

Needs Python >= 3.11, a Postgres 16 you can reach as an admin, and the `psql` client.

```bash
git clone -b elt_bench++ https://github.com/uiuc-kang-lab/ELT-Bench.git
git -C ELT-Bench checkout fcf3129
uv venv && uv pip install -e '.[dev]'

# any throwaway Postgres works, e.g.
docker run -d --name elt-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16

ELT_PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres pytest -v
```

The test uses a small hand-made fixture with the schema of the official
`trains` task (`tests/fixtures/trains`), so it needs no downloads.

## Isolation

Each rollout gets its own Postgres database cloned from a per-task template
and a login role that can only connect to that database. The ground truth and
evaluation SQL never enter the agent's workspace; grading runs with the admin
connection after the episode ends.
