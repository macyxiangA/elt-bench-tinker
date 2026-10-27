"""Acceptance 1 (credential-free): drive the real env/tools/reward with scripted
pipelines on local Postgres. Correct -> 1, wrong -> in (0, 1), empty -> 0, and a
correct transform on an incomplete load -> 0 (stage 1 gate).

Only the model is scripted; everything else (cookbook AgentToolMessageEnv,
Docker sandbox, tools, warehouse, official comparator) is the code used in training.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tinker_cookbook.renderers.base import Message

from elt_rl.env import EltEnvGroupBuilder
from elt_rl.sandbox import DockerSandbox
from elt_rl.scripted import TRAINS_TRANSFORM, PromptOnlyRenderer, call, run_scripted
from elt_rl.task import load_task
from elt_rl.warehouse.postgres import PostgresWarehouse

FIXTURE_DATA = Path(__file__).parent / "fixtures"

LOAD = """
psql -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE trains.train (id int, direction text);
CREATE TABLE trains.cars (id int, train_id int, position int, shape text, len text,
  sides text, roof text, wheels int, load_shape text, load_num int);
\\copy trains.train FROM 'sources/train.csv' CSV HEADER
\\copy trains.cars FROM 'sources/cars.csv' CSV HEADER
SQL
"""
# Same load, but one `cars` row is lost on the way.
SHORT_LOAD = LOAD.replace(
    "FROM 'sources/cars.csv'", "FROM PROGRAM 'head -n -1 sources/cars.csv'"
)

PIPELINES: dict[str, list[Message]] = {
    "correct": [
        call("bash", command=LOAD),
        call("sql", query=TRAINS_TRANSFORM.format(ns="trains", open_roof="none")),
        call("submit"),
    ],
    # Classic mistake (see ELT-Bench-Verified): the data has no 'open' roof value.
    "wrong": [
        call("bash", command=LOAD),
        call("sql", query=TRAINS_TRANSFORM.format(ns="trains", open_roof="open")),
        call("submit"),
    ],
    "empty": [call("submit")],
    "short_load": [
        call("bash", command=SHORT_LOAD),
        call("sql", query=TRAINS_TRANSFORM.format(ns="trains", open_roof="none")),
        call("submit"),
    ],
}


@pytest.mark.asyncio
async def test_reward_separates_pipelines(
    warehouse: PostgresWarehouse, elt_bench_dir: Path
) -> None:
    task = load_task("trains", elt_bench_dir, FIXTURE_DATA)
    builder = EltEnvGroupBuilder(
        task=task,
        warehouse=warehouse,
        renderer=PromptOnlyRenderer(),  # type: ignore[arg-type]
        elt_bench_dir=elt_bench_dir,
        group_size=len(PIPELINES),
    )
    envs = await builder.make_envs()
    try:
        rewards = {}
        for env, (name, script) in zip(envs, PIPELINES.items()):
            rewards[name], metrics = await run_scripted(env, script)
            print(name, rewards[name], metrics)
    finally:
        await builder.cleanup()

    assert rewards["correct"] == 1.0
    assert 0.0 < rewards["wrong"] < 1.0
    assert rewards["empty"] == 0.0
    assert rewards["short_load"] == 0.0


def test_grader_files_stay_out_of_workspace(elt_bench_dir: Path) -> None:
    task = load_task("trains", elt_bench_dir, FIXTURE_DATA)
    visible = set(task.agent_files)
    assert visible == {
        "data_model.yaml",
        "schemas/cars.csv",
        "schemas/train.csv",
        "sources/cars.csv",
        "sources/train.csv",
    }
    assert not any("gt" in p or p.endswith(".sql") for p in visible)


@pytest.mark.asyncio
async def test_sandbox_sees_only_workspace(elt_bench_dir: Path) -> None:
    task = load_task("trains", elt_bench_dir, FIXTURE_DATA)
    sandbox = await DockerSandbox.create(task.agent_files, {}, "host")
    try:
        r = await sandbox.run_command("find . -type f | sort; whoami")
        assert r.stdout.split() == [f"./{p}" for p in sorted(task.agent_files)] + ["agent"]
        r = await sandbox.run_command(f"ls {task.gt_dir.resolve()}")
        assert r.exit_code != 0
        r = await sandbox.run_command("sleep 5", timeout=1)
        assert r.exit_code == 124
    finally:
        await sandbox.cleanup()
