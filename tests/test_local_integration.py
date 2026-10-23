"""Acceptance 1 (credential-free): drive the real env/tools/reward with scripted
pipelines on local Postgres. Correct -> 1, wrong -> in (0, 1), empty -> 0.

Only the model is scripted; everything else (cookbook AgentToolMessageEnv,
tools, warehouse, official comparator) is the code used in training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tinker_cookbook.renderers.base import Message, ToolCall

from elt_rl.env import EltEnvGroupBuilder
from elt_rl.reward import load_comparator
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

TRANSFORM = """
CREATE TABLE trains.trains AS
SELECT t.id AS train_id,
       t.direction,
       CASE WHEN t.direction = 'east' AND COUNT(c.id) < 4 THEN 1 ELSE 0 END
         AS run_east_and_has_less_than_4_cars,
       CASE WHEN SUM(CASE WHEN c.len = 'long' THEN 1 ELSE 0 END) > 2 THEN 1 ELSE 0 END
         AS has_more_than_2_long_cars,
       MAX(CASE WHEN c.position = 3 AND c.sides = 'double' THEN 1 ELSE 0 END)
         AS has_double_sided_cars_in_3rd_position,
       MAX(CASE WHEN c.position = 1 AND c.shape = 'rectangle' THEN 1 ELSE 0 END)
         AS has_rectangular_shaped_head_cars,
       COALESCE(SUM(CASE WHEN c.roof = '{open_roof}' THEN 1 ELSE 0 END), 0)
         AS number_of_open_roof_car
FROM trains.train t LEFT JOIN trains.cars c ON c.train_id = t.id
GROUP BY t.id, t.direction
"""


def call(name: str, **args: str) -> Message:
    tc = ToolCall(function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)))
    return {"role": "assistant", "content": "", "tool_calls": [tc]}


PIPELINES: dict[str, list[Message]] = {
    "correct": [
        call("bash", command=LOAD),
        call("sql", query=TRANSFORM.format(open_roof="none")),
        call("submit"),
    ],
    # Classic mistake (see ELT-Bench-Verified): the data has no 'open' roof value.
    "wrong": [
        call("bash", command=LOAD),
        call("sql", query=TRANSFORM.format(open_roof="open")),
        call("submit"),
    ],
    "empty": [call("submit")],
}


class PromptOnlyRenderer:
    """Stands in for a tokenizer-backed renderer; only builds the prompt prefix."""

    def create_conversation_prefix_with_tools(self, tools, system_prompt: str = ""):
        return [{"role": "system", "content": system_prompt}]

    def get_stop_sequences(self) -> list[str]:
        return []


async def run_scripted(env, script: list[Message]) -> tuple[float, dict]:
    msg_env = env.message_env
    await msg_env.initial_observation()
    for message in script:
        result = await msg_env.step(message)
        if result.episode_done:
            return result.reward, result.metrics
    raise AssertionError("script ended without finishing the episode")


@pytest.mark.asyncio
async def test_reward_separates_pipelines(
    warehouse: PostgresWarehouse, elt_bench_dir: Path
) -> None:
    task = load_task("trains", elt_bench_dir, FIXTURE_DATA)
    builder = EltEnvGroupBuilder(
        task=task,
        warehouse=warehouse,
        renderer=PromptOnlyRenderer(),  # type: ignore[arg-type]
        comparator=load_comparator(elt_bench_dir),
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
