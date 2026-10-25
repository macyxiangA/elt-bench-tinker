"""tinker-cookbook glue: EnvGroupBuilder / RLDataset / RLDatasetBuilder for ELT tasks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import chz
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env

from elt_rl.reward import EltReward, load_comparator
from elt_rl.sandbox import LocalSandbox, make_workspace
from elt_rl.task import Task, load_task
from elt_rl.tools import EltTools
from elt_rl.warehouse.base import Episode, Warehouse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a data engineer. Build an ELT pipeline: load the raw source data into the "
    "warehouse, then write SQL transformations that produce the requested data models. "
    "Use the tools to inspect files, run commands and SQL, and check your results."
)


def instruction(task: Task, episode: Episode) -> str:
    ns = episode.namespace
    return (
        f"Task `{task.name}`. Your workspace has `data_model.yaml` (target models and column "
        "semantics), `schemas/` (source column descriptions) and `sources/` (raw source data, "
        f"one CSV per source table).\n\n"
        f"1. Extract & load: load every source table into `{ns}`, keeping its name "
        f"(e.g. `sources/x.csv` -> table `{ns}.x`).\n"
        f"2. Transform: create each model in `data_model.yaml` as a table `{ns}.<model>` "
        "with exactly the listed columns.\n"
        f"The shell is already connected to your warehouse: {episode.shell_hint} "
        "Call `submit` when done."
    )


class EltEnvGroupBuilder(EnvGroupBuilder):
    """One task, `group_size` independent rollouts; each gets its own namespace + workspace."""

    def __init__(
        self,
        task: Task,
        warehouse: Warehouse,
        renderer: Renderer,
        comparator: ModuleType,
        group_size: int,
        max_turns: int = 30,
        command_timeout: int = 120,
        max_trajectory_tokens: int = 32 * 1024,
    ):
        self.task = task
        self.warehouse = warehouse
        self.renderer = renderer
        self.comparator = comparator
        self.group_size = group_size
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self._episodes: list[Episode] = []
        self._sandboxes: list[LocalSandbox] = []

    def _initial_messages(self, tools: EltTools, episode: Episode) -> list[Message]:
        prefix = self.renderer.create_conversation_prefix_with_tools(
            tools=[t.to_spec() for t in tools.all()], system_prompt=SYSTEM_PROMPT
        )
        return prefix + [{"role": "user", "content": instruction(self.task, episode)}]

    async def make_envs(self) -> Sequence[Env]:
        envs = []
        for _ in range(self.group_size):
            episode = self.warehouse.create_episode(self.task.name)
            sandbox = LocalSandbox(make_workspace(self.task.agent_files), episode.agent_env)
            self._episodes.append(episode)
            self._sandboxes.append(sandbox)
            tools = EltTools(sandbox, self.warehouse, episode, self.command_timeout)
            envs.append(
                build_agent_tool_env(
                    renderer=self.renderer,
                    tools=tools.all(),
                    initial_messages=self._initial_messages(tools, episode),
                    reward_fn=EltReward(self.task, self.warehouse, episode, self.comparator),
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    # Tools share one workspace/namespace, so calls run in order.
                    rollout_config=RolloutConfig(tool_execution="sequential"),
                )
            )
        return envs

    async def cleanup(self) -> None:
        for sandbox in self._sandboxes:
            await sandbox.cleanup()
        for episode in self._episodes:
            try:
                self.warehouse.drop_episode(episode)
            except Exception as e:
                logger.warning("drop_episode(%s) failed: %s", episode.id, e)
        self._sandboxes.clear()
        self._episodes.clear()

    def logging_tags(self) -> list[str]:
        return ["elt", self.task.name]


class EltDataset(RLDataset):
    def __init__(self, builders: list[EltEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self.builders[index * self.batch_size : (index + 1) * self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


def make_warehouse(kind: str, postgres_dsn: str | None = None) -> Warehouse:
    # Imported lazily so each backend's driver is only needed when used.
    if kind == "postgres":
        from elt_rl.warehouse.postgres import PostgresWarehouse

        if not postgres_dsn:
            raise ValueError("warehouse=postgres needs postgres_dsn")
        return PostgresWarehouse(postgres_dsn)
    if kind == "snowflake":
        from elt_rl.warehouse.snowflake import SnowflakeWarehouse

        return SnowflakeWarehouse.from_env()
    raise ValueError(f"unknown warehouse {kind!r}")


@chz.chz
class EltDatasetBuilder(RLDatasetBuilder):
    task_names: list[str]
    elt_bench_dir: str
    data_dir: str
    model_name: str
    # "postgres" (needs postgres_dsn) or "snowflake" (reads ELT_SF_* env vars).
    warehouse: str = "postgres"
    postgres_dsn: str | None = None
    batch_size: int = 1
    group_size: int = 2
    renderer_name: str | None = None
    max_turns: int = 30

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer_utils.get_tokenizer(self.model_name))
        bench, data = Path(self.elt_bench_dir), Path(self.data_dir)
        warehouse = make_warehouse(self.warehouse, self.postgres_dsn)
        comparator = load_comparator(bench)
        builders = [
            EltEnvGroupBuilder(
                task=load_task(name, bench, data),
                warehouse=warehouse,
                renderer=renderer,
                comparator=comparator,
                group_size=self.group_size,
                max_turns=self.max_turns,
            )
            for name in self.task_names
        ]
        return EltDataset(builders, self.batch_size), None
