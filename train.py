"""Acceptance 3: RL on ELT-Bench tasks with tinker-cookbook's rl/train.

Default: one optimizer step on `trains`, one group of 2 rollouts, Snowflake warehouse.
Needs TINKER_API_KEY plus the warehouse's env vars (.env.example).

  set -a; source .env; set +a
  python train.py data_dir=/path/to/elt-data            # one RL step
  python train.py data_dir=/path/to/elt-data dry_run=True  # build envs + prompts, no Tinker calls
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.rl.train import Config, main

from elt_rl.env import EltDatasetBuilder


@chz.chz
class CLIConfig:
    data_dir: str
    elt_bench_dir: str = os.environ.get("ELT_BENCH_DIR") or "ELT-Bench"
    tasks: str = "trains"  # comma-separated
    warehouse: str = "snowflake"
    postgres_dsn: str | None = os.environ.get("ELT_PG_DSN") or None

    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = None
    lora_rank: int = 32
    max_tokens: int = 2048  # per model turn
    max_turns: int = 30

    group_size: int = 2  # >= 2, or the group-centered advantage is always 0
    groups_per_batch: int = 1
    learning_rate: float = 1e-5
    max_steps: int | None = 1

    log_path: str | None = None  # default: <data_dir>/runs/<run name>
    dry_run: bool = False


def dataset_builder(cfg: CLIConfig) -> EltDatasetBuilder:
    return EltDatasetBuilder(
        task_names=[t.strip() for t in cfg.tasks.split(",") if t.strip()],
        elt_bench_dir=cfg.elt_bench_dir,
        data_dir=cfg.data_dir,
        model_name=cfg.model_name,
        warehouse=cfg.warehouse,
        postgres_dsn=cfg.postgres_dsn,
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        renderer_name=cfg.renderer_name,
        max_turns=cfg.max_turns,
    )


async def dry_run(cfg: CLIConfig) -> None:
    """Everything up to the first Tinker call: tokenizer, renderer, episodes, prompts."""
    dataset, _ = await dataset_builder(cfg)()
    for builder in dataset.get_batch(0):
        envs = await builder.make_envs()
        try:
            for env in envs:
                obs, _ = await env.initial_observation()
                print(f"{builder.logging_tags()}: prompt has {obs.length} tokens")
        finally:
            await builder.cleanup()


async def train(cfg: CLIConfig) -> None:
    run = f"elt-{cfg.model_name.replace('/', '-')}-{datetime.now():%Y%m%d-%H%M%S}"
    log_path = cfg.log_path or str(Path(cfg.data_dir) / "runs" / run)
    cli_utils.check_log_dir(log_path, behavior_if_exists="raise")
    await main(Config(
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset_builder(cfg),
        model_name=cfg.model_name,
        recipe_name="recipe_elt_bench",
        renderer_name=cfg.renderer_name,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        log_path=log_path,
        max_steps=cfg.max_steps,
        eval_every=0,
        save_every=0,
    ))


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(dry_run(cfg) if cfg.dry_run else train(cfg))
