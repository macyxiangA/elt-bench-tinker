"""Execution-based reward, following ELT-Bench's two stages.

Stage 1 (EL): every raw source table exists with the expected row count.
Stage 2 (T): run the official eval SQL per model and compare with GT using
ELT-Bench's own comparator (`evaluation/eva_stage2.py`), never a re-implementation.
As in the official evaluator, stage 2 only counts when stage 1 passes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import logging
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

import pandas as pd
from tinker_cookbook.renderers.base import Message

from elt_rl.task import Task
from elt_rl.warehouse.base import Episode, QueryError, Warehouse

logger = logging.getLogger(__name__)


@cache
def load_comparator(elt_bench_dir: Path) -> ModuleType:
    path = elt_bench_dir / "evaluation" / "eva_stage2.py"
    spec = importlib.util.spec_from_file_location("elt_bench_eva_stage2", path)
    assert spec and spec.loader, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_physical(sql: str, task: str, namespace: str) -> str:
    """ELT-Bench eval SQL uses `<task>.<table>`; point it at this episode's namespace."""
    return re.sub(rf"\b{re.escape(task)}\.", f"{namespace}.", sql, flags=re.IGNORECASE)


def _as_strings(df: pd.DataFrame) -> pd.DataFrame:
    # Same CSV round-trip as the official evaluator, so both sides parse alike.
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=True)


def stage1_passes(task: Task, warehouse: Warehouse, episode: Episode) -> bool:
    for table, expected in task.source_rows.items():
        try:
            n = warehouse.fetch_df(episode, f"SELECT COUNT(*) FROM {episode.namespace}.{table}")
        except QueryError as e:
            logger.info("%s stage 1: %s missing: %s", episode.id, table, e)
            return False
        if int(n.iloc[0, 0]) != expected:
            logger.info("%s stage 1: %s has %s rows, expected %s",
                        episode.id, table, n.iloc[0, 0], expected)
            return False
    return True


def score_model(
    cmp: ModuleType, task: Task, model: str, warehouse: Warehouse, episode: Episode
) -> dict:
    """Official per-model result plus `score` = fraction of GT columns matched."""
    gt = pd.read_csv(task.gt_dir / f"{model}.csv", dtype=str, keep_default_na=True)
    try:
        df = _as_strings(warehouse.fetch_df(
            episode, to_physical(task.eval_sql[model], task.name, episode.namespace)
        ))
    except QueryError as e:  # missing table / bad columns => nothing to grade
        logger.info("%s model %s: eval SQL failed: %s", episode.id, model, e)
        return {"match": False, "score": 0.0}
    keys = task.sort_keys.get(model, [])
    res = cmp.check_corretness(cmp.sort_by_keys(gt, keys), cmp.sort_by_keys(df, keys))
    res["score"] = len(res["matched"]) / len(gt.columns)
    return res


@dataclass
class EltReward:
    """cookbook RewardFn: grades warehouse state at episode end; ignores the transcript."""

    task: Task
    warehouse: Warehouse
    episode: Episode
    elt_bench_dir: Path

    def grade(self) -> tuple[float, dict[str, float]]:
        if not stage1_passes(self.task, self.warehouse, self.episode):
            return 0.0, {"reward": 0.0, "stage1": 0.0, "official_match": 0.0,
                         "models_matched": 0.0}
        cmp = load_comparator(self.elt_bench_dir)
        results = [
            score_model(cmp, self.task, m, self.warehouse, self.episode)
            for m in self.task.models
        ]
        reward = sum(r["score"] for r in results) / len(results)
        return reward, {
            "reward": reward,
            "stage1": 1.0,
            "official_match": float(all(r["match"] for r in results)),
            "models_matched": float(sum(r["match"] for r in results)),
        }

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        return await asyncio.to_thread(self.grade)
