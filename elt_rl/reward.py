"""Execution-based reward: run the official eval SQL, compare with GT using
ELT-Bench's own comparator (`evaluation/eva_stage2.py`), never a re-implementation.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd
from tinker_cookbook.renderers.base import Message

from elt_rl.task import Task
from elt_rl.warehouse.base import Episode, Warehouse


def load_comparator(elt_bench_dir: Path) -> ModuleType:
    path = elt_bench_dir / "evaluation" / "eva_stage2.py"
    spec = importlib.util.spec_from_file_location("elt_bench_eva_stage2", path)
    assert spec and spec.loader, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_strings(df: pd.DataFrame) -> pd.DataFrame:
    # Same CSV round-trip as the official evaluator, so both sides parse alike.
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=True)


def score_model(
    cmp: ModuleType, task: Task, model: str, warehouse: Warehouse, episode: Episode
) -> dict:
    """Official per-model result plus `score` = fraction of GT columns matched."""
    gt = pd.read_csv(task.gt_dir / f"{model}.csv", dtype=str, keep_default_na=True)
    try:
        df = _as_strings(warehouse.fetch_df(episode, task.eval_sql[model]))
    except Exception as e:  # missing table / bad SQL => nothing to grade
        return {"status": "error", "error": str(e), "match": False, "score": 0.0}
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
    comparator: ModuleType

    def grade(self) -> tuple[float, dict[str, float]]:
        results = {
            m: score_model(self.comparator, self.task, m, self.warehouse, self.episode)
            for m in self.task.models
        }
        reward = sum(r["score"] for r in results.values()) / len(results)
        metrics = {
            "reward": reward,
            "official_match": float(all(r["match"] for r in results.values())),
            "models_matched": float(sum(r["match"] for r in results.values())),
        }
        return reward, metrics

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        return await asyncio.to_thread(self.grade)
