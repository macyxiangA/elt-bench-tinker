"""Load one ELT-Bench task: what the agent sees, and what the grader needs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Task:
    name: str
    models: list[str]
    # Workspace-relative path -> host file. Everything here is visible to the agent.
    agent_files: dict[str, Path]
    # Grader-only: never copied into the workspace.
    eval_sql: dict[str, str]
    sort_keys: dict[str, list[str]]
    gt_dir: Path


def load_task(name: str, elt_bench_dir: Path, data_dir: Path) -> Task:
    """Read a task from an ELT-Bench checkout plus a data dir.

    `data_dir/<name>/sources/*.csv` holds the raw source data and
    `data_dir/<name>/gt/<model>.csv` the ground truth.
    """
    bench = elt_bench_dir / "elt-bench"
    data_model = bench / "snowflake" / name / "data_model.yaml"
    models = [m["name"] for m in yaml.safe_load(data_model.read_text())["models"]]

    agent_files = {"data_model.yaml": data_model}
    for f in sorted((bench / "schemas" / name).glob("*.csv")):
        agent_files[f"schemas/{f.name}"] = f
    for f in sorted((data_dir / name / "sources").glob("*.csv")):
        agent_files[f"sources/{f.name}"] = f

    sql_dir = elt_bench_dir / "evaluation" / "sql" / name
    eval_sql = {m: (sql_dir / f"{m}.sql").read_text() for m in models}
    all_keys = json.loads((elt_bench_dir / "evaluation" / "sort_key.json").read_text())
    return Task(
        name=name,
        models=models,
        agent_files=agent_files,
        eval_sql=eval_sql,
        sort_keys=all_keys.get(name, {}),
        gt_dir=data_dir / name / "gt",
    )
