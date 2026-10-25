"""Acceptance 2: the official ELT-Bench `trains` task end to end on Snowflake, with real credentials.

1. EL, the official way: Airbyte syncs the REST API stream `cars` (custom connector from
   setup_airbyte.py) and the Drive CSV `train` into the task's template database. Runs once;
   skipped when the template already has the expected row counts (see --reload).
2. Rollout: a scripted reference policy drives the real env/tools on a fresh episode clone.
3. Score: our EltReward, plus the unchanged official evaluator (eva.py, stage 1 + 2) on a
   `TRAINS` clone of the episode, the database name the evaluator expects.
Writes actions, tool outputs and scores to <data-dir>/trains/rollout_<utc time>.json.

Usage: set -a; source .env; set +a
       python scripts/run_official_trains.py --data-dir /path/to/elt-data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from elt_rl.airbyte import Airbyte
from elt_rl.env import EltEnvGroupBuilder
from elt_rl.reward import load_comparator
from elt_rl.scripted import TRAINS_TRANSFORM, PromptOnlyRenderer, call, run_scripted
from elt_rl.task import load_task
from elt_rl.warehouse.snowflake import SCHEMA, SnowflakeWarehouse

TASK = "trains"
# Airbyte's built-in File source and Snowflake destination (ids from ELT-Bench config.yaml).
FILE_SOURCE_DEF = "778daa7c-feaf-4db6-96f3-70fd645acc77"
SNOWFLAKE_DEST_DEF = "424892c4-daac-4491-b35d-c6688ba547ba"


def ensure(ab: Airbyte, kind: str, name: str, body: dict) -> str:
    """Id of the source/destination/connection `name`, creating it if missing."""
    ws = os.environ["AIRBYTE_WORKSPACE_ID"]
    found = ab.find(kind, name, ws)
    if found is None:
        found = ab.public("POST", f"/{kind}", {"name": name, "workspaceId": ws, **body})
    return found[kind[:-1] + "Id"]


def load_template(wh: SnowflakeWarehouse, bench: Path, expected: dict[str, int]) -> list[dict]:
    """Airbyte EL into the template database; returns the finished sync jobs."""
    tmpl = wh.template_database(TASK)
    config = yaml.safe_load((bench / "elt-bench" / "snowflake" / TASK / "config.yaml").read_text())
    ab = Airbyte.from_env()
    e = os.environ
    dest = ensure(ab, "destinations", f"elt-snowflake-{tmpl}", {
        "definitionId": SNOWFLAKE_DEST_DEF,
        "configuration": {
            "host": f"{e['ELT_SF_ACCOUNT']}.snowflakecomputing.com",
            "role": e["ELT_SF_AGENT_ROLE"],
            "warehouse": e["ELT_SF_WAREHOUSE"],
            "database": tmpl,
            "schema": SCHEMA,
            "username": e["ELT_SF_AGENT_USER"],
            "credentials": {"auth_type": "Username and Password",
                            "password": e["ELT_SF_AGENT_PASSWORD"]},
        },
    })
    sources = {
        stream: ensure(ab, "sources", f"elt-{TASK}-api", {
            "definitionId": e["AIRBYTE_API_DEFINITION_ID"], "configuration": {},
        })
        for stream in config["custom_api"]["config"]["tables"]
    }
    for f in config["flat_files"]:
        sources[f["table"]] = ensure(ab, "sources", f"elt-{TASK}-file-{f['table']}", {
            "definitionId": FILE_SOURCE_DEF,
            "configuration": {"dataset_name": f["table"], "format": f["format"], "url": f["path"],
                              "provider": {"storage": "HTTPS", "user_agent": False}},
        })
    jobs = []
    for stream, source in sources.items():
        conn = ensure(ab, "connections", f"elt-{TASK}-{stream}-to-{tmpl}", {
            "sourceId": source,
            "destinationId": dest,
            "namespaceDefinition": "destination",
            "schedule": {"scheduleType": "manual"},
            # Overwrite (official config says append) so a re-run cannot double the rows.
            "configurations": {"streams": [{"name": stream, "syncMode": "full_refresh_overwrite"}]},
        })
        print(f"syncing {stream} -> {tmpl}.{SCHEMA} ...", flush=True)
        jobs.append(ab.sync(conn))
    counts = template_counts(wh, expected)
    if counts != expected:
        raise RuntimeError(f"template row counts {counts} != expected {expected}")
    return jobs


def template_counts(wh: SnowflakeWarehouse, expected: dict[str, int]) -> dict[str, int | None]:
    tmpl = wh.template_database(TASK)
    counts: dict[str, int | None] = {}
    for table in expected:
        try:
            res = wh._run(wh.admin, f"SELECT COUNT(*) FROM {tmpl}.{SCHEMA}.{table}")
            counts[table] = int(res.rows[0][0])
        except Exception:
            counts[table] = None
    return counts


async def rollout(wh: SnowflakeWarehouse, bench: Path, data_dir: Path) -> dict:
    """One scripted episode through the real env; returns transcript + our scores.

    Leaves the episode database in place (caller grades it officially, then drops it).
    """
    task = load_task(TASK, bench, data_dir)
    builder = EltEnvGroupBuilder(task=task, warehouse=wh, renderer=PromptOnlyRenderer(),  # type: ignore[arg-type]
                                 comparator=load_comparator(bench), group_size=1)
    [env] = await builder.make_envs()
    episode = builder._episodes[0]
    ns = episode.namespace
    script = [
        # Check the EL result, then build the model, then submit.
        call("sql", query=f"SELECT 'cars' AS t, COUNT(*) AS n FROM {ns}.cars "
                          f"UNION ALL SELECT 'train', COUNT(*) FROM {ns}.train"),
        call("sql", query=TRAINS_TRANSFORM.format(ns=ns, open_roof="none")),
        call("sql", query=f"SELECT * FROM {ns}.trains ORDER BY train_id LIMIT 5"),
        call("submit"),
    ]
    reward, metrics = await run_scripted(env, script)
    history = env.message_env.history
    for sandbox in builder._sandboxes:
        await sandbox.cleanup()
    return {"episode": episode.id, "reward": reward, "metrics": metrics,
            "transcript": json.loads(json.dumps(history, default=_jsonable))}


def _jsonable(o: object) -> object:
    return o.model_dump() if hasattr(o, "model_dump") else str(o)


def official_eval(wh: SnowflakeWarehouse, bench: Path, data_dir: Path, episode_db: str) -> dict:
    """Run ELT-Bench's eva.py unchanged on a `TRAINS` clone of the episode."""
    run = f"elt_rl_{TASK}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    wh._admin_many([f"CREATE DATABASE {TASK.upper()} CLONE {episode_db}"])  # fails if it exists
    try:
        with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
            cred = Path(tmp) / "snowflake_credential.json"
            cred.write_text(json.dumps(wh.admin))
            cred.chmod(0o600)
            gt_root = Path(tmp) / "gt"
            gt_root.mkdir()
            (gt_root / TASK).symlink_to(data_dir / TASK / "gt")
            subprocess.run(
                [sys.executable, "eva.py", "--folder", run, "--only", TASK,
                 "--credential", str(cred), "--gt-folder", str(gt_root)],
                cwd=bench / "evaluation", check=True,
            )
    finally:
        wh._admin_many([f"DROP DATABASE IF EXISTS {TASK.upper()}"])
    out = bench / "evaluation" / "agent_results" / run
    return {"folder": str(out),
            "stage1": (out / "results.log").read_text().strip().splitlines(),
            "stage2": json.loads((out / "stage2.json").read_text())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", type=Path, required=True,
                   help="has trains/sources/*.csv and trains/gt/trains.csv")
    p.add_argument("--elt-bench-dir", type=Path,
                   default=Path(os.environ.get("ELT_BENCH_DIR") or "ELT-Bench"))
    p.add_argument("--reload", action="store_true", help="re-run the Airbyte syncs")
    args = p.parse_args()
    bench, data_dir = args.elt_bench_dir.resolve(), args.data_dir.resolve()
    expected = json.loads((bench / "evaluation" / "table.json").read_text())[TASK]

    wh = SnowflakeWarehouse.from_env()
    record: dict = {"task": TASK, "started": datetime.now(timezone.utc).isoformat()}
    if args.reload or template_counts(wh, expected) != expected:
        record["airbyte_jobs"] = load_template(wh, bench, expected)
    record["template_counts"] = template_counts(wh, expected)

    result = asyncio.run(rollout(wh, bench, data_dir))
    record.update(result)
    episode = result["episode"]
    try:
        record["official"] = official_eval(wh, bench, data_dir, episode)
    finally:
        wh._admin_many([f"DROP DATABASE IF EXISTS {episode}"])

    out = data_dir / TASK / f"rollout_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    official = record["official"]["stage2"].get(TASK, {}).get(TASK, {})
    print(f"our reward={record['reward']} metrics={record['metrics']}")
    print(f"official stage1={record['official']['stage1']}")
    print(f"official stage2 match={official.get('match')}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
