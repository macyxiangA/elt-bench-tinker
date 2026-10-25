"""Smoke test for the Snowflake warehouse: clone an episode, write as the agent,
read as the grader, drop. Usage: set -a; source .env; set +a; python scripts/check_snowflake.py"""

from elt_rl.warehouse.snowflake import SnowflakeWarehouse


def main() -> None:
    wh = SnowflakeWarehouse.from_env()
    ep = wh.create_episode("trains")
    try:
        wh.execute(ep, f"CREATE TABLE {ep.namespace}.T AS SELECT 1 AS X")
        rows = wh.fetch_df(ep, f"SELECT X FROM {ep.namespace}.T")["X"].tolist()
        assert rows == [1], rows
        print(f"OK: episode {ep.id} created, written by agent, read by grader")
    finally:
        wh.drop_episode(ep)
        print(f"dropped {ep.id}")


if __name__ == "__main__":
    main()
