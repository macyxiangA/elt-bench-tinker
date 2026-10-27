"""Smoke test for the Snowflake warehouse: two episodes from the loaded `trains`
template, each writable by its own agent, invisible to the other one, and read by
the grader. Usage: set -a; source .env; set +a; python scripts/check_snowflake.py"""

from elt_rl.warehouse.base import QueryError
from elt_rl.warehouse.snowflake import SnowflakeWarehouse


def main() -> None:
    wh = SnowflakeWarehouse.from_env()
    a = wh.create_episode("trains")
    b = wh.create_episode("trains")
    try:
        print("source tables:", wh.execute(a, f"SELECT COUNT(*) FROM {a.namespace}.CARS").rows)
        wh.execute(a, f"CREATE TABLE {a.namespace}.T AS SELECT 1 AS X")
        assert wh.fetch_df(a, f"SELECT X FROM {a.namespace}.T")["X"].tolist() == [1]
        for target in (a.namespace, wh.template_name("trains") + ".AIRBYTE_SCHEMA"):
            try:
                wh.execute(b, f"SELECT COUNT(*) FROM {target}.CARS")
            except QueryError:
                continue
            raise AssertionError(f"episode {b.id} can read {target}")
        print(f"OK: {a.id} written by its agent and read by the grader; {b.id} cannot see it")
    finally:
        wh.drop_episode(a)
        wh.drop_episode(b)
        print(f"dropped {a.id} and {b.id}")


if __name__ == "__main__":
    main()
