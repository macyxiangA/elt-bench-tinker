"""Postgres episodes are fresh, writable by their agent, and isolated from each other."""

from __future__ import annotations

from dataclasses import replace

import psycopg
import pytest

from elt_rl.warehouse.postgres import PostgresWarehouse


def test_episode_lifecycle_and_isolation(warehouse: PostgresWarehouse) -> None:
    a = warehouse.create_episode("trains")
    b = warehouse.create_episode("trains")
    try:
        warehouse.execute(a, "CREATE TABLE trains.t AS SELECT 1 AS x")
        assert warehouse.fetch_df(a, "SELECT x FROM trains.t")["x"].tolist() == [1]

        # b is a fresh clone: a's table is not there.
        assert warehouse.execute(b, "SELECT to_regclass('trains.t') AS t").rows == [(None,)]

        # b's credentials cannot open a's database.
        other = dict(b.agent_env, PGDATABASE=a.id)
        with pytest.raises(psycopg.OperationalError):
            warehouse.execute(replace(b, agent_env=other), "SELECT 1")
        # Nor log in as the admin with its own credentials minus the password.
        admin = dict(b.agent_env, PGUSER="postgres", PGPASSWORD="")
        with pytest.raises(psycopg.OperationalError):
            warehouse.execute(replace(b, agent_env=admin), "SELECT 1")
        # Nor the task template.
        other = dict(b.agent_env, PGDATABASE=warehouse._template("trains"))
        with pytest.raises(psycopg.OperationalError):
            warehouse.execute(replace(b, agent_env=other), "SELECT 1")
    finally:
        warehouse.drop_episode(a)
        warehouse.drop_episode(b)

    with warehouse._admin() as conn:
        left = conn.execute(
            "SELECT count(*) FROM pg_database WHERE datname = ANY(%s)", ([a.id, b.id],)
        ).fetchone()
    assert left == (0,)
