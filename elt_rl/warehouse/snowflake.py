"""Snowflake warehouse in ELT-Bench's layout: one database per task, tables in AIRBYTE_SCHEMA.

Each episode is a zero-copy `CLONE` of the task's template database, owned by
the agent role (AIRBYTE_ROLE, the role Airbyte and the agent write with). The
grader uses the admin connection.

Episodes share the agent role, so they are separate but not mutually
isolated; run them one at a time until per-episode roles are added.
"""

from __future__ import annotations

import os
import secrets

import pandas as pd
import snowflake.connector

from elt_rl.warehouse.base import Episode, QueryResult, Warehouse

SCHEMA = "AIRBYTE_SCHEMA"


class SnowflakeWarehouse(Warehouse):
    def __init__(self, admin: dict[str, str], agent: dict[str, str]):
        """`admin`/`agent`: snowflake.connector.connect kwargs (account, user, password,
        role, warehouse). The agent role must be granted to the admin role."""
        self.admin = admin
        self.agent = agent

    @classmethod
    def from_env(cls) -> SnowflakeWarehouse:
        """Read ELT_SF_* variables (see .env.example)."""
        e = os.environ
        common = {"account": e["ELT_SF_ACCOUNT"], "warehouse": e["ELT_SF_WAREHOUSE"]}
        admin = {**common, "user": e["ELT_SF_ADMIN_USER"],
                 "password": e["ELT_SF_ADMIN_PASSWORD"], "role": "SYSADMIN"}
        agent = {**common, "user": e["ELT_SF_AGENT_USER"],
                 "password": e["ELT_SF_AGENT_PASSWORD"], "role": e["ELT_SF_AGENT_ROLE"]}
        return cls(admin, agent)

    def _run(self, conn_kwargs: dict[str, str], sql: str) -> QueryResult:
        with snowflake.connector.connect(**conn_kwargs) as conn, conn.cursor() as cur:
            cur.execute(sql)
            if cur.description is None:
                return QueryResult(columns=[], rows=[])
            return QueryResult(columns=[c.name for c in cur.description], rows=cur.fetchall())

    def _admin_many(self, statements: list[str]) -> None:
        with snowflake.connector.connect(**self.admin) as conn, conn.cursor() as cur:
            for sql in statements:
                cur.execute(sql)

    def _template(self, task: str) -> str:
        return f"ELT_TMPL_{task.upper()}"

    def template_database(self, task: str) -> str:
        """Create (if needed) the task's template and return its name.

        The agent role may write to it, because Airbyte loads the raw sources
        here once with that role; episodes then clone it.
        """
        tmpl = self._template(task)
        role = self.agent["role"]
        self._admin_many(
            [
                f"CREATE DATABASE IF NOT EXISTS {tmpl}",
                f"CREATE SCHEMA IF NOT EXISTS {tmpl}.{SCHEMA}",
                f"GRANT USAGE, CREATE SCHEMA ON DATABASE {tmpl} TO ROLE {role}",
                f"GRANT ALL ON SCHEMA {tmpl}.{SCHEMA} TO ROLE {role}",
            ]
        )
        return tmpl

    def create_episode(self, task: str) -> Episode:
        tmpl = self.template_database(task)
        db = f"ELT_EP_{secrets.token_hex(6).upper()}"
        role = self.agent["role"]
        self._admin_many(
            [
                f"CREATE DATABASE {db} CLONE {tmpl}",
                f"GRANT OWNERSHIP ON SCHEMA {db}.{SCHEMA} TO ROLE {role} COPY CURRENT GRANTS",
                f"GRANT OWNERSHIP ON DATABASE {db} TO ROLE {role} COPY CURRENT GRANTS",
            ]
        )
        agent_env = {
            "SNOWFLAKE_ACCOUNT": self.agent["account"],
            "SNOWFLAKE_USER": self.agent["user"],
            "SNOWFLAKE_PASSWORD": self.agent["password"],
            "SNOWFLAKE_ROLE": role,
            "SNOWFLAKE_WAREHOUSE": self.agent["warehouse"],
            "SNOWFLAKE_DATABASE": db,
            "SNOWFLAKE_SCHEMA": SCHEMA,
        }
        return Episode(
            id=db,
            task=task,
            agent_env=agent_env,
            namespace=f"{db}.{SCHEMA}",
            shell_hint="credentials are in the SNOWFLAKE_* environment variables.",
        )

    def drop_episode(self, episode: Episode) -> None:
        self._admin_many([f"DROP DATABASE IF EXISTS {episode.id}"])

    def execute(self, episode: Episode, sql: str) -> QueryResult:
        return self._run({**self.agent, "database": episode.id, "schema": SCHEMA}, sql)

    def fetch_df(self, episode: Episode, sql: str) -> pd.DataFrame:
        res = self._run({**self.admin, "database": episode.id, "schema": SCHEMA}, sql)
        return pd.DataFrame(res.rows, columns=res.columns)
