"""Snowflake warehouse in ELT-Bench's layout: one database per task, tables in AIRBYTE_SCHEMA.

Airbyte loads the raw sources once into the task's template database (with
AIRBYTE_ROLE, as in ELT-Bench). Each episode is a zero-copy `CLONE` of it,
owned by a throwaway role and user that can reach only that clone. The grader
uses the admin connection. The admin user needs SYSADMIN and USERADMIN.
"""

from __future__ import annotations

import os
import secrets

import pandas as pd
import snowflake.connector

from elt_rl.warehouse.base import Episode, QueryError, QueryResult, Warehouse

SCHEMA = "AIRBYTE_SCHEMA"


class SnowflakeWarehouse(Warehouse):
    sources_preloaded = True
    sandbox_network = "bridge"

    def __init__(self, account: str, warehouse: str, admin_user: str, admin_password: str,
                 loader_role: str, statement_timeout_s: int = 300):
        """`loader_role` is the role Airbyte writes the template with (AIRBYTE_ROLE)."""
        self.statement_timeout_s = statement_timeout_s
        self.account = account
        self.warehouse = warehouse
        self.admin_user = admin_user
        self.admin_password = admin_password
        self.loader_role = loader_role

    @classmethod
    def from_env(cls) -> SnowflakeWarehouse:
        """Read ELT_SF_* variables (see .env.example)."""
        e = os.environ
        return cls(e["ELT_SF_ACCOUNT"], e["ELT_SF_WAREHOUSE"], e["ELT_SF_ADMIN_USER"],
                   e["ELT_SF_ADMIN_PASSWORD"], e["ELT_SF_AIRBYTE_ROLE"])

    def admin_connect_args(self, role: str = "SYSADMIN") -> dict[str, str]:
        return {"account": self.account, "warehouse": self.warehouse,
                "user": self.admin_user, "password": self.admin_password, "role": role}

    def _run(self, conn_kwargs: dict[str, str], statements: list[str]) -> QueryResult:
        """Run statements in order on one connection; return the last one's result."""
        res = QueryResult(columns=[], rows=[])
        with snowflake.connector.connect(**conn_kwargs) as conn, conn.cursor() as cur:
            for sql in statements:
                try:
                    cur.execute(sql)
                except snowflake.connector.errors.ProgrammingError as e:
                    raise QueryError(str(e)) from e
                if cur.description is not None:
                    res = QueryResult([c.name for c in cur.description], cur.fetchall())
        return res

    def admin_run(self, *statements: str, role: str = "SYSADMIN") -> QueryResult:
        return self._run(self.admin_connect_args(role), list(statements))

    def template_name(self, task: str) -> str:
        return f"ELT_TMPL_{task.upper()}"

    def ensure_template(self, task: str) -> str:
        """Create the task's template database for Airbyte to load into; return its name."""
        tmpl = self.template_name(task)
        self.admin_run(
            f"CREATE DATABASE IF NOT EXISTS {tmpl}",
            f"CREATE SCHEMA IF NOT EXISTS {tmpl}.{SCHEMA}",
            f"GRANT USAGE ON DATABASE {tmpl} TO ROLE {self.loader_role}",
            f"GRANT ALL ON SCHEMA {tmpl}.{SCHEMA} TO ROLE {self.loader_role}",
        )
        return tmpl

    def create_episode(self, task: str) -> Episode:
        tmpl = self.template_name(task)
        try:
            n = self.admin_run(
                f"SELECT COUNT(*) FROM {tmpl}.INFORMATION_SCHEMA.TABLES "
                f"WHERE TABLE_SCHEMA = '{SCHEMA}'"
            ).rows[0][0]
        except QueryError:
            n = 0
        if not n:
            raise RuntimeError(f"{tmpl} has no source tables; load it first "
                               "(scripts/run_official_trains.py does this for trains)")
        ep = f"ELT_EP_{secrets.token_hex(6).upper()}"
        # Snowflake's default password policy wants a digit and both cases.
        password = secrets.token_urlsafe(24) + "aA1"
        try:
            self.admin_run(
                f"CREATE ROLE {ep}",
                f"GRANT ROLE {ep} TO ROLE SYSADMIN",
                f"CREATE USER {ep} PASSWORD = '{password}' DEFAULT_ROLE = {ep} "
                f"DEFAULT_WAREHOUSE = {self.warehouse}",
                f"ALTER USER {ep} SET STATEMENT_TIMEOUT_IN_SECONDS = {self.statement_timeout_s}",
                f"GRANT ROLE {ep} TO USER {ep}",
                role="USERADMIN",
            )
            self.admin_run(
                f"CREATE DATABASE {ep} CLONE {tmpl}",
                f"GRANT OWNERSHIP ON ALL TABLES IN SCHEMA {ep}.{SCHEMA} TO ROLE {ep} "
                "COPY CURRENT GRANTS",
                f"GRANT OWNERSHIP ON SCHEMA {ep}.{SCHEMA} TO ROLE {ep} COPY CURRENT GRANTS",
                f"GRANT OWNERSHIP ON DATABASE {ep} TO ROLE {ep} COPY CURRENT GRANTS",
                f"GRANT USAGE ON WAREHOUSE {self.warehouse} TO ROLE {ep}",
            )
        except Exception:
            self._drop(ep)
            raise
        agent_env = {
            "SNOWFLAKE_ACCOUNT": self.account,
            "SNOWFLAKE_USER": ep,
            "SNOWFLAKE_PASSWORD": password,
            "SNOWFLAKE_ROLE": ep,
            "SNOWFLAKE_WAREHOUSE": self.warehouse,
            "SNOWFLAKE_DATABASE": ep,
            "SNOWFLAKE_SCHEMA": SCHEMA,
        }
        return Episode(
            id=ep,
            task=task,
            agent_env=agent_env,
            namespace=f"{ep}.{SCHEMA}",
            shell_hint="credentials are in the SNOWFLAKE_* environment variables "
            "(python has snowflake-connector-python).",
        )

    def _drop(self, ep: str) -> None:
        self.admin_run(f"DROP DATABASE IF EXISTS {ep}")
        self.admin_run(f"DROP USER IF EXISTS {ep}", f"DROP ROLE IF EXISTS {ep}", role="USERADMIN")

    def drop_episode(self, episode: Episode) -> None:
        self._drop(episode.id)

    def execute(self, episode: Episode, sql: str) -> QueryResult:
        e = episode.agent_env
        return self._run({
            "account": e["SNOWFLAKE_ACCOUNT"], "user": e["SNOWFLAKE_USER"],
            "password": e["SNOWFLAKE_PASSWORD"], "role": e["SNOWFLAKE_ROLE"],
            "warehouse": e["SNOWFLAKE_WAREHOUSE"], "database": e["SNOWFLAKE_DATABASE"],
            "schema": e["SNOWFLAKE_SCHEMA"],
        }, [sql])

    def fetch_df(self, episode: Episode, sql: str) -> pd.DataFrame:
        res = self._run({**self.admin_connect_args(), "database": episode.id, "schema": SCHEMA},
                        [sql])
        return pd.DataFrame(res.rows, columns=res.columns)
