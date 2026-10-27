"""Postgres warehouse: one database + one login role per episode.

The task template database is created once and then closed to connections;
each episode is a `CREATE DATABASE ... TEMPLATE` copy owned by a throwaway
role that cannot connect anywhere else. The grader uses the admin DSN.
"""

from __future__ import annotations

import secrets

import pandas as pd
import psycopg
from psycopg import sql as pgsql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from elt_rl.warehouse.base import Episode, QueryError, QueryResult, Warehouse


class PostgresWarehouse(Warehouse):
    sources_preloaded = False
    # Local test setup: the sandbox reaches Postgres on the host's localhost.
    sandbox_network = "host"

    def __init__(self, admin_dsn: str, statement_timeout_ms: int = 60_000):
        self.admin_dsn = admin_dsn
        self.statement_timeout_ms = statement_timeout_ms

    def _admin(self, dbname: str | None = None) -> psycopg.Connection:
        dsn = make_conninfo(self.admin_dsn, dbname=dbname) if dbname else self.admin_dsn
        return psycopg.connect(dsn, autocommit=True)

    def _template(self, task: str) -> str:
        return f"elt_tmpl_{task}"

    def ensure_template(self, task: str) -> None:
        """Template = empty target schema named after the task (ELT-Bench's `<task>.<model>`)."""
        db = pgsql.Identifier(self._template(task))
        with self._admin() as conn:
            if conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (self._template(task),)
            ).fetchone():
                return
            conn.execute(pgsql.SQL("CREATE DATABASE {}").format(db))
        with self._admin(self._template(task)) as conn:
            conn.execute(pgsql.SQL("CREATE SCHEMA {}").format(pgsql.Identifier(task)))
        with self._admin() as conn:
            conn.execute(pgsql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(db))
            conn.execute(pgsql.SQL("ALTER DATABASE {} WITH ALLOW_CONNECTIONS false").format(db))

    def create_episode(self, task: str) -> Episode:
        self.ensure_template(task)
        ep = f"elt_ep_{secrets.token_hex(6)}"
        password = secrets.token_urlsafe(16)
        db, role = pgsql.Identifier(ep), pgsql.Identifier(ep)
        with self._admin() as conn:
            conn.execute(
                pgsql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(role, pgsql.Literal(password))
            )
            conn.execute(
                pgsql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(
                    role, pgsql.Literal(self.statement_timeout_ms)
                )
            )
            conn.execute(
                pgsql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE {}").format(
                    db, role, pgsql.Identifier(self._template(task))
                )
            )
            conn.execute(pgsql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(db))
        with self._admin(ep) as conn:
            conn.execute(
                pgsql.SQL("ALTER SCHEMA {} OWNER TO {}").format(pgsql.Identifier(task), role)
            )
        admin = conninfo_to_dict(self.admin_dsn)
        agent_env = {
            "PGHOST": str(admin.get("host", "localhost")),
            "PGPORT": str(admin.get("port", "5432")),
            "PGUSER": ep,
            "PGPASSWORD": password,
            "PGDATABASE": ep,
        }
        return Episode(
            id=ep,
            task=task,
            agent_env=agent_env,
            namespace=task,
            shell_hint="`psql` is preconfigured; `\\copy` reads workspace files.",
        )

    def drop_episode(self, episode: Episode) -> None:
        with self._admin() as conn:
            conn.execute(
                pgsql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    pgsql.Identifier(episode.id)
                )
            )
            conn.execute(pgsql.SQL("DROP ROLE IF EXISTS {}").format(pgsql.Identifier(episode.id)))

    def _agent_dsn(self, episode: Episode) -> str:
        e = episode.agent_env
        return make_conninfo(
            host=e["PGHOST"],
            port=e["PGPORT"],
            user=e["PGUSER"],
            password=e["PGPASSWORD"],
            dbname=e["PGDATABASE"],
        )

    def _run(self, dsn: str, query: str) -> QueryResult:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
            try:
                cur = conn.execute(query)
            except psycopg.Error as e:
                raise QueryError(str(e)) from e
            if cur.description is None:
                return QueryResult(columns=[], rows=[])
            return QueryResult(columns=[c.name for c in cur.description], rows=cur.fetchall())

    def execute(self, episode: Episode, sql: str) -> QueryResult:
        return self._run(self._agent_dsn(episode), sql)

    def fetch_df(self, episode: Episode, sql: str) -> pd.DataFrame:
        res = self._run(make_conninfo(self.admin_dsn, dbname=episode.id), sql)
        return pd.DataFrame(res.rows, columns=res.columns)
