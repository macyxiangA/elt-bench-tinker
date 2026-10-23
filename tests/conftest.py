"""Shared fixtures: a Postgres admin DSN and an ELT-Bench checkout, both from env vars."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest

from elt_rl.warehouse.postgres import PostgresWarehouse


@pytest.fixture(scope="session")
def elt_bench_dir() -> Path:
    path = Path(os.environ.get("ELT_BENCH_DIR", "ELT-Bench"))
    if not (path / "evaluation" / "eva_stage2.py").exists():
        pytest.fail(f"ELT-Bench checkout not found at {path}; set ELT_BENCH_DIR (see README)")
    return path


@pytest.fixture(scope="session")
def warehouse() -> PostgresWarehouse:
    dsn = os.environ.get("ELT_PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
    try:
        psycopg.connect(dsn, connect_timeout=5).close()
    except psycopg.OperationalError as e:
        pytest.fail(f"cannot reach Postgres at ELT_PG_DSN={dsn!r}: {e}")
    return PostgresWarehouse(dsn)
