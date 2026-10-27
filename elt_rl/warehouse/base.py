"""Warehouse interface. Everything warehouse-specific lives behind this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class QueryError(Exception):
    """A statement failed (bad SQL, missing table, timeout). Connection errors are not this."""


@dataclass(frozen=True)
class Episode:
    """One isolated namespace in the warehouse, owned by one rollout."""

    id: str
    task: str
    # Env vars the agent's shell gets to reach its own namespace (e.g. PGHOST...).
    agent_env: dict[str, str]
    # Qualified prefix of the agent's tables, e.g. `trains` or `EP_1.AIRBYTE_SCHEMA`.
    # ELT-Bench eval SQL names tables `<task>.<table>`; the grader maps that prefix here.
    namespace: str
    # One line telling the agent how its shell reaches the warehouse.
    shell_hint: str


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]


class Warehouse(ABC):
    # True when episodes start with the raw source tables already loaded (ELT-Bench's
    # Airbyte EL step ran on the template); False when the agent must load them itself.
    sources_preloaded: bool
    # Docker network for the agent's sandbox, so its shell can reach this warehouse.
    sandbox_network: str

    @abstractmethod
    def create_episode(self, task: str) -> Episode:
        """Fresh namespace cloned from the task's template; the agent may only touch this."""

    @abstractmethod
    def drop_episode(self, episode: Episode) -> None: ...

    @abstractmethod
    def execute(self, episode: Episode, sql: str) -> QueryResult:
        """Run SQL with the episode's own (restricted) credentials."""

    @abstractmethod
    def fetch_df(self, episode: Episode, sql: str) -> pd.DataFrame:
        """Run grader SQL with the admin credentials."""
