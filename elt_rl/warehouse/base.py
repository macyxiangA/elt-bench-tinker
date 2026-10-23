"""Warehouse interface. Everything warehouse-specific lives behind this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Episode:
    """One isolated namespace in the warehouse, owned by one rollout."""

    id: str
    task: str
    # Env vars the agent's shell gets to reach its own namespace (e.g. PGHOST...).
    agent_env: dict[str, str]


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]


class Warehouse(ABC):
    @abstractmethod
    def create_episode(self, task: str) -> Episode:
        """Fresh namespace cloned from the task's template; the agent may only touch this."""

    @abstractmethod
    def drop_episode(self, episode: Episode) -> None: ...

    @abstractmethod
    def execute(self, episode: Episode, sql: str) -> QueryResult:
        """Run SQL with the agent's (restricted) credentials."""

    @abstractmethod
    def fetch_df(self, episode: Episode, sql: str) -> pd.DataFrame:
        """Run grader SQL with the grader's credentials."""
