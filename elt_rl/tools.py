"""The agent's three tools: bash (in its sandbox), sql (in its namespace), submit."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated

from tinker_cookbook.tool_use import Tool, ToolResult, simple_tool_result, tool

from elt_rl.sandbox import DockerSandbox
from elt_rl.warehouse.base import Episode, QueryError, Warehouse

MAX_OUTPUT_BYTES = 16_384
MAX_SQL_ROWS = 50
MAX_CELL_CHARS = 200


class EltTools:
    def __init__(
        self,
        sandbox: DockerSandbox,
        warehouse: Warehouse,
        episode: Episode,
        command_timeout: int = 120,
    ):
        self._sandbox = sandbox
        self._warehouse = warehouse
        self._episode = episode
        self._command_timeout = command_timeout

    @tool
    async def bash(self, command: Annotated[str, "The bash command to execute."]) -> ToolResult:
        """Run a bash command in the task workspace.

        The warehouse connection is preconfigured in the environment (e.g. `psql` works).
        """
        r = await self._sandbox.run_command(
            command, timeout=self._command_timeout, max_output_bytes=MAX_OUTPUT_BYTES
        )
        return simple_tool_result(
            f"exit_code: {r.exit_code}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )

    @tool
    async def sql(self, query: Annotated[str, "SQL to run in the warehouse."]) -> ToolResult:
        """Run SQL in the warehouse and return up to 50 rows."""
        try:
            res = await asyncio.to_thread(self._warehouse.execute, self._episode, query)
        except QueryError as e:  # SQL errors are feedback for the agent, not crashes
            return simple_tool_result(f"error: {str(e)[:MAX_OUTPUT_BYTES]}")
        rows = [[str(v)[:MAX_CELL_CHARS] for v in row] for row in res.rows[:MAX_SQL_ROWS]]
        out = {"columns": res.columns, "rows": rows, "row_count": len(res.rows)}
        return simple_tool_result(json.dumps(out))

    @tool
    async def submit(self) -> ToolResult:
        """Submit the pipeline. Call this once the target tables are built; it ends the episode."""
        return simple_tool_result("Submitted.", should_stop=True)

    def all(self) -> list[Tool]:
        return [self.bash, self.sql, self.submit]
