"""Per-episode workspace + local subprocess backend for cookbook's SandboxInterface.

A local directory is enough for the credential-free test. A container backend
(Docker on the lab server, Modal) can replace it behind the same interface.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import tempfile
from pathlib import Path

from tinker_cookbook.sandbox import SandboxResult

DEFAULT_MAX_OUTPUT_BYTES = 16_384


def make_workspace(files: dict[str, Path], root: Path | None = None) -> Path:
    """Copy the task's agent-visible files into a fresh directory."""
    ws = Path(tempfile.mkdtemp(prefix="elt_ws_", dir=root))
    for rel, src in files.items():
        dst = ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return ws


def _cap(data: bytes, limit: int) -> str:
    text = data[:limit].decode(errors="replace")
    return text + (f"\n... [truncated {len(data) - limit} bytes]" if len(data) > limit else "")


class LocalSandbox:
    def __init__(self, workdir: Path, env: dict[str, str]):
        self.workdir = workdir
        # Minimal env: no host secrets leak into the agent's shell.
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(workdir), **env}

    @property
    def sandbox_id(self) -> str:
        return str(self.workdir)

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        limit = max_output_bytes or DEFAULT_MAX_OUTPUT_BYTES
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=workdir or self.workdir,
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            return SandboxResult(stdout="", stderr=f"timed out after {timeout}s", exit_code=124)
        return SandboxResult(
            stdout=_cap(out, limit), stderr=_cap(err, limit), exit_code=proc.returncode or 0
        )

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        try:
            data = (self.workdir / path).read_bytes()
        except OSError as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        return SandboxResult(stdout=_cap(data, max_bytes or len(data)), stderr="", exit_code=0)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        target = self.workdir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode() if isinstance(content, str) else content)
        if executable:
            target.chmod(0o755)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)
