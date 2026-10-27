"""The agent's shell: one Docker container per episode (image built from sandbox/Dockerfile).

The container gets a copy of the task's agent-visible files in /workspace and
the episode's warehouse credentials as env vars. It sees none of the host's
files (ground truth, .env, other episodes). The network is the warehouse's
choice: Docker's default bridge for a cloud warehouse, the host network for a
local Postgres (whose admin login must then require a password).
"""

from __future__ import annotations

import asyncio
import secrets
import shutil
import tempfile
from pathlib import Path

from tinker_cookbook.sandbox import SandboxResult

IMAGE = "elt-sandbox"
DEFAULT_MAX_OUTPUT_BYTES = 16_384


def _cap(data: bytes, limit: int) -> str:
    text = data[:limit].decode(errors="replace")
    return text + (f"\n... [truncated {len(data) - limit} bytes]" if len(data) > limit else "")


async def _docker(*args: str, stdin: bytes | None = None) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(stdin)
    return proc.returncode or 0, out, err


class DockerSandbox:
    def __init__(self, name: str):
        self.name = name

    @classmethod
    async def create(
        cls, files: dict[str, Path], env: dict[str, str], network: str, image: str = IMAGE
    ) -> DockerSandbox:
        name = f"elt-sbx-{secrets.token_hex(6)}"
        env_args = [a for k, v in env.items() for a in ("-e", f"{k}={v}")]
        code, _, err = await _docker(
            "run", "-d", "--rm", "--name", name, "--network", network,
            "--memory", "4g", "--cpus", "2", "--pids-limit", "512",
            *env_args, image, "sleep", "infinity",
        )
        if code:
            raise RuntimeError(f"docker run failed: {err.decode()}")
        sandbox = cls(name)
        with tempfile.TemporaryDirectory() as tmp:
            for rel, src in files.items():
                dst = Path(tmp) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            # Stream a tar so the files land owned by the container user.
            tar = await asyncio.create_subprocess_exec(
                "tar", "-C", tmp, "-cf", "-", ".", stdout=asyncio.subprocess.PIPE
            )
            data, _ = await tar.communicate()
        code, _, err = await _docker(
            "exec", "-i", name, "tar", "-C", "/workspace", "-xf", "-", stdin=data
        )
        if code:
            await sandbox.cleanup()
            raise RuntimeError(f"copying workspace failed: {err.decode()}")
        return sandbox

    async def run_command(
        self, command: str, timeout: int = 60, max_output_bytes: int | None = None
    ) -> SandboxResult:
        limit = max_output_bytes or DEFAULT_MAX_OUTPUT_BYTES
        # `timeout` exits 124 when it had to stop the command; -k kills it if TERM is ignored.
        code, out, err = await asyncio.wait_for(
            _docker("exec", self.name, "timeout", "-k", "5", str(timeout), "bash", "-c", command),
            timeout + 30,
        )
        if code == 124:
            err += f"\ntimed out after {timeout}s".encode()
        return SandboxResult(stdout=_cap(out, limit), stderr=_cap(err, limit), exit_code=code)

    async def cleanup(self) -> None:
        await _docker("rm", "-f", self.name)
