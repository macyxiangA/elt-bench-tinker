"""Minimal Airbyte API client (application token auth), enough for ELT-Bench's EL step."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class Airbyte:
    def __init__(self, base: str, client_id: str, client_secret: str):
        self.base = base.rstrip("/")
        self.token: str | None = None
        self.token = self._call("POST", "/api/v1/applications/token", {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant-type": "client_credentials",
        })["access_token"]

    @classmethod
    def from_env(cls) -> Airbyte:
        e = os.environ
        return cls(e.get("AIRBYTE_URL") or "http://localhost:8000",
                   e["AIRBYTE_CLIENT_ID"], e["AIRBYTE_CLIENT_SECRET"])

    def _call(self, method: str, path: str, body: dict | None = None) -> Any:
        req = urllib.request.Request(
            self.base + path, method=method, data=json.dumps(body).encode() if body else None
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode()[:2000]}") from None
        return json.loads(data) if data else {}

    def public(self, method: str, path: str, body: dict | None = None) -> Any:
        """Call the public API (`/api/public/v1` + path)."""
        return self._call(method, "/api/public/v1" + path, body)

    def find(self, kind: str, name: str, workspace: str) -> dict | None:
        """Existing source/destination/connection named `name` (kind = 'sources', ...)."""
        data = self.public("GET", f"/{kind}?workspaceIds={workspace}&limit=100")["data"]
        return next((x for x in data if x["name"] == name), None)

    def sync(self, connection_id: str, poll_seconds: int = 10, timeout: int = 1800) -> dict:
        """Run a sync job and wait for it; raise unless it succeeds."""
        job = self.public("POST", "/jobs", {"connectionId": connection_id, "jobType": "sync"})
        deadline = time.time() + timeout
        while job["status"] in ("pending", "running", "incomplete"):
            if time.time() > deadline:
                raise TimeoutError(f"sync job {job['jobId']} still {job['status']}")
            time.sleep(poll_seconds)
            job = self.public("GET", f"/jobs/{job['jobId']}")
        if job["status"] != "succeeded":
            raise RuntimeError(f"sync job {job['jobId']} ended {job['status']}")
        return job
