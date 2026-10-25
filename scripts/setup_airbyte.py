"""One-time Airbyte setup for ELT-Bench: register the REST API connector
(ELT-Bench's setup/elt_snowflake.yaml) and print the workspace and definition ids.
Usage: set -a; source .env; set +a; python scripts/setup_airbyte.py"""

import os
import sys
from pathlib import Path

import yaml

from elt_rl.airbyte import Airbyte

NAME = "elt-bench-rest-api"


def main() -> None:
    ab = Airbyte.from_env()
    workspace = ab.public("GET", "/workspaces")["data"][0]["workspaceId"]

    defs = f"/workspaces/{workspace}/definitions/declarative_sources"
    existing = [d for d in ab.public("GET", defs).get("data", []) if d["name"] == NAME]
    if existing:
        definition = existing[0]["id"]
    else:
        bench = Path(os.environ.get("ELT_BENCH_DIR") or "ELT-Bench")
        manifest = yaml.safe_load((bench / "setup" / "elt_snowflake.yaml").read_text())
        definition = ab.public("POST", defs, {"name": NAME, "manifest": manifest})["id"]
    print(f"AIRBYTE_WORKSPACE_ID={workspace}")
    print(f"AIRBYTE_API_DEFINITION_ID={definition}")


if __name__ == "__main__":
    sys.exit(main())
