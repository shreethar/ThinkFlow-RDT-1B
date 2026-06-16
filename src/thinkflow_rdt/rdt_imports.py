from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Type


def import_rdt_runner(rdt_repo: str | Path):
    """Import RDTRunner from a local clone of the official RDT repository."""
    repo = Path(rdt_repo).expanduser().resolve()
    if not repo.exists():
        env_repo = os.environ.get("RDT_REPO")
        if env_repo:
            repo = Path(env_repo).expanduser().resolve()
    required = repo / "models" / "rdt_runner.py"
    if not required.exists():
        raise FileNotFoundError(
            f"Could not find {required}. Clone the official RDT repository and "
            "set rdt_repo in the YAML config or RDT_REPO in the environment."
        )
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from models.rdt_runner import RDTRunner  # type: ignore

    return RDTRunner
