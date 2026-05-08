from __future__ import annotations

import os
import shutil
import stat
from inspect import signature
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_NAMES = {"__pycache__", ".pytest_cache", "outputs", "checkpoints"}
RMTREE_RETRY_ARG = "onexc" if "onexc" in signature(shutil.rmtree).parameters else "onerror"


def _make_writable_and_retry(function, path: str, _exc_info) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    function(path)


def remove_path(path: Path) -> None:
    """Remove a generated artifact path inside the repository."""
    resolved = path.resolve()
    if ROOT not in resolved.parents and resolved != ROOT:
        raise RuntimeError(f"refusing to remove path outside repo: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved, **{RMTREE_RETRY_ARG: _make_writable_and_retry})
    elif resolved.exists():
        resolved.chmod(stat.S_IWRITE | stat.S_IREAD)
        resolved.unlink()


def main() -> None:
    removed: list[str] = []
    for name in ("outputs", "checkpoints", ".pytest_cache"):
        path = ROOT / name
        if path.exists():
            remove_path(path)
            removed.append(str(path.relative_to(ROOT)))
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir():
            remove_path(path)
            removed.append(str(path.relative_to(ROOT)))
    for path in ROOT.rglob("*.pyc"):
        if path.is_file():
            remove_path(path)
            removed.append(str(path.relative_to(ROOT)))
    if removed:
        for item in removed:
            print(f"removed {item}")
    else:
        print("no generated artifacts found")


if __name__ == "__main__":
    main()
