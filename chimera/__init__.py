"""Repo-root import shim for direct Python commands.

The actual package lives in src/chimera. This shim keeps `python -c "import
chimera"` working from the repository root without duplicating source modules.
"""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "chimera"
__path__ = [str(_SRC_PACKAGE)]

_SOURCE_INIT = _SRC_PACKAGE / "__init__.py"
exec(compile(_SOURCE_INIT.read_text(encoding="utf-8"), str(_SOURCE_INIT), "exec"), globals())
