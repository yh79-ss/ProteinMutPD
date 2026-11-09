"""The result type shared by every preflight check. Pure data; no imports from the rest of the package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

LEVELS = ("ok", "warn", "error", "skip")


@dataclass(frozen=True)
class Check:
    """One finding of `mpd doctor` / `mpd plan` / `mpd submit`.

    ``level``: ok | warn | error | skip. ``fix`` is a command or one-line instruction the user can act on.
    """

    name: str
    level: str
    message: str
    fix: str | None = None

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"check level must be one of {LEVELS}; got {self.level!r}")


def counts(checks: Iterable[Check]) -> dict[str, int]:
    """Number of checks per level (every level is present, possibly 0)."""
    out = dict.fromkeys(LEVELS, 0)
    for c in checks:
        out[c.level] += 1
    return out


def has_errors(checks: Iterable[Check]) -> bool:
    return any(c.level == "error" for c in checks)
