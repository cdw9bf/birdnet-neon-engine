"""Helpers the plain-script tests in tests/ share (no test framework needed).

Each test prints `ok   NAME` or `FAIL NAME` plus a detail line per check, and
ends with "N passed, M failed"; its exit code is 0 only when nothing failed.
"""
from collections.abc import Callable, Sequence
import os
import sys
from typing import Any

from .compiler import cliargs


class Checks:
    """Counts and prints check results.

    Attributes:
        passed: the number of passed checks.
        failed: the names of the failed checks.
    """

    def __init__(self, detail_limit: int = 600):
        self.passed = 0
        self.failed: list[str] = []
        self.detail_limit = detail_limit

    def check(self, name: str, cond: Any, detail: Any = "") -> None:
        """Records one check; on failure prints the start of the detail."""
        if cond:
            self.passed += 1
            print(f"ok   {name}")
        else:
            self.failed.append(name)
            print(f"FAIL {name}\n     {str(detail)[:self.detail_limit]}")

    def summary(self) -> int:
        """Prints the totals; returns the exit code (1 if anything failed)."""
        names = ": " + ", ".join(self.failed) if self.failed else ""
        print(f"\n{self.passed} passed, {len(self.failed)} failed{names}")
        return 1 if self.failed else 0


def opt(key: str, default: Any = None,
        argv: Sequence[str] | None = None) -> Any:
    """The value after option `key` on the command line, else default."""
    return cliargs.opt(sys.argv if argv is None else argv, key, default)


def opt_all(key: str, argv: Sequence[str] | None = None) -> list[str]:
    """Every value given with a repeatable option (`--model A --model B`)."""
    return cliargs.opt_all(sys.argv if argv is None else argv, key)


def raises(fn: Callable[[], Any],
           exc_type: type[BaseException]) -> BaseException | None:
    """The exception fn() raises if it is an exc_type, else None."""
    try:
        fn()
    except exc_type as e:  # the expected failure
        return e
    return None


def default_models(root: str, names: Sequence[str]) -> list[str]:
    """The files of names present in <root>/weights, in that order."""
    paths = (os.path.join(root, "weights", m) for m in names)
    return [p for p in paths if os.path.isfile(p)]
