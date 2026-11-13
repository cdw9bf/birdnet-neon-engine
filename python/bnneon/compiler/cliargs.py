"""Command-line helpers shared by the bnc commands.

The commands read their options straight from the argument list (no parser
object), so option lookup, the --json output and the choice of text stream
live here.
"""
from collections.abc import Callable, Sequence
import json
import os
import sys
from typing import Any

import numpy as np

from . import errs


def opt(args: Sequence[str], key: str, default: Any = None) -> Any:
    """Returns the value that follows an option.

    Args:
        args: the command line (the command name first).
        key: the option, e.g. "--labels".
        default: what to return when the option is absent or has no value.

    Returns:
        The argument after the first occurrence of `key`, else `default`.
    """
    if key not in args:
        return default
    i = args.index(key) + 1
    return args[i] if i < len(args) else default


def opt_all(args: Sequence[str], key: str) -> list[str]:
    """Returns every value given with a repeatable option.

    Args:
        args: the command line (the command name first; args[0] is never
            read as a value).
        key: the option, e.g. "--audio".

    Returns:
        The argument after each occurrence of `key`, in order.
    """
    return [a for i, a in enumerate(args) if i > 0 and args[i - 1] == key]


def opt_int(args: Sequence[str], key: str, default: int) -> int:
    """Returns an integer option.

    Args:
        args: the command line.
        key: the option.
        default: the value when the option is absent.

    Returns:
        The option's value as an int, or `default`.

    Raises:
        errs.Usage: the value is not an integer.
    """
    value = opt(args, key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise errs.Usage(f"{key} {value}: expected an integer") from None


def read_indices(path: str) -> np.ndarray:
    """A class-index file an option names (--classes, a regional indices
    file): one integer per non-blank line, as int64."""
    with open(path, encoding="utf-8") as f:
        return np.array([int(line) for line in f if line.strip()], np.int64)


def json_to_stdout(args: Sequence[str]) -> bool:
    """True when --json has no FILE: JSON on stdout, the text on stderr."""
    return "--json" in args and (opt(args, "--json") or "-").startswith("-")


def print_stderr(*values: Any) -> None:
    """print() to stderr."""
    print(*values, file=sys.stderr)


def text_printer(args: Sequence[str]) -> Callable[..., None]:
    """The print function for a command's text report (see json_to_stdout)."""
    return print_stderr if json_to_stdout(args) else print


def json_out(args: Sequence[str], obj: Any) -> None:
    """Writes the --json output: to FILE when one is given, else to stdout.

    Args:
        args: the command line (holds --json [FILE]).
        obj: the JSON document; values JSON cannot hold are written with str().
    """
    path = opt(args, "--json")
    if path and not path.startswith("-"):
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=str)
    else:
        print(json.dumps(obj, indent=1, default=str))
