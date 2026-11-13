"""bnc: the bnneon model compiler's command line (report, compile, check).

The usage text (printed for -h / --help, exit 0, and for a missing or unknown
command, exit 3) is usage.txt next to this module. Exit codes are errs.EXIT;
every command returns one.
"""
from collections.abc import Sequence
import importlib.resources
import os
import sys
import traceback

from . import cliargs
from . import errs

try:  # the commands need onnx and ONNX Runtime: the optional compiler extra
    from . import cmd_check
    from . import cmd_compile
    from . import cmd_report
except ModuleNotFoundError as missing:
    if errs.missing_extra(missing) is None:
        raise
    print(errs.missing_extra(missing), file=sys.stderr)
    sys.exit(errs.EXIT["usage"])

USAGE = importlib.resources.files(__package__).joinpath(
    "usage.txt").read_text(encoding="utf-8")
COMMANDS = ("report", "compile", "check")
HELP = ("-h", "--help")


def _wants_help(a: Sequence[str]) -> bool:
    """True for `-h` / `--help`, alone or after a command."""
    return bool(a) and (a[0] in HELP
                        or (a[0] in COMMANDS and len(a) >= 2 and a[1] in HELP))


def _is_command(a: Sequence[str]) -> bool:
    """True for `<command> <model.onnx> ...`."""
    return (bool(a) and a[0] in COMMANDS and len(a) >= 2
            and not a[1].startswith("-"))


def _internal_error(a: Sequence[str], e: Exception) -> int:
    """Reports a compiler bug or an unforeseen graph; returns its exit code.

    The message names the place; with --json the traceback goes into the JSON.
    """
    tb = traceback.extract_tb(e.__traceback__)[-1]
    msg = (f"internal error: {type(e).__name__}: {str(e)[:300]} (at "
           f"{os.path.basename(tb.filename)}:{tb.lineno}); hint: report this "
           f"with the model's `bnc.py report --json` output")
    print(msg)
    if "--json" in a:
        cliargs.json_out(a, dict(model=a[1], status="ERROR",
                                 exit=errs.EXIT["internal"], next_step=msg,
                                 traceback=traceback.format_exc()))
    return errs.EXIT["internal"]


def _run_command(a: Sequence[str]) -> int:
    """Runs the command; any exception but the expected ones is internal."""
    try:
        if a[0] == "report":
            return cmd_report.cmd_report(a)
        if a[0] == "check":
            return cmd_check.cmd_check(a)
        return cmd_compile.cmd_compile(a)
    except (errs.Usage, errs.ModelError, FileNotFoundError, SystemExit,
            KeyboardInterrupt):
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        # a compiler bug or an unforeseen graph: still a defined exit code
        return _internal_error(a, e)


def main(a: Sequence[str] | None = None) -> int:
    """Runs bnc; returns the exit code.

    Args:
        a: the arguments after the program name (default: sys.argv[1:], as
            the `bnc` console script calls it).
    """
    if a is None:
        a = sys.argv[1:]
    if _wants_help(a):
        print(USAGE)
        return errs.EXIT["ok"]
    if not _is_command(a):
        print(USAGE)
        return errs.EXIT["usage"]
    try:
        return _run_command(a)
    except errs.LabelsMismatch as e:
        print(f"labels: {e}")
        return errs.EXIT["labels"]
    except errs.Usage as e:
        print(f"usage error: {e}")
        return errs.EXIT["usage"]
    except errs.ModelError as e:
        print(f"model error: {e}")
        return errs.EXIT["model"]
    except FileNotFoundError as e:
        print(f"usage error: file not found: {e.filename}")
        return errs.EXIT["usage"]


if __name__ == "__main__":
    sys.exit(main())
