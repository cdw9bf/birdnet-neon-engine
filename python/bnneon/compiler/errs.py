"""Errors and exit codes shared by the compiler modules.

One module, so `bnc.py` run as a script and `bnc` imported as a module see
the same classes.
"""

EXIT = dict(ok=0, warn=1, unsupported=2, usage=3, model=4, parity=5,
            runtime=6, labels=7, internal=8)
# the modules of the optional compiler extra (pip install -e '.[compiler]')
COMPILER_EXTRA = ("onnx", "onnxruntime")


def missing_extra(e: ModuleNotFoundError) -> str | None:
    """The one-line install hint when e is a module of the compiler extra,
    else None."""
    if (e.name or "").split(".")[0] not in COMPILER_EXTRA:
        return None
    return (f"bnc: no module named '{e.name}'; hint: install the compiler "
            f"extra: pip install -e '.[compiler]'")


class ModelError(Exception):
    """The ONNX file cannot be used at all.

    Not a graph the compiler could ever take.
    """


class Usage(Exception):
    """Wrong or missing arguments."""


class LabelsMismatch(Usage):
    """The label file does not belong to the head."""
