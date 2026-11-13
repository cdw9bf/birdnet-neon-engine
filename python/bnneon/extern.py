"""Loads a Python module from a tree outside this repository by file path.

The gate scripts reuse code from trees this repository does not own (the
tuning tree's scripts/infer.py and simulate.py, the v3 harness's
simulate5s.py, perch-runtime's perchgate.py). Instead of editing sys.path,
they load such a module from its file with load_module(), which registers it
in sys.modules under its plain name, so that a later `import name` inside
that external code finds the same module object.

Several of those modules read sys.argv when they are imported:
load_with_argv() sets it for the import and restores it afterwards.
"""
import importlib.util
import os
import sys
import types


def load_module(directory: str, name: str) -> types.ModuleType:
    """Loads <directory>/<name>.py and registers it as sys.modules[name].

    Args:
        directory: the directory that holds the module file.
        name: the module name (the file name without .py).

    Returns:
        The loaded module. If a module of that name is already loaded, it
        is returned as it is (a plain `import name` would do the same).

    Raises:
        ModuleNotFoundError: there is no <directory>/<name>.py (the same
            exception and message as a failed `import name`).
    """
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(directory, name + ".py")
    if not os.path.isfile(path):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load_with_argv(directory: str, name: str, *argv: str) -> types.ModuleType:
    """load_module() with sys.argv = [program, *argv] during the import.

    Args:
        directory: the directory that holds the module file.
        name: the module name.
        *argv: the arguments the module reads from sys.argv[1:] when it is
            imported.

    Returns:
        The loaded module; sys.argv is restored, also when the import fails.
    """
    saved = sys.argv
    sys.argv = [saved[0], *argv]
    try:
        return load_module(directory, name)
    finally:
        sys.argv = saved
