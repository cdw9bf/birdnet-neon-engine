"""Compiles and runs the C example (quickstart.c) in docs/REFERENCE.md.

  python -I tests/test_readme_example.py [--lib build/libbnneon.so]
      [--model FILE ...] [--cc cc]

(-I with bnneon installed in the interpreter; for a checkout that is not
installed: PYTHONPATH=<checkout>/python python -P -s tests/...)

So the documented example cannot drift from the API: extracts the first
```c block of docs/REFERENCE.md (the README links to it), compiles it with
-Wall -Wextra -Werror against src/bnneon.h and the built library, runs it on
each model (default: the packages and the legacy .bn24 in weights/ this CPU
can run) and on a missing file, and checks the output: five classes with
confidences in [0, 1], and the library's message on failure. A missing
library, compiler or model is a clean skip (exit 0 with a "skip" line).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import bnneon  # the label column is checked against the binding's labels()
from bnneon import testutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the page whose first ```c block is the C example (quickstart.c)
EXAMPLE_DOC = os.path.join(ROOT, "docs", "REFERENCE.md")
DEFAULT_MODELS = ["birdnet_v2.4-neon-fp16.bnpk", "birdnet_v2.4-neon-fp32.bnpk",
                  "battybirdnet_eu_256khz-neon-fp32.bnpk",
                  "perch_v2_nae-neon-fp16-s8.bnpk",
                  "perch_v2-neon-fp16-s8.bnpk", "v24.bn24"]
CHECKS = testutil.Checks(detail_limit=800)
check = CHECKS.check


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, check=False,
                          **kwargs)


def readme_example():
    """The first ```c block of docs/REFERENCE.md, or None."""
    with open(EXAMPLE_DOC, encoding="utf-8") as f:
        m = re.search(r"```c\n(.*?)```", f.read(), re.S)
    return m.group(1) if m else None


def build_example(cc, lib, code, tmp):
    """Compiles the example; returns the executable, or None if it failed."""
    src = os.path.join(tmp, "quickstart.c")
    exe = os.path.join(tmp, "quickstart")
    with open(src, "w", encoding="utf-8") as f:
        f.write(code)
    # link the library by its absolute path (it has no soname / install name
    # to search for); run from ROOT so a macOS build's relative install name
    # (build/libbnneon.so) resolves too
    r = _run([cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror", "-I",
              os.path.join(ROOT, "src"), src, lib,
              f"-Wl,-rpath,{os.path.dirname(lib)}", "-lm", "-o", exe])
    check("example compiles with -Wall -Wextra -Werror", r.returncode == 0,
          r.stdout + r.stderr)
    return None if r.returncode else exe


def check_model(exe, path, lib):
    """Runs the example on one model; False when this CPU cannot run it."""
    name = os.path.basename(path)
    r = _run([exe, path], cwd=ROOT)
    if r.returncode == 1 and "can run none" in r.stderr:
        print(f"skip {name}: not runnable on this CPU")
        return False
    rows = [ln.split(None, 2) for ln in r.stdout.splitlines()[1:]]
    ok = (r.returncode == 0 and len(rows) == 5
          and all(0.0 <= float(x[1]) <= 1.0 for x in rows))
    check(f"{name}: five classes with confidences", ok, r.stdout + r.stderr)
    with bnneon.Model(path, 0, lib) as m:
        labels = m.labels()
    if ok and labels is not None:
        got = [x[2] if len(x) > 2 else "" for x in rows]
        want = [labels[int(x[0])] for x in rows]
        check(f"{name}: labels match the package's label file", got == want,
              f"{got} vs {want}")
    print("     " + "\n     ".join(r.stdout.splitlines()[:3]))
    return True


def main():
    default_lib = os.path.join(ROOT, "build", "libbnneon.so")
    lib = os.path.abspath(testutil.opt("--lib", default_lib))
    cc = testutil.opt("--cc", os.environ.get("CC", "cc"))
    if not os.path.isfile(lib) or not shutil.which(cc):
        print(f"skip: no library at {lib} or no C compiler '{cc}' (run `make`, "
              f"or pass --lib / --cc)")
        return 0
    code = readme_example()
    check("docs/REFERENCE.md has a C example", code is not None)
    if code is None:
        return 1
    models = testutil.opt_all("--model") or testutil.default_models(
        ROOT, DEFAULT_MODELS)
    with tempfile.TemporaryDirectory(prefix="bnn_readme_") as tmp:
        exe = build_example(cc, lib, code, tmp)
        if exe is None:
            return 1
        r = _run([exe, os.path.join(tmp, "missing.bnpk")], cwd=ROOT)
        check("missing file: exit 1 with the library's message",
              r.returncode == 1 and "cannot open" in r.stderr, r.stderr)
        ran = sum(check_model(exe, p, lib) for p in models)
    if not ran:
        print("skip: no model file in weights/ runs here (pass --model FILE)")
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
