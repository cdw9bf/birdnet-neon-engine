"""Runs the bnc model compiler: `python -I compiler/bnc.py COMMAND ...`.

The compiler is the bnneon.compiler package in python/; `pip install -e
'.[compiler]'` from the repository root installs it with onnx and ONNX Runtime
(and the `bnc` command, the same program). To run a checkout without installing
it, set PYTHONPATH=<checkout>/python and run `python -P -s compiler/bnc.py ...`
(`-I` ignores PYTHONPATH).
"""
import sys

try:
    from bnneon.compiler import bnc
except ModuleNotFoundError as e:
    if e.name != "bnneon":
        raise  # a missing dependency (numpy, onnx, ...): its own traceback
    print("cannot import the bnneon package; hint: run `pip install -e .` "
          "from the repository root, or set PYTHONPATH=<checkout>/python",
          file=sys.stderr)
    sys.exit(3)

if __name__ == "__main__":
    sys.exit(bnc.main(sys.argv[1:]))
