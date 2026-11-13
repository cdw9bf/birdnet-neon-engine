"""Runs bnneon.gate: `python -I scripts/gate_generic.py declare|score ...`.

The generic accuracy gate (docs/ADDING_A_MODEL.md section 9); run it without
arguments for its usage, or see python/bnneon/gate.py. `pip install -e .` from
the repository root installs the package (or set
PYTHONPATH=<checkout>/python and run with `python -P -s`).
"""
import sys

try:
    from bnneon import gate
except ModuleNotFoundError as e:
    if e.name != "bnneon":
        raise  # a missing dependency (numpy, onnx, ...): its own traceback
    print("cannot import the bnneon package; hint: run `pip install -e .` "
          "from the repository root, or set PYTHONPATH=<checkout>/python",
          file=sys.stderr)
    sys.exit(3)

if __name__ == "__main__":
    sys.exit(gate.main(sys.argv[1:]))
