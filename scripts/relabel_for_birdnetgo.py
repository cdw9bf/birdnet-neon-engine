"""Runs bnneon.labels: `python -I scripts/relabel_for_birdnetgo.py IN OUT ...`.

See python/bnneon/labels.py for the options; `pip install -e .` from the
repository root installs the package (or set PYTHONPATH=<checkout>/python and
run with `python -P -s`).
"""
import sys

try:
    from bnneon import labels
except ModuleNotFoundError as e:
    if e.name != "bnneon":
        raise  # a missing dependency (numpy, onnx, ...): its own traceback
    print("cannot import the bnneon package; hint: run `pip install -e .` "
          "from the repository root, or set PYTHONPATH=<checkout>/python",
          file=sys.stderr)
    sys.exit(3)

if __name__ == "__main__":
    sys.exit(labels.main(sys.argv[1:]))
