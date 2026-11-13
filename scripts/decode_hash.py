"""Hashes the decoded 48 kHz float32 stream of one tuning-set stream.

The audio is decoded exactly as tuning/scripts/infer.load_stream produces it
(its first file only); the line printed names the soxr and numpy versions.

usage: python -I decode_hash.py <tuning_dir> <stream name>
"""
import hashlib
import os
import sys

import numpy as np
import soxr

from bnneon import extern
from bnneon import gate_v3


def main(argv: list[str]) -> int:
    """Prints `NAME n_samples sha256[:16] soxr VERSION numpy VERSION`."""
    tuning, name = argv[1], argv[2]
    # infer.py reads sys.argv[1] (the tuning dir) when it is imported
    infer = extern.load_with_argv(os.path.join(tuning, "scripts"), "infer",
                                  tuning)
    path = os.path.join(tuning, "results", "streams.json")
    st = gate_v3.read_streams(path, env_filter=False)[name]
    x = infer.load_stream(st["files"][:1])
    print(name, len(x), hashlib.sha256(x.tobytes()).hexdigest()[:16], "soxr",
          soxr.__version__, "numpy", np.__version__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
