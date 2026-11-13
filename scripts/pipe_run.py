"""Runs windows through the engine CLI's --pipe mode and saves the outputs.

The windows file holds n x n_samples float32; the CLI runs as
`cli <weights> <mode> --pipe[-emb]`.

usage: python pipe_run.py <cli> <weights> <mode> <windows.f32> <n_samples>
           <out.f32> [--emb]
"""
import subprocess
import sys

import numpy as np

from bnneon import pipe


def main(argv: list[str]) -> int:
    """Scores the windows file; prints the output size per window.

    Args:
        argv: the full command line (argv[0] is the program).

    Returns:
        0 (a failing CLI raises CalledProcessError; its stderr is
        discarded).
    """
    cli, w, mode, win, ns, out = argv[1:7]
    ns = int(ns)
    x = np.fromfile(win, np.float32).reshape(-1, ns)
    n = len(x)
    cmd = [cli, w, mode, "--pipe-emb" if "--emb" in argv else "--pipe"]
    offs = np.arange(n, dtype=np.int32) * ns
    rc, y = pipe.pipe_once(cmd, offs, x, stderr=subprocess.DEVNULL)
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)
    with open(out, "wb") as f:
        f.write(y)
    print(out, len(y) // 4 // n, "floats per window")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
