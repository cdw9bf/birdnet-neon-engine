"""Gate scorer for the shared harness's `extern` command.

Run as `modelgate.py extern ... -- <this> ... {in} {off} {out}` (the harness
modelgate.py is not part of this repository): streams a float32 stream and
int32 window offsets from host files into `cli <weights> <mode> --pipe[-emb]`
running inside the arm64 eval container (docker exec -i; the container never
reads files on the shared mount) and writes the n x D float32 outputs to
{out}.

usage: python pipe_score.py <container> <cli> <weights> <mode> <in.f32>
           <off.i32> <out.f32> [--emb] [--lse DIR]
  --lse DIR: also save the per-window log-sum-exp of the outputs (float64 ->
             float32) to DIR/<stream>.lse.f32 (the softmax normalizer in
             absolute units; calibration analyses need it, the gate does
             not)
"""
import os
import sys

import numpy as np

from bnneon import pipe


def save_lse(fout: str, fin: str, n: int, d: str) -> None:
    """Saves the per-window log-sum-exp of the outputs to d/<stream>.lse.f32.

    Args:
        fout: the n x D float32 outputs file.
        fin: the input stream file (its name, without .f32, names the
            output).
        n: the number of windows.
        d: the output directory (created if needed).
    """
    os.makedirs(d, exist_ok=True)
    y = np.fromfile(fout, np.float32).reshape(n, -1).astype(np.float64)
    m = y.max(1)
    lse = m + np.log(np.exp(y - m[:, None]).sum(1))
    base = os.path.basename(fin)
    name = base[:-4] if fin.endswith(".f32") else base
    lse.astype(np.float32).tofile(os.path.join(d, name + ".lse.f32"))


def main(argv: list[str]) -> int:
    """Scores one stream in the container; returns the CLI's exit code.

    Args:
        argv: the full command line (argv[0] is the program).

    Returns:
        The exit code of the engine command.
    """
    if len(argv) < 8:
        raise SystemExit(f"pipe_score: bad arguments {argv}")
    ctr, cli, w, mode, fin, foff, fout = argv[1:8]
    off = np.fromfile(foff, np.int32)
    x = np.memmap(fin, dtype=np.float32, mode="r")
    flag = "--pipe-emb" if "--emb" in argv else "--pipe"
    with open(fout, "wb") as out:
        p, t = pipe.start(["docker", "exec", "-i", ctr, cli, w, mode, flag],
                          off, x, stdout=out)
        t.join()
        rc = p.wait()
    if rc == 0 and "--lse" in argv:
        save_lse(fout, fin, len(off), argv[argv.index("--lse") + 1])
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
