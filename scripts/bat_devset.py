"""Bat diagnosis windows drawn from track T's eval set.

n windows per source drawn at random (seed) from track T's eval set (CC0;
PCM16 at 256 kHz through BirdNET-Go's resampler). NOTE: these windows are a
subset of the bat gate set; they are used to locate the fp16-sensitive
layers only, and any fix is re-gated on the full set.

usage: python -I bat_devset.py <evalset_dir> <out.f32> <per_source> [seed]
Writes the windows (float32, row-major) and <out>.txt with
"source file offset" per window.
"""
import json
import os
import sys

import numpy as np


def by_source(streams: list[dict]) -> dict[str, list[dict]]:
    """The index's streams grouped by source, in index order."""
    by = {}
    for s in streams:
        by.setdefault(s["source"], []).append(s)
    return by


def candidates(ev: str, streams: list[dict], w: int,
               hop: int) -> list[tuple[str, int]]:
    """Every (file, sample offset) of a full window in the streams."""
    cand = []
    for s in streams:
        n = os.path.getsize(os.path.join(ev, "streams", s["file"])) // 2
        n_win = max(0, (n - w) // hop + 1)
        cand += [(s["file"], k * hop) for k in range(n_win)]
    return cand


def read_window(ev: str, f: str, o: int, w: int) -> np.ndarray:
    """w PCM16 samples at sample offset o of a stream file, / 32768."""
    pcm = np.fromfile(os.path.join(ev, "streams", f), dtype="<i2", count=w,
                      offset=2 * o)
    return pcm.astype(np.float32) / 32768.0


def main(argv: list[str]) -> int:
    """Writes the windows and their index file."""
    ev, out, per = argv[1], argv[2], int(argv[3])
    seed = int(argv[4]) if len(argv) > 4 else 7
    with open(os.path.join(ev, "index.json"), encoding="utf-8") as f:
        idx = json.load(f)
    w, hop = idx["window"], idx["hop"]
    rng = np.random.default_rng(seed)
    by = by_source(idx["streams"])
    xs, meta = [], []
    for src in sorted(by):
        cand = candidates(ev, by[src], w, hop)
        pick = rng.choice(len(cand), size=min(per, len(cand)), replace=False)
        for j in sorted(pick):
            f, o = cand[j]
            xs.append(read_window(ev, f, o, w))
            meta.append(f"{src} {f} {o}")
    np.stack(xs).astype(np.float32).tofile(out)
    with open(out + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(meta) + "\n")
    print(out, len(xs), "windows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
