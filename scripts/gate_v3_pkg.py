"""BirdNET v3.0 package gate scorer (gates V002 and K2B-G1).

Every v3 gate window (gate grid, overlap 2.2 -> stride 8/6 s; 92,829
windows) is scored by one or more engine commands (the CLI's --pipe on a
.bnpk package, which returns logits: the sigmoid is applied here in float64)
and by fp32 ONNX Runtime (CPU) of the same model on this host, once per
window for all candidates, in the same run.

Full outputs are never stored. Per stream and candidate it writes
<out>/<label>/scores/<stream>.npz with the top-10 of each side over the
model's classes as GLOBAL label indices (--map: the regional indices file
maps class c to global index map[c]; the v3 harness's form), idx int16,
conf float16, and accumulates window-level statistics over all the model's
classes (and, with --subset, over a class subset and its complement): logit
SQNR (logits of confidences clipped to [1e-4, 1-1e-4], as the v3 harness),
top-1 agreement, max / mean |dconf|, hit-set Jaccard and flips at 0.84 with
the reference-side margins. Written to <out>/<label>/window_stats.json.
ORT_THREADS (default 3) sets ONNX Runtime's threads; STREAMS (a comma list)
limits a smoke run.

usage: python -I gate_v3_pkg.py <evalset_dir> <model.onnx> <out_dir> <jobs>
           [--map IDX] [--subset IDX]
           -- <label> <engine cmd ...> [-- <label> <engine cmd ...>]
  e.g. ... -- fp32 docker exec -i -e BNN_THREADS=2 bnneon-eval /root/bS/cli
           /root/g/nae-fp32.bnpk 2 --pipe
Audio decoding = the v3 harness's (ffmpeg s16le / 32768, soxr VHQ to
32 kHz).

A stream whose engine output arrives short, or whose engine exits nonzero, is
scored again from its first window (up to STREAM_TRIES times; every retry is
printed). A stream's statistics and scores are only added once it is complete,
so a retry cannot change a result: it only replaces a run that lost bytes on
the way from the engine.
"""
import dataclasses
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np

from bnneon import decode_v3
from bnneon import gate_v3
from bnneon import pipe
from bnneon import refrun
from bnneon.compiler import cliargs

META_KEYS = ("command", "model")
# a stream is scored at most this many times while its output is incomplete
STREAM_TRIES = 3


@dataclasses.dataclass
class Setup:
    """The command line, the class maps and the ORT session options."""
    ev: str
    model: str
    out: str
    jobs: int
    cands: list[tuple[str, list[str]]]  # (label, engine command)
    sub: np.ndarray | None
    cmap: np.ndarray | None  # regional class -> global label index
    so: Any  # onnxruntime.SessionOptions
    sessions: gate_v3.SessionPool | None = None  # made by setup_from


class StreamIncomplete(Exception):
    """The engine's output for a stream was short, or the engine failed."""


def split_cands(rest: list[str]) -> list[tuple[str, list[str]]]:
    """`<label> <cmd ...> [-- <label> <cmd ...>]` as (label, cmd) pairs."""
    cands = []
    while rest:
        k = rest.index("--") if "--" in rest else len(rest)
        cands.append((rest[0], rest[1:k]))
        rest = rest[k + 1:]
    return cands


def setup_from(argv: list[str]) -> Setup:
    """Parses the command line; creates <out>/<label>/scores."""
    cut = argv.index("--")
    ev, model, out, jobs = argv[0], argv[1], argv[2], int(argv[3])
    opts = argv[4:cut]
    mapf = opts[opts.index("--map") + 1] if "--map" in opts else None
    subf = opts[opts.index("--subset") + 1] if "--subset" in opts else None
    cands = split_cands(argv[cut + 1:])
    decode_v3.ffmpeg_exe()  # resolved first: fails before any I/O
    for lab, _ in cands:
        os.makedirs(os.path.join(out, lab, "scores"), exist_ok=True)
    sub = cliargs.read_indices(subf) if subf else None
    cmap = cliargs.read_indices(mapf) if mapf else None
    su = Setup(ev=ev, model=model, out=out, jobs=jobs, cands=cands, sub=sub,
               cmap=cmap, so=gate_v3.session_options())
    su.sessions = gate_v3.SessionPool(
        lambda: refrun.session(su.model, su.so), jobs)
    return su


def stat_columns(sub: np.ndarray | None, nc: int) -> dict[str, np.ndarray]:
    """The columns of the all / subset / rest statistics."""
    sub = sub if sub is not None else np.arange(nc)
    return {"all": np.arange(nc), "subset": sub,
            "rest": np.setdiff1d(np.arange(nc), sub)}


def start_engines(su: Setup, offs: np.ndarray, x: np.ndarray) -> tuple:
    """Starts every candidate's engine and its stdin feeder thread."""
    started = [pipe.start(cmd, offs, x) for _, cmd in su.cands]
    return [p for p, _ in started], [t for _, t in started]


def sigmoid_conf(raw: bytes, n: int, nc: int) -> np.ndarray:
    """float32 sigmoid (computed in float64) of n x nc float32 logits."""
    lg = np.frombuffer(raw, np.float32).reshape(n, nc).astype(np.float64)
    return (1.0 / (1.0 + np.exp(-lg))).astype(np.float32)


def read_outputs(procs: list, p: subprocess.Popen, nc: int, n: int,
                 what: str, b: int, total: int) -> bytes:
    """n x nc float32 from p's stdout (windows b.. of total).

    Raises:
        StreamIncomplete: the engine stopped short (every engine of the
            stream is killed first, so the retry starts clean).
    """
    raw = pipe.read_exact(p.stdout, 4 * nc * n)
    if len(raw) != 4 * nc * n:
        for q in procs:
            q.kill()
        got = 4 * nc * b + len(raw)
        raise StreamIncomplete(f"{what}: engine output short at window {b}: "
                               f"{got} of {4 * nc * total} bytes "
                               f"(rc {p.poll()})")
    return raw


def finish(su: Setup, name: str, procs: list, ths: list, x: np.ndarray,
           starts: np.ndarray, ref_top: tuple, cand_top: list) -> None:
    """Waits for the engines and writes each candidate's top-10 cache.

    Raises:
        StreamIncomplete: an engine exited nonzero (nothing is written).
    """
    for t in ths:
        t.join()
    ri, rc = ref_top
    for j, p in enumerate(procs):
        p.wait()
        if p.returncode != 0:
            raise StreamIncomplete(f"{name} {su.cands[j][0]}: engine rc "
                                   f"{p.returncode}")
    for j in range(len(procs)):
        ci, cc = cand_top[j]
        np.savez_compressed(os.path.join(su.out, su.cands[j][0], "scores",
                                         name + ".npz"),
                            starts=starts, duration=len(x) / gate_v3.SR,
                            ref_idx=np.concatenate(ri),
                            ref_conf=np.concatenate(rc),
                            g_idx=np.concatenate(ci),
                            g_conf=np.concatenate(cc))


def run(su: Setup, name: str, x: np.ndarray) -> tuple:
    """Scores one stream on the reference and every candidate.

    Returns:
        (name, number of windows, [{stat set: gate_v3.Acc}] per candidate).
    """
    with su.sessions.borrow() as sess:
        return run_with(su, sess, name, x)


def _add_batch(acc: dict, cols: dict, ref: np.ndarray, cand: np.ndarray,
               name: str, b: int, aliased: bool) -> None:
    """Adds one batch to a candidate's statistics, the logits converted once
    (aliased: no --subset, the subset is all classes and is not counted
    twice)."""
    lr, lc = gate_v3.logit(ref), gate_v3.logit(cand)
    for k, a in acc.items():
        if len(cols[k]) and not (aliased and k == "subset"):
            c = cols[k]
            a.add_logits(ref[:, c], cand[:, c], lr[:, c], lc[:, c], name, b,
                         c)


def run_with(su: Setup, sess: Any, name: str, x: np.ndarray) -> tuple:
    """run() on a borrowed reference session."""
    starts, offs = gate_v3.grid(x)
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    procs, ths = start_engines(su, offs, x)
    acc = [{} for _ in su.cands]
    ref_top = ([], [])
    cand_top = [([], []) for _ in su.cands]
    nc = None
    for b in range(0, len(starts), gate_v3.BATCH):
        ob = offs[b:b + gate_v3.BATCH]
        ref = sess.run([oname], {iname: gate_v3.windows(x, ob)})[0]
        if nc is None:
            nc = ref.shape[1]
            cols = stat_columns(su.sub, nc)
            for a in acc:
                for k in cols:
                    a[k] = gate_v3.Acc()
        _append_top(ref_top, gate_v3.topk(ref, su.cmap))
        for j, p in enumerate(procs):
            what = f"{name} {su.cands[j][0]}"
            raw = read_outputs(procs, p, nc, len(ob), what, b, len(starts))
            cand = sigmoid_conf(raw, len(ob), nc)
            _add_batch(acc[j], cols, ref, cand, name, b, su.sub is None)
            _append_top(cand_top[j], gate_v3.topk(cand, su.cmap))
    finish(su, name, procs, ths, x, starts, ref_top, cand_top)
    return name, len(starts), acc


def run_complete(su: Setup, name: str, x: np.ndarray) -> tuple:
    """run(), again from the start while the engine's output is incomplete."""
    for attempt in range(1, STREAM_TRIES + 1):
        try:
            return run(su, name, x)
        except StreamIncomplete as e:
            print(f"RETRY {attempt} of {STREAM_TRIES}: {e}", flush=True)
    raise SystemExit(f"{name}: incomplete engine output {STREAM_TRIES} times")


def _append_top(lists: tuple[list, list], top: tuple) -> None:
    """Appends a batch's (classes, confidences) to the two lists."""
    lists[0].append(top[0])
    lists[1].append(top[1])


def write_results(su: Setup, tot: list[dict]) -> None:
    """Writes each candidate's window_stats.json and prints its lines."""
    for j, (lab, cmd) in enumerate(su.cands):
        res = {}
        for k, v in tot[j].items():
            if k == "subset" and su.sub is None:
                if "all" in res:  # no --subset: the subset is all classes
                    res[k] = res["all"]
            elif v.rows:
                res[k] = v.report()
        res["command"] = cmd
        res["model"] = su.model
        with open(os.path.join(su.out, lab, "window_stats.json"), "w",
                  encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        for k, r in res.items():
            if k in META_KEYS:
                continue
            print(gate_v3.stats_line(f"{lab} {k:6s}", r))


def main(argv: list[str]) -> int:
    """Scores every stream, writes the statistics, prints the summary."""
    su = setup_from(argv[1:])
    streams = gate_v3.read_streams(os.path.join(su.ev, "results",
                                                "streams.json"))
    t0 = time.time()
    tot = None

    def load(files):
        return decode_v3.load_stream(files, su.ev)

    def score(name, x):
        return run_complete(su, name, x)

    for name, n, acc in gate_v3.pooled(streams, su.jobs, load, score):
        if tot is None:
            tot = [{k: gate_v3.Acc() for k in a} for a in acc]
        line = []
        for j, a in enumerate(acc):
            for k in tot[j]:
                tot[j][k].merge(a[k])
            r = a["all"].report()
            line.append(f"{su.cands[j][0]} {r['sqnr_db']:.1f} dB top-1 "
                        f"{r['top1']:.4f} flips {r['flips']}")
        print(f"{name}: {n} windows; " + "; ".join(line)
              + f" ({time.time() - t0:.0f} s)", flush=True)
    write_results(su, tot)
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
