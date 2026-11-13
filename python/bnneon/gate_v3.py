"""Shared parts of the BirdNET v3.0 gate scorers.

scripts/gate_v3_pkg.py scores every v3 gate window (gate grid: overlap
2.2, 1/6 s units, stride 8 units) with an engine command over --pipe, on
audio decoded as the v3 harness does (bnneon.decode_v3). This module holds
the window grid, the top-10 cache form and the window-level statistics; the
pipe protocol and the job pool are bnneon.pipe's (feed and pooled are
re-exported here). It needs numpy only.
"""
from collections.abc import Callable, Iterator
import contextlib
import json
import os
import queue
import threading
from typing import Any

import numpy as np

from . import agree
from . import pipe
from . import refrun

SR, WIN, SU, TH, TOPK, BATCH = 32000, 160000, 8, 0.84, 10, 8
STAT_SETS = ("all", "subset", "rest")


def read_streams(path: str, env_filter: bool = True) -> dict[str, Any]:
    """streams.json; with env_filter, only the STREAMS (a comma list, for a
    smoke run) when that environment variable is set."""
    with open(path, encoding="utf-8") as f:
        streams = json.load(f)
    if env_filter and os.environ.get("STREAMS"):
        keep = os.environ["STREAMS"].split(",")
        streams = {k: v for k, v in streams.items() if k in keep}
    return streams


def session_options() -> Any:
    """Quiet ONNX Runtime CPU options with ORT_THREADS (default 3) intra-op
    threads, the scorers' reference setting."""
    return refrun.session_options(int(os.environ.get("ORT_THREADS", "3")))


def grid(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(window starts in 1/6 s units, int32 sample offsets) of a stream."""
    n_units = int(np.floor((len(x) - WIN) / (SR / 6)))
    starts = np.arange(0, n_units + 1, SU, dtype=np.int32)
    offs = np.array([int(round(s * SR / 6)) for s in starts], np.int32)
    return starts, offs


def windows(x: np.ndarray, ob: np.ndarray) -> np.ndarray:
    """The windows at the sample offsets ob, stacked."""
    return np.stack([x[o:o + WIN] for o in ob])


# the --pipe input writer and the stream pool (the tests call gate_v3.pooled)
feed = pipe.feed
pooled = pipe.pooled


def topk(c: np.ndarray,
         mapping: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The top-10 per row, descending: (int16 classes, float16 confidences).

    Args:
        c: [n, classes] confidences.
        mapping: maps a class to the index stored (regional -> global), or
            None.

    Returns:
        ([n, TOPK] int16, [n, TOPK] float16).
    """
    top = np.argpartition(-c, TOPK, axis=1)[:, :TOPK]
    tc = np.take_along_axis(c, top, axis=1)
    o = np.argsort(-tc, axis=1)
    i = np.take_along_axis(top, o, axis=1)
    if mapping is not None:
        i = mapping[i]
    return i.astype(np.int16), np.take_along_axis(tc, o, axis=1).astype(
        np.float16)


def logit(a: np.ndarray) -> np.ndarray:
    """float64 logits of confidences clipped to [1e-4, 1-1e-4]."""
    a = np.clip(a.astype(np.float64), 1e-4, 1 - 1e-4)
    return np.log(a / (1 - a))


class Acc:
    """Window-level statistics of reference against candidate confidences.

    Logit SQNR, top-1 agreement, max / mean |dconf| (with the worst entry),
    and hit-set Jaccard and flips at TH with the reference-side margins.
    """

    def __init__(self):
        self.sig = self.noise = self.sabs = 0.0
        self.n = self.inter = self.union = self.flips = self.top1 = 0
        self.rows = 0
        self.dmax = 0.0
        self.worst = None
        self.margins = []

    def add(self, r: np.ndarray, c: np.ndarray, name: str, i0: int,
            cols: np.ndarray) -> None:
        """Adds a batch: reference r and candidate c over columns cols."""
        self.add_logits(r, c, logit(r), logit(c), name, i0, cols)

    def add_logits(self, r: np.ndarray, c: np.ndarray, lr: np.ndarray,
                   lc: np.ndarray, name: str, i0: int,
                   cols: np.ndarray) -> None:
        """add() with the logits lr = logit(r), lc = logit(c) given (the
        scorer converts a batch once and hands each column set a slice)."""
        self.sig += float((lr ** 2).sum())
        self.noise += float(((lr - lc) ** 2).sum())
        d = np.abs(r.astype(np.float64) - c)
        self.sabs += float(d.sum())
        self.n += d.size
        k = np.unravel_index(d.argmax(), d.shape)
        if d[k] > self.dmax:
            self.dmax = float(d[k])
            self.worst = (name, int(i0 + k[0]), int(cols[k[1]]), float(r[k]),
                          float(c[k]))
        inter, union, fl = agree.hits(r, c, TH)
        self.inter += inter
        self.union += union
        self.flips += int(fl.sum())
        self.margins.extend(np.abs(r[fl] - TH).astype(float).tolist())
        self.top1 += int((r.argmax(1) == c.argmax(1)).sum())
        self.rows += len(r)

    def merge(self, o: "Acc") -> None:
        """Adds another accumulator's counts."""
        for a in ("sig", "noise", "sabs", "n", "inter", "union", "flips",
                  "top1", "rows"):
            setattr(self, a, getattr(self, a) + getattr(o, a))
        self.margins += o.margins
        if o.dmax > self.dmax:
            self.dmax, self.worst = o.dmax, o.worst

    def report(self) -> dict[str, Any]:
        """The statistics as a dict (window_stats.json form)."""
        le_0005, le_002, gt_005, top = agree.margin_counts(
            np.array(self.margins))
        return dict(windows=self.rows,
                    sqnr_db=10 * np.log10(self.sig / max(self.noise, 1e-30)),
                    top1=self.top1 / max(1, self.rows), max_dconf=self.dmax,
                    mean_dconf=self.sabs / max(1, self.n),
                    jaccard=self.inter / max(1, self.union),
                    hits_union=self.union, flips=self.flips,
                    margin_le_0005=le_0005, margin_le_002=le_002,
                    margin_gt_005=gt_005, max_margin=top, worst=self.worst)


class SessionPool:
    """Reference sessions for the scorer's streams, made once and reused.

    At most n sessions exist (one per job); a stream borrows one for its
    whole run, so no session is used by two threads at a time.
    """

    def __init__(self, make: Callable[[], Any], n: int):
        self.make = make
        self.n = n
        self.made = 0
        self.free: queue.Queue = queue.Queue()
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def borrow(self) -> Iterator[Any]:
        """A session for the duration of the with block."""
        with self.lock:
            fresh = self.free.empty() and self.made < self.n
            if fresh:
                self.made += 1
        if fresh:
            try:
                s = self.make()
            except BaseException:
                with self.lock:  # the slot is free again
                    self.made -= 1
                raise
        else:
            s = self.free.get()
        try:
            yield s
        finally:
            self.free.put(s)


def stats_line(head: str, r: dict[str, Any]) -> str:
    """The summary line of one statistics dict."""
    return (f"{head}: {r['windows']} windows, SQNR {r['sqnr_db']:.1f} dB, "
            f"top-1 {r['top1']:.4f}, max|dconf| {r['max_dconf']:.4f} (mean "
            f"{r['mean_dconf']:.2e}), Jaccard@{TH} {r['jaccard']:.4f}, flips "
            f"{r['flips']} (<=0.005 {r['margin_le_0005']}, <=0.02 "
            f"{r['margin_le_002']}, >0.05 {r['margin_gt_005']}, max "
            f"{r['max_margin']:.4f}); worst {r['worst']}")
