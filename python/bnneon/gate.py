"""The generic accuracy gate: a compiled package against its reference model.

`python -I scripts/gate_generic.py declare|score ...` runs main(); USAGE (the
text printed for a bad command line) says what the gate computes and how it
decides. In short: `declare` writes the pass criteria once, before any
scoring; `score` runs every candidate package through the engine CLI and the
reference model (ONNX Runtime or LiteRT) on the same windows and reports
window-level agreement, and with a truth file, precision and recall with
paired-bootstrap intervals.
"""
from collections.abc import Callable, Sequence
import dataclasses
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, NoReturn

import numpy as np

from . import agree
from . import refrun
from .compiler import cliargs
from .compiler import errs
from .compiler import parity
from .compiler import runner
from .compiler import windows

USAGE = (
    "gate_generic: an agreement gate for a compiled package against its "
    "reference model, on your own audio.\n"
    "\n"
    "Declare the pass criteria BEFORE scoring (the criteria file is written "
    "once and never overwritten), then score every\n"
    "candidate against the reference on the same windows and report each "
    "one, pass or fail:\n"
    "\n"
    "  python -I scripts/gate_generic.py declare criteria.json "
    "[--thresholds 0.1,0.5] [--jaccard 0.97] [--flip-margin 0.05]\n"
    "        [--mean-pp -0.5] [--max-dconf X] [--sqnr-db X] [--top1 X] "
    "[--note TEXT]\n"
    "  python -I scripts/gate_generic.py score criteria.json --ref "
    "REF.onnx|REF.tflite --cand PKG.bnpk[:MODE] [--cand ...]\n"
    "        --cli build/cli --post sigmoid|softmax (--windows-file "
    "W.npy|W.f32 | --audio A.wav ...) [--sample-rate HZ]\n"
    "        [--ref-output TENSOR] [--truth truth.csv --labels labels.txt] "
    "[--label LABEL] [--json OUT.json]\n"
    "\n"
    "Statistics (per candidate, per declared threshold): window top-1 "
    "agreement; hit-set Jaccard over (window, class)\n"
    "detections; flips (detections on one side only) with their margins "
    "|reference conf - threshold| (<= 0.005, <= 0.02,\n"
    "> 0.05, max); max |dconf|; logit SQNR. With --truth (CSV: "
    "window_index,label per row; labels by name from --labels or\n"
    "by class index), precision and recall at each threshold for reference "
    "and candidate, with paired-bootstrap 95 %\n"
    "intervals of dP and dR over windows (2000 resamples).\n"
    "\n"
    "Pass per candidate, as the published gates' rule: at every threshold "
    "hit-set Jaccard > jaccard and no flip margin >\n"
    "flip_margin; with truth also dP and dR mean >= mean_pp with CI upper "
    "bound >= 0. Three optional, stricter lines\n"
    "(off unless declared): max |dconf| <= max_dconf, SQNR >= sqnr_db, "
    "top-1 agreement >= top1. A detection is\n"
    "conf > threshold (BirdNET-Go's rule); a threshold with no detection on "
    "either side counts as agreement (Jaccard 1).\n"
    "This is the window-level half of the published gates (the other half, "
    "an external harness, also replays BirdNET-Go's\n"
    "detection post-processing over annotated field recordings and "
    "bootstraps over recording periods, not windows); it\n"
    "says whether a package reproduces its reference on the audio you have, "
    "not what the model's accuracy is. The\n"
    "criteria file is created once (never overwritten, made read-only) and "
    "`score` records its sha256: keep both files\n"
    "with the results. Exit 0 = every candidate passes, 1 = some fail, 3 = "
    "usage."
)

Truth = dict[int, set[int]]


# option lookup as in the bnc commands (the tests call gate.opt / opt_all)
opt = cliargs.opt
opt_all = cliargs.opt_all


def die(msg: str, code: int = 3) -> NoReturn:
    """Prints `error: msg` to stderr and exits with code."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def sha(b: bytes) -> str:
    """The hex sha256 of b."""
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ declare
def _float_opt(args: Sequence[str], k: str) -> float | None:
    """Option k as a float, or None when it is not given."""
    return float(opt(args, k)) if opt(args, k) is not None else None


def _criteria(args: Sequence[str]) -> dict[str, Any]:
    """The criteria dict of a `declare` command line (key order is kept)."""
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    thresholds = opt(args, "--thresholds", "0.1,0.5").split(",")
    return dict(declared_utc=now,
                thresholds=[float(x) for x in thresholds],
                jaccard=float(opt(args, "--jaccard", "0.97")),
                flip_margin=float(opt(args, "--flip-margin", "0.05")),
                mean_pp=float(opt(args, "--mean-pp", "-0.5")),
                max_dconf=_float_opt(args, "--max-dconf"),
                sqnr_db=_float_opt(args, "--sqnr-db"),
                top1=_float_opt(args, "--top1"),
                note=opt(args, "--note", ""))


def cmd_declare(args: Sequence[str]) -> int:
    """`declare CRITERIA.json [options]`: writes the criteria file once.

    Args:
        args: the command line after the program name (args[0] is
            "declare").

    Returns:
        0. Exits with 3 (via die) if the file exists already.
    """
    path = args[1]
    crit = _criteria(args)
    try:
        # created once: an existing file is never overwritten
        with open(path, "x", encoding="utf-8") as f:
            json.dump(crit, f, indent=1)
    except FileExistsError:
        die(f"{path} exists: criteria are declared once; write a new file "
            "for new criteria (and say why)")
    os.chmod(path, 0o444)
    with open(path, "rb") as f:
        digest = sha(f.read())
    print(f"declared {path} (read-only; sha256 {digest}; record it with the "
          f"results): {json.dumps(crit)}")
    return 0


# --------------------------------------------------------------- references
def _tflite_logits(path: str, x: np.ndarray) -> np.ndarray:
    """LiteRT outputs of every window, one window per invoke."""
    run = refrun.tflite_runner(path)
    y = run(x) if run is not None else None
    if y is None:
        die(f"{path}: a .tflite reference needs ai_edge_litert or tensorflow")
    return y


# the graph node that applies each --post (bnc check reads the logits before it)
POST_OPS = {"sigmoid": "Sigmoid", "softmax": "Softmax"}


def _run_windows(s: Any, o: str, x: np.ndarray) -> np.ndarray:
    """Tensor o of session s for every window, one window per run."""
    i = s.get_inputs()[0].name
    return np.stack([s.run([o], {i: w.reshape(1, -1)})[0].reshape(-1)
                     for w in x])


def _logits_candidates(m: Any) -> list[str]:
    """The inputs of the Sigmoid / Softmax nodes that give graph outputs."""
    outs = [o.name for o in m.graph.output]
    cands = []
    for o in outs:
        p = refrun.producer(m, o)
        if p is not None and p.op_type in POST_OPS.values():
            cands.append(f"{p.input[0]} (before the {p.op_type} that gives "
                         f"'{o}')")
    return cands


def _session_at(m: Any, path: str, o: str, outs: list[str]) -> Any:
    """A session of model m (loaded from path) that outputs tensor o.

    Exits with 3 (via die) when the model has no such tensor.
    """
    if o not in outs and refrun.producer(m, o) is None:
        hint = "; ".join(_logits_candidates(m))
        die(f"--ref-output {o}: {path} has no tensor of that name; graph "
            f"outputs: {', '.join(outs)}"
            + (f"; logits: {hint}" if hint else ""))
    return refrun.session_at(m, o)


def _looks_activated(y: np.ndarray, post: str) -> bool:
    """Whether outputs y look like --post already applied.

    Values in [0, 1], and rows summing to 1 for softmax.
    """
    if post not in POST_OPS or y.min() < 0 or y.max() > 1:
        return False
    return post == "sigmoid" or bool(np.allclose(y.sum(1), 1, atol=1e-3))


def _pre_activation(m: Any, o: str, post: str) -> str | None:
    """The input of the --post node that produces o in model m, or None."""
    p = refrun.producer(m, o)
    if p is None or p.op_type != POST_OPS[post]:
        return None
    return p.input[0]


def _onnx_logits(path: str, x: np.ndarray, out_name: str | None,
                 post: str = "none") -> tuple[np.ndarray, str]:
    """ONNX Runtime outputs of every window; returns (outputs, tensor).

    out_name may be an internal tensor (it is added as a graph output on an
    in-memory copy of the model). When the tensor read is the output of the
    --post activation itself (a Sigmoid with --post sigmoid, a Softmax with
    --post softmax), the gate reads that node's input instead, as bnc check
    takes the logits, so that the activation is not applied twice.
    """
    s = refrun.ort_session(path)
    outs = [t.name for t in s.get_outputs()]
    o = out_name or outs[-1]
    m = None  # the graph, loaded only when an internal tensor is needed
    if o not in outs:
        m = refrun.onnx_model(path)
        s = _session_at(m, path, o, outs)
    y = _run_windows(s, o, x)
    if _looks_activated(y, post):
        if m is None:
            m = refrun.onnx_model(path)
        logits = _pre_activation(m, o, post)
        if logits is not None:
            print(f"note: '{o}' is the {POST_OPS[post]} of '{logits}': "
                  f"scoring '{logits}' (the logits; --post {post} is applied "
                  f"to both sides here)", file=sys.stderr)
            o = logits
            y = _run_windows(_session_at(m, path, o, outs), o, x)
    return y, o


def ref_logits(path: str, x: np.ndarray, out_name: str | None = None,
               post: str = "none") -> tuple[np.ndarray, str]:
    """The reference model's outputs for every window.

    Args:
        path: a .tflite model (LiteRT) or an ONNX model (ONNX Runtime).
        x: the windows, [n, n_samples] float32.
        out_name: the ONNX tensor to read (default: the last graph output).
        post: the --post activation (an ONNX output that is that activation
            is read before it; see _onnx_logits).

    Returns:
        ([n, outputs] array, a description of the runtime and tensor).
    """
    if path.endswith(".tflite"):
        return _tflite_logits(path, x), "tflite"
    y, o = _onnx_logits(path, x, out_name, post)
    return y, f"onnxruntime:{o}"


# ------------------------------------------------------------------ windows
def load_windows(args: Sequence[str], nsamp: int) -> np.ndarray:
    """The windows to score, from --windows-file or --audio files.

    Args:
        args: the `score` command line.
        nsamp: the package's window length in samples.

    Returns:
        [n, nsamp] float32 windows. Exits with 3 (via die) on bad input.
    """
    wf = opt(args, "--windows-file")
    audio = opt_all(args, "--audio")
    if not wf and not audio:
        die("give --windows-file or --audio")
    sr = int(opt(args, "--sample-rate", "0"))
    try:
        if wf:
            return windows.from_file(wf, nsamp)
        # every full window, silent or not (mono mix, no resampling)
        return windows.from_audio(audio, nsamp, 0, sr, cliargs.print_stderr,
                                  skip_silence=False)
    except errs.Usage as e:
        die(str(e))


# ------------------------------------------------------------------ scoring
# the confidence function for --post and the logit SQNR, as bnc check's
post_fn = parity.post_fn
sqnr_db = parity.sqnr_db


def run_cli(cli: str, pkg: str, mode: int, win: str, n: int, nc: int,
            out: str) -> np.ndarray:
    """Scores the windows file with `cli PKG MODE WIN OUT 1`.

    Args:
        cli: the engine CLI.
        pkg: the package file.
        mode: the engine mode.
        win: the float32 windows file.
        n: the number of windows.
        nc: the number of classes.
        out: the output file the CLI writes.

    Returns:
        [n, nc] float32 outputs. Exits with 1 (via die) if the CLI fails or
        writes the wrong number of floats.
    """
    r = runner.run_cli(cli, pkg, mode, win, out)
    if r.returncode:
        die(f"{pkg} (mode {mode}): {r.stderr.strip() or r.stdout.strip()}", 1)
    y, msg = runner.read_output(out, n, nc)
    if y is None:
        die(f"{pkg}: {msg} (is this the right reference?)", 1)
    return y


def _label_index(labels: Sequence[str] | None) -> dict[str, int]:
    """Class index by label: the whole line, its scientific or common part."""
    by_name: dict[str, int] = {}
    for i, label in enumerate(labels or []):
        for key in {label} | set(label.split("_")[:2]):
            by_name.setdefault(key, i)
    return by_name


def _truth_class(lab: str, by_name: dict[str, int], nc: int) -> int:
    """The class of a truth label (a name or an index); dies if invalid."""
    c = by_name.get(lab)
    if c is None:
        if lab.isdigit():
            c = int(lab)
        else:
            die(f"truth: label '{lab}' is not in the label file (give "
                "--labels, or use class indices)")
    if not 0 <= c < nc:
        die(f"truth: class {c} out of range (0..{nc - 1})")
    return c


def read_truth(path: str, labels: Sequence[str] | None, nc: int,
               nwin: int) -> Truth:
    """Reads a truth CSV of rows window_index,label.

    Args:
        path: the CSV file (an optional header row, # comments allowed).
        labels: the label lines, or None (labels are then class indices).
        nc: the number of classes.
        nwin: the number of windows scored.

    Returns:
        {window: set of classes}. Exits with 3 (via die) on a bad row.
    """
    t: Truth = {}
    by_name = _label_index(labels)
    with open(path, encoding="utf-8") as f:
        for k, ln in enumerate(f):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            w, lab = ln.split(",", 1)
            if k == 0 and not w.strip().isdigit():
                continue  # a header row (window_index,label)
            w, lab = int(w), lab.strip()
            if not 0 <= w < nwin:
                die(f"truth: window {w} is outside 0..{nwin - 1} ({nwin} "
                    "windows scored)")
            c = _truth_class(lab, by_name, nc)
            t.setdefault(w, set()).add(c)
    return t


def prf_window(hits: np.ndarray, truth: Truth, n: int) -> np.ndarray:
    """Per-window (tp, fp, fn) of the detection sets against the truth.

    Args:
        hits: [n, nc] detections.
        truth: {window: set of classes}, every class < nc.
        n: the number of windows.

    Returns:
        [n, 3] int64.
    """
    pairs = [(w, c) for w, cs in truth.items() for c in cs]
    rows = np.array([w for w, _ in pairs], np.int64)
    cols = np.array([c for _, c in pairs], np.int64)
    tp = np.bincount(rows, weights=hits[rows, cols], minlength=n)
    tp = tp.astype(np.int64)
    n_true = np.bincount(rows, minlength=n).astype(np.int64)
    n_hit = hits[:n].sum(1).astype(np.int64)
    return np.stack([tp, n_hit - tp, n_true - tp], axis=1)


# precision / recall of tp / fp / fn counts (the tests call gate.prf)
prf = agree.prf


def _package_shape(cli: str, cand: str) -> tuple[int, int]:
    """(n_samples, n_classes) of a candidate from `cli PKG --info`."""
    info = subprocess.run([cli, cand.split(":")[0], "--info"],
                          capture_output=True, text=True, check=False).stdout
    try:
        nsamp = int(info.split("n_samples ")[1].split()[0])
        nc = int(info.split("n_classes ")[1].split()[0])
    except (IndexError, ValueError):
        die(f"{cand}: cannot read n_samples / n_classes from `cli --info`: "
            f"{info.strip()[:200]}")
    return nsamp, nc


def _read_labels(path: str | None) -> list[str] | None:
    """The non-blank lines of the --labels file, or None without one."""
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return [label for label in f.read().splitlines() if label.strip()]


def _strict_checks(res: dict[str, Any],
                   crit: dict[str, Any]) -> tuple[bool, list[str]]:
    """The finite-logits check and the three optional stricter criteria."""
    ok = res["finite"]
    why = []
    if not res["finite"]:
        why.append("non-finite logits")
    if crit.get("top1") is not None and res["top1"] < crit["top1"]:
        ok = False
        why.append(f"top-1 {res['top1']:.3f} < {crit['top1']}")
    if crit.get("sqnr_db") is not None and res["sqnr_db"] < crit["sqnr_db"]:
        ok = False
        why.append(f"SQNR {res['sqnr_db']} dB < {crit['sqnr_db']}")
    if (crit.get("max_dconf") is not None
            and res["max_dconf"] > crit["max_dconf"]):
        ok = False
        why.append(f"max |dconf| {res['max_dconf']:.4f} > {crit['max_dconf']}")
    return ok, why


def _agreement(pr: np.ndarray, pc: np.ndarray,
               th: float) -> tuple[float, dict[str, Any]]:
    """Hit-set Jaccard and flip statistics at one threshold.

    Returns:
        (the Jaccard, unrounded; the statistics dict, Jaccard rounded).
    """
    # a detection is conf > threshold (BirdNET-Go)
    inter, union, fl = agree.hits(pr, pc, th)
    jac = 1.0 if union == 0 else inter / union
    le_005, le_02, gt_05, flip_max = agree.margin_counts(np.abs(pr[fl] - th))
    return jac, dict(jaccard=round(jac, 4), union=union, flips=int(fl.sum()),
                     flip_le_005=le_005, flip_le_02=le_02, flip_gt_05=gt_05,
                     flip_max=flip_max)


@dataclasses.dataclass
class _RefTruth:
    """The reference side of one threshold's truth statistics, the same for
    every candidate: the resamples, its counts and bootstrapped P / R."""
    bidx: np.ndarray
    p0: np.ndarray
    prec0: np.ndarray
    rec0: np.ndarray


def _ref_truth(pr: np.ndarray, thresholds: Sequence[float], truth: Truth,
               n: int) -> list[_RefTruth]:
    """_RefTruth per declared threshold, in order; the resamples are drawn as
    each candidate drew them before (one generator seeded 0, thresholds in
    order)."""
    rng = np.random.default_rng(0)
    out = []
    for th in thresholds:
        bidx = agree.bootstrap_index(rng, n)
        p0 = prf_window(pr > th, truth, n)
        prec0, rec0 = agree.boot_prf(p0, bidx)
        out.append(_RefTruth(bidx=bidx, p0=p0, prec0=prec0, rec0=rec0))
    return out


def _truth_stats(rt: _RefTruth, hb: np.ndarray, truth: Truth,
                 n: int) -> dict[str, Any]:
    """Counts and paired-bootstrap intervals of dP and dR over windows."""
    p0 = rt.p0
    p1 = prf_window(hb, truth, n)
    prec1, rec1 = agree.boot_prf(p1, rt.bidx)
    dp, dr = 100 * (prec1 - rt.prec0), 100 * (rec1 - rt.rec0)
    tp0, fp0, fn0 = agree.counts(p0)
    tp1, fp1, fn1 = agree.counts(p1)
    return dict(dP=[float(v) for v in agree.ci(dp)],
                dR=[float(v) for v in agree.ci(dr)],
                dP_mean=float(dp.mean()), dR_mean=float(dr.mean()),
                ref=dict(tp=tp0, fp=fp0, fn=fn0),
                cand=dict(tp=tp1, fp=fp1, fn=fn1))


def _counts_line(side: str, c: dict[str, int]) -> str:
    """`     reference TP/FP/FN ...  P x %  R y %` for one side."""
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p, r = agree.pr_percent(tp, fp, fn)
    return f"     {side} TP/FP/FN {tp}/{fp}/{fn}  P {p:.2f} %  R {r:.2f} %"


def _truth_lines(ci: dict[str, Any], n: int) -> list[str]:
    """The three report lines of the truth statistics."""
    return [_counts_line("reference", ci["ref"]),
            _counts_line("candidate", ci["cand"]),
            f"     dP {ci['dP_mean']:+.2f} pp [{ci['dP'][0]:+.2f}, "
            f"{ci['dP'][1]:+.2f}]  dR {ci['dR_mean']:+.2f} pp "
            f"[{ci['dR'][0]:+.2f}, {ci['dR'][1]:+.2f}]"
            f"  ({n} windows; bootstrap over windows)"]


def _truth_ok(ci: dict[str, Any], crit: dict[str, Any]) -> bool:
    """Both CI upper bounds >= 0 and both means >= mean_pp."""
    return (ci["dP"][1] >= 0 and ci["dR"][1] >= 0
            and ci["dP_mean"] >= crit["mean_pp"]
            and ci["dR_mean"] >= crit["mean_pp"])


def _score_threshold(pr: np.ndarray, pc: np.ndarray, th: float,
                     crit: dict[str, Any], truth: Truth | None,
                     rt: _RefTruth | None, lines: list[str],
                     why: list[str]) -> tuple[dict[str, Any], bool]:
    """One threshold of one candidate: statistics, report lines, verdict.

    Args:
        pr: the reference confidences, [n, nc].
        pc: the candidate confidences, [n, nc].
        th: the threshold.
        crit: the declared criteria.
        truth: the truth sets, or None.
        rt: the threshold's reference truth side (with truth).
        lines: the report lines (appended to).
        why: the candidate's failure reasons (appended to).

    Returns:
        (the threshold's statistics, whether it passes).
    """
    jac, t = _agreement(pr, pc, th)
    union = t["union"]
    tok = ((union == 0 or jac > crit["jaccard"])
           and t["flip_max"] <= crit["flip_margin"])
    if not tok:
        why.append(f"@{th}: Jaccard {jac:.4f} / max flip margin "
                   f"{t['flip_max']:.4f}")
    lines.append(f"  @{th}: hit-set Jaccard {jac:.4f} (union {union}); flips "
                 f"{t['flips']} (margin <=0.005 {t['flip_le_005']}, "
                 f"<=0.02 {t['flip_le_02']}, >0.05 {t['flip_gt_05']}, max "
                 f"{t['flip_max']:.4f})")
    if truth is not None:
        n = len(pr)
        ci = _truth_stats(rt, pc > th, truth, n)
        t["truth"] = ci
        lines.extend(_truth_lines(ci, n))
        pok = _truth_ok(ci, crit)
        if not pok:
            why.append(f"@{th}: dP/dR {ci['dP_mean']:+.2f}/"
                       f"{ci['dR_mean']:+.2f} pp")
        tok = tok and pok
    return t, tok


@dataclasses.dataclass
class _Scoring:
    """What every candidate of one `score` run is compared against."""
    crit: dict[str, Any]
    cli: str
    ref: np.ndarray  # the reference outputs, [n, nc]
    pr: np.ndarray  # the reference confidences
    pf: Callable[[np.ndarray], np.ndarray]
    truth: Truth | None
    ref_truth: list[_RefTruth] | None  # per declared threshold, with truth
    win: str  # the windows file the CLI reads
    tmpd: str
    n: int
    nc: int


def _not_run(pkg: str, mode: int, lines: list[str]) -> dict[str, Any]:
    """The result of a candidate whose package did not run."""
    lines.append(f"{os.path.basename(pkg)} (mode {mode}): did not run: see "
                 "above\n  gate: FAIL (the package did not run)")
    return dict(package=pkg, mode=mode, ran=False, **{"pass": False},
                why=["did not run"])


def _score_candidate(sc: _Scoring, pkg: str, mode: int,
                     lines: list[str]) -> dict[str, Any]:
    """Scores one candidate package; appends its report lines.

    Args:
        sc: the run's reference side.
        pkg: the package file.
        mode: the engine mode.
        lines: the report lines (appended to).

    Returns:
        The candidate's result dict (key "pass" holds the verdict).
    """
    try:
        y = run_cli(sc.cli, pkg, mode, sc.win, sc.n, sc.nc,
                    os.path.join(sc.tmpd, "out.f32"))
    except SystemExit:
        return _not_run(pkg, mode, lines)
    pc = sc.pf(y)
    res = dict(package=pkg, mode=mode, ran=True,
               top1=float((sc.ref.argmax(1) == y.argmax(1)).mean()),
               sqnr_db=round(sqnr_db(sc.ref, y), 1),
               max_dconf=float(np.abs(sc.pr - pc).max()), thresholds={},
               finite=bool(np.isfinite(y).all()))
    ok, why = _strict_checks(res, sc.crit)
    lines.append(f"{os.path.basename(pkg)} (mode {mode}): top-1 "
                 f"{res['top1']:.4f}, logit SQNR {res['sqnr_db']} dB, max "
                 f"|dconf| {res['max_dconf']:.4f}")
    for k, th in enumerate(sc.crit["thresholds"]):
        rt = sc.ref_truth[k] if sc.ref_truth else None
        t, tok = _score_threshold(sc.pr, pc, th, sc.crit, sc.truth, rt,
                                  lines, why)
        res["thresholds"][str(th)] = t
        ok = ok and tok
    res["pass"], res["why"] = ok, why
    reasons = f" ({'; '.join(why)})" if why else ""
    lines.append(f"  gate: {'PASS' if ok else 'FAIL'}" + reasons)
    return res


def _read_criteria(cpath: str) -> tuple[bytes, dict[str, Any]]:
    """(raw bytes, parsed criteria) of the criteria file; dies if missing."""
    if not os.path.exists(cpath):
        die(f"{cpath}: declare the criteria first (`declare`)")
    with open(cpath, "rb") as f:
        craw = f.read()
    return craw, json.loads(craw)


def _reference(args: Sequence[str], ref_path: str, post: str,
               x: np.ndarray, nc: int) -> tuple[np.ndarray, str, Callable]:
    """(reference outputs, kind, post function); dies on a class mismatch."""
    ref, ref_kind = ref_logits(ref_path, x, opt(args, "--ref-output"), post)
    if ref.shape[1] != nc:
        die(f"reference produces {ref.shape[1]} outputs, the package {nc} "
            f"classes; {_logits_hint(ref_path)}")
    return ref, ref_kind, post_fn(post)


def _logits_hint(ref_path: str) -> str:
    """Which --ref-output to pass after a class-count mismatch."""
    if ref_path.endswith(".tflite"):
        return "use the reference model of this package's head"
    cands = _logits_candidates(refrun.onnx_model(ref_path))
    if not cands:
        return "use --ref-output for the logits tensor"
    return "use --ref-output with the logits tensor: " + "; ".join(cands)


def _overall_line(allok: bool, results: list[dict[str, Any]], ncands: int,
                  n: int) -> str:
    """The OVERALL report line."""
    npass = sum(c["pass"] for c in results)
    return (f"OVERALL: {'PASS' if allok else 'FAIL'} ({npass}/{ncands} "
            f"candidates pass; {n} windows: an agreement gate on your audio, "
            "not an accuracy measurement)")


def cmd_score(args: Sequence[str]) -> int:
    """`score CRITERIA.json --ref ... --cand ... --cli ... --post ...`.

    Args:
        args: the command line after the program name (args[0] is "score").

    Returns:
        0 if every candidate passes, else 1. Exits with 3 (via die) on bad
        input.
    """
    cpath = args[1]
    craw, crit = _read_criteria(cpath)
    ref_path = opt(args, "--ref")
    cli = opt(args, "--cli")
    post = opt(args, "--post")
    cands = opt_all(args, "--cand")
    if (not ref_path or not cli or not cands
            or post not in ("sigmoid", "softmax", "none")):
        die("need --ref, --cli, at least one --cand and --post "
            "sigmoid|softmax|none")
    # the window length from the first candidate's metadata
    nsamp, nc = _package_shape(cli, cands[0])
    x = load_windows(args, nsamp)
    n = len(x)
    ref, ref_kind, pf = _reference(args, ref_path, post, x, nc)
    pr = pf(ref)
    labels = _read_labels(opt(args, "--labels"))
    truth_path = opt(args, "--truth")
    truth = read_truth(truth_path, labels, nc, n) if truth_path else None
    tmpd = tempfile.mkdtemp(prefix="gate_")
    win = os.path.join(tmpd, "windows.f32")
    x.tofile(win)
    # the reference side of the truth statistics, once for every candidate
    ref_truth = (_ref_truth(pr, crit["thresholds"], truth, n)
                 if truth is not None else None)
    sc = _Scoring(crit=crit, cli=cli, ref=ref, pr=pr, pf=pf, truth=truth,
                  ref_truth=ref_truth, win=win, tmpd=tmpd, n=n, nc=nc)
    out = dict(criteria=cpath, criteria_sha256=sha(craw),
               criteria_declared=crit.get("declared_utc"), reference=ref_path,
               reference_kind=ref_kind, windows=n, classes=nc, post=post,
               candidates=[])
    lines = [f"criteria {cpath} (sha256 {out['criteria_sha256'][:16]}, "
             f"declared {crit.get('declared_utc')}); reference {ref_path} "
             f"({ref_kind}); {n} windows x {nc} classes; post {post}"]
    allok = True
    for cand in cands:
        pkg, _, mode = cand.partition(":")
        res = _score_candidate(sc, pkg, int(mode or 0), lines)
        allok = allok and res["pass"]
        out["candidates"].append(res)
    lines.append(_overall_line(allok, out["candidates"], len(cands), n))
    out["overall"] = "PASS" if allok else "FAIL"
    print("\n".join(lines))
    if opt(args, "--json"):
        with open(opt(args, "--json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    shutil.rmtree(tmpd, ignore_errors=True)
    return 0 if allok else 1


def main(argv: Sequence[str]) -> int:
    """Runs `declare` or `score`; prints USAGE (returns 0 for --help, else
    3) otherwise.

    Args:
        argv: the command line without the program name.

    Returns:
        The exit code.
    """
    a = list(argv)
    if "--help" in a or "-h" in a:
        print(USAGE)
        return 0
    if len(a) < 2 or a[0] not in ("declare", "score"):
        print(USAGE)
        return 3
    return cmd_declare(a) if a[0] == "declare" else cmd_score(a)
