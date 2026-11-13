"""bnc check: is a model ready to compile and ship?

Coverage, one package per path, the package run through the runtime on a few
windows, compared with ONNX Runtime (logit SQNR, max |dconf|, top-1), and a
PASS / WARN / FAIL verdict with the next step. The steps run in that order;
the first failing one ends the check with its exit code.
"""
from collections.abc import Sequence
import os
import time
from typing import Any

import numpy as np

from . import cliargs
from . import cmd_compile
from . import cmd_report
from . import errs
from . import lower
from . import parity
from . import prep
from . import recog
from . import runner
from . import windows

# the SQNR (dB) a path must reach against its reference (lower.PATHS
# need_db); below FAIL_DB fails. int8 GEMMs with dynamic per-row activations
# (the neon-int8 path, and the dotprod family on neon-fp16) reach 46-57 dB on
# v3.0 audio and show large single deviations; they are judged by the int8
# floor.
NEED_INT8_DB = 40.0
FAIL_DB = 35
# the fp32 path's allowance when ONNX Runtime's DFT matrix limits the reference
MATRIX_LIMITED_DB = 40.0
# options check passes on to compile (with a value / without one)
COMPILE_VALUE_OPTIONS = ("--append", "--at", "--logits", "--emb", "--dense",
                         "--classes", "--int8", "--wino")
COMPILE_FLAGS = ("--proto-s8", "--proto32-h16", "--no-fold-bn", "--no-canon")
# what a runtime failure says when the CPU, not the package, is the problem
CPU_REFUSALS = ("can run none of the package", "lacks")
VERDICT_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


class Finished(Exception):
    """Ends a check early: carries the exit code finish() returned."""

    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


class Check:
    """One bnc check run: its result document and its printer.

    Attributes:
        args: the command line.
        log: the print function (stderr when the JSON takes stdout).
        res: the --json document, filled step by step.
    """

    def __init__(self, args: Sequence[str]):
        self.args = args
        self.log = cliargs.text_printer(args)
        self.t0 = time.time()
        self.res: dict[str, Any] = {
            "model": args[1], "status": "PASS", "exit": errs.EXIT["ok"],
            "steps": [], "paths": {}, "next_step": "", "warnings": []}

    def opt(self, key: str, default: Any = None) -> Any:
        """The value of a check option (cliargs.opt)."""
        return cliargs.opt(self.args, key, default)

    def finish(self, status: str, code: str, next_step: str) -> int:
        """Records and prints the verdict, writes --json; returns the exit."""
        self.res["status"] = status
        self.res["exit"] = errs.EXIT[code]
        self.res["next_step"] = next_step
        self.res["seconds"] = round(time.time() - self.t0, 1)
        self.log(f"\n{status}: {next_step}")
        if "--json" in self.args:
            cliargs.json_out(self.args, self.res)
        return self.res["exit"]

    def stop(self, status: str, code: str, next_step: str) -> Finished:
        """finish(), as an exception that ends the run (raise the result)."""
        return Finished(self.finish(status, code, next_step))

    def warn(self, msg: str) -> None:
        """Records and prints a warning."""
        self.res["warnings"].append(msg)
        self.log(f"WARN: {msg}")

    def step(self, name: str, ok: bool, **extra: Any) -> None:
        """Records a step of the --json document."""
        self.res["steps"].append(dict(step=name, ok=ok, **extra))


def _parse_basics(c: Check) -> tuple[str, int, int | None]:
    """--classes refusal, --post, --windows and --sample-rate.

    Returns:
        (post, number of windows, sample rate or None).
    """
    if "--classes" in c.args:
        raise c.stop("FAIL", "usage",
                     "check does not take --classes: check the full head "
                     "first, then compile the slice with --classes and compare "
                     "it against the full package on the kept classes")
    post = c.opt("--post", "sigmoid")
    if post not in ("none", "sigmoid", "softmax"):
        raise c.stop("FAIL", "usage",
                     f"--post {post}: expected none, sigmoid or softmax")
    try:
        n = cliargs.opt_int(c.args, "--windows", 4)
        sr = cliargs.opt_int(c.args, "--sample-rate", 0) or None
    except errs.Usage as e:
        raise c.stop("FAIL", "usage", str(e)) from None
    return post, n, sr


def _first_primary(cov: dict[str, Any]) -> str:
    """The FAIL message for the first primary unsupported node."""
    p = cov["primary"][0]
    if p["index"] is not None:
        where = f"node {p['index']} {p['op_type']} '{p['name']}'"
    else:
        where = "the graph input"
    msg = (f"unsupported op {p['op_type']} at {where} ({p['reason']}): "
           f"{p['next_step']}")
    if len(cov["primary"]) > 1:
        msg += (f"; {len(cov['primary']) - 1} more primary unsupported "
                f"node(s) above")
    return msg


def _load_and_cover(c: Check):
    """Step 1: load, recognize, the coverage report and the logits tensor.

    Returns:
        (graph, IR ops, logits tensor, number of classes).
    """
    try:
        g, nf = prep.load(c.args)
    except (errs.ModelError, errs.Usage) as e:
        c.step("load", False, error=str(e))
        code = "model" if isinstance(e, errs.ModelError) else "usage"
        raise c.stop("FAIL", code,
                     f"the model cannot be loaded: {e}") from None
    r, ops = prep.recognize(g, "--no-fold-bn" not in c.args)
    cov = cmd_report.coverage(g, r, ops)
    lt = c.opt("--logits", cov["logits_default"])
    # only the nodes the requested outputs depend on count (a Sigmoid or an
    # unsupported tail after --logits is ignored)
    cone = prep.problems_in_cone(g, r, [lt, c.opt("--emb")])
    skipped = len(cov["unsupported"]) - len(cone)
    cov["unsupported"] = cone
    cov["primary"] = [p for p in cone if not p["downstream_of"]]
    c.res["coverage"] = cov
    cmd_report.print_coverage(cov, nf, c.log, "--verbose" in c.args,
                              "--quiet" in c.args)
    if skipped:
        c.log(f"({skipped} unsupported node(s) after the requested outputs "
              f"ignored)")
    c.step("coverage", not cov["unsupported"])
    if cov["primary"]:
        raise c.stop("FAIL", "unsupported", _first_primary(cov))
    if lt not in r.val:
        if lt not in g.prod and lt not in g.const:
            raise c.stop("FAIL", "usage",
                         f"--logits {lt}: no tensor of that name in the graph "
                         f"(outputs: {g.outputs})")
        raise c.stop("FAIL", "unsupported",
                     f"the logits tensor '{lt}' is not produced by a supported "
                     f"op; hint: pass --logits <pre-activation tensor> (the "
                     f"engine returns logits; the host applies --post)")
    nc = r.val[lt].C
    c.log(f"logits: '{lt}' [{nc}]; window {g.nsamp} samples")
    return g, ops, lt, nc


def _labels(c: Check, ops: Sequence[dict], nsamp: int, nc: int,
            sr: int | None, workdir: str) -> str:
    """Step 2: the label file (or placeholders); returns its path.

    The label-form warning is compile's: for the family a package at this
    rate gets on the v2.4 backbone (24 at 48000 Hz, the default).
    """
    labels = c.opt("--labels")
    if not labels:
        labels = os.path.join(workdir, "labels_placeholder.txt")
        with open(labels, "w", encoding="utf-8") as f:
            f.write("".join(f"class_{i}\n" for i in range(nc)))
        c.warn(f"no --labels: {nc} placeholder labels written to {labels}; "
               f"the package's label hash is meaningless")
        c.step("labels", True)
        return labels
    with open(labels, "rb") as f:
        raw = f.read()
    problem = prep.label_rule_error(labels, raw)
    if problem:
        c.step("labels", False)
        raise c.stop("FAIL", "labels", problem)
    lines = prep.class_labels(raw)
    c.res["labels"] = dict(file=labels, lines=len(lines), classes=nc)
    # compile's default on the v2.4 backbone is family 24 (256000 Hz: 224)
    family = prep.family_at_rate(sr or 48000) or "24"
    if prep.birdnetgo_labels(ops, nsamp, family):
        warning = prep.labels_warning(lines, labels)
        if warning:
            c.warn(warning)
    if len(lines) != nc:
        c.step("labels", False)
        raise c.stop("FAIL", "labels",
                     f"label file {labels} has {len(lines)} labels but the "
                     f"logits have {nc} classes; hint: use the label file of "
                     f"this head (one per line, output order) or check "
                     f"--logits / --classes")
    c.step("labels", True)
    return labels


def _windows(c: Check, nsamp: int, n: int, sr: int | None,
             workdir: str) -> tuple[np.ndarray, str]:
    """Step 3: the windows; returns them and the file they are written to."""
    audio = cliargs.opt_all(c.args, "--audio")
    wfile = c.opt("--windows-file")
    count = n if "--windows" in c.args or not wfile else 0
    try:
        x = windows.make_windows(nsamp, count, audio, sr, c.log, wfile)
    except errs.Usage as e:
        raise c.stop("FAIL", "usage", str(e)) from None
    win = os.path.join(workdir, "windows.f32")
    x.tofile(win)
    source = "file" if wfile else "audio" if audio else "synthetic"
    c.res["windows"] = dict(n=len(x), source=source)
    return x, win


def _check_rate(c: Check, sr: int | None, fe_v24: bool) -> None:
    """Refuses a sample rate the package could not honestly carry."""
    if not sr:
        c.warn("no --sample-rate: assuming 48000 Hz (the package metadata "
               "will say so)")
    elif sr <= 0:
        raise c.stop("FAIL", "usage",
                     f"--sample-rate {sr}: expected a positive rate in Hz")
    elif (fe_v24 and sr not in prep.FAMILY_RATE.values()
          and "--force" not in c.args):
        raise c.stop("FAIL", "usage",
                     f"--sample-rate {sr} with the BirdNET v2.4 front end "
                     f"(144000-sample window): that backbone takes 48000 Hz "
                     f"(BirdNET, family 24) or 256000 Hz (BattyBirdNET, family "
                     f"224) and the rate is baked into the package; use one of "
                     f"those (or --force)")


def _choose_post(c: Check, g, lt: str, post: str) -> str:
    """--post, else what the graph applies to the logits, else sigmoid."""
    if "--post" in c.args:
        return post
    graph_post = prep.default_post(g, lt)
    if graph_post:
        c.log(f"the graph applies {graph_post} to '{lt}': using --post "
              f"{graph_post} (the host applies it; the package returns logits)")
        return graph_post
    c.warn("no --post: assuming sigmoid (the host applies it to the logits); "
           "pass --post softmax for a softmax head")
    return post


def scratch_model_id(stem: str) -> str:
    """The model id of check's packages for a model file named stem.

    The package files are named after it too (<id>-<path>.bnpk), so the
    library's hint for a path this CPU cannot run names a file that exists.
    """
    return f"check_{stem}"[:60]


def compile_args(args: Sequence[str], labels: str, stem: str, post: str,
                 sr: int | None, path: str, out_pkg: str) -> list[str]:
    """The compile command line check runs for one path.

    The packages are throwaway, so --force: no license, attribution or
    family is needed.
    """
    cargs = ["compile", args[1], "--labels", labels,
             "--model-id", scratch_model_id(stem), "--family", "0",
             "--post", post, "--sample-rate", str(sr or 48000),
             "--paths", path, "-o", out_pkg]
    for k in COMPILE_VALUE_OPTIONS:
        if cliargs.opt(args, k):
            cargs += [k, cliargs.opt(args, k)]
    for k in COMPILE_FLAGS:
        if k in args:
            cargs.append(k)
    cargs.append("--force")
    return cargs


def _compile_paths(c: Check, g, labels: str, post: str, sr: int | None,
                   workdir: str) -> dict[str, str]:
    """Step 4: one package per path from the loaded graph g; returns {path:
    package file}."""
    paths = c.opt("--paths", "neon-fp16,neon-fp32").split(",")
    stem = os.path.splitext(os.path.basename(c.args[1]))[0]
    pkgs = {}
    for pth in paths:
        out_pkg = os.path.join(workdir,
                               f"{scratch_model_id(stem)}-{pth}.bnpk")
        cargs = compile_args(c.args, labels, stem, post, sr, pth, out_pkg)
        try:
            cmd_compile.compile_loaded(cargs, g, quiet=True)
            pkgs[pth] = out_pkg
            c.res["paths"][pth] = dict(package=out_pkg, compiled=True)
        except (recog.Unsupported, errs.Usage) as e:
            c.res["paths"][pth] = dict(compiled=False, error=str(e))
            c.log(f"path {pth}: cannot compile: {e}")
    c.step("compile", bool(pkgs))
    if not pkgs:
        e = next(iter(c.res["paths"].values()))["error"]
        raise c.stop("FAIL", "unsupported", f"no path compiles: {e}")
    return pkgs


def _start_runner(c: Check) -> runner.Runner:
    """Step 5: find or build the engine CLI."""
    try:
        return runner.Runner(c.args, c.log)
    except (RuntimeError, errs.Usage) as e:
        c.step("build", False, error=str(e))
        raise c.stop("FAIL", "runtime", str(e)) from None


def cmd_check(args: Sequence[str]) -> int:
    """bnc check: see the module docstring and the usage text.

    Returns:
        The exit code (errs.EXIT): ok, warn, unsupported, usage, model,
        parity, runtime or labels.
    """
    c = Check(args)
    try:
        post, n, sr = _parse_basics(c)
        g, ops, lt, nc = _load_and_cover(c)
        workdir = c.opt("--workdir") or os.path.join(
            runner.ROOT, "eval_tmp",
            "check-" + os.path.splitext(os.path.basename(args[1]))[0])
        os.makedirs(workdir, exist_ok=True)
        labels = _labels(c, ops, g.nsamp, nc, sr, workdir)
        x, win = _windows(c, g.nsamp, n, sr, workdir)
        _check_rate(c, sr, prep.is_v24_backbone(ops, g.nsamp))
        post = _choose_post(c, g, lt, post)
        pkgs = _compile_paths(c, g, labels, post, sr, workdir)
        engine = _start_runner(c)
    except Finished as done:
        return done.code
    try:  # whatever happens from here on, a throwaway container is removed
        return check_compare(c, g, ops, pkgs, engine, x, lt, post, nc, win,
                             workdir)
    except Finished as done:
        return done.code
    finally:
        engine.close()


def _references(c: Check, g, x: np.ndarray, lt: str):
    """Step 6: ONNX Runtime logits, and TFLite ones with --tflite.

    Returns:
        (ONNX Runtime logits [n][nc], TFLite logits or None).
    """
    try:
        sess = g.session([lt])
        ref = np.stack([sess.run([lt], {g.inp: w.reshape(1, -1)})[0].reshape(-1)
                        for w in x])
    except Exception as e:  # pylint: disable=broad-exception-caught
        # ONNX Runtime raises its own exception types for a graph it cannot run
        first = str(e).splitlines()[0][:300]
        raise c.stop("FAIL", "model",
                     f"ONNX Runtime fails on the windows: {first}") from None
    tfl = None
    if c.opt("--tflite"):
        tfl = parity.tflite_logits(c.opt("--tflite"), x, c.log)
    if tfl is not None and tfl.shape != ref.shape:
        c.warn(f"--tflite output {tfl.shape} differs from the ONNX logits "
               f"{ref.shape}: skipped")
        tfl = None
    return ref, tfl


def _run_paths(c: Check, pkgs: dict[str, str], engine: runner.Runner,
               win: str, n: int, nc: int,
               workdir: str) -> dict[str, np.ndarray]:
    """Step 7: runs every package; returns {path: logits} of those that ran."""
    got = {}
    for pth, pkg in pkgs.items():
        y, err, _ = engine.run(pkg, lower.path_info(pth).cli_mode, win, n, nc,
                               workdir)
        if y is None:
            c.res["paths"][pth].update(ran=False, error=err)
            c.step(f"run {pth}", False, error=err)
            if any(s in err for s in CPU_REFUSALS):
                c.warn(f"path {pth} cannot run on this CPU: {err}")
                continue
            raise c.stop("FAIL", "runtime",
                         f"the runtime refused or failed the {pth} package: "
                         f"{err}")
        got[pth] = y
    if not got:
        raise c.stop("FAIL", "runtime", "no path ran on this CPU")
    return got


def need_db(pth: str, int8_path: bool) -> float:
    """The SQNR a path needs (int8_path: its GEMMs run in int8)."""
    if int8_path:
        return NEED_INT8_DB
    return lower.path_info(pth).need_db


def judged_sqnr(pth: str, need: float, s_onnx: float, s_tflite: float | None,
                fe_matrix: bool, own16: float | None,
                s32: float | None) -> tuple[float, float, str]:
    """The SQNR a path is judged by, the dB it needs, and any allowance.

    Args:
        pth: the path.
        need: the SQNR it needs (need_db).
        s_onnx: its SQNR against ONNX Runtime.
        s_tflite: its SQNR against TFLite (rounded), None without --tflite;
            the exact reference decides when present.
        fe_matrix: the model has the v2.4 front end, whose DFT ONNX Runtime
            computes with an fp32 matrix (the engine computes it exactly).
        own16: the fp16 path's SQNR against the engine's fp32 path, or None.
        s32: the fp32 path's SQNR against ONNX Runtime, or None.

    Returns:
        (judged SQNR, needed SQNR, allowance note or "").
    """
    if s_tflite is not None:
        return s_tflite, need, ""
    if fe_matrix and pth == "neon-fp32" and s_onnx < need:
        return s_onnx, MATRIX_LIMITED_DB, (
            f"reference-limited: {s_onnx:.2f} dB is under this path's 60 dB "
            f"because ONNX Runtime computes the DFT with an fp32 matrix and "
            f"the engine exactly; 40 dB accepted, pass --tflite for the exact "
            f"comparison")
    if (fe_matrix and pth == "neon-fp16" and own16 is not None
            and s_onnx < need and s32 is not None and s32 < 60):
        return own16, need, (
            f"judged by its own loss against the engine's fp32 path "
            f"({own16:.1f} dB), the ONNX reference being matrix-limited "
            f"({s32:.1f} dB)")
    return s_onnx, need, ""


def _at_least_warn(status: str) -> str:
    return "WARN" if status == "PASS" else status


def path_verdict(int8_path: bool, nonfinite: bool, judged: float, need: float,
                 maxd: float, top1: int, n: int) -> tuple[str, list[str]]:
    """One path's verdict and the reasons for it.

    Args:
        int8_path: its GEMMs run in int8 (large single confidence deviations
            are expected then).
        nonfinite: the logits hold NaN or inf.
        judged: the SQNR the path is judged by (judged_sqnr).
        need: the SQNR it needs.
        maxd: the largest |confidence difference| against the reference.
        top1: windows whose top-1 class agrees.
        n: windows.

    Returns:
        ("PASS" | "WARN" | "FAIL", reasons).
    """
    status, why = "PASS", []
    if nonfinite:
        status, why = "FAIL", ["non-finite logits"]
    if judged < need:
        status = "FAIL" if judged < FAIL_DB else "WARN"
        why.append(f"{judged:.1f} dB < {need:.0f} dB")
    if maxd > 0.05 and not int8_path:
        status = "FAIL"
        why.append(f"max |dconf| {maxd:.3f} > 0.05")
    elif maxd > 0.05:  # int8 (dynamic per-row activations): the gate decides
        status = _at_least_warn(status)
        why.append(f"max |dconf| {maxd:.3f} > 0.05 (int8: expected on real "
                   f"audio)")
    elif maxd > 0.01:
        status = _at_least_warn(status)
        why.append(f"max |dconf| {maxd:.3f} > 0.01")
    if top1 != n:
        status = _at_least_warn(status)
        why.append(f"top-1 agrees on {top1}/{n} windows")
    return status, why


def _path_line(pth: str, pr: dict[str, Any], s_onnx: float, maxd: float,
               top1: int, n: int, has_tflite: bool, int8_path: bool) -> str:
    """The printed result line of one path (an int8-GEMM path names the
    floor it is judged by)."""
    line = f"path {pth}: {s_onnx:.1f} dB vs ONNX Runtime"
    if int8_path:
        line += (f" (int8 GEMMs: judged by the int8 floor, "
                 f"{pr['need_db']:.0f} dB)")
    if has_tflite:
        line += f", {pr['sqnr_vs_tflite_db']} dB vs TFLite"
    line += f", max |dconf| {maxd:.4f}, top-1 {top1}/{n}: {pr['verdict']}"
    if pr["why"]:
        line += f" ({'; '.join(pr['why'])})"
    if pr["allowance"]:
        line += f" [{pr['allowance']}]"
    return line


def _judge_path(c: Check, pth: str, y: np.ndarray, ref: np.ndarray,
                tfl: np.ndarray | None, ctx: dict[str, Any]) -> str:
    """Measures one path against the reference, records and prints it.

    Args:
        c: the check.
        pth: the path.
        y: its logits.
        ref: the ONNX Runtime logits.
        tfl: the TFLite logits, or None.
        ctx: post (the post function), ops (the IR), fe_matrix, own16 and s32
            (see judged_sqnr).

    Returns:
        The path's verdict.
    """
    n = len(y)
    pr = c.res["paths"][pth]
    s_onnx = parity.sqnr_db(ref, y)
    d = ctx["post"](y) - ctx["post"](ref)
    maxd = float(np.abs(d).max())
    top1 = int((y.argmax(1) == ref.argmax(1)).sum())
    pr.update(ran=True, sqnr_db=round(s_onnx, 1), max_dconf=round(maxd, 4),
              top1_agree=f"{top1}/{n}",
              nonfinite=bool(not np.isfinite(y).all()))
    if tfl is not None:
        pr["sqnr_vs_tflite_db"] = round(parity.sqnr_db(tfl, y), 1)
    # the SQNR the verdict uses, and any allowance: both on the path's line
    int8_path = (pth == "neon-int8"
                 or cmd_compile.lowering_class(pth, ctx["ops"]).INT8_GEMMS)
    judged, need, note = judged_sqnr(
        pth, need_db(pth, int8_path), s_onnx, pr.get("sqnr_vs_tflite_db"),
        ctx["fe_matrix"], ctx["own16"], ctx["s32"])
    pr["judged_sqnr_db"], pr["need_db"], pr["allowance"] = (round(judged, 1),
                                                            need, note)
    pr["verdict"], pr["why"] = path_verdict(int8_path, pr["nonfinite"], judged,
                                            need, maxd, top1, n)
    c.log(_path_line(pth, pr, s_onnx, maxd, top1, n, tfl is not None,
                     int8_path))
    c.step(f"run {pth}", pr["verdict"] != "FAIL")
    return pr["verdict"]


def skipped_paths(paths: dict[str, dict], pkgs: Sequence[str]) -> list[str]:
    """The compiled paths this CPU could not run (ran False)."""
    return [p for p in pkgs if paths.get(p, {}).get("ran") is False]


def skipped_note(paths: dict[str, dict], pkgs: Sequence[str]) -> str:
    """"; skipped: <path> (cannot run here), ..." for the compiled paths this
    CPU could not run (ran False), else ""."""
    skipped = skipped_paths(paths, pkgs)
    if not skipped:
        return ""
    return "; skipped: " + ", ".join(f"{p} (cannot run here)" for p in skipped)


def named_skip_step(skipped: Sequence[str], status: str, step: str) -> str:
    """The next step when paths named with --paths could not run here.

    Args:
        skipped: the named paths this CPU could not run.
        status: the verdict of the paths that ran (PASS or WARN).
        step: its next step.
    """
    names = ", ".join(skipped)
    them = "it" if len(skipped) == 1 else "them"
    if status == "PASS":
        return (f"--paths names {names}, which this CPU cannot run (not "
                f"checked); every other path agrees with ONNX Runtime on "
                f"these windows; next: check {names} on a CPU that runs "
                f"{them}, or leave {them} out of --paths")
    if "; skipped: " in step:
        return step
    return f"{step}; --paths names {names}, which this CPU cannot run"


def overall(paths: dict[str, dict], worst: str, reasons: Sequence[str],
            pkgs: Sequence[str],
            named: bool = False) -> tuple[str, str, str]:
    """The check's verdict: from the per-path verdicts, and a named path
    this CPU could not run.

    Args:
        paths: res["paths"] (each with a "verdict" when it ran, ran False
            when this CPU could not run it).
        worst: the worst path verdict.
        reasons: "<path>: <why>" for every path that has reasons.
        pkgs: the paths that compiled.
        named: the paths were named with --paths: one this CPU cannot run
            makes a PASS a WARN (without --paths it is a note on the PASS).

    Returns:
        (status, exit code name, next step).
    """
    status, code, step = verdict_of_paths(paths, worst, reasons, pkgs)
    skipped = skipped_paths(paths, pkgs)
    if not named or not skipped or status == "FAIL":
        return status, code, step
    return "WARN", "warn", named_skip_step(skipped, status, step)


def verdict_of_paths(paths: dict[str, dict], worst: str,
                     reasons: Sequence[str],
                     pkgs: Sequence[str]) -> tuple[str, str, str]:
    """The check's verdict from the per-path verdicts.

    Args:
        paths: res["paths"] (each with a "verdict" when it ran, ran False
            when this CPU could not run it).
        worst: the worst path verdict.
        reasons: "<path>: <why>" for every path that has reasons.
        pkgs: the paths that compiled.

    Returns:
        (status, exit code name, next step).
    """
    why = "; ".join(reasons)
    skipped = skipped_note(paths, pkgs)
    v16 = paths.get("neon-fp16", {}).get("verdict")
    v32 = paths.get("neon-fp32", {}).get("verdict")
    v8 = paths.get("neon-int8", {}).get("verdict")
    if v32 == "FAIL":
        return "FAIL", "parity", (
            f"the fp32 path disagrees with ONNX Runtime ({why}): this is a "
            f"lowering or recognizer bug, not precision; localize it per layer "
            f"(guide section 8) before going on")
    if (v16 == "WARN" and v32 == "PASS"
            and paths["neon-fp16"].get("need_db") == NEED_INT8_DB):
        return "WARN", "warn", (
            f"the fp32 path passes; the fp16 path, whose GEMMs run in int8 "
            f"(judged by the int8 floor), differs from it on these windows "
            f"({why}), which int8 may do: gate it on real audio before "
            f"shipping it (guide sections 2.1 and 9)")
    if v16 in ("FAIL", "WARN") and v32 == "PASS":
        return "WARN", "warn", (
            f"the fp32 path passes; the fp16 path does not ({why}): ship the "
            f"neon-fp32 package, or screen the fp16 precision per layer class "
            f"(guide section 7)")
    if v8 == "FAIL" and v32 == "PASS":
        return "WARN", "warn", (
            f"the fp32 path passes; the int8 path does not ({why}): ship the "
            f"neon-fp32 package, or put fewer layer classes on int8 (--int8, "
            f"e.g. without pwl; guide sections 2.1 and 7)")
    if v8 == "WARN" and v32 == "PASS" and worst == "WARN":
        return "WARN", "warn", (
            f"the fp32 path passes; the int8 path differs from it on these "
            f"windows ({why}), which int8 may do: gate it on real audio before "
            f"shipping it (guide sections 2.1 and 9)")
    if worst == "FAIL":
        others = ",".join(p for p in pkgs if p != "neon-fp32")
        return "FAIL", "parity", (
            f"{why}: with no passing fp32 path to compare against, treat this "
            f"as a lowering bug first (guide section 8); run the exact path "
            f"next to it (--paths neon-fp32,{others})")
    if worst == "WARN":
        return "WARN", "warn", (
            f"{why}{skipped}: run with more real audio (--audio), and gate "
            f"before shipping (guide section 9)")
    every = "every path this CPU can run" if skipped else "every path"
    return "PASS", "ok", (
        f"{every} agrees with ONNX Runtime on these windows{skipped}; next: "
        f"compile the shipping packages with --labels/--license-file (guide "
        f"section 5) and gate on real data (section 9)")


def _clean_workdir(c: Check, workdir: str) -> None:
    """Removes the packages and float files unless --keep or --workdir."""
    if "--keep" not in c.args and "--workdir" not in c.args:
        for f in os.listdir(workdir):
            if f.endswith((".bnpk", ".f32")):
                os.remove(os.path.join(workdir, f))
    else:
        c.log(f"packages kept in {workdir}")


def check_compare(c: Check, g, ops: Sequence[dict], pkgs: dict[str, str],
                  engine: runner.Runner, x: np.ndarray, lt: str, post: str,
                  nc: int, win: str, workdir: str) -> int:
    """Steps 6-8: references, run every path, judge; returns the exit code.

    Raises:
        Finished: a step failed (the exit code is in it).
    """
    ref, tfl = _references(c, g, x, lt)
    n = len(x)
    if post == "none":
        c.warn("--post none: max |dconf| is measured on the raw outputs (logit "
               "units), so the 0.01 / 0.05 lines are only meaningful for a "
               "sigmoid or softmax head")
    if tfl is not None:
        c.res["reference"] = dict(
            onnx_vs_tflite_sqnr_db=round(parity.sqnr_db(tfl, ref), 1))
        c.log(f"ONNX Runtime vs TFLite: "
              f"{c.res['reference']['onnx_vs_tflite_sqnr_db']} dB")
    # run every path, then judge (the fp16 verdict may use the engine's own
    # fp32 path as its reference)
    got = _run_paths(c, pkgs, engine, win, n, nc, workdir)
    own16 = None
    if "neon-fp16" in got and "neon-fp32" in got:
        own16 = parity.sqnr_db(got["neon-fp32"], got["neon-fp16"])
        c.res["fp16_vs_fp32_sqnr_db"] = round(own16, 1)
    # s32: how far ONNX Runtime is from the engine's exact path
    s32 = parity.sqnr_db(ref, got["neon-fp32"]) if "neon-fp32" in got else None
    ctx = dict(post=parity.post_fn(post), ops=ops,
               fe_matrix=prep.has_v24_front_end(ops), own16=own16, s32=s32)
    worst, reasons = "PASS", []
    for pth, y in got.items():
        verdict = _judge_path(c, pth, y, ref, tfl, ctx)
        if VERDICT_ORDER[verdict] > VERDICT_ORDER[worst]:
            worst = verdict
        if c.res["paths"][pth]["why"]:
            reasons.append(f"{pth}: {'; '.join(c.res['paths'][pth]['why'])}")
    c.step("compare", worst != "FAIL")
    if own16 is not None:
        c.log(f"engine fp16 vs engine fp32: {own16:.1f} dB (the fp16 path's "
              f"own loss, reference-free)")
    _clean_workdir(c, workdir)
    return c.finish(*overall(c.res["paths"], worst, reasons, list(pkgs),
                             named="--paths" in c.args))
