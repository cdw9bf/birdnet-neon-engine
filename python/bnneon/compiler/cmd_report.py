"""bnc report: the coverage report.

What the recognizer made of the graph: the IR op listing, the unsupported
nodes, the graph outputs no supported op produces and the uncovered nodes.
"""
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import cliargs
from . import errs
from . import onnxg
from . import prep
from . import recog

# op fields the IR listing leaves out (weights, tables, node lists)
HIDDEN_FIELDS = ("nodes", "w", "b", "window", "mel", "dft", "branches", "P",
                 "cw", "cb", "w1", "b1", "w2", "b2", "s", "t", "bn_s", "bn_b",
                 "tap")


def _array_tag(a: np.ndarray) -> str:
    return f"array{list(a.shape)}"


def _shown_item(x: Any) -> Any:
    """A list element of an op field: arrays (also in tuples) by their shape."""
    if isinstance(x, np.ndarray):
        return _array_tag(x)
    if isinstance(x, tuple):
        return tuple(_array_tag(y) if isinstance(y, np.ndarray) else y
                     for y in x)
    return x


def _shown_value(v: Any) -> Any:
    """An op field as the listing shows it: any array by its shape only."""
    if isinstance(v, np.ndarray):
        return _array_tag(v)
    if isinstance(v, (list, tuple)) and any(isinstance(x, np.ndarray)
                                            for x in v):
        return [_shown_item(x) for x in v]
    return v


def op_fields(o: dict) -> dict:
    """The fields of one IR op the listing prints (a weight by its shape)."""
    d = {k: v for k, v in o.items() if k not in HIDDEN_FIELDS}
    if "w" in o:
        d["w"] = list(o["w"].shape)
    return {k: _shown_value(v) for k, v in d.items()}


def summary(ops: Sequence[dict]) -> list[str]:
    """The IR listing: one line per op."""
    return [f"  {o['op']:12s} {op_fields(o)}  ({len(o['nodes'])} nodes)"
            for o in ops]


def _no_body_rule(r: recog.Recognizer, left: Sequence[Any]) -> dict[str, int]:
    """Op types behind an unrecognized front end that no body rule handles.

    Returns:
        {op type: count}, most frequent first (empty unless the front end
        failed: the report then stops there, so this shows what lies behind).
    """
    census: dict[str, int] = {}
    if r.fe_failed is not None:
        for n in left:
            if n.op_type not in recog.BODY_OPS:
                census[n.op_type] = census.get(n.op_type, 0) + 1
    return dict(sorted(census.items(), key=lambda kv: -kv[1]))


def coverage(g: onnxg.Graph, r: recog.Recognizer,
             ops: Sequence[dict]) -> dict[str, Any]:
    """The report's structured content (also the --json document)."""
    outs = [r.val[o].id for o in g.outputs if o in r.val]
    fused = recog.fold_vec(recog.fuse(list(ops), outs), outs) if ops else []
    probs = recog.problem_report(g, r)
    left = [n for n in g.nodes if id(n) not in r.covered]
    return {
        "no_body_rule": _no_body_rule(r, left),
        "model": g.path,
        "nodes": len(g.nodes),
        "input": g.inp,
        "n_samples": g.nsamp,
        "outputs": g.outputs,
        "ir_ops": len(fused),
        "logits_default": prep.default_logits(g, r),
        "ir": [line.strip() for line in summary(fused)],
        "unsupported": probs,
        "primary": [p for p in probs if not p["downstream_of"]],
        "missing_outputs": [o for o in g.outputs if o not in r.val],
        "uncovered": len(left),
        "canonicalised": g.canon,
        "canon_err": g.canon_err,
    }


def _totals_line(cov: dict[str, Any]) -> str:
    """The report's last line: IR ops, unsupported and uncovered nodes."""
    n = len(cov["unsupported"])
    line = f"{cov['ir_ops']} IR ops; {n} unsupported nodes"
    if n:
        line += f" ({len(cov['primary'])} primary, the rest downstream)"
    line += f"; {cov['uncovered']} uncovered nodes"
    if cov["uncovered"]:
        line += (" (no IR op covers them; only nodes the logits / embedding "
                 "depend on matter)")
    return line


def print_coverage(cov: dict[str, Any], nf: int,
                   out: Callable[..., None] = print, verbose: bool = False,
                   quiet: bool = False) -> None:
    """Prints the coverage report.

    Args:
        cov: what coverage() returned.
        nf: the number of constant nodes folded.
        out: the print function.
        verbose: list every unsupported node, not only the first few.
        quiet: leave out the IR listing.
    """
    out(f"{cov['model']}: {cov['nodes']} nodes after folding {nf} constant "
        f"nodes; input {cov['input']} [{cov['n_samples']}]; "
        f"outputs {cov['outputs']}")
    if cov.get("canonicalised"):
        rewrites = ", ".join(f"{k} x{n}" for k, n in cov["canonicalised"])
        out(f"canonicalized: {rewrites} (ONNX Runtime output unchanged: "
            f"relative max |d| {cov['canon_err']:.1e})")
    if not quiet:
        for line in cov["ir"]:
            out("  " + line)
    for line in recog.format_problems(cov["unsupported"], verbose):
        out(line)
    for o in cov["missing_outputs"]:
        out(f"graph output {o} not produced by a supported op (fine when it "
            f"is a final Sigmoid/Softmax after the logits: the host applies "
            f"--post; the package returns the logits)")
    if cov.get("no_body_rule"):
        types = ", ".join(f"{k} x{v}" for k, v in cov["no_body_rule"].items())
        out("behind the front end, op types no body rule handles (each needs "
            "a rule and usually a kernel, guide section 6): " + types)
    out(_totals_line(cov))


def cmd_report(args: Sequence[str]) -> int:
    """bnc report: prints the coverage report (and writes it with --json).

    Returns:
        errs.EXIT["unsupported"] when a node or graph output is not supported,
        else errs.EXIT["ok"].
    """
    out = cliargs.text_printer(args)
    g, nf = prep.load(args)
    r, ops = prep.recognize(g, "--no-fold-bn" not in args)
    cov = coverage(g, r, ops)
    print_coverage(cov, nf, out, "--verbose" in args)
    bad = bool(cov["unsupported"] or cov["missing_outputs"])
    cov["status"] = "unsupported" if bad else "ok"
    cov["exit"] = errs.EXIT["unsupported"] if bad else errs.EXIT["ok"]
    if "--json" in args:
        cliargs.json_out(args, cov)
    return cov["exit"]
