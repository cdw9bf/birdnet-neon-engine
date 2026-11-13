"""Model preparation shared by report, compile and check.

Loading (ONNX import, constant folding, canonicalization verified with ONNX
Runtime), recognition, the default output tensors and post-activation, and the
label file helpers.
"""
from collections.abc import Sequence
import itertools
import os

import numpy as np

from .. import labels
from . import canon
from . import cliargs
from . import errs
from . import onnxg
from . import recog

# canonicalization may move the graph outputs by at most this much (relative)
CANON_TOLERANCE = 1e-4
# the header line of Perch v2's label file (not a class)
PERCH_LABEL_HEADER = "inat2024_fsd50k"
# the BirdNET v2.4 backbone: its window, and the rate the hosts feed each
# family on it (24 BirdNET v2.4, 224 BattyBirdNET)
V24_SAMPLES = 144000
FAMILY_RATE = {"24": 48000, "224": 256000}


def _append_spec(args: Sequence[str]) -> list[tuple[str, str]]:
    """The --append HEAD --at TENSOR pair, as onnxg.Graph takes it."""
    if "--append" not in args:
        return []
    if not cliargs.opt(args, "--at"):
        raise errs.Usage("--append needs --at TENSOR (the backbone tensor the "
                         "head's input is wired to)")
    return [(cliargs.opt(args, "--append"), cliargs.opt(args, "--at"))]


def _fold_constants(g: onnxg.Graph) -> int:
    """Folds constant subgraphs until none is left; returns the count."""
    folded = 0
    # Shape -> arithmetic -> Slice/Reshape parameters fold in a few rounds
    while True:
        k = g.fold()
        folded += k
        if not k:
            return folded


def max_relative_change(before: dict, after: dict,
                        outputs: Sequence[str]) -> float:
    """The largest max |after - before| / max |before| over the outputs."""
    worst = 0.0
    for o in outputs:
        a = np.asarray(before[o], np.float64)
        b = np.asarray(after[o], np.float64)
        worst = max(worst,
                    float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12)))
    return worst


def _canonicalize_verified(g: onnxg.Graph, src: str, enabled: bool) -> None:
    """Canonicalizes g and checks with ONNX Runtime that the outputs held.

    Sets g.canon (the rewrites applied; [] when not enabled) and, when there
    were any, g.canon_err. The ONNX Runtime run on the check window happens
    either way (it also proves the graph runs).

    Raises:
        errs.ModelError: a rewrite changed the outputs (a compiler bug).
    """
    # exporter artifacts -> the forms the recognizer matches, on one window
    x = np.random.default_rng(1).standard_normal((1, g.nsamp)) * 0.05
    x = x.astype(np.float32)
    before = g.run_shapes(x, want_values=g.outputs)
    g.canon = canon.canonicalize(g) if enabled else []
    if not g.canon:
        return
    after = g.run_shapes(x, want_values=g.outputs)
    worst = max_relative_change(before, after, g.outputs)
    g.canon_err = worst
    # pylint: disable-next=unnecessary-negation
    if not worst <= CANON_TOLERANCE:  # written so that NaN fails too
        raise errs.ModelError(
            f"{src}: canonicalization changed the graph output (relative max "
            f"|d| {worst:.2e}) after {g.canon}; hint: this is a compiler bug: "
            f"run with --no-canon and report it")


def load(args: Sequence[str]) -> tuple[onnxg.Graph, int]:
    """Loads the model named on the command line, ready for recognition.

    Args:
        args: the command line: args[1] is the ONNX file; --append/--at add a
            head, --no-canon skips canonicalization.

    Returns:
        (graph, number of constant nodes folded).

    Raises:
        errs.Usage: --append without --at.
        errs.ModelError: the model cannot be used, or canonicalization changed
            its outputs.
    """
    src = args[1]
    g = onnxg.Graph(src, _append_spec(args))
    g.run_shapes()
    folded = _fold_constants(g)
    _canonicalize_verified(g, src, "--no-canon" not in args)
    return g, folded


def recognize(g: onnxg.Graph,
              fold_bn: bool = True) -> tuple[recog.Recognizer, list[dict]]:
    """Runs the recognizer; returns it and its IR op list."""
    r = recog.Recognizer(g, fold_bn=fold_bn)
    ops = r.walk()
    return r, ops


def default_logits(g: onnxg.Graph, r: recog.Recognizer) -> str:
    """The logits tensor when --logits is not given.

    Returns:
        The graph output a dense or prototype head produces and nothing
        consumes further (the last one if several); else the input of a final
        Sigmoid/Softmax whose output is a graph output; else the last graph
        output.
    """
    used = {i for o in r.ops for i in o["ins"]}
    heads = {o["out"] for o in r.ops
             if o["op"] in ("DENSE", "PROTO") and o["out"] not in used}
    cands = [o for o in g.outputs if o in r.val and r.val[o].id in heads]
    if cands:
        return cands[-1]
    for o in reversed(g.outputs):
        p = g.prod.get(o)
        if (p is not None and p.op_type in ("Sigmoid", "Softmax")
                and p.input[0] in r.val):
            return p.input[0]
    return g.outputs[-1]


def default_post(g: onnxg.Graph, lt: str) -> str | None:
    """"sigmoid" / "softmax" when the graph applies one to lt, else None."""
    tail = [c.op_type for c in g.cons.get(lt, [])
            if c.op_type in ("Softmax", "Sigmoid")]
    return tail[0].lower() if tail else None


def problems_in_cone(g: onnxg.Graph, r: recog.Recognizer,
                     tensors: Sequence[str | None]) -> list[dict]:
    """The unsupported nodes the given output tensors depend on.

    Nodes after the tensors are ignored; all problems count when no tensor is
    given or the front end was not recognized.
    """
    want = [t for t in tensors if t]
    cone = g.ancestors(want) if want else None
    probs = recog.problem_report(g, r)
    if cone is None or r.fe_failed is not None:
        return probs
    return [p for p in probs
            if p["index"] is None or id(g.nodes[p["index"]]) in cone]


def has_v24_front_end(ops: Sequence[dict]) -> bool:
    """True when the front end is the BirdNET v2.4 one (FE_STFT2_POW)."""
    return any(o["op"] == "FE_STFT2_POW" for o in ops)


def is_v24_backbone(ops: Sequence[dict], nsamp: int) -> bool:
    """The BirdNET v2.4 front end on its 144000-sample window."""
    return has_v24_front_end(ops) and nsamp == V24_SAMPLES


def family_at_rate(sr: int) -> str | None:
    """The v2.4-backbone family fed at sr Hz (FAMILY_RATE), or None."""
    for family, rate in FAMILY_RATE.items():
        if rate == sr:
            return family
    return None


def birdnetgo_labels(ops: Sequence[dict], nsamp: int, family: str) -> bool:
    """Whether BirdNET-Go reads this package's labels in its
    `Scientific name_Common name` form: family 24 on the v2.4 backbone."""
    return is_v24_backbone(ops, nsamp) and family == "24"


def nonempty_lines(raw: bytes) -> list[str]:
    """The label lines of a label file (UTF-8; bad bytes replaced).

    A line ends at "\\n" and loses one "\\r" before it; a line of only
    spaces, tabs and "\\r" is blank and is not a label.
    """
    text = raw.decode("utf-8", "replace")
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    return [line for line in lines if line.strip(" \t\r")]


def class_labels(raw: bytes) -> list[str]:
    """One label per class: the label lines without Perch's header line.

    The rule every reader of a package's label file follows. Twin:
    bnneon.binding.class_labels (the binding is self-contained; the README's
    C example spells the same rule); tests/test_label_rule.py checks that
    all three agree.
    """
    lines = nonempty_lines(raw)
    if lines and lines[0] == PERCH_LABEL_HEADER:
        lines = lines[1:]
    return lines


def label_rule_error(path: str, raw: bytes) -> str | None:
    """Why a label file could be numbered two ways, or None.

    Python's str.splitlines() also breaks lines at a lone "\\r", "\\v",
    "\\f", "\\x1c"-"\\x1e", "\\x85", "\\u2028" and "\\u2029", str.strip()
    also removes other white space (the compiler counted labels that way
    before the rule above), and a C host's string ends at a NUL byte. A
    file that any of these would read differently is refused, so that no
    host numbers its labels other than the compiler did.
    """
    text = raw.decode("utf-8", "replace")
    ours = nonempty_lines(raw)
    other = [line for line in text.splitlines() if line.strip()]
    if ours == other and "\0" not in text:
        return None
    pairs = itertools.zip_longest(ours, other)
    i = next((k for k, (a, b) in enumerate(pairs) if a != b), None)
    if i is None:
        i = next((k for k, line in enumerate(ours) if "\0" in line), 0)
    line = ours[i] if i < len(ours) else other[i]
    return (f"label file {path}: non-blank line {i + 1} ({line[:60]!r}) "
            f"holds a line break other than \\n or \\r\\n, a NUL byte, or is "
            f"blank only by white space other than spaces and tabs; hint: "
            f"save the file as plain text, one label per line, \\n or \\r\\n "
            f"line ends")


def label_form(lines: Sequence[str]) -> tuple[int, str | None]:
    """Counts the lines not in BirdNET-Go's `Scientific name_Common name` form.

    The rule is bnneon.labels' (a line its rewrite would change).

    Returns:
        (count, the first such line or None).
    """
    bad = [line for line in lines if labels.rewrite(line, {}) != line]
    return len(bad), (bad[0] if bad else None)


def labels_warning(lines: Sequence[str], path: str) -> str | None:
    """The warning for a label file BirdNET-Go would misread, or None."""
    n, first = label_form(lines)
    if not n:
        return None
    return (f"{n} of {len(lines)} label lines are not in BirdNET-Go's "
            f"`Scientific name_Common name` form (first: '{first}'); the "
            f"package embeds {os.path.basename(path)} as it is: rewrite it "
            f"with scripts/relabel_for_birdnetgo.py before compiling if "
            f"BirdNET-Go will read it")
