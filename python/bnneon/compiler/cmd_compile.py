"""bnc compile: lower the recognized model per path, write the .bnpk package.

The steps, in order: load and recognize; refuse unsupported nodes; pick the
logits and embedding tensors; fuse and prune the IR; apply a class subset;
settle --post, --family and --sample-rate; lower every requested path; check
the label file; write the package with its metadata.
"""
from collections.abc import Sequence
import hashlib
import os

import numpy as np

from . import cliargs
from . import errs
from . import lower
from . import lower_dot
from . import onnxg
from . import pkgfmt
from . import prep
from . import recog

POST_CODES = {"none": 0, "sigmoid": 1, "softmax": 2}
# lowering policy options with a value, in the order they are applied:
# int8: the neon-int8 path's layer classes on int8 (pw, pwl, head, sed, lin);
# wino: dense 3x3 s1 convs with >= this many input channels by Winograd F(4,3);
# dense: f16 | f32 for the fp16 path (default: f16 above 64 classes)
POLICY_OPTIONS = ("int8", "wino", "dense")
# kernel families with a Lowering of their own: (applies(path, ops), class)
LOWERING_FAMILIES = ((lower_dot.applies, lower_dot.LoweringDot),)


def lowering_class(path: str, ops: list[dict]) -> type[lower.Lowering]:
    """The Lowering for a path and a fused IR: the first family that applies.

    Returns:
        That family's class, else lower.Lowering.
    """
    for applies, cls in LOWERING_FAMILIES:
        if applies(path, ops):
            return cls
    return lower.Lowering


def labels_info(path: str, raw: bytes, n_classes: int) -> bytes:
    """Checks a label file against the head; returns the file's sha256.

    Args:
        path: the label file (for the message).
        raw: its bytes.
        n_classes: the head's number of classes.

    Raises:
        errs.LabelsMismatch: the file has another number of labels, or
            could be numbered two ways (prep.label_rule_error).
    """
    problem = prep.label_rule_error(path, raw)
    if problem:
        raise errs.LabelsMismatch(problem)
    lines = prep.class_labels(raw)
    if len(lines) != n_classes:
        raise errs.LabelsMismatch(
            f"label file {path} has {len(lines)} non-empty lines but the "
            f"model's logits have {n_classes} classes; hint: pass the label "
            f"file that belongs to this head (one label per line, same order "
            f"as the outputs), or check --logits / --classes")
    return hashlib.sha256(raw).digest()


def cmd_compile(a: Sequence[str]) -> int:
    """bnc compile; returns the exit code.

    Raises:
        errs.Usage: (and errs.LabelsMismatch) bnc.main maps them to exit codes.
    """
    try:
        return compile_(a)
    except recog.Unsupported as e:
        print(f"unsupported: {e}")
        return errs.EXIT["unsupported"]


def _require_options(a: Sequence[str]) -> None:
    for k in ("--labels", "-o"):
        if not cliargs.opt(a, k):
            raise errs.Usage(f"{k} is required")
    if not cliargs.opt(a, "--model-id"):
        raise errs.Usage("--model-id is required (a short identifier such as "
                         "my_model)")


def _logits_value(g, r: recog.Recognizer, lt: str) -> int:
    """The IR value of the logits tensor.

    Raises:
        errs.Usage: no tensor of that name.
        recog.Unsupported: no supported op produces it.
    """
    if lt not in r.val and lt not in g.prod and lt not in g.const:
        raise errs.Usage(f"--logits {lt}: no tensor of that name in the graph "
                         f"(outputs: {g.outputs})")
    if lt not in r.val:
        raise recog.Unsupported(
            f"logits tensor '{lt}' is not produced by a supported op (graph "
            f"outputs: {g.outputs}); hint: name the pre-activation tensor with "
            f"--logits (the engine returns logits; the host applies --post "
            f"sigmoid|softmax), and run `bnc.py report` for the unsupported "
            f"nodes")
    return r.val[lt].id


def _embedding_value(g, r: recog.Recognizer, ops: Sequence[dict],
                     et: str | None) -> int | None:
    """The IR value of --emb, else of the last spatial mean (None: none)."""
    if et:
        if et not in g.prod and et not in g.const:
            raise errs.Usage(f"--emb {et}: no tensor of that name in the graph "
                             f"(outputs: {g.outputs})")
        if et not in r.val:
            raise recog.Unsupported(
                f"embedding tensor {et} is not produced by a supported op")
        return r.val[et].id
    means = [o for o in ops if o["op"] in ("MEAN", "GEM")]
    return means[-1]["out"] if means else None


def _fuse(r: recog.Recognizer, ops: list[dict], lt: str, lv: int,
          ev: int | None, et: str | None) -> list[dict]:
    """Fuses and prunes the IR to what the outputs need, and checks it."""
    want = [lv] + ([ev] if ev else [])
    ops = recog.dce(recog.fold_vec(recog.fuse(ops, want), want), want)
    for o in ops:
        if (o["op"] == "ACT" and r.val.get(lt, None) is not None
                and any(v.id == o["ins"][0] and v.kind == "vec"
                        for v in r.val.values())):
            raise recog.Unsupported(
                f"--logits / --emb names a dense layer's output that a ReLU "
                f"consumes ({lower.src_name(o)}): name the ReLU's output "
                f"instead (the engine fuses the ReLU into the dense layer)")
    if ev and not any(o["out"] == ev for o in ops):
        raise recog.Unsupported(
            f"embedding tensor '{et or 'the last spatial mean'}' is not "
            f"computed by any op after fusion; hint: name a tensor that a "
            f"dense layer, a mean or a prototype head produces with --emb")
    return ops


def _apply_classes(ops: Sequence[dict], lv: int, path: str) -> None:
    """--classes: keeps these output classes, in this order, of the head.

    The op producing the logits (a dense or prototype head) is sliced in
    place: a regional slice of a larger catalog.
    """
    idx = cliargs.read_indices(path)  # 0-based classes of the head
    head = [o for o in ops if o["out"] == lv]
    if len(head) != 1 or head[0]["op"] not in ("PROTO", "DENSE"):
        raise recog.Unsupported(
            "--classes needs the logits to come from a dense or prototype head")
    h = head[0]
    n = h["cb"].size if h["op"] == "PROTO" else h["w"].shape[0]
    if (idx.size == 0 or idx.min() < 0 or idx.max() >= n
            or len(set(idx.tolist())) != idx.size):
        raise recog.Unsupported("--classes: indices out of range or repeated")
    if h["op"] == "PROTO":
        ppc = h["ppc"]
        rows = (idx[:, None] * ppc + np.arange(ppc)[None, :]).reshape(-1)
        h["P"] = h["P"][rows].copy()
        h["cw"] = h["cw"][idx].copy()
        h["cb"] = h["cb"][idx].copy()
    else:
        h["w"] = h["w"][idx].copy()
        h["b"] = h["b"][idx].copy()


def _post(a: Sequence[str], g, lt: str, v24: bool) -> str:
    """--post, else what the graph applies, else sigmoid on a v2.4 backbone."""
    graph_post = prep.default_post(g, lt)
    v24_post = "sigmoid" if v24 else None
    post = cliargs.opt(a, "--post") or graph_post or v24_post
    if "--post" not in a:
        if post is None:
            raise errs.Usage(
                "--post is required (none, sigmoid or softmax: what the host "
                "applies to the logits); the graph does not apply one and the "
                "backbone is not BirdNET v2.4")
        why = (f" (the graph applies it to '{lt}')" if graph_post
               else " (a BirdNET v2.4 backbone)")
        print(f"--post not given: using {post}{why}")
    if post not in POST_CODES:
        raise errs.Usage(f"--post {post}: expected none, sigmoid or softmax "
                         f"(what the host applies to the logits)")
    return post


def _family(a: Sequence[str], v24: bool, force: bool) -> str:
    """--family, else 24 on a v2.4 backbone, else 0 with --force."""
    family = cliargs.opt(a, "--family")
    if family is not None:
        return family
    if v24:
        print("--family not given: using 24 (BirdNET v2.4 backbone "
              "recognized; BirdNET-Go refuses unknown families; a BattyBirdNET "
              "head on this backbone is --family 224 --sample-rate 256000)")
        return "24"
    if force:
        return "0"
    raise errs.Usage("--family is required (24 BirdNET v2.4, 120 Perch v2, 224 "
                     "BattyBirdNET; hosts refuse families they do not know); "
                     "--force writes family 0")


def _sample_rate(a: Sequence[str], v24: bool, family: str,
                 force: bool) -> int:
    """--sample-rate (default 48000), checked against the v2.4 families."""
    try:
        sr = int(cliargs.opt(a, "--sample-rate", "48000"))
    except ValueError:
        raise errs.Usage(f"--sample-rate {cliargs.opt(a, '--sample-rate')}: "
                         f"expected an integer rate in Hz") from None
    if sr <= 0:
        raise errs.Usage(f"--sample-rate {sr}: expected a positive rate in Hz")
    want = prep.FAMILY_RATE.get(family)
    if v24 and want is not None and sr != want and not force:
        raise errs.Usage(
            f"--sample-rate {sr} with family {family} on the BirdNET v2.4 "
            f"front end (144000-sample window): that family takes {want} Hz "
            f"and the rate is baked into the package; use --sample-rate {want} "
            f"(or --force; BattyBirdNET is --family 224 --sample-rate 256000)")
    return sr


def license_file(a: Sequence[str]) -> str | None:
    """--license-file, or its old spelling --licence-file (still accepted)."""
    return cliargs.opt(a, "--license-file") or cliargs.opt(a, "--licence-file")


def _check_credits(a: Sequence[str], force: bool, quiet: bool) -> None:
    """--license-file and --attribution are required unless --force."""
    given = {"--license-file": license_file(a),
             "--attribution": cliargs.opt(a, "--attribution")}
    missing = [k for k, v in given.items() if not v]
    if missing and not force:
        verb = "is" if len(missing) == 1 else "are"
        raise errs.Usage(
            f"{' and '.join(missing)} {verb} required: the package carries the "
            f"model's license text and credit line (guide section 5 has a "
            f"template for a custom classifier); --force writes it without")
    if missing and not quiet:
        print(f"WARN: --force: writing the package without "
              f"{' / '.join(missing)}")


def _set_policy(a: Sequence[str], lowering: lower.Lowering) -> None:
    """The lowering policy options of one path."""
    if "--proto-s8" in a:
        lowering.policy["proto"] = "s8"
    if "--proto32-h16" in a:
        lowering.policy["proto32"] = "h16"  # fp32 path: fp16-stored prototypes
    for key in POLICY_OPTIONS:
        if cliargs.opt(a, f"--{key}"):
            lowering.policy[key] = cliargs.opt(a, f"--{key}")


def wino_note(a: Sequence[str], pth: str, rops: Sequence[tuple]) -> str | None:
    """The note for a --wino that no conv of this path took, or None.

    --wino only changes dense 3x3 stride-1 convs of the fp32-activation
    lowerings (CONVW records); on another path it is accepted and has no
    effect, so old command lines keep working.
    """
    wino = cliargs.opt(a, "--wino")
    if not wino or any(op[0] & 0xFF == lower.OP["CONVW"] for op in rops):
        return None
    return (f"note: --wino {wino}: no conv of path {pth} takes the Winograd "
            f"form, so the option has no effect on this path")


def _lower_path(a: Sequence[str], pkg: pkgfmt.Package, pth: str,
                ops: list[dict], vshape: dict, nsamp: int, lv: int,
                ev: int | None) -> tuple[int, int]:
    """Lowers the IR for one path and adds it to the package.

    Args:
        a: the command line (the policy options).
        pkg: the package.
        pth: the path.
        ops: the fused IR.
        vshape: {value id: (H, W)}.
        nsamp: samples per window.
        lv: the logits value.
        ev: the embedding value, or None.

    Returns:
        (number of classes, embedding size or 0).
    """
    lowering = lowering_class(pth, ops)(pkg, ops, pth, nsamp)
    _set_policy(a, lowering)
    bufs, rops = lowering.run(vshape)
    note = wino_note(a, pth, rops)
    if note:
        print(note)
    ob = lowering.vbuf[lv]
    eb = lowering.vbuf.get(ev) if ev else None
    n_classes = bufs[ob][3]
    emb_dim = bufs[eb][3] if eb is not None else 0
    hw = lower.path_info(pth).hwcap
    if any(op[0] >> 8 == 2 for op in rops):
        hw |= 2
    flat = sum(b[6] * (2 if b[0] == 1 else 4) for b in bufs[1:])
    bufs, shared = lower.plan_slots(bufs, rops, [ob, eb])
    pkg.add_path(pth, hw, bufs, rops, 0, ob, eb)
    # debug metadata: one source node per op (the runtime's errors name it)
    pkg.set(f"src_nodes/{pth}", "\n".join(lowering.src))
    slots = len(set(b[7] for b in bufs))
    print(f"path {pth}: {len(rops)} ops, {len(bufs)} buffers in {slots} slots, "
          f"{shared / 1e6:.1f} MB activations ({flat / 1e6:.1f} MB unshared)")
    return n_classes, emb_dim


def _warn_label_form(labels: str, raw: bytes) -> None:
    """BirdNET-Go's label contract (v2.4 classifiers): warn about the form."""
    warning = prep.labels_warning(prep.nonempty_lines(raw), labels)
    if warning:
        print("WARN: " + warning)


def _file_sha256(path: str) -> str:
    """The hex sha256 of a file, read in chunks (ONNX files are large)."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _set_metadata(a: Sequence[str], pkg: pkgfmt.Package, meta: dict) -> None:
    """The package metadata, in the order the format has always had it.

    Args:
        a: the command line.
        pkg: the package.
        meta: family, sample_rate, n_samples, n_classes, emb_dim, post,
            labels_sha256 (bytes), labels (the file's path) and labels_raw
            (its bytes).
    """
    labels = meta["labels"]
    src_sha = _file_sha256(a[1])
    if "--append" in a:
        src_sha += " " + _file_sha256(cliargs.opt(a, "--append"))
    lic = ""
    if license_file(a):
        with open(license_file(a), encoding="utf-8") as f:
            lic = f.read().strip()
    pkg.set("model_id", cliargs.opt(a, "--model-id"))
    pkg.set("family", int(meta["family"]))
    pkg.set("sample_rate", meta["sample_rate"])
    pkg.set("n_samples", meta["n_samples"])
    pkg.set("hop_samples", int(cliargs.opt(a, "--hop", "0")))
    pkg.set("n_classes", int(meta["n_classes"]))
    pkg.set("emb_dim", int(meta["emb_dim"]))
    pkg.set("post", POST_CODES[meta["post"]])
    pkg.set("logit_scale", float(cliargs.opt(a, "--logit-scale", "1.0")))
    pkg.set("labels_sha256", meta["labels_sha256"])
    pkg.set("labels_file", os.path.basename(labels))
    # the label file itself (bytes as shipped; sha256 = labels_sha256)
    pkg.set("labels", meta["labels_raw"])
    pkg.set("license", lic)
    pkg.set("attribution", cliargs.opt(a, "--attribution", ""))
    pkg.set("source_sha256", src_sha)
    pkg.set("input", f"float32 PCM in [-1, 1) at sample_rate, "
                     f"{meta['n_samples']} samples per window")
    pkg.set("compiler", lower.COMPILER_VERSION)
    if cliargs.opt(a, "--notes"):
        pkg.set("notes", cliargs.opt(a, "--notes"))
    if cliargs.opt(a, "--classes"):
        pkg.set("class_subset", _read_bytes(cliargs.opt(a, "--classes")))


def compile_(a: Sequence[str], quiet: bool = False) -> int:
    """Compiles the model on the command line into a package.

    Args:
        a: the compile command line.
        quiet: leave out the label and license warnings (check prints its own).

    Returns:
        errs.EXIT["unsupported"] after printing the problems when nodes the
        outputs need are not supported, else 0 once the package is written.

    Raises:
        errs.Usage: missing or wrong options (errs.LabelsMismatch: the label
            file does not fit the head).
        recog.Unsupported: the outputs or a lowering are not supported.
    """
    g, _ = prep.load(a)
    return compile_loaded(a, g, quiet)


def compile_loaded(a: Sequence[str], g: onnxg.Graph,
                   quiet: bool = False) -> int:
    """compile_() on a graph already loaded with prep.load(a).

    bnc check loads the model once and compiles every path from it: each
    call recognizes the graph afresh, so no IR op is shared between paths.

    Args:
        a: the compile command line (its model, --append / --at and
            --no-canon must be the ones the graph was loaded with).
        g: the loaded graph.
        quiet: as for compile_().

    Returns:
        As compile_().

    Raises:
        As compile_() (the load errors excepted).
    """
    r, ops = prep.recognize(g, "--no-fold-bn" not in a)
    lt = cliargs.opt(a, "--logits", prep.default_logits(g, r))
    probs = prep.problems_in_cone(g, r, [lt, cliargs.opt(a, "--emb")])
    if probs:
        for line in recog.format_problems(probs, "--verbose" in a):
            print(line)
        return errs.EXIT["unsupported"]
    # outputs: --logits TENSOR (default: the dense / prototype head's output,
    # or the input of a final Sigmoid/Softmax) and --emb TENSOR (default: the
    # last spatial mean, i.e. the pooled embedding)
    lv = _logits_value(g, r, lt)
    _require_options(a)
    et = cliargs.opt(a, "--emb")
    ev = _embedding_value(g, r, ops, et)
    ops = _fuse(r, ops, lt, lv, ev, et)
    if cliargs.opt(a, "--classes"):
        _apply_classes(ops, lv, cliargs.opt(a, "--classes"))
    force = "--force" in a
    v24 = prep.is_v24_backbone(ops, g.nsamp)
    post = _post(a, g, lt, v24)
    family = _family(a, v24, force)
    sr = _sample_rate(a, v24, family, force)
    _check_credits(a, force, quiet)
    if "--paths" not in a:
        print("no --paths: both paths go into one file; the usual shipping "
              "form is one package per path (--paths neon-fp16, then --paths "
              "neon-fp32)")
    vshape = {v.id: (v.H, v.W) for v in r.val.values()}
    pkg = pkgfmt.Package()
    for pth in cliargs.opt(a, "--paths", "neon-fp16,neon-fp32").split(","):
        n_classes, emb_dim = _lower_path(a, pkg, pth, ops, vshape, g.nsamp,
                                         lv, ev)
    labels = cliargs.opt(a, "--labels")
    labels_raw = _read_bytes(labels)
    sha = labels_info(labels, labels_raw, n_classes)
    if prep.birdnetgo_labels(ops, g.nsamp, family) and not quiet:
        _warn_label_form(labels, labels_raw)
    _set_metadata(a, pkg, dict(family=family, sample_rate=sr,
                               n_samples=g.nsamp, n_classes=n_classes,
                               emb_dim=emb_dim, post=post, labels_sha256=sha,
                               labels=labels, labels_raw=labels_raw))
    out = cliargs.opt(a, "-o")
    size = pkg.write(out)
    print(f"wrote {out}: {size / 1e6:.1f} MB, {len(pkg.tensors)} tensors, "
          f"{n_classes} classes, emb {emb_dim}")
    return 0
