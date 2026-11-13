"""The Python binding (python/bnneon/binding.py) against a built library.

  python -I tests/test_python_binding.py [--lib build/libbnneon.so]
      [--cli build/cli] [--model FILE ...] [--windows N]

(-I with bnneon installed in the interpreter; for a checkout that is not
installed: PYTHONPATH=<checkout>/python python -P -s tests/...)

For every model found (default: a few packages and the legacy .bn24 in
weights/, those this CPU can run), the binding's scores must equal the C CLI's
output bit for bit on N synthetic windows (default 3), and the metadata,
labels, embedding and error paths must behave as documented. Exit 0 when every
check passes; a missing library, CLI or model is a clean skip (exit 0 with a
"skip" line).
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

import bnneon
from bnneon import testutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODELS = ["birdnet_v2.4-neon-fp16.bnpk", "birdnet_v2.4-neon-fp32.bnpk",
                  "battybirdnet_eu_256khz-neon-fp32.bnpk",
                  "perch_v2_nae-neon-fp16-s8.bnpk", "v24.bn24",
                  "birdnet_v3.0_north-america-east-neon-fp16.bnpk"]
CHECKS = testutil.Checks()
check = CHECKS.check


def raises(fn, kind, text=""):
    """(ok, detail): fn() raised a BnnError of this kind whose text has text."""
    try:
        fn()
    except bnneon.BnnError as e:
        return e.kind == kind and text in str(e), f"{e.kind}: {e}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        # any other exception is a failed check, reported with its type
        return False, f"{type(e).__name__}: {e}"
    return False, "no exception"


def windows(n, ns, sr, seed=0):
    """Deterministic test windows: a rising chirp plus noise, amplitude ~0.3."""
    t = np.arange(ns) / sr
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f0 = sr * (0.02 + 0.03 * i)
        chirp = np.sin(2 * np.pi * (f0 * t + 0.1 * sr * t * t / (2 * t[-1])))
        x = 0.25 * chirp + 0.05 * rng.standard_normal(ns)
        out.append(x.astype(np.float32))
    return np.stack(out)


def check_metadata(name, path, m, pi, h):
    """Info against probe, license metadata, an absent key."""
    inf = m.info
    check(f"{name}: info agrees with probe",
          (inf.n_classes, inf.n_samples, inf.sample_rate, inf.model_id)
          == (pi.n_classes, pi.n_samples, pi.sample_rate, pi.model_id),
          repr(inf))
    check(f"{name}: license metadata", bool(m.meta("license")) and
          bnneon.probe_meta(path, "license", h) == m.meta("license"))
    check(f"{name}: the old key spelling 'licence' reads the same",
          m.meta("licence") == m.meta("license")
          and bnneon.probe_meta(path, "licence", h) == m.meta("license"))
    check(f"{name}: absent key -> None", m.meta("no_such_key") is None)


def check_labels(name, path, m, h, tmp):
    """The embedded label file: count, verification, a changed copy."""
    inf = m.info
    labels = m.labels()
    count = None if labels is None else len(labels)
    check(f"{name}: labels inside = n_classes", labels is not None
          and len(labels) == inf.n_classes and all(labels),
          f"{count} vs {inf.n_classes}")
    lf = os.path.join(tmp, "labels.txt")
    with open(lf, "wb") as f:
        f.write(m.meta("labels", raw=True))
    try:
        bnneon.verify_labels(path, lf, h)
        vl = None
    except bnneon.BnnError as e:
        vl = f"{e.kind}: {e}"
    check(f"{name}: the embedded label file verifies", vl is None, vl)
    with open(lf, "ab") as f:
        f.write(b"extra\n")
    check(f"{name}: a changed label file -> LABELS",
          *raises(lambda: bnneon.verify_labels(path, lf, h), "LABELS",
                  "is not the label file"))


def expected_confidence(post, score):
    """The confidence of one score under the model's post (None: softmax)."""
    if post == bnneon.POST_SIGMOID:
        return 1 / (1 + np.exp(-score))
    if post == bnneon.POST_SOFTMAX:
        return None
    return score  # NONE: identity


def check_scores(name, m, x):
    """infer() on the windows: embedding, repeatability, confidences, length.

    Returns:
        The scores [n][n_classes].
    """
    inf = m.info
    res = [m.infer(w) for w in x]
    scores = np.stack([r[0] for r in res])
    emb = res[0][1]
    emb_ok = emb is None or (emb.shape == (inf.emb_dim,)
                             and np.isfinite(emb).all())
    check(f"{name}: embedding", (emb is None) == (inf.emb_dim == 0) and emb_ok,
          f"emb_dim {inf.emb_dim}")
    check(f"{name}: repeatable", np.array_equal(m.infer(x[0])[0], scores[0]))
    c = m.confidences(scores)
    k = int(np.argmax(scores[0]))
    want = expected_confidence(inf.post, float(scores[0, k]))
    ok = (np.isfinite(c).all() and c.min() >= 0 and c.max() <= 1
          and c.shape == scores.shape)
    if inf.post == bnneon.POST_SOFTMAX:
        ok = ok and np.allclose(c.sum(axis=-1), 1, atol=1e-4)
    else:
        ok = ok and abs(float(c[0, k]) - want) < 1e-6
    check(f"{name}: confidences (post {inf.post})", ok, f"{c[0, k]} vs {want}")
    try:
        m.infer(x[0][:-1])
        short = "no exception"
    except ValueError:
        short = None
    check(f"{name}: wrong window length -> ValueError", short is None, short)
    return scores


def check_cli(name, path, m, x, scores, cli, tmp):
    """The binding's scores are the CLI's, bit for bit."""
    wf = os.path.join(tmp, "w.f32")
    of = os.path.join(tmp, "o.f32")
    x.tofile(wf)
    r = subprocess.run([cli, path, "0", wf, of], capture_output=True, text=True,
                       check=False)
    if r.returncode == 0:
        ref = np.fromfile(of, np.float32)
    else:
        ref = np.zeros(0, np.float32)
    if r.returncode:
        detail = r.stderr
    elif ref.size == scores.size:
        detail = f"max |d| {np.abs(ref.reshape(scores.shape) - scores).max()}"
    else:
        detail = "max |d| size"
    check(f"{name}: {len(x)} windows bit-identical to the CLI "
          f"({m.info.path.decode()})", ref.tobytes() == scores.tobytes(),
          detail)


def check_threads(name, m, x, scores):
    """3 threads give the same bits; a legacy file refuses threads."""
    if m.info.format_version >= 1:
        m.set_threads(3)
        t3 = np.stack([m.infer(w)[0] for w in x])
        m.set_threads(1)
        check(f"{name}: 3 threads bit-identical to 1",
              t3.tobytes() == scores.tobytes())
    else:
        check(f"{name}: threads on a legacy file -> USAGE",
              *raises(lambda: m.set_threads(2), "USAGE"))


def check_model(path, h, cli, nwin, tmp):
    """Every check on one model; False when this CPU runs none of its paths."""
    name = os.path.basename(path)
    pi = bnneon.probe(path, h)
    if not pi.path:
        print(f"skip {name}: this CPU can run none of its paths")
        return False
    with bnneon.Model(path, bnneon.MODE_DEFAULT, h) as m:
        check_metadata(name, path, m, pi, h)
        if m.info.format_version >= 1:
            check_labels(name, path, m, h, tmp)
        x = windows(nwin, m.info.n_samples, m.info.sample_rate)
        scores = check_scores(name, m, x)
        check_cli(name, path, m, x, scores, cli, tmp)
        check_threads(name, m, x, scores)
    check(f"{name}: closed model raises USAGE",
          *raises(lambda: m.infer(x[0]), "USAGE", "closed"))
    return True


def check_errors(models, h, tmp):
    """Error paths: kind and message come from the library."""
    check("missing file -> PACKAGE",
          *raises(lambda: bnneon.Model(os.path.join(tmp, "none.bnpk"), 0, h),
                  "PACKAGE", "cannot open"))
    bnv3 = os.path.join(tmp, "old.bnv3")
    with open(bnv3, "wb") as f:
        # a legacy BirdNET v3.0 header: magic, version 1, 2 tensors, 800
        # classes (the engine that read it is retired)
        f.write(b"BNV3ENG1" + struct.pack("<5I", 1, 2, 800, 1, 0))
    for what, fn in (("probe", lambda: bnneon.probe(bnv3, h)),
                     ("Model", lambda: bnneon.Model(bnv3, 0, h))):
        check(f"legacy .bnv3 {what} -> PACKAGE, names the v3.0 packages",
              *raises(fn, "PACKAGE", "neon-fp16.bnpk"))
    pkg = next((p for p in models if p.endswith(".bnpk")), None)
    if not pkg:
        return
    check("mode 1 on a package -> USAGE",
          *raises(lambda: bnneon.Model(pkg, bnneon.MODE_REFERENCE, h), "USAGE"))
    if "fp16" in os.path.basename(pkg):
        other = bnneon.MODE_FP32
    else:
        other = bnneon.MODE_FP16
    check("path not in the package -> CPU",
          *raises(lambda: bnneon.Model(pkg, other, h), "CPU",
                  "not in this package"))


def check_label_parsing():
    """binding.class_labels: blank lines skipped, Perch's header line.

    tests/test_label_rule.py checks the whole rule against the compiler's.
    """
    cl = bnneon.binding.class_labels
    check("labels: blank lines are not labels (as the compiler counts)",
          cl("A_a\n\nB_b\r\n  \nC_c\n\n") == ["A_a", "B_b", "C_c"])
    check("labels: Perch's header line is not a class",
          cl("inat2024_fsd50k\nA_a\n\nB_b\n") == ["A_a", "B_b"])
    check("labels: another first line is a class",
          cl("header\nA_a\n") == ["header", "A_a"])
    check("labels: no final newline", cl("A_a\nB_b") == ["A_a", "B_b"])


def main():
    check_label_parsing()
    lib = testutil.opt("--lib", os.path.join(ROOT, "build", "libbnneon.so"))
    cli = testutil.opt("--cli", os.path.join(ROOT, "build", "cli"))
    nwin = int(testutil.opt("--windows", "3"))
    if not (os.path.isfile(lib) and os.access(cli, os.X_OK)):
        print(f"skip: no built library/CLI at {lib} / {cli} (run `make`, or "
              f"pass --lib and --cli)")
        return 0
    models = testutil.opt_all("--model") or testutil.default_models(
        ROOT, DEFAULT_MODELS)
    if not models:
        print("skip: no model file in weights/ (pass --model FILE)")
        return 0
    h = bnneon.load_library(lib)
    check("api version", h.bnn_api_version() == bnneon.API_VERSION)
    with tempfile.TemporaryDirectory(prefix="bnn_py_") as tmp:
        ran = sum(check_model(path, h, cli, nwin, tmp) for path in models)
        if not ran:
            print("skip: no model in the list runs on this CPU")
            return 0
        check_errors(models, h, tmp)
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
