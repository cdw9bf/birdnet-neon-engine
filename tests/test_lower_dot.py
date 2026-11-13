"""Tests of the dotprod + fp16 lowering and, optionally, of its kernels.

  python -I tests/test_lower_dot.py [--model V3.onnx --labels LABELS.txt
      --cli build/cli [--windows W.f32]]

(-I with bnneon installed in the interpreter; for a checkout that is not
installed: PYTHONPATH=<checkout>/python python -P -s tests/...)

The lowering is python/bnneon/compiler/lower_dot.py, its kernels
src/rt_ops_dot.c.

Part A (always): the padding helpers, the SED head check, which Lowering class
a path gets, and that every layer the family has no kernel for is refused at
compile time with a reason (never left for the runtime to reject).
Part B (with --model, a BirdNET v3.0 fp32 ONNX, --labels and --cli, a CLI built
for a CPU with fp16 + dotprod): the model compiles to a neon-fp16 package, the
runtime loads it, and its output bytes are the same at 1, 2 and 3 threads.
With --windows W.f32 (real 5 s windows at 32 kHz) as well, `bnc check`
compares the package with ONNX Runtime on the first 8 of them: at least
SQNR_FLOOR_DB of logit SQNR and the same top class on every window. (The
family's int8 layers reach 46-53 dB on real audio; a mis-lowered layer falls
far below the floor.)
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from bnneon import testutil
from bnneon.compiler import cmd_compile
from bnneon.compiler import lower
from bnneon.compiler import lower_dot
from bnneon.compiler import pkgfmt
from bnneon.compiler import recog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BNC = os.path.join(ROOT, "compiler", "bnc.py")
# -P keeps the current directory off sys.path; the bnneon package comes from
# the install or PYTHONPATH
PYFLAGS = ("-P", "-s")
SQNR_FLOOR_DB = 40.0
CHECKS = testutil.Checks(detail_limit=800)
check = CHECKS.check


def refusal(fn, *args):
    """The Unsupported message fn(*args) raises, or None if it raises none."""
    try:
        fn(*args)
    except recog.Unsupported as e:
        return str(e)
    return None


def conv(ins, out, cout, cin, k, stride, pads, act="swish", res=None):
    """A CONV IR op with zero weights."""
    weights = np.zeros((cout, cin, k, k), np.float32)
    op = dict(op="CONV", ins=[ins], out=out, w=weights,
              b=np.zeros(cout, np.float32), k=[k, k], s=[stride, stride],
              pads=pads, act=act)
    if res is not None:
        op["res"] = res
    return op


def sed(ins, out, red, length, channels=32, hidden=32, classes=16,
        pool=(3, 1), clip=lower_dot.SED_CLIP):
    """A SED head IR op with zero weights and the kernel's average pool."""
    avg = (np.full((channels, 3), lower_dot.AVG3, np.float32), 1,
           np.zeros(channels, np.float32))
    return dict(op="SED", ins=[ins], out=out, red=red, L=length, C=channels,
                pool=pool, dconv=avg, clip=clip, scale=0.5,
                w1=np.zeros((hidden, channels), np.float32),
                b1=np.zeros(hidden, np.float32),
                wa=np.zeros((classes, hidden), np.float32),
                ba=np.zeros(classes, np.float32),
                wc=np.zeros((classes, hidden), np.float32),
                bc=np.zeros(classes, np.float32))


def lowering(ops, shapes):
    """A LoweringDot on these ops with the given value shapes."""
    low = lower_dot.LoweringDot(pkgfmt.Package(), ops, "neon-fp16", 160000)
    low.vshape = shapes
    return low


def test_helpers():
    check("TF-same pads, even size, stride 2",
          lower_dot.tf_same_pads(224, 3, 2) == (0, 1))
    check("TF-same pads, odd size, stride 2",
          lower_dot.tf_same_pads(225, 3, 2) == (1, 1))
    check("TF-same pads, stride 1", lower_dot.tf_same_pads(56, 3, 1) == (1, 1))
    op = conv(1, 2, 96, 24, 3, 2, [0, 0, 1, 1])
    check("is_tf_same accepts bottom / right pads on an even image",
          lower_dot.is_tf_same(op, 112, 234))
    check("is_tf_same refuses them where TF-same pads the left",
          not lower_dot.is_tf_same(op, 56, 117))
    check("SED head the kernel implements",
          lower_dot.sed_reason(sed(1, 2, 3, 7), 32) is None)
    reason = lower_dot.sed_reason(sed(1, 2, 3, 7, pool=(5, 2)), 32)
    check("SED head with another pool", "max-pool" in (reason or ""))
    reason = lower_dot.sed_reason(sed(1, 2, 3, 7, clip=(-5.0, 5.0)), 32)
    check("SED head with another clip", "clip" in (reason or ""))


def test_class_choice():
    dbimg, logmel = [{"op": "FE_DBIMG"}], [{"op": "FE_LOGMEL"}]
    choose = cmd_compile.lowering_class
    check("neon-fp16 of a dB-image model: the dotprod family",
          choose("neon-fp16", dbimg) is lower_dot.LoweringDot)
    check("neon-fp32 of the same model: the base lowering",
          choose("neon-fp32", dbimg) is lower.Lowering)
    check("neon-fp16 of a log-mel model: the base lowering",
          choose("neon-fp16", logmel) is lower.Lowering)


def test_stem():
    even = conv(1, 2, 24, 3, 3, 2, [0, 0, 1, 1])
    low = lowering([even], {2: (112, 234)})
    low.vbuf[1] = low.buf(lower_dot.F32, 224, 468, 3, ld=3)
    low.cur = even
    low.l_conv(0, even)
    code, bufs, _, ints = low.out[-1]
    stem_code = lower_dot.OP["STEM3"] | lower_dot.DOT << 8
    check("stem on an even image: STEM3 in the dotprod variant",
          code == stem_code, hex(code))
    check("stem op integers", ints == [224, 468, 112, 234], ints)
    out_dtype, _, _, _, out_ld, out_halo, _ = low.bufs[bufs[1]]
    check("stem output: zero-haloed fp16, 24 channels",
          (out_dtype, out_ld, out_halo) == (lower_dot.F16, 24, 1))
    odd = conv(1, 2, 24, 3, 3, 2, [1, 0, 1, 1])
    low = lowering([odd], {2: (113, 234)})
    low.vbuf[1] = low.buf(lower_dot.F32, 225, 468, 3, ld=3)
    low.cur = odd
    msg = refusal(low.l_conv, 0, odd)
    check("stem on an odd image height is refused with the reason",
          msg is not None and "even" in msg, msg)


def test_depthwise_input():
    dw = dict(op="DWCONV", ins=[1], out=2,
              w=np.zeros((32, 1, 3, 3), np.float32),
              b=np.zeros(32, np.float32), k=[3, 3], s=[1, 1],
              pads=[1, 1, 1, 1], act="swish")
    low = lowering([dw], {2: (14, 30)})
    low.vbuf[1] = low.buf(lower_dot.F32, 14, 30, 32, ld=32)
    low.cur = dw
    msg = refusal(low.l_dwconv, 0, dw)
    check("depthwise conv on fp32 rows is refused at compile time",
          msg is not None and "depthwise" in msg, msg)
    low = lowering([dw], {2: (14, 30)})
    low.vbuf[1] = low.buf(lower_dot.F16, 14, 30, 32, ld=32)
    low.cur = dw
    low.l_dwconv(0, dw)
    names = [code & 0xff for code, _, _, _ in low.out]
    padded_dw = [lower.OP["PAD"], lower.OP["DWK"]]
    check("depthwise 3x3 on unpadded fp16 rows lowers as a padding copy + "
          "fp16 depthwise", names == padded_dw, names)
    dw5 = dict(dw, w=np.zeros((32, 1, 5, 5), np.float32), k=[5, 5],
               pads=[2, 2, 2, 2])
    low = lowering([dw5], {2: (14, 30)})
    low.vbuf[1] = low.buf(lower_dot.F16, 14, 30, 32, ld=32, halo=1)
    low.cur = dw5
    msg = refusal(low.l_dwconv, 0, dw5)
    check("depthwise 5x5 on a halo-1 image is refused at compile time",
          msg is not None and "3x3" in msg, msg)


def test_head_sum():
    dense = dict(op="DENSE", ins=[1], out=3,
                 w=np.zeros((16, 32), np.float32), b=np.zeros(16, np.float32))
    sed_h = sed(2, 4, 3, 7)
    add = dict(op="VADD", ins=[3, 4], out=5)
    low = lowering([dense, sed_h, add], {})
    low.l_dense(0, dense)
    low.l_sed(1, sed_h)
    msg = refusal(low.l_vadd, 2, add)
    check("dense + one SED head only is refused at compile time",
          msg is not None and "width" in msg, msg)
    lone = dict(op="DENSE", ins=[1], out=3,
                w=np.zeros((16, 32), np.float32), b=np.zeros(16, np.float32))
    low = lowering([lone], {})
    msg = refusal(low.l_dense, 0, lone)
    check("a dense head that is not part of the sum is refused",
          msg is not None, msg)


def run_package(cli, package, windows, out, threads):
    """Runs the CLI in mode 3 with BNN_THREADS; returns (ok, stderr)."""
    env = dict(os.environ, BNN_THREADS=str(threads))
    done = subprocess.run([cli, package, "3", windows, out, "1"], env=env,
                          capture_output=True, text=True, check=False)
    return done.returncode == 0, done.stderr


def test_against_onnxruntime(model, labels, cli, windows):
    """`bnc check` on real windows: the neon-fp16 package vs ONNX Runtime."""
    with tempfile.TemporaryDirectory() as work:
        report = os.path.join(work, "check.json")
        command = [sys.executable, *PYFLAGS, BNC, "check", model, "--labels",
                   labels, "--sample-rate", "32000", "--post", "sigmoid",
                   "--emb", "embeddings", "--paths", "neon-fp16", "--cli", cli,
                   "--windows-file", windows, "--windows", "8", "--quiet",
                   "--workdir", work, "--json", report]
        done = subprocess.run(command, capture_output=True, text=True,
                              check=False)
        if not os.path.isfile(report):
            check("bnc.py check writes its report", False,
                  done.stdout + done.stderr)
            return
        with open(report, encoding="utf-8") as f:
            path = json.load(f)["paths"].get("neon-fp16", {})
        sqnr = path.get("sqnr_db", 0.0)
        check(f"neon-fp16 vs ONNX Runtime on real windows: {sqnr} dB >= "
              f"{SQNR_FLOOR_DB} dB", sqnr >= SQNR_FLOOR_DB, path)
        check("same top class as ONNX Runtime on every window",
              path.get("top1_agree") == "8/8", path)
        check("check does not FAIL the path",
              path.get("verdict") in ("PASS", "WARN"), path)


def test_package(model, labels, cli):
    """A v3.0 graph compiles to neon-fp16; the output is thread-invariant."""
    with tempfile.TemporaryDirectory() as work:
        package = os.path.join(work, "v3-fp16.bnpk")
        windows = os.path.join(work, "w.f32")
        command = [sys.executable, *PYFLAGS, BNC, "compile", model, "--labels",
                   labels, "--model-id", "test_v3", "--family", "30", "--post",
                   "sigmoid", "--sample-rate", "32000", "--emb", "embeddings",
                   "--paths", "neon-fp16", "--force", "-o", package]
        done = subprocess.run(command, capture_output=True, text=True,
                              check=False)
        check("a BirdNET v3.0 graph compiles to a neon-fp16 package",
              done.returncode == 0, done.stdout + done.stderr)
        if done.returncode:
            return
        rng = np.random.default_rng(3)
        x = rng.standard_normal((2, 160000)) * 0.05
        x.astype(np.float32).tofile(windows)
        outputs = []
        for threads in (1, 2, 3):
            out = os.path.join(work, f"o{threads}.f32")
            ok, err = run_package(cli, package, windows, out, threads)
            check(f"the runtime runs it on {threads} thread(s)", ok, err)
            if ok:
                with open(out, "rb") as f:
                    outputs.append(f.read())
        same = (len(outputs) == 3 and outputs[0] == outputs[1] == outputs[2]
                and len(outputs[0]) > 0)
        check("output bytes are the same at 1, 2 and 3 threads", same)


def main():
    test_helpers()
    test_class_choice()
    test_stem()
    test_depthwise_input()
    test_head_sum()
    model = testutil.opt("--model")
    labels = testutil.opt("--labels")
    cli = testutil.opt("--cli")
    if model and labels and cli:
        test_package(model, labels, cli)
        if testutil.opt("--windows"):
            test_against_onnxruntime(model, labels, cli,
                                     testutil.opt("--windows"))
        else:
            print("skip: the ONNX Runtime comparison needs --windows W.f32 "
                  "(real windows)")
    else:
        print("skip: part B needs --model V3.onnx --labels LABELS.txt --cli "
              "build/cli")
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
