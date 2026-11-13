"""Unit tests for the lowering, ONNX import, canonicalization and package code.

  python -P -s tests/test_lower_units.py     (bnneon installed, or
                                               PYTHONPATH=<checkout>/python)

Covers the GEMM packers, int8 quantization, buffer slot planning, the
per-op kernel checks, the canonicalization rules on tiny graphs (one ONNX
Runtime run each) and the .bnpk record layout. Exit 0 when every check
passes.
"""
# The private helpers (_lifetimes, _pick_slot, _text, _head_name, the record
# packers) are the units under test:
# pylint: disable=protected-access
import os
import struct
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from bnneon.compiler import canon
from bnneon.compiler import errs
from bnneon.compiler import lower
from bnneon.compiler import lower_dot
from bnneon.compiler import onnxg
from bnneon.compiler import pkgfmt
from bnneon.compiler import recog
from bnneon import testutil

CHECKS = testutil.Checks()
check = CHECKS.check


raises = testutil.raises


def lowering(path="neon-fp32"):
    """A Lowering over an empty op list with a 16-sample input."""
    return lower.Lowering(pkgfmt.Package(), [], path, 16)


def test_paths():
    f16, f32, s8 = (lower.PATHS[p] for p in ("neon-fp16", "neon-fp32",
                                             "neon-int8"))
    check("path dtypes and variants", (f16.act, f16.var, f32.act, f32.var,
                                       s8.act, s8.var)
          == (lower.F16, 1, lower.F32, 0, lower.F32, 0))
    check("path tensor prefixes",
          [p.prefix for p in (f16, f32, s8)] == ["f16/", "f32/", "s8n/"])
    check("path hwcap and CLI modes", [(p.hwcap, p.cli_mode)
                                       for p in (f16, f32, s8)]
          == [(1, 3), (0, 2), (0, 4)])
    check("path SQNR floors", [p.need_db for p in (f16, f32, s8)]
          == [50.0, 60.0, 60.0])
    check("another path name is neon-fp32",
          lower.path_info("neon-x") is lower.PATHS["neon-fp32"])
    for name in ("neon-fp16", "neon-fp32", "neon-int8"):
        low = lowering(name)
        info = lower.PATHS[name]
        check(f"Lowering reads PATHS ({name})", (low.adt, low.var, low.pre)
              == (info.act, info.var, info.prefix)
              and low.f16 == (name == "neon-fp16")
              and low.int8 == (name == "neon-int8"))
    check("only the dotprod family declares int8 GEMMs",
          lower_dot.LoweringDot.INT8_GEMMS and not lower.Lowering.INT8_GEMMS)
    check("the dotprod family takes the fp16 path only",
          lower_dot.applies("neon-fp16", [{"op": "FE_DBIMG"}])
          and not lower_dot.applies("neon-fp32", [{"op": "FE_DBIMG"}])
          and not lower_dot.applies("neon-fp16", [{"op": "PW"}]))


def _pack_h16_loop(w, k_pad):
    """The per-row form pack_h16 had (kept here as the oracle)."""
    n, k = w.shape
    n_pad = lower.r8(n)
    n16 = n_pad // 16
    panels = np.zeros((((n_pad * k_pad * 2) + 127) & ~63) // 2, np.float16)
    w16 = w.astype(np.float16)
    for o in range(n):
        nb, j = divmod(o, 16)
        if nb < n16:
            panels[(nb * k_pad + np.arange(k)) * 16 + j] = w16[o]
        else:
            panels[n16 * k_pad * 16 + np.arange(k) * 8 + j] = w16[o]
    return panels


def _pack_f32_loop(w, k_pad):
    """The per-row form pack_f32 had (kept here as the oracle)."""
    n, k = w.shape
    n_pad = lower.r8(n)
    panels = np.zeros((((n_pad * k_pad * 4) + 127) & ~63) // 4, np.float32)
    for o in range(n):
        nb, j = divmod(o, 8)
        panels[(nb * k_pad + np.arange(k)) * 8 + j] = w[o]
    return panels


def test_pack_vectorized():
    rng = np.random.default_rng(3)
    same = True
    for n, k, k_pad in ((1, 1, 8), (7, 5, 8), (8, 16, 16), (16, 3, 8),
                        (17, 33, 40), (24, 10, 16), (40, 64, 64),
                        (100, 37, 48)):
        w = rng.standard_normal((n, k)).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        h = lower.pack_h16(w, b, k_pad)[0]
        f = lower.pack_f32(w, b, k_pad)[0]
        h_old = _pack_h16_loop(w, k_pad).view(np.uint16)
        f_old = _pack_f32_loop(w, k_pad).view(np.uint32)
        same = (same and np.array_equal(h.view(np.uint16), h_old)
                and np.array_equal(f.view(np.uint32), f_old))
    check("pack_h16 / pack_f32 equal the per-row forms bit for bit", same)


def test_pack():
    check("r8", [lower.r8(x) for x in (0, 1, 8, 9)] == [0, 8, 8, 16])
    w = np.arange(15, dtype=np.float32).reshape(3, 5)
    b = np.array([1, 2, 3], np.float32)
    panels, bb, shp = lower.pack_f32(w, b, 8)
    check("pack_f32 shape", shp == (3, 5, 8, 8), shp)
    check("pack_f32 size", panels.size == 80 and panels.dtype == np.float32,
          panels.size)
    ok = all(panels[k * 8 + j] == w[j, k] for j in range(3) for k in range(5))
    check("pack_f32 layout [Np/8][Kp][8]", ok)
    check("pack_f32 K padding is zero", not panels[5 * 8:8 * 8].any())
    check("pack_f32 bias", bb.tolist()[:4] == [1, 2, 3, 0] and bb.size == 16)
    w = np.arange(60, dtype=np.float32).reshape(20, 3)
    panels, b32, b16, shp = lower.pack_h16(w, np.zeros(20, np.float32), 4)
    check("pack_h16 shape", shp == (20, 3, 4, 24), shp)
    check("pack_h16 panel row", panels[2 * 16 + 5] == np.float16(w[5, 2]))
    tail = 1 * 4 * 16  # after the one full 16-row panel
    check("pack_h16 tail row", panels[tail + 1 * 8 + 2] == np.float16(w[18, 1]))
    check("pack_h16 biases", b32.size == 40 and b16.dtype == np.float16)


def test_s8():
    lw = lowering("neon-int8")
    w = np.array([[0.5, -1.0, 0.25], [0.0, 0.0, 0.0]], np.float32)
    tl, shp = lw.s8_tensors("t", w, np.array([1.0, 2.0], np.float32))
    check("s8 shape", shp == (2, 3, 16, 8), shp)
    names = [t[0] for t in lw.pkg.tensors]
    check("s8 tensor names", names == ["s8n/t.q", "s8n/t.sw", "s8n/t.cs",
                                       "s8n/t.b"], names)
    q = lw.pkg.tensors[tl[0]][1]
    check("s8 packed [Np/4][Kp/16][4][16]", q.shape == (2, 1, 4, 16), q.shape)
    # 0.5 * 127 = 63.5 rounds away from zero to 64; 0.25 * 127 = 31.75 -> 32
    check("s8 rounding", q[0, 0, 0, :3].tolist() == [64, -127, 32],
          q[0, 0, 0, :3])
    check("s8 zero row", not q[0, 0, 1].any())
    cs = lw.pkg.tensors[tl[2]][1]
    check("s8 column sums", cs[:2].tolist() == [64 - 127 + 32, 0])
    sw = lw.pkg.tensors[tl[1]][1]
    check("s8 scales", np.isclose(sw[0], 1 / 127) and sw[1] == 1.0, sw[:2])


def test_slots():
    f32 = 0
    bufs = [(f32, 1, 1, 16, 16, 0, 16)] + [(f32, 1, 1, 8, 8, 0, 8)] * 3
    bufs.append((1, 1, 1, 8, 8, 0, 8))  # fp16, named by no op
    ops = [(1, [0, 1], [], []), (1, [1, 2], [], []), (1, [2, 3], [], [])]
    out, total = lower.plan_slots(bufs, ops, [None])
    slots = [b[-1] for b in out]
    check("plan_slots reuse after lifetime", slots == [0, 1, 2, 1, 3], slots)
    check("plan_slots bytes", total == 64, total)
    out, _ = lower.plan_slots(bufs, ops, [3])
    slots = [b[-1] for b in out]
    check("plan_slots pinned own slot", slots[3] == 1 and slots[1] == 2
          and slots[2] == 3, slots)
    first, last = lower._lifetimes(ops)
    check("lifetimes", first == {0: 0, 1: 0, 2: 1, 3: 2}
          and last == {0: 0, 1: 1, 2: 2, 3: 2})
    slots = [None, [0, 100, 0], [0, 40, 0], [0, 10, 5]]
    pick = lower._pick_slot
    check("pick best fit", pick(slots, 0, 30, 1) == 2)
    check("pick largest when none fits", pick(slots, 0, 500, 1) == 1)
    check("pick skips busy", pick(slots, 0, 5, 3) == 2)
    check("pick new slot", pick(slots, 1, 5, 3) == 4 and len(slots) == 5)


def test_names():
    check("src_name", lower.src_name({"nodes": ["a", "b", "c"]}) == "'a' +2")
    check("src_name one", lower.src_name({"nodes": ["a"]}) == "'a'")
    check("src_name none", lower.src_name(None) == "?")
    check("src_name empty names", lower.src_name({"nodes": [""]}) == "'?'")
    check("dense_kxk", lower.dense_kxk({"op": "CONV", "k": [3, 3]})
          and not lower.dense_kxk({"op": "CONV", "k": [1, 1]}))
    check("act_code", lower.act_code({"act": "swish"}) == 1
          and lower.act_code({}) == 0)
    e = raises(lambda: lower.act_code({"act": "sigmoid"}), recog.Unsupported)
    check("act_code unsupported", e and "'sigmoid'" in str(e), e)


def test_checks():
    e = raises(lambda: lower.Lowering.pw_check(12, 8, 0), recog.Unsupported)
    check("pw stride check", e and "stride >= 16" in str(e), e)
    e = raises(lambda: lower.Lowering.pw_check(8, 8, 1), recog.Unsupported)
    check("pw halo check", e and "halo 1" in str(e), e)
    check("pw ok", lower.Lowering.pw_check(8, 8, 0) is None)
    dw = {"k": [3, 3], "s": [1, 1], "pads": [1, 1, 1, 1], "act": "relu"}
    e = raises(lambda: lower.Lowering.dw_check(dw), recog.Unsupported)
    check("dw act check", e and "swish only" in str(e), e)
    dw.update(act="swish", k=[4, 4])
    e = raises(lambda: lower.Lowering.dw_check(dw), recog.Unsupported)
    check("dw even k", e and "square odd k" in str(e), e)
    conv = {"w": np.zeros((8, 4, 3, 3)), "s": [3, 3], "pads": [1, 1, 1, 1]}
    e = raises(lambda: lower.Lowering.convk_check(conv, (8, 8, 4, 4), (3, 3)),
               recog.Unsupported)
    check("convk stride check", e and "equal strides <= 2" in str(e), e)
    e = raises(lambda: lower.Lowering.convk_check(conv, (8, 8, 2, 4), (3, 3)),
               recog.Unsupported)
    check("convk channel check", e and "4 input channels on a 2" in str(e), e)
    lw = lowering()
    lw.vbuf[7] = lw.buf(0, 4, 4, 8)
    check("residual none", lw.residual_buf({}, (4, 4, 8), "x") == -1)
    check("residual ok", lw.residual_buf({"res": 7}, (4, 4, 8), "x") == 1)
    e = raises(lambda: lw.residual_buf({"res": 7}, (4, 4, 16), "y"),
               recog.Unsupported)
    check("residual mismatch", e and str(e).endswith("4x4x16 y"), e)
    e = raises(lambda: lower.Lowering.sed_smoothing(
        {"pool": (2, 1), "dconv": None}, 8), recog.Unsupported)
    check("sed even pool", e and "keep the length" in str(e), e)
    pk, dw_, _, dk = lower.Lowering.sed_smoothing(
        {"pool": None, "dconv": None}, 8)
    check("sed no smoothing", (pk, dk, dw_.shape) == (0, 0, (8, 1)))


def test_dot():
    check("r16", [lower.r16(x) for x in (0, 1, 16, 17)] == [0, 16, 16, 32])
    check("tf same pads", lower_dot.tf_same_pads(7, 3, 2) == (1, 1)
          and lower_dot.tf_same_pads(8, 3, 2) == (0, 1))
    o = {"k": [3, 3], "s": [2, 2], "pads": [0, 0, 1, 1]}
    check("is_tf_same", lower_dot.is_tf_same(o, 8, 8)
          and not lower_dot.is_tf_same(o, 7, 7))
    sed = {"pool": (3, 1), "clip": lower_dot.SED_CLIP,
           "dconv": (np.full((4, 3), lower_dot.AVG3), 1, np.zeros(4))}
    check("sed_reason ok", lower_dot.sed_reason(sed, 4) is None)
    sed["clip"] = (-5.0, 5.0)
    check("sed_reason clip", "clip" in lower_dot.sed_reason(sed, 4))
    check("applies", lower_dot.applies("neon-fp16", [{"op": "FE_DBIMG"}])
          and not lower_dot.applies("neon-fp32", [{"op": "FE_DBIMG"}]))


def save(d, name, nodes, inits, out_shape):
    """Writes a one-input [1, 16] float32 graph; returns its path."""
    g = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        [numpy_helper.from_array(v, k) for k, v in inits.items()])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    p = os.path.join(d, name + ".onnx")
    onnx.save(m, p)
    return p


def test_canon():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        inits = {"shape": np.array([1, 1, 4, 4], np.int64),
                 "pads": np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64),
                 "w": rng.standard_normal((2, 1, 3, 3)).astype(np.float32)}
        nodes = [helper.make_node("Reshape", ["x", "shape"], ["img"]),
                 helper.make_node("Cast", ["img"], ["imgc"], to=1),
                 helper.make_node("Pad", ["imgc", "pads"], ["padded"]),
                 helper.make_node("Conv", ["padded", "w"], ["conv"]),
                 helper.make_node("GlobalAveragePool", ["conv"], ["y"]),
                 helper.make_node("Relu", ["x"], ["unused"])]
        g = onnxg.Graph(save(d, "canon", nodes, inits, [1, 2, 1, 1]))
        x = rng.standard_normal((1, 16)).astype(np.float32)
        before = g.run_shapes(x, want_values=["y"])["y"]
        fired = canon.canonicalize(g)
        expect = [("float->float Cast", 1), ("zero Pad folded into a Conv", 1),
                  ("GlobalAveragePool -> ReduceMean", 1),
                  ("dead node removed", 1)]
        check("canon rules fired", fired == expect, fired)
        ops = [n.op_type for n in g.nodes]
        check("canon nodes", ops == ["Reshape", "Conv", "ReduceMean"], ops)
        conv = g.nodes[1]
        check("canon conv pads", g.attrs(conv)["pads"] == [1, 1, 1, 1])
        after = g.run_shapes(x, want_values=["y"])["y"]
        check("canon output unchanged",
              np.allclose(before, after, rtol=1e-5, atol=1e-6))
        check("canon fixed point", canon.canonicalize(g) == [])
    check("attribute text", canon._text(b"NOTSET") == "NOTSET"
          and canon._text("x") == "x")


def test_graph():
    with tempfile.TemporaryDirectory() as d:
        nodes = [helper.make_node("Relu", ["x"], ["y"])]
        g = onnxg.Graph(save(d, "relu", nodes, {}, [1, 16]))
        check("graph input", (g.inp, g.nsamp, g.outputs) == ("x", 16, ["y"]))
        check("graph declared attrs", (g.shape, g.dtype, g.canon, g.canon_err)
              == ({}, {}, [], 0.0))
        g.run_shapes()
        check("graph shapes", g.shape["y"] == [1, 16]
              and g.dtype["y"] == np.float32)
        check("single consumer", g.single_cons("x") is g.nodes[0])
        check("other input", g.other_in(g.nodes[0], "x") == "x")
        check("where", g.where(g.nodes[0]) == "node 0 Relu ''")
        e = raises(lambda: onnxg.Graph(os.path.join(d, "missing.onnx")),
                   errs.ModelError)
        check("missing file", e and str(e).endswith("missing.onnx: file not "
                                                    "found"), e)
        bad = os.path.join(d, "bad.onnx")
        with open(bad, "wb") as f:
            f.write(b"not onnx")
        e = raises(lambda: onnxg.Graph(bad), errs.ModelError)
        check("not onnx", e and "not a loadable ONNX file" in str(e), e)
    check("head name", onnxg._head_name("h_in", "at", "h_in", "p/") == "at"
          and onnxg._head_name("t", "at", "h_in", "p/") == "p/t"
          and onnxg._head_name("", "at", "h_in", "p/") == "")


def test_package():
    check("pad8", [len(pkgfmt.pad8(b"x" * n)) for n in (0, 1, 8, 9)]
          == [0, 8, 8, 16])
    pkg = pkgfmt.Package()
    pkg.set("model_id", "m")
    pkg.set("n", 3)
    i = pkg.tensor("t", np.arange(4, dtype=np.float32))
    check("tensor dedup", pkg.tensor("t", np.arange(4, dtype=np.float32)) == i)
    ops = [(3 | 1 << 8, [0, 1], [i], [7, 8, 9])]
    pkg.add_path("neon-fp32", 0, [(0, 1, 1, 16, 16, 0, 16, 0)], ops, 0, 0,
                 None)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.bnpk")
        size = pkg.write(p)
        with open(p, "rb") as f:
            raw = f.read()
    check("package size", size == len(raw) and size % 64 == 0, size)
    magic, bom, ver = raw[:8], *struct.unpack_from("<IH", raw, 8)
    check("package header", (magic, bom, ver)
          == (b"BNNPKG01", 0x01020304, pkgfmt.FORMAT_VERSION))
    counts = struct.unpack_from("<4I", raw, 16)
    check("package counts", counts == (2, 1, 1, 0), counts)
    meta_off, tensor_off, path_off, file_size = struct.unpack_from("<4Q", raw,
                                                                   32)
    check("package offsets", meta_off == 64 and file_size == size)
    rec = pkgfmt._op_record(*ops[0])
    head = struct.unpack_from("<6H", rec)
    check("op record", head == (0x103, 40, 2, 1, 3, 0) and len(rec) == 40,
          head)
    check("op record body", struct.unpack_from("<6i", rec, 12)
          == (0, 1, 0, 7, 8, 9))
    check("meta record", pkgfmt._meta_record("n", 3)
          == pkgfmt.pad8(struct.pack("<HHI", 1, 1, 8) + b"n"
                         + struct.pack("<q", 3)))
    off = struct.unpack_from("<Q", raw, tensor_off + 48 + 24)[0]
    check("tensor blob", raw[off:off + 16]
          == np.arange(4, dtype=np.float32).tobytes() and off % 64 == 0)
    check("path name", raw[path_off:path_off + 9] == b"neon-fp32")


def main():
    test_paths()
    test_pack_vectorized()
    test_pack()
    test_s8()
    test_slots()
    test_names()
    test_checks()
    test_dot()
    test_canon()
    test_graph()
    test_package()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
