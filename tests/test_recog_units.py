"""Unit tests for the pure helpers and the IR passes of compiler/recog.py.

  python -P -s tests/test_recog_units.py     (bnneon installed, or
                                               PYTHONPATH=<checkout>/python)

No model, library or ONNX Runtime run is needed: graphs are small stand-ins.
Exit 0 when every check passes.
"""
import sys
import types

import numpy as np

from bnneon.compiler import recog
from bnneon import testutil

CHECKS = testutil.Checks()
check = CHECKS.check
raises = testutil.raises


def node(op_type, inputs, outputs, name=None):
    """A stand-in for an ONNX node."""
    return types.SimpleNamespace(op_type=op_type,
                                 input=list(inputs),
                                 output=list(outputs),
                                 name=name or outputs[0])


class FakeGraph:
    """The parts of onnxg.Graph the recognizer helpers read."""

    def __init__(self, nodes, const=None, shape=None):
        self.nodes = nodes
        self.const = const or {}
        self.shape = shape or {}
        self.cons = {}
        for n in nodes:
            for t in n.input:
                self.cons.setdefault(t, []).append(n)

    def where(self, n):
        """Text naming a node in messages."""
        return f"node '{n.name}'"


def test_scalars():
    """first_float, decode_attr, take."""
    check("first_float of a list", recog.first_float([2.5, 7]) == 2.5)
    value = recog.first_float(np.array([[3]], np.int64))
    check("first_float gives a float", isinstance(value, float) and value == 3)
    check("decode_attr decodes bytes",
          recog.decode_attr(b"reflect") == "reflect")
    check("decode_attr keeps str", recog.decode_attr("linear") == "linear")
    check("decode_attr keeps numbers", recog.decode_attr(3) == 3)
    nodes = []
    n = node("Relu", ["a"], ["b"])
    check("take returns the output", recog.take(nodes, n) == "b")
    check("take records the node", nodes == [n])


def test_slice_indices():
    """Indices of a single-axis Slice."""
    check("slice head", recog.slice_indices(6, 0, 3, 1) == [0, 1, 2])
    check("slice tail to a huge end",
          recog.slice_indices(6, 3, 1 << 63, 1) == [3, 4, 5])
    check("slice reverse",
          recog.slice_indices(4, -1, -(1 << 63), -1) == [3, 2, 1, 0])
    check("slice negative start", recog.slice_indices(5, -2, 5, 1) == [3, 4])
    check("slice step 2", recog.slice_indices(5, 0, 5, 2) == [0, 2, 4])


def dft_kernel(n_fft, flip_sin=False):
    """A [2F][1][N] window * cos | window * sin kernel and its window."""
    n_bins = n_fft // 2 + 1
    win = np.hanning(n_fft) + 0.1
    f = np.arange(n_bins)[:, None]
    k = np.arange(n_fft)[None, :]
    cos = win * np.cos(2 * np.pi * f * k / n_fft)
    sin = win * np.sin(2 * np.pi * f * k / n_fft)
    if flip_sin:
        sin = -sin
    return np.concatenate([cos, sin])[:, None, :], win


def test_windowed_dft():
    """The DFT-kernel check of the dB-image front end."""
    w, win = dft_kernel(16)
    check("dft kernel accepted", recog.is_windowed_dft(w, win, 16, 9, 1e-5))
    w_neg, _ = dft_kernel(16, flip_sin=True)
    check("dft kernel with -sin accepted",
          recog.is_windowed_dft(w_neg, win, 16, 9, 1e-5))
    bad_cos = w.copy()
    bad_cos[2, 0, 5] += 0.01
    check("perturbed cos row refused",
          not recog.is_windowed_dft(bad_cos, win, 16, 9, 1e-5))
    bad_sin = w.copy()
    bad_sin[12, 0, 5] += 0.01
    check("perturbed sin row refused",
          not recog.is_windowed_dft(bad_sin, win, 16, 9, 1e-5))
    check("perturbation inside a loose tolerance accepted",
          recog.is_windowed_dft(bad_cos, win, 16, 9, 0.1))


def test_einsum():
    """Einsum equations that are a matrix product."""
    check("einsum over a middle axis refused",
          not recog.einsum_is_matmul("bft,fm->btm"))
    check("einsum x[..,F] M[F,m]", recog.einsum_is_matmul("btf,fm->btm"))
    check("einsum 2-D", recog.einsum_is_matmul("ij,jk->ik"))
    check("einsum wrong output", not recog.einsum_is_matmul("btf,fm->bmt"))
    check("einsum implicit output", not recog.einsum_is_matmul("ij,jk"))
    check("einsum three operands", not recog.einsum_is_matmul("a,b,c->d"))
    check("einsum 3-D matrix", not recog.einsum_is_matmul("btf,fmk->btmk"))


def test_stft2_layout():
    """Image layout of the dual-STFT Concat output."""
    check("layout mel first",
          recog.stft2_image_layout([96, 511, 1, 2], 96) == (96, 511, "hw1c"))
    check("layout mel third",
          recog.stft2_image_layout([1, 511, 96, 2], 96) == (96, 511, "1whc"))
    e = raises(lambda: recog.stft2_image_layout([2, 96, 511, 1], 96),
               recog.Unsupported)
    check("layout refused with its shape",
          e is not None and
          str(e) == "spectrogram image layout [2, 96, 511, 1]", e)


def test_conv_pads():
    """Explicit pads from auto_pad / pads."""
    pads = recog.conv_pads({"auto_pad": b"SAME_UPPER"}, (5, 5), (3, 3), (3, 3),
                           (2, 2))
    check("SAME_UPPER even total", pads == [1, 1, 1, 1], pads)
    pads = recog.conv_pads({"auto_pad": b"SAME_UPPER"}, (4, 4), (2, 2), (3, 3),
                           (2, 2))
    check("SAME_UPPER odd total pads the end", pads == [0, 0, 1, 1], pads)
    pads = recog.conv_pads({"auto_pad": "SAME_LOWER"}, (4, 4), (2, 2), (3, 3),
                           (2, 2))
    check("SAME_LOWER odd total pads the start", pads == [1, 1, 0, 0], pads)
    pads = recog.conv_pads({"auto_pad": b"SAME_UPPER"}, (8, 4), (8, 4), (1, 1),
                           (1, 1))
    check("SAME on a 1x1 conv is no pad", pads == [0, 0, 0, 0], pads)
    pads = recog.conv_pads({"auto_pad": b"VALID", "pads": [1, 1, 1, 1]},
                           (5, 5), (3, 3), (3, 3), (1, 1))
    check("VALID ignores pads", pads == [0, 0, 0, 0], pads)
    pads = recog.conv_pads({"pads": (0, 1, 2, 3)}, (5, 5), (3, 3), (3, 3),
                           (1, 1))
    check("explicit pads as a list", pads == [0, 1, 2, 3], pads)
    check("no attribute is no pad",
          recog.conv_pads({}, (5, 5), (3, 3), (3, 3), (1, 1)) == [0, 0, 0, 0])


def test_json_attrs():
    """Attributes as JSON-friendly values."""
    attrs = recog.json_attrs({
        "mode": b"reflect",
        "axes": (np.int64(1), 2),
        "value": np.array([1.5, 2.5]),
        "alpha": 0.5,
        "other": {1, },
    })
    check("json_attrs decodes bytes", attrs["mode"] == "reflect")
    check("json_attrs makes int lists",
          attrs["axes"] == [1, 2] and isinstance(attrs["axes"][0], int))
    check("json_attrs lists arrays", attrs["value"] == [1.5, 2.5])
    check("json_attrs keeps numbers", attrs["alpha"] == 0.5)
    check("json_attrs stringifies the rest", attrs["other"] == "{1}")
    check("json_attrs keeps key order",
          list(attrs) == ["mode", "axes", "value", "alpha", "other"])


def test_guidance():
    """Nearest pattern and next step of an unsupported node."""
    near, step = recog.guidance("Softmax", "no rule", None)
    check("guidance of a known op",
          (near, step) == recog.NEAREST["Softmax"])
    near, step = recog.guidance("Frobnicate", "no rule", None)
    check("guidance of an unknown op",
          (near, step) == recog.NEAREST_DEFAULT)
    near, step = recog.guidance("Conv", "front end: window length", None)
    check("guidance of a front-end failure",
          near.startswith("the BirdNET v2.4 dual STFT") and
          "fe_stft2 / fe_logmel" in step)
    _, step = recog.guidance("Pad", "front end: a Conv-framed DFT ('c')", None)
    check("guidance of a Conv-framed DFT",
          step.startswith("stock BirdNET v3.0 compiles"))
    _, step = recog.guidance("Conv", "no rule", "t7")
    check("guidance downstream",
          step == "fix the unsupported node producing 't7' first; this node "
          "may then match", step)


def test_format_problems():
    """Text lines of a problem report."""
    prob = dict(index=4,
                op_type="Erf",
                name="erf_1",
                reason="no rule",
                attrs={"k": 1},
                inputs=[dict(name="a", shape=[1, 2], const=False),
                        dict(name="c", shape=[2], const=True)],
                outputs=[dict(name="b", shape=[1, 2])],
                downstream_of=None,
                nearest="NEAR",
                next_step="STEP")
    lines = recog.format_problem(prob)
    check("format_problem lines", lines == [
        "unsupported op Erf at node 4 Erf 'erf_1': no rule",
        "    attributes: {'k': 1}",
        "    inputs: a [1, 2], c [2] (const)",
        "    outputs: b [1, 2]",
        "    nearest supported pattern: NEAR",
        "    next step: STEP",
    ], lines)
    down = dict(prob, index=5, name="erf_2", downstream_of="b")
    lines = recog.format_problem(down)
    check("format_problem downstream", lines == [
        "unsupported op Erf at node 5 Erf 'erf_2': no rule",
        "    (downstream of the unsupported node producing 'b')",
    ], lines)
    at_input = dict(prob, index=None, attrs={}, inputs=[], outputs=[])
    check("format_problem at the graph input",
          recog.format_problem(at_input)[0] ==
          "unsupported op Erf at graph input: no rule")
    lines = recog.format_problems([prob, down, down, down, down])
    check("format_problems counts the downstream nodes",
          len(lines) == 7 and lines[-1] ==
          "(4 downstream node(s) not listed: Erf 'erf_2', Erf 'erf_2', "
          "Erf 'erf_2', ...; --verbose lists them)", lines[-1])
    lines = recog.format_problems([prob, down], verbose=True)
    check("format_problems verbose lists them", len(lines) == 8, lines)


def test_val_and_chan_vec():
    """Val and the per-channel constant check."""
    act = recog.Val("act", 3, 4, 5, "nchw")
    vec = recog.Val("vec", c=7)
    check("Val ids count up", vec.id == act.id + 1)
    check("Val repr of an activation", repr(act) == f"v{act.id}:act[3x4x5]")
    check("Val repr of a vector", repr(vec) == f"v{vec.id}:vec[7]")
    check("Val defaults", (vec.H, vec.W, vec.C, vec.layout) == (1, 1, 7, None))

    r = recog.Recognizer(None)
    check("chan_vec of None", r.chan_vec(None, 3, "nchw") is None)
    scalar = r.chan_vec(np.float32(2), 3, "nchw")
    check("chan_vec broadcasts a scalar", list(scalar) == [2.0, 2.0, 2.0])
    nchw = np.arange(3.0).reshape(1, 3, 1, 1)
    check("chan_vec NCHW [1,C,1,1]",
          list(r.chan_vec(nchw, 3, "nchw")) == [0.0, 1.0, 2.0])
    check("chan_vec NCHW [C,1,1]",
          r.chan_vec(np.zeros((3, 1, 1)), 3, "nchw") is not None)
    check("chan_vec NCHW refuses a 1-D [C]",
          r.chan_vec(np.zeros(3), 3, "nchw") is None)
    check("chan_vec NHWC takes a 1-D [C]",
          r.chan_vec(np.zeros(3), 3, "nhwc") is not None)
    check("chan_vec NHWC refuses [1,C,1,1]",
          r.chan_vec(nchw, 3, "nhwc") is None)
    check("chan_vec refuses another size",
          r.chan_vec(np.zeros(4), 3, "nhwc") is None)


def test_graph_helpers():
    """only_cons, _layout, _slice_axis, opt_const, bias on a stand-in."""
    # pylint: disable=protected-access
    a = node("Reshape", ["x"], ["a"])
    b = node("Transpose", ["a"], ["b"])
    c = node("Conv", ["b", "w", "bias"], ["c"])
    d1 = node("Relu", ["c"], ["d1"])
    d2 = node("Neg", ["c"], ["d2"])
    sl = node("Slice", ["s", "st", "en", "ax", "sp"], ["s2"])
    g = FakeGraph([a, b, c, d1, d2, sl],
                  const={
                      "w": np.zeros((2, 1, 1, 1)),
                      "bias": np.array([1.0, 2.0]),
                      "st": np.array([-1]),
                      "en": np.array([-(1 << 63)]),
                      "ax": np.array([1]),
                      "sp": np.array([-1]),
                  },
                  shape={"s": [1, 4, 9]})
    r = recog.Recognizer(g)
    check("only_cons finds the consumer", r.only_cons("x") is a)
    check("only_cons remembers it", r.at is a)
    check("only_cons filters by type", r.only_cons("x", "Conv") is None)
    check("only_cons takes a tuple of types",
          r.only_cons("x", ("Conv", "Reshape")) is a)
    check("only_cons refuses two consumers", r.only_cons("c") is None)
    check("only_cons blames the first of them", r.at is d1)
    check("only_cons of an unused tensor", r.only_cons("d1") is None)

    seen = []
    t, n = r._layout("x", seen)
    check("_layout steps over layout ops",
          (t, n) == ("b", c) and seen == [a, b])
    seen = []
    t, n = r._layout("x", seen, ("Transpose",))
    check("_layout stops at other ops", (t, n) == ("x", a) and not seen)

    check("opt_const of a present input", r.opt_const(c, 2) is g.const["bias"])
    check("opt_const of a missing input", r.opt_const(d1, 2) is None)
    check("bias of a conv with one",
          list(r.bias(c, 2, np.float32)) == [1.0, 2.0])
    zeros = r.bias(d1, 3, np.float64)
    check("bias of a node without one",
          list(zeros) == [0.0, 0.0, 0.0] and zeros.dtype == np.float64)

    check("_cint reads integers", r._cint(sl, 4) == [-1])
    check("_cint default for a missing input", r._cint(d1, 3, [0]) == [0])
    e = raises(lambda: r._cint(c, 0), recog.Unsupported)
    check("_cint refuses a non-constant input",
          e is not None and e.node is c and
          str(e) == "non-constant input 0 of node 'c'", e)
    check("_slice_axis of a reversal",
          r._slice_axis(sl, "s") == (-2, [3, 2, 1, 0]))


def conv(out, ins, **more):
    """A CONV op dict."""
    return dict(op="CONV", ins=ins, out=out, act=None, nodes=[f"n{out}"],
                **more)


def test_fuse_and_dce():
    """fuse and dce on small op lists."""
    ops = [conv(2, [1]), dict(op="ACT", act="relu", ins=[2], out=3,
                              nodes=["r"])]
    fused = recog.fuse(ops, {3})
    check("fuse conv + act", len(fused) == 1 and fused[0]["act"] == "relu" and
          fused[0]["nodes"] == ["n2", "r"], fused)
    check("fuse gives the op the requested id", fused[0]["out"] == 3)

    ops = [conv(2, [1]), dict(op="ACT", act="relu", ins=[2], out=3,
                              nodes=["r"])]
    check("fuse keeps a requested conv output apart",
          len(recog.fuse(ops, {2})) == 2)

    ops = [conv(2, [1]), conv(3, [2]),
           dict(op="ADD", ins=[3, 2], out=4, nodes=["add"]),
           dict(op="MEAN", ins=[4], out=5, nodes=["m"])]
    fused = recog.fuse(ops, {5})
    check("fuse conv + residual", [o["op"] for o in fused] ==
          ["CONV", "CONV", "MEAN"] and fused[1]["res"] == 2 and
          fused[1]["nodes"] == ["n3", "add"], fused)
    check("fuse rewires the consumer of the ADD", fused[2]["ins"] == [3])

    ops = [dict(conv(2, [1]), act="relu"),
           dict(op="POOL", kind="max", k=[1, 2], s=[1, 2], ins=[2], out=3,
                nodes=["mp"]),
           dict(op="POOL", kind="avg", k=[1, 2], s=[1, 2], ins=[2], out=4,
                nodes=["ap"]),
           dict(op="CONCAT", ins=[3, 4], out=5, nodes=["cat"])]
    fused = recog.fuse(ops, {5})
    check("fuse conv + pool pair + concat", len(fused) == 1 and
          fused[0]["pool"] == "maxavg_w2" and fused[0]["out"] == 5 and
          fused[0]["nodes"] == ["n2", "mp", "ap", "cat"], fused)

    ops = [conv(2, [1]), conv(3, [1]), conv(4, [3], res=2), conv(5, [1])]
    kept = recog.dce(ops, {4})
    check("dce keeps inputs and residuals", [o["out"] for o in kept] ==
          [2, 3, 4], kept)
    check("dce of nothing wanted", not recog.dce(ops, set()))


def test_fold_vec():
    """fold_vec on a two-head sum."""
    w1 = np.array([[1, 2], [3, 4]], np.float32)
    w2 = np.array([[10, 20], [30, 40]], np.float32)
    b1 = np.array([1, 2], np.float32)
    b2 = np.array([5, 6], np.float32)
    ops = [
        dict(op="DENSE", ins=[1], out=2, w=w1, b=b1, nodes=["a"]),
        dict(op="VAFFINE", ins=[2], out=3, kind="mul", c=2.0, nodes=["m"]),
        dict(op="DENSE", ins=[1], out=4, w=w2, b=b2, nodes=["b"]),
        dict(op="VADD", ins=[3, 4], out=5, nodes=["s"]),
        dict(op="VAFFINE", ins=[5], out=6, kind="sub",
             c=np.array([1, 1], np.float32), nodes=["d"]),
    ]
    out = recog.fold_vec(ops, {6})
    last = out[-1]
    check("fold_vec keeps the op count", len(out) == 5)
    check("fold_vec folds the chain into one DENSE",
          last["op"] == "DENSE" and last["ins"] == [1] and last["out"] == 6 and
          last["folded"] == 2, last)
    check("fold_vec weights", np.array_equal(last["w"], 2 * w1 + w2) and
          last["w"].dtype == np.float32, last["w"])
    check("fold_vec bias", np.array_equal(last["b"], 2 * b1 + b2 - 1) and
          last["b"].dtype == np.float32, last["b"])
    check("fold_vec node order", last["nodes"] == ["a", "b", "d"],
          last["nodes"])
    check("fold_vec leaves the folded values to dce",
          [o["out"] for o in recog.dce(out, {6})] == [6])
    check("fold_vec key order", list(last) ==
          ["op", "ins", "out", "w", "b", "nodes", "folded"])

    div = [
        dict(op="DENSE", ins=[1], out=2, w=w1, b=b1, nodes=["a"]),
        dict(op="VAFFINE", ins=[2], out=3, kind="div", c=4.0, nodes=["q"]),
    ]
    out = recog.fold_vec(div, {3})
    check("fold_vec division", np.array_equal(out[1]["w"], w1 / 4) and
          np.array_equal(out[1]["b"], b1 / 4), out[1])

    plain = [conv(2, [1])]
    check("fold_vec without dense layers returns its input",
          recog.fold_vec(plain, {2}) is plain)


def main():
    test_scalars()
    test_slice_indices()
    test_windowed_dft()
    test_einsum()
    test_stft2_layout()
    test_conv_pads()
    test_json_attrs()
    test_guidance()
    test_format_problems()
    test_val_and_chan_vec()
    test_graph_helpers()
    test_fuse_and_dce()
    test_fold_vec()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
