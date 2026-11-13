"""Unit tests for the pure helpers of the converters, generators, benchmarks
and small tools under scripts/, and of tests/test_errors.py.

  python -P -s tests/test_scripts_units_s2.py     (bnneon installed, or
                                                   PYTHONPATH=<checkout>/python)

The scripts are standalone programs; they are loaded by file with
bnneon.extern.load_module (no sys.path edits). Exit 0 when every check
passes. A checkout without bnneon.extern, matplotlib or the shipped weights
skips the tests that need them (a "skip" line, exit 0).
"""
import contextlib
import hashlib
import io
import os
import struct
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper

from bnneon import testutil

try:
    from bnneon import extern
except ImportError:
    extern = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
TESTS = os.path.join(ROOT, "tests")
GENERATED = ("src/gemm_f16_kblock.h", "src/gemm_f32_kblock.h",
             "bench/peak_kern.c", "bench/peak_ops.c")
CHECKS = testutil.Checks()
check = CHECKS.check


def load(name, directory=SCRIPTS):
    """A script loaded as a module."""
    return extern.load_module(directory, name)


def raises(exc, fn, *args, **kw):
    """Does fn(*args, **kw) raise exc?"""
    try:
        fn(*args, **kw)
    except exc:
        return True
    return False


def read_bytes(path):
    """A file's bytes."""
    with open(path, "rb") as f:
        return f.read()


def write_text(path, text):
    """Writes a text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def exit_code(fn, *args):
    """fn(*args)'s return value or SystemExit code, and what it wrote to
    stderr."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            code = fn(*args)
        except SystemExit as e:
            code = e.code
    return code, err.getvalue()


# ------------------------------------------------------------- exporters
def _fake_model(m, nodes, values):
    """A stand-in for the exporter's Model with the given nodes and values."""
    fake = m.Model.__new__(m.Model)
    fake.nodes = nodes
    fake.values = values
    return fake


def test_export_weights():
    m = load("export_weights")
    out = m.Export()
    out.add("x", np.arange(6).reshape(2, 3))
    out.blocks.append((1, 24, 72, 36, 0, 0, 0, 0))
    labels = b"a\n" * 6522
    hdr = m.file_header(out, labels)
    head = struct.unpack_from("<8s5I", hdr)
    lic_len = head[5]
    check("bn24 header: magic, counts, license padded to 8",
          head[:5] == (b"BN24ENG1", 1, 1, 6522, 1) and lic_len % 8 == 0
          and lic_len - len(m.LICENSE) < 8, head)
    check("bn24 header: label sha256 after the license, then the blocks",
          hdr[28 + lic_len:60 + lic_len] == hashlib.sha256(labels).digest()
          and struct.unpack_from("<8i", hdr, 60 + lic_len)
          == (1, 24, 72, 36, 0, 0, 0, 0))
    ents, datas, end = m.tensor_directory(out.tensors, 64)
    check("bn24 directory: 88-byte entries, 64-byte aligned data",
          len(ents[0]) == 88 and datas[0] == (64, out.tensors[0][1].tobytes())
          and end == 128, (len(ents[0]), datas[0][0], end))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.txt")
        write_text(p, "a\n" * 6522 + "\n\n")
        check("bn24 read_labels counts non-empty lines",
              m.read_labels(p) == read_bytes(p))
        write_text(p, "a\n" * 6521)
        check("bn24 read_labels refuses 6521 lines",
              raises(AssertionError, m.read_labels, p))
    nodes = [helper.make_node("Conv", ["x", "w"], ["y"], strides=[2, 2]),
             helper.make_node("Relu", ["y"], ["z"]),
             helper.make_node("Conv", ["z", "w2", "b2"], ["o"])]
    values = {"w": np.ones((4, 3, 1, 1), np.float32),
              "w2": np.ones((5, 4, 1, 1), np.float32),
              "b2": np.arange(5, dtype=np.float32)}
    fake = _fake_model(m, nodes, values)
    convs = m.ConvStream(fake)
    n, w, b, at = convs.take()
    check("ConvStream.take: zeros for a missing bias, the attributes",
          n is nodes[0] and w.shape == (4, 3, 1, 1)
          and np.array_equal(b, np.zeros(4)) and b.dtype == np.float32
          and at == {"strides": [2, 2]})
    _, _, b, _ = convs.take()
    check("ConvStream.take: the bias", np.array_equal(b, np.arange(5)))
    check("ConvStream.check_done passes when every Conv was read",
          not raises(SystemExit, convs.check_done))
    convs = m.ConvStream(fake)
    convs.take()
    check("ConvStream.check_done refuses an unread Conv",
          raises(SystemExit, convs.check_done))
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        code, err = exit_code(m.main)
    finally:
        sys.argv = argv
    check("export_weights without arguments: the usage and exit 2",
          code == 2 and err.startswith("usage:"), (code, err))


def test_export_perch():
    m = load("export_perch")
    check("perch conv_key stem / head / block",
          [m.conv_key("/Stem_0/BatchNorm_0/Mul"),
           m.conv_key("/Head_0/BatchNorm_0/Mul"),
           m.conv_key("/MBConv_7/DepthwiseBatchNorm_0/Mul")]
          == [("stem", None), ("head", None), (7, "Depthwise")])
    one = helper.make_node("Max", ["a"], ["b"], name="n1")
    check("perch only() returns the one node", m.only([one], "max") is one)
    check("perch only() refuses two",
          raises(AssertionError, m.only, [one, one], "max"))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.txt")
        write_text(p, "inat2024_fsd50k\na\n\nb\n")
        check("perch read_labels skips the catalog header line",
              m.read_labels(p, 2) == hashlib.sha256(read_bytes(p)).digest())
        check("perch read_labels refuses a wrong count",
              raises(AssertionError, m.read_labels, p, 3))
    out = m.Export()
    out.add("p", np.zeros((2, 3)))
    out.blocks.append((3, 2, 40, 0, 24, 10, 0, 1 | 1 << 8 | 1 << 16 | 1 << 24))
    sha = b"\x07" * 32
    hdr = m.file_header(out, 14795, sha)
    head = struct.unpack_from("<8s5I", hdr)
    lic_len = head[5]
    check("bnp2 header: magic, counts, license padded to 8, sha, blocks",
          head[:5] == (b"BNP2ENG1", 1, 1, 14795, 1) and lic_len % 8 == 0
          and hdr[28 + lic_len:60 + lic_len] == sha
          and struct.unpack_from("<8i", hdr, 60 + lic_len)[7]
          == 0x01010101, head)
    entries, blobs, end = m.tensor_directory(out.tensors, 64)
    fields = struct.unpack("<48sII4IQQ", entries[0])
    check("bnp2 directory pads dims with 0",
          fields[3:7] == (2, 3, 0, 0) and fields[7:] == (64, 24)
          and blobs[0][0] == 64 and end == 128, (fields, end))


# ------------------------------------------------------------ dump_graph
def _attrs(n):
    return {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}


def test_dump_graph():
    m = load("dump_graph")
    conv = helper.make_node("Conv", ["x", "w"], ["y"], name="c", group=1,
                            strides=[2, 2], pads=[1, 1, 1, 1])
    shp = {"x": [1, 3, 8, 8], "y": [1, 4, 4, 4], "a": [4, 10],
           "b": [3, 10, 5], "s": [4], "q": [2, 2]}
    init = {"w": np.zeros((4, 3, 3, 3), np.float32)}
    mac, line = m.compact_entry(0, conv, shp, init, _attrs(conv))
    check("dump conv MACs = out pixels x channels x kernel",
          mac == 4 * 4 * 4 * 27 and line == (
              "  0 Conv  in [1, 3, 8, 8] w [4, 3, 3, 3] g 1 s [2, 2] "
              "p [1, 1, 1, 1] out [1, 4, 4, 4] MAC 0.0M  c\n"), line)
    mm = helper.make_node("MatMul", ["a", "b"], ["o"], name="mm")
    mac, line = m.compact_entry(1, mm, shp, init, {})
    check("dump batched MatMul MACs", mac == 40 * 5 * 3
          and line == "  1 MatMul [4, 10] x [3, 10, 5] MAC 0.0M  mm\n", line)
    shape = helper.make_node("Shape", ["x"], ["s"], name="sh")
    check("dump hides rank-1 shape plumbing",
          m.compact_entry(2, shape, shp, init, {}) == (0, None))
    cast = helper.make_node("Cast", ["x"], ["q"], name="ca", to=1)
    check("dump lists a rank-2 Cast",
          m.compact_entry(3, cast, shp, init, _attrs(cast))
          == (0, "  3 Cast       [[1, 3, 8, 8]] -> [2, 2]  ca\n"))
    full = m.full_entry(3, cast, shp, {"to": 1, "mode": b"linear",
                                       "value": 5}, 0)
    check("dump full entry decodes bytes and drops 'value'",
          full == ("  3 Cast               ca\n"
                   "     in  [('x', [1, 3, 8, 8])]\n"
                   "     out [('q', [2, 2])]\n"
                   "     att {'to': 1, 'mode': 'linear'} MAC 0.00M\n"), full)
    g = helper.make_graph([conv, mm, shape, cast, conv], "g", [], [])
    check("dump intermediate names: in order, once, no empty names",
          m.intermediate_names(g) == ["y", "o", "s", "q"])


# ------------------------------------------------------------ generators
def test_generators():
    kb = load("gen_kblock")
    check("gen clobbers", kb.clobbers([0, 1, 16]) == '"v0", "v1", "v16"')
    check("gen accumulator comment",
          kb.acc_comment(1) == "accumulators start at 0"
          and kb.acc_comment(0) == "accumulators loaded from cb")
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src"))
        os.makedirs(os.path.join(d, "bench"))
        cwd = os.getcwd()
        argv = sys.argv
        os.chdir(d)
        sys.argv = [sys.argv[0]]  # gen_kblock_f32 takes an output path
        try:
            for name in ("gen_kblock", "gen_kblock_f32", "gen_peak_kern",
                         "gen_peak_ops"):
                load(name).main()
            load("gen_kblock_f32").main()
        finally:
            os.chdir(cwd)
            sys.argv = argv
        for f in GENERATED:
            check(f"tracked {f} is what its generator writes",
                  read_bytes(os.path.join(d, f))
                  == read_bytes(os.path.join(ROOT, f)))


# --------------------------------------------------- comparison tools, plot
def test_tools():
    cmp_ = load("cmp")
    check("cmp sigmoid", cmp_.sigmoid(np.array([0.0])) == 0.5)
    pc = load("perch_cmp")
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    sm = pc.softmax(z)
    check("perch_cmp softmax rows sum to 1 and are shift-invariant",
          np.allclose(sm.sum(1), 1) and np.allclose(sm, pc.softmax(z + 50)))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.f32")
        np.arange(6, dtype=np.float32).tofile(p)
        a = pc.load(p, 3)
        check("perch_cmp load reshapes to [n][C] float64",
              a.shape == (2, 3) and a.dtype == np.float64 and a[1, 0] == 3)
        check("cmp load needs 6522 columns", raises(ValueError, cmp_.load, p))


def test_replay_args():
    m = load("gate_sim_v3_pkg")
    got = m.parse(["p", "t", "e", "s", "--site", "10,20"])
    check("replay defaults", got
          == (["t", "e", "s"], False, "fp32 ORT (this run)", (10.0, 20.0)),
          got)
    got = m.parse(["p", "t", "--ref-name", "fp32 ORT global", "e", "s",
                   "--site", "-1.5,2", "lab", "--regional"])
    check("replay --ref-name, --site and --regional anywhere",
          got == (["t", "e", "s", "lab"], True, "fp32 ORT global",
                  (-1.5, 2.0)), got)
    check("replay needs --site", raises(ValueError, m.parse,
                                        ["p", "t", "e", "s"]))
    code, err = exit_code(m.main, ["p", "t", "e", "s"])
    check("replay without --site: a message and exit 2",
          code == 2 and "--site LAT,LON is required" in err, (code, err))
    code, err = exit_code(m.main, ["p", "t", "--site", "10,20"])
    check("replay with too few arguments: the usage and exit 2",
          code == 2 and err.startswith("usage:"), (code, err))


def test_plot():
    try:
        m = load("plot_final_perf")
    except ImportError as e:
        print(f"skip plot_final_perf helpers ({e})")
        return
    check("plot format_ms", [m.format_ms(58.7), m.format_ms(146.8),
                             m.format_ms(2321.8)] == ["58.7", "147", "2,322"])
    check("plot rows: default, 1 thread, 4 threads per model",
          all(len(row) == 4 for _, rows in m.BOARDS for row in rows))


# ------------------------------------------------------------ benchmarks
def test_benchmarks():
    tb = load("stock_threads_bench")
    check("threads bench: the prediction output is not the embedding",
          tb.prediction_size({"shape": [1, 1280]}) == -1
          and tb.prediction_size({"shape": [1, 6522]}) == 6522)
    a = tb.parse_args(["m.onnx", "--npy", "w", "--out", "o"])
    check("threads bench defaults",
          (a.threads, a.runs, a.warm, a.ov) == (1, 10, 2, False))
    cb = load("stock_cadence_bench")
    a = cb.parse_args(["m", "--stride", "1.5", "--npy", "w", "--out", "o"])
    check("cadence bench defaults",
          (a.threads, a.stride, a.duration) == (1, 1.5, 600))


# ---------------------------------------------------------- test_errors
def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def test_test_errors_helpers():
    te = load("test_errors", TESTS)
    buf = struct.pack("<7I4xQ", 1, 2, 3, 4, 5, 6, 7, 99)  # a 40-byte record
    rec = te._read_buf(buf, 0)  # pylint: disable=protected-access
    check("Pkg buffer record",
          rec == dict(at=0, dtype=1, H=2, W=3, C=4, ld=5, halo=6, slot=7,
                      elems=99), rec)
    op_bytes = struct.pack("<6H", 0x103, 28, 2, 1, 1, 0) + struct.pack(
        "<4i", 5, 6, 7, -5)
    op, rec_len = te._read_op(op_bytes, 0)  # pylint: disable=protected-access
    check("Pkg op record: counts, offsets and argument lists",
          rec_len == 28 and op["code"] == 0x103 and op["b"] == [5, 6]
          and op["t"] == [7] and op["p"] == [-5] and op["t_at"] == 20
          and op["p_at"] == 24, op)
    mm = helper.make_node("MatMul", ["f", "W"], ["pre"], name="mm")
    add = helper.make_node("Add", ["pre", "b"], ["logits"], name="add")
    g = helper.make_graph([mm, add], "g", [_vi("f", [1, 4])],
                          [_vi("logits", [1, 2])])
    check("find_head through the bias Add", te.find_head(g, "logits") == [mm])
    check("find_head directly", te.find_head(g, "pre") == [mm])
    rm = helper.make_node("ReduceMean", ["x"], ["y"], axes=[-2, -1],
                          keepdims=1)
    check("reduce_mean_axes from the attribute, mod 4",
          te.reduce_mean_axes(rm, {}) == [2, 3])
    rm2 = helper.make_node("ReduceMean", ["x", "ax"], ["y"], keepdims=1)
    check("reduce_mean_axes from an initializer",
          te.reduce_mean_axes(rm2, {"ax": np.array([3, 2])}) == [2, 3])
    rm3 = helper.make_node("ReduceMean", ["x"], ["y"], axes=[2, 3],
                           keepdims=0)
    check("reduce_mean_axes refuses keepdims 0",
          te.reduce_mean_axes(rm3, {}) is None)
    convs = [helper.make_node("Conv", ["x", "w"], [f"c{i}"], name=f"conv{i}")
             for i in range(4)]
    convs[3].attribute.append(helper.make_attribute("dilations", [1, 1]))
    m = helper.make_model(helper.make_graph(convs, "g", [], []))
    conv = te.dilate_conv(m)
    check("dilate_conv replaces the fourth Conv's dilations",
          conv.name == "conv3" and _attrs(conv)["dilations"] == [2, 2]
          and len(conv.attribute) == 1)
    te.dilate_conv(helper.make_model(helper.make_graph(convs[:3] + [
        helper.make_node("Conv", ["x", "w"], ["c"], name="plain")], "g", [],
        [])))
    info = ("model_id t family 0\nbest path for this CPU: neon-fp32\n"
            "mode 0: ok\n")
    check("parse_best_path: the path the info names",
          te.parse_best_path(info) == "neon-fp32")
    check("parse_best_path: (none) is NO_PATH, not a missing answer",
          te.parse_best_path("best path for this CPU: (none)\n") == te.NO_PATH)
    check("parse_best_path: a missing line is None",
          te.parse_best_path("cannot open x.bnpk\n") is None
          and te.parse_best_path("model_id t family 0\nmode 0: ok\n") is None)
    name = te.sized_name("bnv3.bnpk")
    check("sized_name: the bare name as a path of the tested length",
          name.endswith("/bnv3.bnpk") and abs(len(name) - te.PATH_ROOM) <= 1
          and os.path.normpath(name) == "bnv3.bnpk", name)
    check("sized_name: a name longer than the size stays as it is",
          te.sized_name("x" * 120) == "x" * 120)
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, "sub")
        os.makedirs(sub)
        write_text(os.path.join(sub, "a.bnpk"), "x")
        write_text(os.path.join(d, "b.onnx"), "x")
        te.drop(sub, os.path.join(d, "b.onnx"), os.path.join(d, "missing"))
        check("drop: removes files and directories, a missing one is fine",
              os.listdir(d) == [], os.listdir(d))
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.onnx")
        onnx.save(helper.make_model(g), base)
        env = te.Env(d, None, base, None)
        p = te.model_with_tail(env, "logits", "Softmax", "sm", "probs",
                               "t.onnx", only=True, axis=-1)
        tailed = onnx.load(p).graph
        check("model_with_tail: node appended, outputs replaced",
              [n.op_type for n in tailed.node] == ["MatMul", "Add", "Softmax"]
              and [o.name for o in tailed.output] == ["probs"]
              and tailed.node[2].input[0] == "logits")


def fake_info_cli(d, name, info, code=0):
    """A stand-in CLI in d that prints info and exits code; its path."""
    path = os.path.join(d, name)
    write_text(path, f"#!/bin/sh\nprintf '%s' '{info}'\nexit {code}\n")
    os.chmod(path, 0o755)
    return path


def quiet_call(fn, *args):
    """fn(*args)'s return value and what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        value = fn(*args)
    return value, out.getvalue()


def answer_and_failures(te, fn, *args):
    """fn(*args)'s return value, its output, and the failed checks of
    test_errors it recorded."""
    before = len(te.CHECKS.failed)
    value, printed = quiet_call(fn, *args)
    return value, printed, te.CHECKS.failed[before:]


def test_best_path():
    te = load("test_errors", TESTS)
    with tempfile.TemporaryDirectory() as d:
        fp32 = fake_info_cli(d, "fp32", "best path for this CPU: neon-fp32\n")
        none = fake_info_cli(d, "none", "best path for this CPU: (none)\n")
        silent = fake_info_cli(d, "silent", "model_id t family 0\n")
        failing = fake_info_cli(d, "failing",
                                "best path for this CPU: neon-fp32\n", 1)
        value, _, failed = answer_and_failures(te, te.best_path, fp32, "p")
        check("best_path: the named path, no failure",
              value == "neon-fp32" and not failed, (value, failed))
        value, _, failed = answer_and_failures(te, te.best_path, none, "p")
        check("best_path: (none) is an answer, not a failure",
              value == te.NO_PATH and not failed, (value, failed))
        value, _, failed = answer_and_failures(te, te.best_path, silent, "p")
        check("best_path: no best-path line is a recorded failure",
              value is None and len(failed) == 1, (value, failed))
        value, _, failed = answer_and_failures(te, te.best_path, failing, "p")
        check("best_path: a non-zero --info exit is a recorded failure",
              value is None and len(failed) == 1, (value, failed))
        value, printed, failed = answer_and_failures(
            te, te.runs_here, te.Env(d, none, None, None), "p", "neon-fp16")
        check("runs_here: a neon-fp16 package this CPU cannot run is a skip",
              value is False and not failed and printed.startswith("skip C2"),
              (value, printed, failed))
        value, _, failed = answer_and_failures(
            te, te.runs_here, te.Env(d, none, None, None), "p", "neon-fp32")
        check("runs_here: a neon-fp32 package with no path is a failure",
              value is False and len(failed) == 1, (value, failed))
        value, _, failed = answer_and_failures(
            te, te.runs_here, te.Env(d, silent, None, None), "p", "neon-fp16")
        check("runs_here: no answer from --info is a failure, not a skip",
              value is False and len(failed) == 1, (value, failed))
        value, _, failed = answer_and_failures(
            te, te.runs_here, te.Env(d, fp32, None, None), "p", "neon-fp32")
        check("runs_here: the package's own path runs",
              value is True and not failed, (value, failed))


def test_pkg_reader():
    te = load("test_errors", TESTS)
    path = os.path.join(ROOT, "weights", "birdnet_v2.4-neon-fp16.bnpk")
    if not os.path.exists(path):
        print("skip Pkg on the shipped v2.4 package (weights/ not linked)")
        return
    pkg = te.Pkg(path)
    o, _, _ = pkg.meta("n_classes")
    check("Pkg reads the shipped package: n_classes 6522",
          struct.unpack_from("<q", pkg.b, o)[0] == 6522)
    check("Pkg: one path with buffers and ops, tensors named",
          len(pkg.paths) == 1 and pkg.paths[0]["name"] == "neon-fp16"
          and pkg.paths[0]["bufs"] and pkg.paths[0]["ops"]
          and all(t["name"] for t in pkg.tensors))
    check("Pkg.meta of a missing key is None", pkg.meta("no_such") is None)


def main():
    if extern is None:
        print("skip test_scripts_units_s2: bnneon.extern (the module loader "
              "of the gate scripts) is not in this checkout")
        return CHECKS.summary()
    test_export_weights()
    test_export_perch()
    test_dump_graph()
    test_generators()
    test_tools()
    test_replay_args()
    test_plot()
    test_benchmarks()
    test_test_errors_helpers()
    test_best_path()
    test_pkg_reader()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
