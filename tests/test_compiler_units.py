"""Unit tests for the pure helpers of the bnc command modules.

  python -P -s tests/test_compiler_units.py     (bnneon installed, or
                                                  PYTHONPATH=<checkout>/python)

No model, library or ONNX Runtime run is needed. Exit 0 when every check
passes.
"""
import contextlib
import io
import os
import sys
import tempfile
import wave

import numpy as np

from bnneon.compiler import bnc
from bnneon.compiler import cliargs
from bnneon.compiler import cmd_check
from bnneon.compiler import cmd_compile
from bnneon.compiler import cmd_report
from bnneon.compiler import errs
from bnneon.compiler import lower
from bnneon.compiler import parity
from bnneon.compiler import prep
from bnneon.compiler import runner
from bnneon.compiler import windows
from bnneon import testutil

CHECKS = testutil.Checks()
check = CHECKS.check


raises = testutil.raises


def test_cliargs():
    a = ["check", "m.onnx", "--labels", "l.txt", "--windows", "x", "--json"]
    check("opt value", cliargs.opt(a, "--labels") == "l.txt")
    check("opt absent", cliargs.opt(a, "--emb", "d") == "d")
    check("opt last without value", cliargs.opt(a, "--json", "d") == "d")
    check("opt_int default", cliargs.opt_int(a, "--hop", 7) == 7)
    e = raises(lambda: cliargs.opt_int(a, "--windows", 4), errs.Usage)
    check("opt_int non-integer", str(e) == "--windows x: expected an integer",
          e)
    check("json to stdout", cliargs.json_to_stdout(a))
    check("json to a file", not cliargs.json_to_stdout(["r", "m", "--json",
                                                        "f.json"]))
    check("json absent", not cliargs.json_to_stdout(["r", "m"]))
    check("text printer", cliargs.text_printer(a) is cliargs.print_stderr
          and cliargs.text_printer(["r", "m"]) is print)


def test_prep():
    raw = "inat2024_fsd50k\n\nA a_B\n  \nC c_D\n".encode()
    check("nonempty lines", prep.nonempty_lines(raw) == ["inat2024_fsd50k",
                                                         "A a_B", "C c_D"])
    check("class labels drop the Perch header",
          prep.class_labels(raw) == ["A a_B", "C c_D"])
    lines = ["Turdus merula_Common Blackbird", "Common_Name_Genus_species"]
    check("label form", prep.label_form(lines)
          == (1, "Common_Name_Genus_species"))
    check("labels warning names the file",
          "x/l.txt" not in (prep.labels_warning(lines, "x/l.txt") or "")
          and "l.txt as it is" in prep.labels_warning(lines, "x/l.txt"))
    check("labels warning none", prep.labels_warning(lines[:1], "l") is None)
    before = {"a": np.array([1.0, -2.0]), "b": np.array([4.0])}
    after = {"a": np.array([1.0, -2.5]), "b": np.array([4.0])}
    worst = prep.max_relative_change(before, after, ["a", "b"])
    check("max relative change", abs(worst - 0.25) < 1e-9, worst)
    fe = [{"op": "FE_STFT2_POW"}, {"op": "CONV"}]
    check("v2.4 backbone: front end and window",
          prep.is_v24_backbone(fe, 144000)
          and not prep.is_v24_backbone(fe, 160000)
          and not prep.is_v24_backbone(fe[1:], 144000))
    check("family at a rate", (prep.family_at_rate(48000),
                               prep.family_at_rate(256000),
                               prep.family_at_rate(32000))
          == ("24", "224", None))
    check("BirdNET-Go label form: family 24 on the v2.4 backbone only",
          prep.birdnetgo_labels(fe, 144000, "24")
          and not prep.birdnetgo_labels(fe, 144000, "224")
          and not prep.birdnetgo_labels(fe, 160000, "24"))


def test_report():
    o = {"op": "DENSE", "ins": [1], "out": 2, "w": np.zeros((3, 4)),
         "b": np.zeros(3), "nodes": [0, 1], "tab": np.zeros(5),
         "parts": [np.zeros(2), (np.zeros(1), 3), 4]}
    f = cmd_report.op_fields(o)
    check("op fields", f == {"op": "DENSE", "ins": [1], "out": 2,
                             "tab": "array[5]",
                             "parts": ["array[2]", ("array[1]", 3), 4],
                             "w": [3, 4]}, f)
    check("op fields keep order", list(f) == ["op", "ins", "out", "tab",
                                              "parts", "w"])
    line = cmd_report.summary([o])[0]
    check("summary line", line.startswith("  DENSE        {")
          and line.endswith("  (2 nodes)"), line)
    cov = {"ir_ops": 5, "unsupported": [1, 2], "primary": [1],
           "uncovered": 0}
    totals = cmd_report._totals_line(cov)  # pylint: disable=protected-access
    check("totals line", totals
          == "5 IR ops; 2 unsupported nodes (1 primary, the rest "
             "downstream); 0 uncovered nodes")


def test_parity():
    z = np.array([[0.0, 1.0, 2.0]])
    check("softmax sums to 1", abs(parity.post_fn("softmax")(z).sum() - 1)
          < 1e-12)
    check("sigmoid", parity.post_fn("sigmoid")(np.zeros(1))[0] == 0.5)
    check("identity", parity.post_fn("none")(z) is z)
    a = np.array([1.0, 2.0], np.float32)
    check("sqnr exact", parity.sqnr_db(a, a) == parity.SQNR_EXACT_DB)
    check("sqnr zero ref", parity.sqnr_db(np.zeros(2), a)
          == parity.SQNR_ZERO_REF_DB)
    s = parity.sqnr_db(np.array([10.0]), np.array([9.0]))
    check("sqnr 20 dB", abs(s - 20.0) < 1e-9, s)


def test_windows():
    x1 = windows.synthetic(1000, 3, 16000)
    x2 = windows.synthetic(1000, 3, 16000)
    check("synthetic is deterministic", np.array_equal(x1, x2))
    check("synthetic shape and range", x1.shape == (3, 1000)
          and x1.dtype == np.float32 and np.abs(x1).max() <= 0.99)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.f32")
        np.arange(10, dtype=np.float32).tofile(p)
        e = raises(lambda: windows.from_file(p, 4, 0), errs.Usage)
        check("windows file not a multiple", e is not None
              and "not a multiple of 4" in str(e), e)
        check("windows file keeps n", windows.from_file(p, 5, 1).shape
              == (1, 5))
        q = os.path.join(d, "w.npy")
        np.save(q, np.zeros((2, 3), np.int16))
        e = raises(lambda: windows.from_file(q, 3, 0), errs.Usage)
        check("windows file integer dtype", e is not None
              and "dtype int16" in str(e), e)
        check("windows file n defaults to all", windows.from_file(p, 5).shape
              == (2, 5))


def _write_wav(path, pcm, rate):
    """A mono 16-bit wav file of the int16 samples pcm."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype(np.int16).tobytes())


def test_audio_windows():
    loud = np.full(4, 1000)
    pcm = np.concatenate([loud, np.zeros(4), loud, loud[:2]])
    warned = []
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.wav")
        _write_wav(p, pcm, 8000)
        x = windows.from_audio([p], 4, 5, 8000, warned.append)
        check("audio: silent windows skipped", x.shape == (2, 4), x.shape)
        x = windows.from_audio([p], 4, 0, 16000, warned.append,
                               skip_silence=False)
        check("audio: every full window with skip_silence=False, n=0",
              x.shape == (3, 4) and not x[1].any(), x.shape)
        check("audio: the rate warning", len(warned) == 1
              and "16000 Hz" in warned[0], warned)
        x = windows.from_audio([p, p], 4, 3, None, warned.append)
        check("audio: n windows across files", x.shape == (3, 4), x.shape)
        q = os.path.join(d, "z.wav")
        _write_wav(q, np.zeros(8), 8000)
        e = raises(lambda: windows.from_audio([q], 4, 2, None, print),
                   errs.Usage)
        check("audio: no window with signal", e is not None
              and "with signal" in str(e), e)
        e = raises(lambda: windows.from_audio([q], 8 * 2, 0, None, print,
                                              skip_silence=False), errs.Usage)
        check("audio: no full window", e is not None
              and "with signal" not in str(e), e)


def test_runner_output():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "out.f32")
        np.arange(6, dtype=np.float32).tofile(p)
        y, msg = runner.read_output(p, 2, 3)
        check("read_output shape", y is not None and y.shape == (2, 3)
              and msg == "", msg)
        y, msg = runner.read_output(p, 2, 4)
        check("read_output size mismatch", y is None
              and msg == "the CLI wrote 6 floats, expected 2 x 4", msg)


def test_check_verdicts():
    check("need: int8 floor", cmd_check.need_db("neon-fp16", True) == 40.0)
    check("need: fp16", cmd_check.need_db("neon-fp16", False) == 50.0)
    check("need: fp32", cmd_check.need_db("neon-fp32", False) == 60.0)
    js = cmd_check.judged_sqnr
    check("judged: tflite decides", js("neon-fp16", 50.0, 30.0, 55.0, True,
                                       70.0, 50.0) == (55.0, 50.0, ""))
    judged, need, note = js("neon-fp32", 60.0, 45.0, None, True, None, 45.0)
    check("judged: fp32 matrix allowance", (judged, need) == (45.0, 40.0)
          and note.startswith("reference-limited"), note)
    judged, need, note = js("neon-fp16", 50.0, 45.0, None, True, 70.0, 45.0)
    check("judged: fp16 by its own loss", (judged, need) == (70.0, 50.0)
          and "70.0 dB" in note and "45.0 dB" in note, note)
    check("judged: plain", js("neon-int8", 40.0, 45.0, None, False, None, None)
          == (45.0, 40.0, ""))
    pv = cmd_check.path_verdict
    check("verdict pass", pv(False, False, 80, 60, 0.001, 4, 4)
          == ("PASS", []))
    check("verdict low sqnr warns", pv(False, False, 50, 60, 0, 4, 4)[0]
          == "WARN")
    check("verdict very low sqnr fails",
          pv(False, False, 30, 60, 0, 4, 4)[0] == "FAIL")
    check("verdict dconf fails", pv(False, False, 80, 50, 0.06, 4, 4)
          == ("FAIL", ["max |dconf| 0.060 > 0.05"]))
    check("verdict int8 dconf warns", pv(True, False, 80, 40, 0.06, 4, 4)[0]
          == "WARN")
    check("verdict top-1", pv(False, False, 80, 60, 0, 3, 4)
          == ("WARN", ["top-1 agrees on 3/4 windows"]))
    check("verdict nonfinite", pv(False, True, 80, 60, 0, 4, 4)
          == ("FAIL", ["non-finite logits"]))
    paths = {"neon-fp32": {"verdict": "PASS"}, "neon-fp16": {"verdict": "WARN"}}
    st, code, msg = cmd_check.overall(paths, "WARN", ["neon-fp16: x"],
                                      ["neon-fp16", "neon-fp32"])
    check("overall: ship fp32", (st, code) == ("WARN", "warn")
          and "ship the neon-fp32 package" in msg, msg)
    paths["neon-fp16"]["need_db"] = cmd_check.NEED_INT8_DB
    st, code, msg = cmd_check.overall(paths, "WARN", ["neon-fp16: x"],
                                      ["neon-fp16", "neon-fp32"])
    check("overall: an int8-GEMM fp16 path's WARN goes to the gate",
          (st, code) == ("WARN", "warn") and "gate it on real audio" in msg
          and "ship the" not in msg, msg)
    paths["neon-fp16"]["verdict"] = "FAIL"
    st, code, msg = cmd_check.overall(paths, "FAIL", ["neon-fp16: x"],
                                      ["neon-fp16", "neon-fp32"])
    check("overall: an int8-GEMM fp16 FAIL still says ship fp32",
          (st, code) == ("WARN", "warn")
          and "ship the neon-fp32 package" in msg, msg)
    st, code, msg = cmd_check.overall({"neon-fp16": {"verdict": "FAIL"}},
                                      "FAIL", ["neon-fp16: y"], ["neon-fp16"])
    check("overall: no fp32 to compare", (st, code) == ("FAIL", "parity")
          and msg.endswith("(--paths neon-fp32,neon-fp16)"), msg)
    check("overall: pass", cmd_check.overall({"neon-fp32": {"verdict": "PASS"}},
                                             "PASS", [], ["neon-fp32"])[:2]
          == ("PASS", "ok"))
    paths = {"neon-fp32": {"verdict": "PASS", "ran": True},
             "neon-fp16": {"ran": False}}
    st, code, msg = cmd_check.overall(paths, "PASS", [],
                                      ["neon-fp16", "neon-fp32"])
    check("overall: a path this CPU cannot run is named as skipped, not as "
          "agreeing", (st, code) == ("PASS", "ok") and msg.startswith(
              "every path this CPU can run agrees with ONNX Runtime on these "
              "windows; skipped: neon-fp16 (cannot run here); next:"), msg)
    check("skipped note: none when every path ran",
          cmd_check.skipped_note({"neon-fp32": {"ran": True}},
                                 ["neon-fp32"]) == "")
    check("scratch model id: check_ prefix, at most 60 characters",
          cmd_check.scratch_model_id("m") == "check_m"
          and len(cmd_check.scratch_model_id("x" * 80)) == 60)
    cargs = cmd_check.compile_args(
        ["check", "m.onnx", "--emb", "e", "--no-canon", "--logits", "l"],
        "lab.txt", "m", "sigmoid", None, "neon-fp32", "o.bnpk")
    check("compile args", cargs == [
        "compile", "m.onnx", "--labels", "lab.txt", "--model-id", "check_m",
        "--family", "0", "--post", "sigmoid", "--sample-rate", "48000",
        "--paths", "neon-fp32", "-o", "o.bnpk", "--logits", "l", "--emb", "e",
        "--no-canon", "--force"], cargs)


def test_check_named_paths():
    paths = {"neon-fp32": {"verdict": "PASS", "ran": True},
             "neon-fp16": {"ran": False}}
    st, code, msg = cmd_check.overall(paths, "PASS", [],
                                      ["neon-fp16", "neon-fp32"], named=True)
    check("overall: a path named with --paths that cannot run here warns",
          (st, code) == ("WARN", "warn") and msg.startswith(
              "--paths names neon-fp16, which this CPU cannot run (not "
              "checked); every other path agrees with ONNX Runtime") and
          msg.endswith("or leave it out of --paths"), msg)
    st, code, msg = cmd_check.overall({"neon-fp32": {"verdict": "PASS",
                                                     "ran": True}},
                                      "PASS", [], ["neon-fp32"], named=True)
    check("overall: every path named with --paths ran: pass",
          (st, code) == ("PASS", "ok") and "skipped" not in msg, msg)
    st, code, msg = cmd_check.overall(paths, "WARN", ["neon-fp32: x"],
                                      ["neon-fp16", "neon-fp32"], named=True)
    check("overall: a WARN already names the path it skipped",
          (st, code) == ("WARN", "warn")
          and msg.count("neon-fp16") == 1
          and "skipped: neon-fp16 (cannot run here)" in msg, msg)
    paths8 = {"neon-fp32": {"verdict": "PASS", "ran": True},
              "neon-int8": {"verdict": "FAIL", "ran": True},
              "neon-fp16": {"ran": False}}
    st, code, msg = cmd_check.overall(paths8, "FAIL", ["neon-int8: x"],
                                      list(paths8), named=True)
    check("overall: another WARN gains the named path it could not run",
          (st, code) == ("WARN", "warn") and msg.endswith(
              "; --paths names neon-fp16, which this CPU cannot run"), msg)
    paths["neon-fp32"]["verdict"] = "FAIL"
    st, code, _ = cmd_check.overall(paths, "FAIL", ["neon-fp32: x"],
                                    ["neon-fp16", "neon-fp32"], named=True)
    check("overall: a FAIL stays a FAIL with a named path skipped",
          (st, code) == ("FAIL", "parity"))
    check("skipped paths: the compiled paths that did not run",
          cmd_check.skipped_paths(paths8, list(paths8)) == ["neon-fp16"])


def test_compile_options():
    # pylint: disable=protected-access
    check("family given", cmd_compile._family(["c", "m", "--family", "120"],
                                              False, False) == "120")
    check("family forced", cmd_compile._family(["c", "m"], False, True) == "0")
    check("family required", raises(lambda: cmd_compile._family(
        ["c", "m"], False, False), errs.Usage) is not None)
    check("rate default", cmd_compile._sample_rate(["c", "m"], False, "0",
                                                   False) == 48000)
    e = raises(lambda: cmd_compile._sample_rate(
        ["c", "m", "--sample-rate", "32000"], True, "24", False), errs.Usage)
    check("rate refused for family 24", e is not None
          and "takes 48000 Hz" in str(e), e)
    e = raises(lambda: cmd_compile._sample_rate(
        ["c", "m", "--sample-rate", "x"], False, "0", False), errs.Usage)
    check("rate not an integer", str(e) == "--sample-rate x: expected an "
          "integer rate in Hz", e)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "idx.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("3\n\n1\n")
        idx = cliargs.read_indices(p)
        check("class indices", idx.tolist() == [3, 1]
              and idx.dtype == np.int64, idx)


def test_wino_note():
    a = ["compile", "m.onnx", "--wino", "24"]
    convw = (lower.OP["CONVW"], [], [], [])
    pw = (lower.OP["PW"] | 1 << 8, [], [], [])
    note = cmd_compile.wino_note(a, "neon-fp16", [pw])
    check("wino note: no Winograd conv on the path",
          note is not None and "--wino 24" in note and "neon-fp16" in note,
          note)
    check("wino note: none when a conv took it",
          cmd_compile.wino_note(a, "neon-fp32", [pw, convw]) is None)
    check("wino note: none without --wino",
          cmd_compile.wino_note(a[:2], "neon-fp16", [pw]) is None)


def test_bnc_entry():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc_help = bnc.main(["--help"])
        rc_h = bnc.main(["check", "-h"])
        rc_none = bnc.main([])
        rc_bad = bnc.main(["frobnicate", "m.onnx"])
    check("--help and -h print the usage and exit 0; a missing or unknown "
          "command exits 3", (rc_help, rc_h, rc_none, rc_bad) == (0, 0, 3, 3)
          and out.getvalue().startswith("bnc: the bnneon model compiler"),
          (rc_help, rc_h, rc_none, rc_bad))
    hint = errs.missing_extra(ModuleNotFoundError("x", name="onnx"))
    check("missing_extra: onnx names the compiler extra",
          hint == "bnc: no module named 'onnx'; hint: install the compiler "
          "extra: pip install -e '.[compiler]'", hint)
    check("missing_extra: an onnxruntime submodule is the extra too",
          errs.missing_extra(ModuleNotFoundError(
              "x", name="onnxruntime.capi")) is not None)
    check("missing_extra: any other module is not",
          errs.missing_extra(ModuleNotFoundError("x", name="numpy")) is None
          and errs.missing_extra(ModuleNotFoundError("x")) is None)


def main():
    test_cliargs()
    test_prep()
    test_report()
    test_parity()
    test_windows()
    test_audio_windows()
    test_runner_output()
    test_check_verdicts()
    test_check_named_paths()
    test_compile_options()
    test_wino_note()
    test_bnc_entry()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
