"""Unit tests for the gate helpers the scripts share.

bnneon.gate (the generic gate), bnneon.extern (loading a module by path),
bnneon.replay_v3 (the v3 detection replays) and bnneon.gate_v3 (the v3 gate
scorers).

  python -P -s tests/test_scripts_units_s1.py     (bnneon installed, or
                                                   PYTHONPATH=<checkout>/python)

Exit 0 when every check passes.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types

import numpy as np

from bnneon import extern
from bnneon import gate
from bnneon import gate_v3
from bnneon import replay_v3
from bnneon import testutil

CHECKS = testutil.Checks()
check = CHECKS.check


def _exits(fn, *args):
    """(exit code, stderr) of a call that should exit; (None, '') if not."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            fn(*args)
    except SystemExit as e:
        return e.code, err.getvalue()
    return None, err.getvalue()


def test_gate_options():
    args = ["score", "c.json", "--cand", "a", "--x", "--cand", "b", "--k"]
    check("opt value", gate.opt(args, "--x") == "--cand")
    check("opt absent", gate.opt(args, "--y", "d") == "d")
    check("opt without a value", gate.opt(args, "--k", "d") == "d")
    check("opt_all", gate.opt_all(args, "--cand") == ["a", "b"])
    check("opt_all ignores args[0]", gate.opt_all(["--cand", "x"], "x") == [])
    # the private helpers are the units under test here
    # pylint: disable-next=protected-access
    crit = gate._criteria(["declare", "c.json", "--thresholds", "0.2,0.6",
                           "--top1", "0.9"])
    check("criteria key order", list(crit) == [
        "declared_utc", "thresholds", "jaccard", "flip_margin", "mean_pp",
        "max_dconf", "sqnr_db", "top1", "note"], list(crit))
    check("criteria values", crit["thresholds"] == [0.2, 0.6]
          and crit["top1"] == 0.9 and crit["max_dconf"] is None
          and crit["jaccard"] == 0.97 and crit["note"] == "", crit)
    check("sha", gate.sha(b"") == "e3b0c44298fc1c149afbf4c8996fb924"
          "27ae41e4649b934ca495991b7852b855")


def test_gate_stats():
    z = np.array([[0.0, 1.0, -1.0], [2.0, 2.0, 2.0]], np.float32)
    sm = gate.post_fn("softmax")(z)
    check("softmax rows sum to 1", np.allclose(sm.sum(1), 1.0), sm)
    check("sigmoid", np.allclose(gate.post_fn("sigmoid")(z),
                                 1 / (1 + np.exp(-z))))
    check("post none", gate.post_fn("none")(z) is z)
    a = np.array([1.0, 2.0, 3.0])
    check("sqnr equal", gate.sqnr_db(a, a) == 200.0)
    check("sqnr all-zero reference (the parity constant)",
          gate.sqnr_db(np.zeros(3), a) == -99.0)
    check("sqnr", abs(gate.sqnr_db(a, a + 0.1) - 10 * np.log10(14 / 0.03))
          < 1e-9)
    hits = np.array([[True, False, True], [False, False, False]])
    rng = np.random.default_rng(4)
    many = rng.random((20, 6)) > 0.6
    truth = {w: set(rng.choice(6, size=w % 3, replace=False).tolist())
             for w in range(0, 20, 2)}
    want = np.zeros((20, 3), np.int64)
    for w in range(20):  # the per-window form prf_window had
        ts = truth.get(w, set())
        ps = set(np.nonzero(many[w])[0].tolist())
        want[w] = (len(ts & ps), len(ps - ts), len(ts - ps))
    check("prf_window equals the per-window set form",
          np.array_equal(gate.prf_window(many, truth, 20), want))
    pp = gate.prf_window(hits, {0: {0, 1}, 1: {2}}, 2)
    check("prf_window", pp.tolist() == [[1, 1, 1], [0, 0, 1]], pp)
    p, r = gate.prf(pp)
    check("prf", p == 0.5 and r == 1 / 3, (p, r))
    p, r = gate.prf(np.zeros((4, 3), np.int64))
    check("prf of nothing", p == 0 and r == 0)
    pr = np.array([[0.9, 0.1], [0.52, 0.2]])
    pc = np.array([[0.9, 0.1], [0.49, 0.2]])
    jac, t = gate._agreement(pr, pc, 0.5)  # pylint: disable=protected-access
    check("agreement", jac == 0.5 and t["union"] == 2 and t["flips"] == 1
          and t["flip_le_02"] == 0 and abs(t["flip_max"] - 0.02) < 1e-12, t)
    jac, t = gate._agreement(pr, pc, 0.95)  # pylint: disable=protected-access
    check("agreement without detections", jac == 1.0 and t["union"] == 0
          and t["flip_max"] == 0.0, t)


def test_gate_truth():
    labels = ["Turdus merula_Common Blackbird", "Erithacus rubecula_Robin"]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("window_index,label\n0,Robin\n# c\n\n1,Turdus merula\n"
                    "1,0\n2,Erithacus rubecula_Robin\n")
        t = gate.read_truth(path, labels, 2, 3)
        check("read_truth by name, part and index", t == {0: {1}, 1: {0},
                                                          2: {1}}, t)
        with open(path, "w", encoding="utf-8") as f:
            f.write("0,Nope\n")
        code, err = _exits(gate.read_truth, path, labels, 2, 3)
        check("read_truth unknown label", code == 3
              and "not in the label file" in err, err)
        with open(path, "w", encoding="utf-8") as f:
            f.write("5,1\n")
        code, err = _exits(gate.read_truth, path, None, 2, 3)
        check("read_truth window range", code == 3 and "outside 0..2" in err,
              err)
        with open(path, "w", encoding="utf-8") as f:
            f.write("0,7\n")
        code, err = _exits(gate.read_truth, path, None, 2, 3)
        check("read_truth class range", code == 3 and "out of range" in err,
              err)


def test_gate_declare():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = gate.cmd_declare(["declare", path, "--note", "n"])
        with open(path, encoding="utf-8") as f:
            crit = json.load(f)
        check("declare writes", rc == 0 and crit["note"] == "n"
              and out.getvalue().startswith(f"declared {path} (read-only"))
        check("declare read-only", not os.access(path, os.W_OK)
              or os.geteuid() == 0)
        code, err = _exits(gate.cmd_declare, ["declare", path])
        check("declare twice", code == 3 and "declared once" in err, err)
    with contextlib.redirect_stdout(io.StringIO()):
        check("main usage", gate.main([]) == 3
              and gate.main(["frob", "x"]) == 3)


def test_extern():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "s1_ext_mod.py"), "w",
                  encoding="utf-8") as f:
            f.write("import sys\nARG = sys.argv[1]\n")
        saved = sys.argv
        sys.argv = ["prog", "tuning"]
        try:
            m = extern.load_module(d, "s1_ext_mod")
        finally:
            sys.argv = saved
        check("load_module runs the module", m.ARG == "tuning")
        check("load_module registers it", sys.modules["s1_ext_mod"] is m
              and m.__file__ == os.path.join(d, "s1_ext_mod.py"))
        check("load_module returns the loaded one",
              extern.load_module("/nonexistent", "s1_ext_mod") is m)
        del sys.modules["s1_ext_mod"]
        try:
            extern.load_module(d, "s1_missing_mod")
            missing = None
        except ModuleNotFoundError as e:
            missing = e
        check("load_module missing", missing is not None
              and str(missing) == "No module named 's1_missing_mod'"
              and "s1_missing_mod" not in sys.modules)
        with open(os.path.join(d, "s1_bad_mod.py"), "w",
                  encoding="utf-8") as f:
            f.write("raise ValueError('boom')\n")
        try:
            extern.load_module(d, "s1_bad_mod")
        except ValueError:
            pass
        check("load_module failure unregisters", "s1_bad_mod"
              not in sys.modules)
        with open(os.path.join(d, "s1_argv_mod.py"), "w",
                  encoding="utf-8") as f:
            f.write("import sys\nARGS = sys.argv[1:]\n")
        before = list(sys.argv)
        m = extern.load_with_argv(d, "s1_argv_mod", "t", "e")
        check("load_with_argv sets sys.argv for the import",
              m.ARGS == ["t", "e"], m.ARGS)
        check("load_with_argv restores sys.argv", sys.argv == before)
        del sys.modules["s1_argv_mod"]
        try:
            extern.load_with_argv(d, "s1_bad_mod", "x")
        except ValueError:
            pass
        check("load_with_argv restores sys.argv after a failed import",
              sys.argv == before)


def test_testutil():
    e = testutil.raises(lambda: int("x"), ValueError)
    check("testutil.raises returns the exception", isinstance(e, ValueError))
    check("testutil.raises: none raised", testutil.raises(int, ValueError)
          is None)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "weights"))
        for m in ("b.bnpk", "a.bnpk"):
            with open(os.path.join(d, "weights", m), "wb"):
                pass
        got = testutil.default_models(d, ["a.bnpk", "c.bnpk", "b.bnpk"])
        check("testutil.default_models: present files, in order",
              got == [os.path.join(d, "weights", m)
                      for m in ("a.bnpk", "b.bnpk")], got)


def _fake_sim():
    """A stand-in for the v3 harness's simulate5s module."""
    sim = types.SimpleNamespace()
    sim.SPEC = {"v3": {"human": np.array([False, True, False, False]),
                       "geo": np.array([0, -1, 1, 2]),
                       "sid": np.array([10, 11, 12, 100001])}}
    sim.RF_THR = 0.05
    return sim


def test_replay_v3():
    sim = _fake_sim()
    v = replay_v3.variant(sim, np.array([[0, 1], [2, 1]], np.int16),
                          np.array([[0.9, 0.3], [0.2, 0.6]], np.float16),
                          np.array([0, 8], np.int32))
    check("variant form", list(v) == ["fam", "grid", "unit", "win", "lab",
                                      "conf", "hum"] and v["lab"].dtype
          == np.int32 and v["conf"].dtype == np.float32)
    check("variant human max", np.allclose(v["hum"], [0.3, 0.6], atol=1e-3),
          v["hum"])
    geo = {"s": np.array([0.5, 0.01, 0.2])}
    m = replay_v3.site_mask(sim, geo, "s")
    check("site mask", m.tolist() == [True, False, False, True], m)
    sim.include_mask = lambda fam, n, kind: ("own", fam, n, kind)
    check("mask own", replay_v3.mask(sim, geo, "s", "own")
          == ("own", "v3", "s", "geo"))
    sim.names = ["a", "b"]
    sim.events = lambda v, ov, th, m: (v, ov, th, m)
    sim.simulate = lambda evs, lv: (evs[0], lv)
    sim.score = lambda sims: sims
    got = replay_v3.replay(sim, {"a": 1, "b": 2}, 2.2, 0.84, 3,
                           lambda n: n + "m")
    check("replay wiring", got == {"a": ((1, 2.2, 0.84, "am"), 3),
                                   "b": ((2, 2.2, 0.84, "bm"), 3)}, got)


def test_replay_site():
    check("parse_site", replay_v3.parse_site("12.5,-3") == (12.5, -3.0))
    for bad in ("12.5", "1,2,3", "x,1", "91,0", "0,181"):
        try:
            replay_v3.parse_site(bad)
        except ValueError:
            continue
        check(f"parse_site refuses {bad!r}", False)
    args = ["t", "--site", "1.5,2", "e"]
    check("take_site", replay_v3.take_site(args) == (1.5, 2.0)
          and args == ["t", "e"], args)
    for bad in (["t", "e"], ["t", "--site"]):
        try:
            replay_v3.take_site(list(bad))
        except ValueError:
            continue
        check(f"take_site refuses {bad!r}", False)
    check("site_text", replay_v3.site_text((1.0, -2.5)) == "(1.00, -2.50)")


def test_session_pool():
    made = []

    def make():
        made.append(len(made))
        return f"s{len(made) - 1}"

    pool = gate_v3.SessionPool(make, 2)
    with pool.borrow() as a:
        with pool.borrow() as b:
            check("pool: one session per concurrent borrower",
                  {a, b} == {"s0", "s1"})
    with pool.borrow() as c:
        check("pool: sessions are reused, none made beyond n",
              c in ("s0", "s1") and len(made) == 2, (c, made))
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("no model")
        return "ok"

    pool = gate_v3.SessionPool(flaky, 1)
    try:
        with pool.borrow():
            pass
    except OSError:
        pass
    with pool.borrow() as s:
        check("pool: a failed make frees its slot", s == "ok", s)
    acc_a, acc_b = gate_v3.Acc(), gate_v3.Acc()
    r = np.array([[0.9, 0.1], [0.3, 0.85]], np.float32)
    c = np.array([[0.8, 0.2], [0.3, 0.9]], np.float32)
    cols = np.arange(2)
    acc_a.add(r, c, "x", 0, cols)
    acc_b.add_logits(r, c, gate_v3.logit(r), gate_v3.logit(c), "x", 0, cols)
    check("Acc.add_logits equals add", acc_a.report() == acc_b.report())


def test_gate_v3():
    x = np.zeros(gate_v3.WIN + 3 * gate_v3.SR, np.float32)
    starts, offs = gate_v3.grid(x)
    check("grid", starts.tolist() == [0, 8, 16] and offs.tolist()
          == [0, 42667, 85333] and offs.dtype == np.int32, (starts, offs))
    c = np.tile(np.linspace(0, 1, 12, dtype=np.float32), (2, 1))
    i, conf = gate_v3.topk(c)
    check("topk", i.dtype == np.int16 and conf.dtype == np.float16
          and i[0].tolist() == list(range(11, 1, -1)), i)
    i, _ = gate_v3.topk(c, np.arange(12) + 100)
    check("topk mapped", i[0, 0] == 111, i)
    lg = gate_v3.logit(np.array([0.0, 0.5, 1.0]))
    check("logit clips", np.allclose(lg, [np.log(1e-4 / (1 - 1e-4)), 0,
                                          np.log((1 - 1e-4) / 1e-4)]), lg)
    r = np.array([[0.9, 0.1], [0.5, 0.86]], np.float32)
    cand = np.array([[0.9, 0.1], [0.5, 0.80]], np.float32)
    a = gate_v3.Acc()
    a.add(r, cand, "s", 4, np.array([7, 9]))
    rep = a.report()
    check("Acc", rep["windows"] == 2 and rep["flips"] == 1
          and rep["hits_union"] == 2 and rep["jaccard"] == 0.5
          and rep["worst"][:3] == ("s", 5, 9) and rep["top1"] == 1.0, rep)
    b = gate_v3.Acc()
    b.merge(a)
    check("Acc merge", b.report() == rep)
    with np.errstate(divide="ignore"):  # SQNR of nothing: log10(0)
        empty = gate_v3.Acc().report()
    check("Acc empty report", empty["max_margin"] == 0.0
          and empty["windows"] == 0)
    line = gate_v3.stats_line("all   ", rep)
    check("stats_line", line.startswith("all   : 2 windows, SQNR ")
          and "Jaccard@0.84 0.5000, flips 1 (<=0.005 0" in line, line)
    out = list(gate_v3.pooled({"a": {"files": [1]}, "b": {"files": [2]}}, 1,
                              lambda f: f[0] * 10, lambda n, v: (n, v)))
    check("pooled order", out == [("a", 10), ("b", 20)], out)


def main():
    test_gate_options()
    test_gate_stats()
    test_gate_truth()
    test_gate_declare()
    test_extern()
    test_testutil()
    test_replay_v3()
    test_replay_site()
    test_gate_v3()
    test_session_pool()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
