"""Unit tests for bnneon.agree (the gates' agreement statistics).

  python -P -s tests/test_agree.py     (bnneon installed, or
                                        PYTHONPATH=<checkout>/python)

Exit 0 when every check passes.
"""
import sys

import numpy as np

from bnneon import agree
from bnneon import testutil

CHECKS = testutil.Checks()
check = CHECKS.check


def test_hits():
    ref = np.array([[0.9, 0.1, 0.6], [0.2, 0.7, 0.5]], np.float32)
    cand = np.array([[0.8, 0.6, 0.4], [0.2, 0.9, 0.5]], np.float32)
    inter, union, fl = agree.hits(ref, cand, 0.5)
    check("hits intersection and union", (inter, union) == (2, 4),
          (inter, union))
    check("hits flip mask", fl.tolist() == [[False, True, True],
                                            [False, False, False]], fl)
    inter, union, fl = agree.hits(ref, cand, 0.95)
    check("hits with no detection", (inter, union) == (0, 0)
          and not fl.any())
    check("a value equal to the threshold is no detection",
          agree.hits(np.array([0.5]), np.array([0.5]), 0.5)[1] == 0)


def test_margins():
    m = np.array([0.001, 0.005, 0.01, 0.03, 0.06])
    check("margin counts", agree.margin_counts(m) == (2, 3, 1, 0.06),
          agree.margin_counts(m))
    empty = agree.margin_counts(np.array([]))
    check("margin counts of no flip", empty == (0, 0, 0, 0.0)
          and isinstance(empty[3], float), empty)


def test_counts():
    pp = np.array([[1, 0, 2], [3, 1, 0]])
    check("counts", agree.counts(pp) == (4, 1, 2))
    check("counts are ints", all(isinstance(v, int) for v in agree.counts(pp)))
    check("pr_percent", agree.pr_percent(4, 1, 2) == (80.0, 400 / 6))
    check("pr_percent of nothing", agree.pr_percent(0, 0, 0) == (0.0, 0.0))
    p, r = agree.prf(pp)
    check("prf", p == 0.8 and r == 4 / 6, (p, r))
    p, r = agree.prf(np.stack([pp, np.zeros_like(pp)]))
    check("prf over a leading axis", p.tolist() == [0.8, 0.0]
          and r.tolist() == [4 / 6, 0.0], (p, r))


def test_bootstrap():
    bidx = agree.bootstrap_index(np.random.default_rng(0), 5)
    again = np.random.default_rng(0).integers(0, 5, size=(agree.B, 5))
    check("bootstrap_index is one draw of (B, n)", bidx.shape == (agree.B, 5)
          and np.array_equal(bidx, again))
    pr = np.array([[1, 1, 0], [0, 0, 1], [2, 0, 0]])
    dp, dr = agree.paired_delta(pr, pr, bidx[:10, :3] % 3)
    check("paired_delta of equal sides is 0", not dp.any() and not dr.any())
    pn = np.array([[1, 0, 0], [0, 0, 1], [2, 0, 0]])
    dp, dr = agree.paired_delta(pr, pn, np.array([[0, 1, 2]]))
    check("paired_delta in percentage points",
          np.allclose(dp, [100 * (3 / 3 - 3 / 4)]) and np.allclose(dr, 0),
          (dp, dr))
    big = np.random.default_rng(1).integers(0, 9, size=(30, 3))
    bidx = agree.bootstrap_index(np.random.default_rng(2), 30)
    p_all, r_all = agree.prf(big[bidx])
    p_chk, r_chk = agree.boot_prf(big, bidx, chunk=7)
    check("boot_prf equals prf of the whole gather",
          np.array_equal(p_all, p_chk) and np.array_equal(r_all, r_chk))
    lo, hi = agree.ci(np.arange(1001.0))
    check("ci is the 2.5 / 97.5 percentiles", (lo, hi) == (25.0, 975.0),
          (lo, hi))


def main():
    test_hits()
    test_margins()
    test_counts()
    test_bootstrap()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
