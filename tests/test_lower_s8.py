"""Tests of the int8 prototype-head packing helpers in bnneon.compiler.lower.

  python -I tests/test_lower_s8.py     (bnneon installed, or
                                        PYTHONPATH=<checkout>/python -P -s)

quant_rows_s8 (per-row symmetric int8) and n4_panel_flags (which 4-row panels
the kernel's n=4 K loop may take without int16 overflow). Exit 0 when every
check passes.
"""
import sys

import numpy as np

from bnneon import testutil
from bnneon.compiler import lower

CHECKS = testutil.Checks()
check = CHECKS.check
ZERO = [0] * 32


def panel(rows):
    """int8 [4][32] weights from 4 lists of 32 values."""
    return np.array(rows, np.int8)


def row_with_lane(values, lane=3):
    """32 weights, zero except the 4 bytes of one int16 lane.

    The lane's bytes are l, l + 8, l + 16 and l + 24.
    """
    row = [0] * 32
    for step, value in enumerate(values):
        row[lane + 8 * step] = value
    return row


def test_quant_rows_s8():
    w = np.array([[0.5, -0.25, 0.1, 0.0]], np.float32)
    q, sw = lower.quant_rows_s8(w)
    check("largest value maps to 127", q.tolist() == [[127, -64, 25, 0]]
          and abs(float(sw[0]) - 0.5 / 127) < 1e-9, (q, sw))
    w = np.array([[127.0, 0.5, -0.5, 1.5, -2.5]], np.float32)
    q, _ = lower.quant_rows_s8(w)
    check("ties round away from zero", q.tolist() == [[127, 1, -1, 2, -3]], q)
    q, sw = lower.quant_rows_s8(np.zeros((2, 8), np.float32))
    check("zero row has zero scale and bytes",
          not q.any() and sw.tolist() == [0.0, 0.0], (q, sw))
    rng = np.random.default_rng(0)
    w = -np.abs(rng.standard_normal((16, 64))).astype(np.float32)
    q, _ = lower.quant_rows_s8(w)
    check("never -128", int(q.min()) >= -127, int(q.min()))


def brute_force_flags(q):
    """n4_panel_flags computed lane by lane, for a [rows][96] q."""
    expected = []
    for first in range(0, q.shape[0], 4):
        worst = 0
        for row in range(first, first + 4):
            for step in range(3):
                for lane in range(8):
                    total = sum(abs(int(q[row, 32 * step + lane + 8 * i]))
                                for i in range(4))
                    worst = max(worst, total)
        expected.append(int(worst <= 258))
    return expected


def test_n4_panel_flags():
    limit = lower.N4_LANE_LIMIT
    check("limit is the int16 bound for 127", limit == 258
          and limit * 127 <= 32767 and (limit + 1) * 127 > 32767, limit)
    q = panel([row_with_lane([127, -127, 4, 0]), ZERO, ZERO, ZERO])
    check("lane sum at the limit is safe",
          lower.n4_panel_flags(q).tolist() == [1])
    q = panel([ZERO, ZERO, row_with_lane([127, -127, 4, -1]), ZERO])
    check("lane sum one above the limit is not safe",
          lower.n4_panel_flags(q).tolist() == [0])
    same_lane = [0] * 64
    other_lanes = [0] * 64
    for position in (3, 11, 19, 27):  # one lane of the first step: sum 400
        same_lane[position] = 100
    # different lanes, and the next step: every lane sum <= 100
    for position in (3, 12, 21, 30, 35):
        other_lanes[position] = 100
    z64 = ZERO * 2
    q = np.array([same_lane, z64, z64, z64, other_lanes, z64, z64, z64],
                 np.int8)
    check("lanes are bytes 8 apart within one 32-byte step",
          lower.n4_panel_flags(q).tolist() == [0, 1])
    rng = np.random.default_rng(1)
    q = rng.integers(-20, 21, (8, 64)).astype(np.int8)
    q[5, 7] = q[5, 15] = q[5, 23] = q[5, 31] = 100
    check("one bad row clears its panel only",
          lower.n4_panel_flags(q).tolist() == [1, 0])
    check("K not a multiple of 32 is never flagged",
          lower.n4_panel_flags(np.zeros((8, 48), np.int8)).tolist() == [0, 0])
    rng = np.random.default_rng(2)
    q = rng.integers(-90, 91, (64, 96)).astype(np.int8)
    expected = brute_force_flags(q)
    halved = lower.n4_panel_flags((q // 2).astype(np.int8)).tolist()
    check("brute force agrees", lower.n4_panel_flags(q).tolist() == expected
          and 0 in expected and 1 in halved)


def main():
    test_quant_rows_s8()
    test_n4_panel_flags()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
