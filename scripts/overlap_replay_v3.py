"""H004 (report-only, no verdict): what a LOWER birdnet.overlap costs v3.0.

The cost in detections for BirdNET v3.0, from the fp32 ORT reference scores
of gate G003 (BirdNET v3.0 global, all 32 streams of the 34.4 h set on the
overlap-2.2 grid, 8/6 s stride, top-10 per window). Coarser cadences are
exact sub-grids of that grid: overlap 1.4 = every 2nd window (16/6 s
stride), overlap 0.6 = every 3rd (4 s stride); 1.5 and 0 are not
representable on it (their strides 2.5 s and 5 s are not multiples of 8/6
s), so 1.4 and 0.6 stand in for them. Replayed through the v3 harness
(simulate5s.py, imported read-only: BirdNET-Go 20261014 post-processing,
v3_global form, geomodel range filter 0.05 at the evaluation site given
with --site, each stream's own week), threshold 0.84 (the gated value)
and 0.80, FP filter levels 0 / 2 / 3; the FP filter's minimum detection
count follows the
model's step as in BirdNET-Go. Paired bootstrap (2000 reps over the 37
periods) of dP / dR against overlap 2.2 at the same threshold and level.

usage: python -I overlap_replay_v3.py <tuning_dir> <eval_dir>
           <G003 scores_dir> --site LAT,LON
  --site: the evaluation site of the range filter (required; the published
          results used one site in the eastern US, see docs/GATES.md)
"""
import os
import sys
import types

import numpy as np

from bnneon import agree
from bnneon import replay_v3

BASE_OV = 2.2
OVERLAPS = (2.2, 1.4, 0.6)


def load_ref(sim: types.ModuleType, sc: str) -> replay_v3.Scores:
    """The G003 reference top-10 per stream (all on the 8/6 s grid)."""
    ref = {}
    for n in sim.names:
        d = np.load(os.path.join(sc, n + ".npz"))
        ref[n] = replay_v3.variant(sim, d["ref_idx"], d["ref_conf"],
                                   d["starts"])
        assert (np.diff(d["starts"]) == 8).all(), n
    return ref


def overlap_line(sim: types.ModuleType, pp: np.ndarray, th: float, lv: int,
                 ov: float) -> str:
    """The counts part of one report line."""
    tp, fp, fn = agree.counts(pp)
    p, r = agree.pr_percent(tp, fp, fn)
    step = 5.0 * (3.0 - ov) / 3.0
    return (f"thr {th:.2f} FP L{lv} overlap {ov:.1f} (step {step:.2f} s, "
            f"min_det {sim.min_det(step, lv)}): TP/FP/FN {tp}/{fp}/{fn}  "
            f"P {p:.1f}%  R {r:.1f}%")


def delta_part(base: np.ndarray, pp: np.ndarray, bidx: np.ndarray) -> str:
    """The `vs 2.2` bootstrap part of a report line."""
    dp, dr = agree.paired_delta(base, pp, bidx)
    lo_p, hi_p = agree.ci(dp)
    lo_r, hi_r = agree.ci(dr)
    return (f"  vs 2.2: dP {dp.mean():+.2f} [{lo_p:+.2f}, {hi_p:+.2f}]  dR "
            f"{dr.mean():+.2f} [{lo_r:+.2f}, {hi_r:+.2f}] pp")


def main(argv: list[str]) -> int:
    """Prints the overlap report; returns 0."""
    args = list(argv[1:])
    site = replay_v3.take_site(args)
    tuning, ev, sc = args[0], args[1], args[2]
    sim = replay_v3.load_harness(tuning, ev)
    geo_site = replay_v3.site_geo(sim, site)
    ref = load_ref(sim, sc)

    def mask_of(n):
        return replay_v3.site_mask(sim, geo_site, n)

    rng = np.random.default_rng(0)
    bidx = None
    print("fp32 ORT BirdNET v3.0 global reference (G003 scores), range "
          f"filter at the evaluation site {replay_v3.site_text(site)}")
    for th in (0.84, 0.80):
        for lv in (0, 2, 3):
            base = replay_v3.replay(sim, ref, BASE_OV, th, lv, mask_of)
            if bidx is None:
                bidx = agree.bootstrap_index(rng, len(base))
            for ov in OVERLAPS:
                if ov == BASE_OV:
                    pp = base
                else:
                    pp = replay_v3.replay(sim, ref, ov, th, lv, mask_of)
                line = overlap_line(sim, pp, th, lv, ov)
                if ov != BASE_OV:
                    line += delta_part(base, pp, bidx)
                print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
