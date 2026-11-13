"""Shared parts of the BirdNET v3.0 detection replays.

scripts/gate_sim_v3_pkg.py and overlap_replay_v3.py replay top-10 score
caches through the v3 harness (simulate5s.py: BirdNET-Go 20261014
post-processing, v3_global form = top-10 over all labels) with the v3
geomodel range filter, at a fixed evaluation site (latitude, longitude; each
stream's own week) or at each stream's own location. The site is a required
argument of both scripts (--site LAT,LON, parsed by take_site()); the
published results used one site in the eastern US, whose coordinates are not
recorded in this repository (docs/GATES.md). load_harness() loads the harness
module (it reads its directories from sys.argv when imported); it is passed
in here as `sim`.
"""
from collections.abc import Callable
import os
import types
from typing import Any

import numpy as np

from . import agree
from . import extern

GEO = "BirdNET+_Geomodel_V3.0.2_Global_12K_FP16.onnx"
OV, TH, LV = 2.2, 0.84, 3  # the v3 gate's overlap, threshold, FP level
B = agree.B  # bootstrap resamples
CACHE_NAME = "scores_v3 cache [report-only]"
WHERE = ("site", "own")
SITE_OPTION = "--site"

Site = tuple[float, float]
Variant = dict[str, Any]
Scores = dict[str, Variant]


def parse_site(text: str) -> Site:
    """The (latitude, longitude) of "LAT,LON" in decimal degrees.

    Raises:
        ValueError: not two numbers, or out of range.
    """
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"site {text!r}: expected LAT,LON")
    lat = float(parts[0])
    lon = float(parts[1])
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"site {text!r}: latitude or longitude out of range")
    return lat, lon


def take_site(args: list[str]) -> Site:
    """Removes `--site LAT,LON` from args and returns the site.

    Raises:
        ValueError: the option is missing, has no value or a bad value.
    """
    if SITE_OPTION not in args:
        raise ValueError(f"{SITE_OPTION} LAT,LON is required: the evaluation "
                         "site of the range filter (docs/GATES.md)")
    i = args.index(SITE_OPTION)
    if i + 1 >= len(args):
        raise ValueError(f"{SITE_OPTION} needs a value LAT,LON")
    site = parse_site(args[i + 1])
    del args[i:i + 2]
    return site


def site_text(site: Site) -> str:
    """The site as it appears in a report: `(LAT, LON)`, two decimals."""
    return f"({site[0]:.2f}, {site[1]:.2f})"


def load_harness(tuning: str, ev: str) -> types.ModuleType:
    """The v3 harness module <ev>/scripts/simulate5s.py.

    Args:
        tuning: the tuning directory (the harness's sys.argv[1]).
        ev: the evaluation directory (sys.argv[2]; holds scripts/).
    """
    return extern.load_with_argv(os.path.join(ev, "scripts"), "simulate5s",
                                 tuning, ev)


def site_geo(sim: types.ModuleType, site: Site) -> dict[str, np.ndarray]:
    """The geomodel scores at the evaluation site, per stream (own week)."""
    geo = os.path.join(sim.MD, GEO)
    return {n: sim.rf_scores(geo, site[0], site[1], sim.week[n])
            for n in sim.names}


def site_mask(sim: types.ModuleType, geo_site: dict[str, np.ndarray],
              n: str) -> np.ndarray:
    """The v3 classes passing the range filter at the evaluation site."""
    g = sim.SPEC["v3"]["geo"]
    return np.where(g >= 0, geo_site[n][np.clip(g, 0, None)],
                    0.0) >= sim.RF_THR


def mask(sim: types.ModuleType, geo_site: dict[str, np.ndarray], n: str,
         where: str) -> np.ndarray:
    """The range-filter mask of stream n: "own" location, else the site's."""
    if where == "own":
        return sim.include_mask("v3", n, "geo")
    return site_mask(sim, geo_site, n)


def variant(sim: types.ModuleType, idx: np.ndarray, conf: np.ndarray,
            starts: np.ndarray) -> Variant:
    """A top-10 cache in the harness's v3 form."""
    lab = idx.astype(np.int32)
    conf = conf.astype(np.float32)
    hum = np.where(sim.SPEC["v3"]["human"][lab], conf, 0).max(1)
    return dict(fam="v3", grid=starts, unit=6.0, win=5.0, lab=lab, conf=conf,
                hum=hum)


def replay(sim: types.ModuleType, v: Scores, ov: float, th: float, lv: int,
           mask_of: Callable[[str], np.ndarray]) -> np.ndarray:
    """Per-period (tp, fp, fn) of score set v at one config."""
    ev = {n: sim.events(v[n], ov, th, mask_of(n)) for n in sim.names}
    return sim.score({n: sim.simulate([ev[n]], lv) for n in sim.names})


def _regional_cache(sim: types.ModuleType, c: Any, k: np.ndarray,
                    starts: np.ndarray) -> Variant:
    """The cache's 'reg' (regional) scores as a global-index top-10."""
    r = c["reg"][k].astype(np.float32)
    top = np.argpartition(-r, 10, axis=1)[:, :10]
    conf = np.take_along_axis(r, top, axis=1).astype(np.float16)
    return variant(sim, sim.v3_reg[top], conf, starts)


def load_scores(sim: types.ModuleType, sc_dir: str,
                regional: bool = False) -> tuple[Scores, Scores, Scores]:
    """(reference, candidate, existing reference cache) per stream.

    Args:
        sim: the v3 harness module.
        sc_dir: the scorer's <scores_dir> (ref_idx / ref_conf, g_idx /
            g_conf per stream).
        regional: take the report-only cache's 'reg' (regional) scores
            instead of its global top-10.

    Returns:
        Three {stream: variant} dicts.
    """
    ref, cand, cache = {}, {}, {}
    for n in sim.names:
        d = np.load(os.path.join(sc_dir, n + ".npz"))
        ref[n] = variant(sim, d["ref_idx"], d["ref_conf"], d["starts"])
        cand[n] = variant(sim, d["g_idx"], d["g_conf"], d["starts"])
        # the existing reference cache (report-only)
        c = np.load(os.path.join(sim.OUT, "scores_v3", n + ".npz"))
        pos = {int(s): i for i, s in enumerate(c["starts"])}
        k = np.array([pos[int(s)] for s in d["starts"]])
        if regional:
            cache[n] = _regional_cache(sim, c, k, d["starts"])
        else:
            cache[n] = variant(sim, c["g_idx"][k],
                               c["g_conf"][k].astype(np.float16), d["starts"])
    return ref, cand, cache


def _counts_line(lab: str, pp: np.ndarray) -> str:
    """`  LABEL TP/FP/FN ...  precision x%  recall y%` of per-period counts."""
    tp, fp, fn = agree.counts(pp)
    p, r = agree.pr_percent(tp, fp, fn)
    return (f"  {lab:32s} TP/FP/FN {tp}/{fp}/{fn}  precision {p:.1f}%  "
            f"recall {r:.1f}%")


def _comparison(sim: types.ModuleType, pr: np.ndarray, pn: np.ndarray,
                bidx: np.ndarray, rname: str, label: str) -> None:
    """Prints one reference / candidate comparison (counts, bootstrap,
    the periods that differ)."""
    dp, dr = agree.paired_delta(pr, pn, bidx)
    for lab, pp in ((rname, pr), (label, pn)):
        print(_counts_line(lab, pp))
    lo_p, hi_p = agree.ci(dp)
    lo_r, hi_r = agree.ci(dr)
    print(f"  paired bootstrap ({B} reps over {len(pr)} periods): dP "
          f"{dp.mean():+.2f} pp [{lo_p:+.2f}, {hi_p:+.2f}]"
          f"  dR {dr.mean():+.2f} pp [{lo_r:+.2f}, {hi_r:+.2f}]  -> (a) "
          f"{'ok' if lo_p > -0.5 else 'FAIL'} (b) "
          f"{'ok' if lo_r > -1.0 else 'FAIL'}")
    print(f"  identical period-level outcomes: "
          f"{int((pr == pn).all(1).sum())}/{len(pr)}")
    for k in np.nonzero((pr != pn).any(1))[0]:
        period = sim.periods[k]
        print(f"    period {period[0]} {period[1]:.0f}-{period[2]:.0f}s: ref "
              f"{tuple(int(x) for x in pr[k])} -> {label} "
              f"{tuple(int(x) for x in pn[k])}")


def report(sim: types.ModuleType, site: Site,
           geo_site: dict[str, np.ndarray],
           scores: tuple[Scores, Scores, Scores], ref_name: str,
           label: str) -> None:
    """Prints the replay at both locations against both references.

    Args:
        sim: the v3 harness module.
        site: the evaluation site (named in the report).
        geo_site: site_geo(sim, site).
        scores: (reference, candidate, report-only cache) from load_scores.
        ref_name: the reference's name in the report.
        label: the candidate's name.
    """
    ref, cand, cache = scores
    rng = np.random.default_rng(0)
    bidx = None
    for where in WHERE:
        def mask_of(n, where=where):
            return mask(sim, geo_site, n, where)
        # the candidate's replay depends on the location only
        pn = replay(sim, cand, OV, TH, LV, mask_of)
        for rname, rv in ((ref_name, ref), (CACHE_NAME, cache)):
            pr = replay(sim, rv, OV, TH, LV, mask_of)
            if bidx is None:
                bidx = agree.bootstrap_index(rng, len(pr))
            at = (f"the evaluation site {site_text(site)}"
                  if where == "site" else "each stream")
            print(f"== range filter at {at}; reference {rname}")
            _comparison(sim, pr, pn, bidx, rname, label)


def coverage(sim: types.ModuleType,
             geo_site: dict[str, np.ndarray]) -> None:
    """Prints which truth species the detection gate can exercise."""
    tsp = sorted({lab for n in sim.names for s, e, lab in sim.truth[n]
                  if lab >= 0})
    sid = sim.SPEC["v3"]["sid"]
    inmodel = [t for t in tsp if (sid == t).any()]
    for where in WHERE:
        ok = set()
        for n in sim.names:
            m = mask(sim, geo_site, n, where)
            ok |= {int(s) for s in sid[m] if s < 100000}
        ex = [t for t in inmodel if t in ok]
        names = ", ".join(sim.v24[t].split("_", 1)[1] for t in ex)
        print(f"truth species: {len(tsp)}; in the v3 label set: "
              f"{len(inmodel)}; passing the range filter at {where} for at "
              f"least one stream: {len(ex)}: " + names)
    print(f"classes the detection gate cannot exercise: every class without "
          f"annotated audio here, i.e. all but <= {len(inmodel)} of 11560 "
          "(all non-North-American and non-annotated species); the window "
          "statistics cover them")
