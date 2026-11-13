"""Calibration report for a Perch regional slice (no verdict).

What the slice's confidences mean against the full model's, on the 34.4 h
set, with the external harness's replay (its modelgate module, adapter
perch_v2_na_east). The harness is the birdlab_tools package, which is not
part of this repository: install it next to `pip install -e .` of this
repository.

Score sets over the 999 kept species (all from caches, no inference here):
  F  the FULL model's own confidences for the kept species (softmax over
     all 14,795 classes, NOT renormalized): built from the full fp32
     reference cache (ref_fp32) by keeping the kept columns of its sparse
     entries (conf > 0.005).
  S  plain slice softmax over the 999 logits (what BirdNET-Go computes
     today): the slice reference cache itself.
  R  softmax with the catalog's rest logit appended to the denominator (HF
     models.json score_calibration softmax-rest-logit, rest_logit 9.92 for
     north-america-east): p_R = p_S / (1 + exp(rest - LSE_999)), with
     LSE_999 the per-window log-sum-exp of the 999 slice logits (saved by
     scripts/pipe_score.py --lse during the fp32 slice scoring).
Report: P/R and saved detections of F, S and R at the gated configs; the
threshold at which S and R match F's operating point (same saved-detection
count as F at 0.8 and at 0.5, pi_template / perch_override replay);
inflation stats. Written to results/nae_calibration.txt and printed.

usage: python -I nae_calibration.py <full_ref_dir> <slice_ref_dir> <lse_dir>
           <work_dir> [rest_logit]
"""
import dataclasses
import json
import os
import sys
import types

from birdlab_tools import modelgate
import numpy as np

N_FULL = 14795  # Perch v2 classes
N_KEPT = 999  # the NA-east slice's classes
KEY = 4096  # sparse key = window * KEY + class
# swept thresholds of the operating-point match
THS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
# (overlap, FP level, F's threshold, config name)
OPERATING_POINTS = ((2.2, 2, 0.8, "pi_template"),
                    (2.2, 2, 0.5, "perch_override"),
                    (0.0, 0, 0.8, "upstream_default"))
# the harness's gate lines, dropped: this is a calibration report
GATE_LINES = ("label ", "OVERALL", "   gate:")


@dataclasses.dataclass
class Calib:
    """The command line and the adapter's class mapping."""
    full_dir: str
    slice_dir: str
    lse_dir: str
    work: str
    rest: float
    ad: types.ModuleType
    remap: np.ndarray  # full class -> kept index, -1 if not kept
    names: list[str]
    f_dir: str
    r_dir: str


def setup(argv: list[str]) -> Calib:
    """Parses the command line and loads the perch_v2_na_east adapter."""
    full_dir, slice_dir, lse_dir, work = argv[1:5]
    rest = float(argv[5]) if len(argv) > 5 else 9.92
    ad = modelgate.load_adapter("perch_v2_na_east")
    idx = np.array(ad.indices(), np.int64)
    remap = np.full(N_FULL, -1, np.int64)
    remap[idx] = np.arange(len(idx))
    return Calib(full_dir=full_dir, slice_dir=slice_dir, lse_dir=lse_dir,
                 work=work, rest=rest, ad=ad, remap=remap,
                 names=list(ad.streams()),
                 f_dir=os.path.join(work, "F_full_kept"),
                 r_dir=os.path.join(work, "R_rest_logit"))


def topk_from_sparse(k_win: int, w: np.ndarray, c: np.ndarray,
                     v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window top-K (classes, confidences) from sparse entries.

    Args:
        k_win: the number of windows.
        w: the entries' window rows.
        c: their classes.
        v: their confidences.

    Returns:
        ([k_win, TOPK] int16 classes, [k_win, TOPK] float32 confidences),
        zero-padded.
    """
    ti = np.zeros((k_win, modelgate.TOPK), np.int16)
    tc = np.zeros((k_win, modelgate.TOPK), np.float32)
    o = np.lexsort((-v, w))
    w, c, v = w[o], c[o], v[o]
    start = np.searchsorted(w, np.arange(k_win))
    end = np.searchsorted(w, np.arange(k_win), side="right")
    for k in np.nonzero(end > start)[0]:
        n = min(modelgate.TOPK, end[k] - start[k])
        ti[k, :n] = c[start[k]:start[k] + n]
        tc[k, :n] = v[start[k]:start[k] + n]
    return ti, tc


def write_meta(cb: Calib) -> None:
    """Creates the F and R cache directories with the slice's meta.json."""
    for d in (cb.f_dir, cb.r_dir):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(cb.slice_dir, "meta.json"),
                  encoding="utf-8") as f:
            meta = json.load(f)
        meta["model"] = "calibration:" + os.path.basename(d)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)


def _empty(s: dict) -> dict:
    """The (empty) full-vector part of a cache."""
    return dict(sub_starts=np.zeros(0, np.int32),
                sub_logits=np.zeros((0, N_KEPT), np.float32),
                duration=s["duration"])


def _write_full_kept(cb: Calib, n: str, s: dict,
                     f: dict) -> tuple[np.ndarray, ...]:
    """Writes F: the full model's confidences of the kept species.

    Returns:
        The kept sparse entries (windows, kept classes, confidences).
    """
    keep = cb.remap[f["sp_c"].astype(np.int64)] >= 0
    fw = f["sp_w"][keep]
    fc = cb.remap[f["sp_c"][keep].astype(np.int64)].astype(np.int16)
    fv = f["sp_conf"][keep]
    ti, tc = topk_from_sparse(len(s["starts"]), fw, fc, fv)
    o = np.lexsort((fc, fw))
    np.savez_compressed(os.path.join(cb.f_dir, n + ".npz"),
                        starts=s["starts"], idx=ti, conf=tc, sp_w=fw[o],
                        sp_c=fc[o], sp_conf=fv[o], **_empty(s))
    return fw, fc, fv


def _write_rest_logit(cb: Calib, n: str, s: dict) -> np.ndarray:
    """Writes R: the slice softmax shrunk by the rest logit; returns LSE."""
    lse_path = os.path.join(cb.lse_dir, n + ".lse.f32")
    lse = np.fromfile(lse_path, np.float32).astype(np.float64)
    k_win = len(s["starts"])
    assert len(lse) == k_win, (n, len(lse), k_win)
    sc = 1.0 / (1.0 + np.exp(cb.rest - lse))
    np.savez_compressed(os.path.join(cb.r_dir, n + ".npz"),
                        starts=s["starts"], idx=s["idx"],
                        conf=(s["conf"] * sc[:, None]).astype(np.float32),
                        sp_w=s["sp_w"], sp_c=s["sp_c"],
                        sp_conf=(s["sp_conf"]
                                 * sc[s["sp_w"]]).astype(np.float32),
                        **_empty(s))
    return lse


def _kept_mass(s: dict, fw: np.ndarray, fc: np.ndarray,
               fv: np.ndarray) -> list[float]:
    """p_full / p_slice of each window's top kept entry present on both
    sides (the kept-species mass of the full softmax)."""
    top = s["idx"][:, 0].astype(np.int64)
    key_s = s["sp_w"].astype(np.int64) * KEY + s["sp_c"]
    key_f = fw.astype(np.int64) * KEY + fc
    ks = np.arange(len(s["starts"])) * KEY + top
    ps = dict(zip(key_s.tolist(), s["sp_conf"].tolist()))
    pf = dict(zip(key_f.tolist(), fv.tolist()))
    return [pf[k] / ps[k] for k in ks.tolist() if k in ps and k in pf]


def build(cb: Calib) -> tuple[np.ndarray, np.ndarray]:
    """Writes the F and R caches; returns (kept mass, LSE) over windows."""
    mass, lses = [], []
    write_meta(cb)
    for n in cb.names:
        s = modelgate.load_cache(cb.ad, cb.slice_dir, n)
        f = np.load(os.path.join(cb.full_dir, n + ".npz"))
        assert np.array_equal(s["starts"], f["starts"]), n
        fw, fc, fv = _write_full_kept(cb, n, s, f)
        lse = _write_rest_logit(cb, n, s)
        mass.extend(_kept_mass(s, fw, fc, fv))
        lses.append(lse)
    return np.array(mass), np.concatenate(lses)


def run_sim(cb: Calib, ref: str, cand: str, label: str,
            configs: dict) -> dict:
    """The harness's `sim` of cand against ref at the given configs.

    Returns:
        modelgate.sim_result's dict: the report lines, and per config the
        (tp, fp, fn) and saved-detection counts of both sides.
    """
    cb.ad.CONFIGS = configs
    return modelgate.sim_result(cb.ad, ref, cand, label, None)


def pr(t: tuple[int, int, int]) -> tuple[float, float]:
    """(precision, recall) in % of (tp, fp, fn)."""
    tp, fp, fn = t
    return 100 * tp / max(1, tp + fp), 100 * tp / max(1, tp + fn)


def header_lines(cb: Calib, mass: np.ndarray, lse: np.ndarray) -> list[str]:
    """The report's title and inflation statistics."""
    out = [f"Perch v2 north-america-east slice calibration (no verdict); "
           f"rest logit {cb.rest}; {len(cb.names)} streams",
           f"full-softmax mass of the kept 999 species per window (top kept "
           f"entry, p_full / p_slice): median {np.median(mass):.3f}, p10 "
           f"{np.percentile(mass, 10):.3f}, p90 "
           f"{np.percentile(mass, 90):.3f} (n={len(mass)}); slice "
           f"confidences are inflated by 1/mass: median "
           f"x{1 / np.median(mass):.2f}"]
    lse_stats = (f"LSE of the 999 slice logits: median {np.median(lse):.2f}, "
                 f"p10 {np.percentile(lse, 10):.2f}, p90 "
                 f"{np.percentile(lse, 90):.2f}; ")
    shrink = np.median(1 / (1 + np.exp(cb.rest - lse)))
    out.append(lse_stats + f"rest-logit shrink factor 1/(1+e^(rest-LSE)): "
               f"median {shrink:.3f}")
    return out


def gated_lines(cb: Calib, base: dict) -> list[str]:
    """F against S and against R at the harness's gated configs."""
    out = []
    for cand, lab in ((cb.slice_dir, "S plain slice softmax"),
                      (cb.r_dir, "R rest-logit softmax")):
        out.append(f"\n== F (full model, kept species, not renormalized) vs "
                   f"{lab}: gated configs")
        res = run_sim(cb, cb.f_dir, cand, lab, base)
        out.append("\n".join(line for line in res["lines"]
                             if not line.startswith(GATE_LINES)))
    return out


def operating_point_lines(cb: Calib, ovl: float, fpl: int, thr0: float,
                          nm: str) -> list[str]:
    """F at one deploy threshold against S and R at the swept thresholds."""
    cf = {f"t{t}": dict(overlap=ovl, threshold=t, fp_level=fpl,
                        range_threshold=None) for t in THS}
    out = [f"\n== operating point of F at {nm} (overlap {ovl}, FP L{fpl}, "
           f"threshold {thr0}) vs S / R at swept thresholds"]
    fr = run_sim(cb, cb.f_dir, cb.slice_dir, "S", cf)["configs"]
    rr = run_sim(cb, cb.f_dir, cb.r_dir, "R", cf)["configs"]
    f0 = fr[f"t{thr0}"]["ref"]
    saved0 = fr[f"t{thr0}"]["saved"][0]
    out.append(f"   F @{thr0}: TP/FP/FN {f0}  P {pr(f0)[0]:.2f} %  R "
               f"{pr(f0)[1]:.2f} %  saved {saved0}")
    for t in THS:
        s_, r_ = fr[f"t{t}"]["cand"], rr[f"t{t}"]["cand"]
        s_saved, r_saved = fr[f"t{t}"]["saved"][1], rr[f"t{t}"]["saved"][1]
        out.append(f"   t={t:<5} S: TP/FP/FN {s_} P {pr(s_)[0]:6.2f} R "
                   f"{pr(s_)[1]:6.2f} saved {s_saved:5d} | "
                   f"R: TP/FP/FN {r_} P {pr(r_)[0]:6.2f} R {pr(r_)[1]:6.2f} "
                   f"saved {r_saved:5d}")
    return out


def main(argv: list[str]) -> int:
    """Builds the F and R caches, writes and prints the report."""
    cb = setup(argv)
    os.makedirs(cb.work, exist_ok=True)
    mass, lse = build(cb)
    rep = header_lines(cb, mass, lse)
    base = dict(cb.ad.CONFIGS)
    rep.extend(gated_lines(cb, base))
    # operating-point match: thresholds swept on S and R, F at the deploy
    # thresholds
    for ovl, fpl, thr0, nm in OPERATING_POINTS:
        rep.extend(operating_point_lines(cb, ovl, fpl, thr0, nm))
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "results", "nae_calibration.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
