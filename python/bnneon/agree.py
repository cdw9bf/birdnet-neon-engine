"""Agreement statistics shared by the gates and the detection replays.

Window level: the detection sets of reference and candidate at a threshold
(intersection, union, flips) and the reference-side margins of the flips.
Detection level: tp / fp / fn counts per row (window or recording period),
precision and recall, and the paired bootstrap of their differences.
Each caller keeps its own report format, rounding and verdict rule; this
module only computes. It needs numpy only.
"""
import numpy as np

B = 2000  # paired-bootstrap resamples


def hits(ref: np.ndarray, cand: np.ndarray,
         th: float) -> tuple[int, int, np.ndarray]:
    """The detection sets ref > th and cand > th compared.

    Args:
        ref: the reference's values (confidences or logits).
        cand: the candidate's values, same shape.
        th: the threshold (a detection is a value > th).

    Returns:
        (size of the intersection, size of the union, the flip mask: True
        where exactly one side detects).
    """
    hr = ref > th
    hc = cand > th
    return int((hr & hc).sum()), int((hr | hc).sum()), hr != hc


def margin_counts(m: np.ndarray) -> tuple[int, int, int, float]:
    """Counts of flip margins: (<= 0.005, <= 0.02, > 0.05, the maximum).

    Args:
        m: the margins |reference value - threshold| of the flips, in the
            caller's dtype (the comparisons run in it).

    Returns:
        The three counts and the largest margin (0.0 when there is none).
    """
    top = float(m.max()) if len(m) else 0.0
    return (int((m <= 0.005).sum()), int((m <= 0.02).sum()),
            int((m > 0.05).sum()), top)


def counts(pp: np.ndarray) -> tuple[int, int, int]:
    """(tp, fp, fn) summed over the rows of [rows, 3] counts."""
    tp, fp, fn = (int(v) for v in pp.sum(0))
    return tp, fp, fn


def pr_percent(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """(precision, recall) in percent; a zero denominator counts as 1."""
    return 100 * tp / max(1, tp + fp), 100 * tp / max(1, tp + fn)


def prf(pp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(precision, recall) of (..., rows, 3) tp/fp/fn counts summed over rows.

    Args:
        pp: tp, fp, fn counts in the last axis; summed over the axis before.

    Returns:
        (precision, recall), each tp / max(1, tp + fp) and tp / max(1, tp +
        fn).
    """
    tp = pp[..., 0].sum(-1)
    fp = pp[..., 1].sum(-1)
    fn = pp[..., 2].sum(-1)
    return tp / np.maximum(1, tp + fp), tp / np.maximum(1, tp + fn)


def boot_prf(pp: np.ndarray, bidx: np.ndarray,
             chunk: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """prf(pp[bidx]), summed `chunk` resamples at a time.

    The same integer sums as prf on the whole [B, rows, 3] gather, without
    holding it (8 bytes x 3 x B x rows: 480 MB at 10,000 windows).

    Args:
        pp: [rows, 3] tp / fp / fn counts.
        bidx: the resamples, [B, rows].
        chunk: resamples gathered at once.

    Returns:
        (precision, recall), each [B].
    """
    sums = np.concatenate([pp[bidx[i:i + chunk]].sum(1)
                           for i in range(0, len(bidx), chunk)])
    tp, fp, fn = sums[:, 0], sums[:, 1], sums[:, 2]
    return tp / np.maximum(1, tp + fp), tp / np.maximum(1, tp + fn)


def bootstrap_index(rng: np.random.Generator, n: int) -> np.ndarray:
    """B resamples of n rows with replacement, [B, n] (one draw from rng)."""
    return rng.integers(0, n, size=(B, n))


def paired_delta(pr: np.ndarray, pn: np.ndarray,
                 bidx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap distributions of dP and dR (candidate - reference), in pp.

    Args:
        pr: the reference's [rows, 3] tp / fp / fn counts.
        pn: the candidate's counts on the same rows.
        bidx: the resamples, from bootstrap_index.

    Returns:
        (dP, dR), each [B] percentage points.
    """
    p0, r0 = boot_prf(pr, bidx)
    p1, r1 = boot_prf(pn, bidx)
    return 100 * (p1 - p0), 100 * (r1 - r0)


def ci(x: np.ndarray) -> tuple[float, float]:
    """The 2.5 and 97.5 percentiles of x (a 95 % interval)."""
    return np.percentile(x, 2.5), np.percentile(x, 97.5)
