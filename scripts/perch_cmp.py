"""Compare Perch logits (n x C float32) of a candidate against a reference:
logit SQNR, max |dlogit|, softmax max |dconf|, top-1 agreement, windows
whose top-1 conf crosses thresholds.

usage: python perch_cmp.py <ref.f32> <cand.f32> [C]
"""
import sys

import numpy as np

N_CLASSES = 14795
THRESHOLDS = (0.1, 0.5, 0.8)


def softmax(z):
    """Row-wise softmax (shifted by the row maximum)."""
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def load(path, n_classes):
    """A logits file as float64 [windows][n_classes]."""
    return np.fromfile(path, np.float32).reshape(-1, n_classes).astype(
        np.float64)


def main():
    n_classes = int(sys.argv[3]) if len(sys.argv) > 3 else N_CLASSES
    r = load(sys.argv[1], n_classes)
    c = load(sys.argv[2], n_classes)
    n = min(len(r), len(c))
    r, c = r[:n], c[:n]
    pr, pc = softmax(r), softmax(c)
    d = c - r
    sq = 10 * np.log10((r ** 2).sum() / max((d ** 2).sum(), 1e-300))
    cen = r - r.mean(1, keepdims=True)
    sqc = 10 * np.log10((cen ** 2).sum()
                        / max(((d - d.mean(1, keepdims=True)) ** 2).sum(),
                              1e-300))
    flips = {t: int((((pr >= t) != (pc >= t))).sum()) for t in THRESHOLDS}
    print(f"n {n}  logit SQNR {sq:.1f} dB (mean-removed {sqc:.1f})  "
          f"max|dlogit| {np.abs(d).max():.3g}  "
          f"max|dconf| {np.abs(pc - pr).max():.3g}  "
          f"top1 {np.mean(r.argmax(1) == c.argmax(1)):.4f}  "
          f"flips@0.1/0.5/0.8 {flips}")


if __name__ == "__main__":
    main()
