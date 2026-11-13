"""Compare engine logits with ORT logits.

usage: python -I cmp.py <ref.f32> <cand.f32> [label]
"""
import sys

import numpy as np

N_CLASSES = 6522


def sigmoid(z):
    """1 / (1 + exp(-z))."""
    return 1 / (1 + np.exp(-z))


def load(path):
    """A logits file as float64 [windows][6522]."""
    return np.fromfile(path, np.float32).reshape(-1, N_CLASSES).astype(
        np.float64)


def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ca, cb = sigmoid(a), sigmoid(b)
    sq = 10 * np.log10((a ** 2).sum() / max(((a - b) ** 2).sum(), 1e-30))
    fl = ((ca > 0.8) != (cb > 0.8)).sum()
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    print(f"{label}: {n} windows  logit SQNR {sq:.1f} dB  "
          f"max|dlogit| {np.abs(a-b).max():.4f}  "
          f"max|dconf| {np.abs(ca-cb).max():.4f}  flips@0.8 {fl} "
          f"(ref hits {(ca>0.8).sum()})  "
          f"top1 {(a.argmax(1)==b.argmax(1)).mean():.4f}")


if __name__ == "__main__":
    main()
