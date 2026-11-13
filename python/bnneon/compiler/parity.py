"""Parity measures for bnc check: post-activations, SQNR, TFLite references."""
from collections.abc import Callable

import numpy as np

from .. import refrun

# sqnr_db's values for an exact match and for an all-zero reference
SQNR_EXACT_DB = 200.0
SQNR_ZERO_REF_DB = -99.0


def softmax_rows(z: np.ndarray) -> np.ndarray:
    """Softmax over axis 1."""
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """The logistic function."""
    return 1 / (1 + np.exp(-z))


def identity(z: np.ndarray) -> np.ndarray:
    """--post none."""
    return z


def post_fn(post: str) -> Callable[[np.ndarray], np.ndarray]:
    """The host's post-activation for --post softmax / sigmoid / none."""
    if post == "softmax":
        return softmax_rows
    if post == "sigmoid":
        return sigmoid
    return identity


def sqnr_db(ref: np.ndarray, cand: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio of cand against ref, in dB.

    Returns:
        10 log10(sum ref^2 / sum (ref - cand)^2), computed in float64;
        SQNR_EXACT_DB when the two are equal, SQNR_ZERO_REF_DB when ref is
        all zero (and cand is not).
    """
    ref64 = ref.astype(np.float64)
    num = float((ref64 ** 2).sum())
    den = float(((ref64 - cand.astype(np.float64)) ** 2).sum())
    if den == 0:
        return SQNR_EXACT_DB
    if num == 0:
        return SQNR_ZERO_REF_DB
    return 10 * np.log10(num / den)


def tflite_logits(path: str, x: np.ndarray,
                  log: Callable[..., None]) -> np.ndarray | None:
    """Reference logits from a TFLite file, one window per invoke.

    Args:
        path: the .tflite model.
        x: the windows, [n][samples].
        log: the print function (for the warning when no interpreter exists).

    Returns:
        The first output of every window, [n][outputs]; None when neither
        ai_edge_litert nor tensorflow is installed.
    """
    run = refrun.tflite_runner(path)
    if run is None:
        log("WARN: --tflite given but neither ai_edge_litert nor tensorflow is "
            "importable: skipped")
        return None
    return run(x)
