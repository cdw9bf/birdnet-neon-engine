"""bnneon: the Python side of the NEON engine.

`import bnneon` gives the ctypes binding of libbnneon (bnneon.binding,
re-exported here). The binding is one self-contained module: copy
python/bnneon/binding.py as bnneon.py to use it without this package. The
shared library is loaded on the first call that needs it, so importing bnneon
needs no built library.

Subpackages and modules: bnneon.compiler (the `bnc` model compiler),
bnneon.labels (label files in BirdNET-Go's form), bnneon.gate (the generic
accuracy gate), bnneon.agree (the agreement statistics the gates and replays
share), bnneon.pipe (the engine CLI's --pipe protocol and the stream pool),
bnneon.refrun (the ONNX Runtime and LiteRT references), bnneon.gate_v3,
bnneon.decode_v3 and bnneon.replay_v3 (the BirdNET v3.0 gates),
bnneon.extern (loads modules of the external evaluation harness) and
bnneon.testutil (the plain-script tests' helpers).
"""
from .binding import (API_VERSION, ERR_KINDS, MODE_DEFAULT, MODE_FP16,
                      MODE_FP32, MODE_INT8, MODE_REFERENCE, OUT_LOGITS,
                      OUT_SIGMOID, POST_NONE, POST_SIGMOID, POST_SOFTMAX,
                      BnnError, Info, Model, load_library, probe, probe_meta,
                      verify_labels)

__all__ = [
    "API_VERSION", "ERR_KINDS", "MODE_DEFAULT", "MODE_FP16", "MODE_FP32",
    "MODE_INT8", "MODE_REFERENCE", "OUT_LOGITS", "OUT_SIGMOID", "POST_NONE",
    "POST_SIGMOID", "POST_SOFTMAX", "BnnError", "Info", "Model",
    "load_library", "probe", "probe_meta", "verify_labels",
]
