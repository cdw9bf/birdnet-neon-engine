"""Python binding for libbnneon (ctypes; standard library + numpy only).

    import bnneon
    with bnneon.Model("birdnet_v2.4-neon-fp16.bnpk",
                      lib="build/libbnneon.so") as m:
        scores, emb = m.infer(pcm)    # m.info.n_samples float32 samples
        conf = m.confidences(scores)  # logit_scale and the model's post

pcm holds m.info.n_samples samples at m.info.sample_rate, in [-1, 1);
confidences() applies logit_scale and the model's post (sigmoid / softmax /
none).

The library is found from the `lib` argument, else the BNNEON_LIB environment
variable, else the dynamic loader's search path ("libbnneon.so"); it is loaded
by the first call that needs it, not on import. `scores` is exactly what
bnn_infer writes (logits for .bnpk and .bn24 files; sigmoid confidences for a
legacy .bnv3 on a libbnneon from before its engine was retired: this one
refuses .bnv3 files). Errors raise BnnError carrying bnn_last_error() and
bnn_last_error_kind(). One Model per thread (the C contexts are not
thread-safe; different Models may run on different threads).

This file is self-contained: copy it into a project as bnneon.py to use the
binding without the bnneon package.
"""
import ctypes
import os

import numpy as np

# the bnneon.h generation this binding marshals (bnn_info with the struct_size
# handshake)
API_VERSION = 2
MODE_DEFAULT = 0
MODE_REFERENCE = 1
MODE_FP32 = 2
MODE_FP16 = 3
MODE_INT8 = 4
OUT_LOGITS = 0
OUT_SIGMOID = 1
POST_NONE = 0
POST_SIGMOID = 1
POST_SOFTMAX = 2
ERR_KINDS = {0: "NONE", 1: "PACKAGE", 2: "OOM", 3: "CPU", 4: "WINDOW",
             5: "LABELS", 6: "USAGE"}
# the first line of Perch v2's full label file: a header, not a class
LABEL_HEADER = "inat2024_fsd50k"


class BnnError(RuntimeError):
    """A failed library call.

    str(e) is bnn_last_error() (show it unchanged); e.kind is the BNN_ERR_*
    name ("PACKAGE", "OOM", "CPU", "WINDOW", "LABELS", "USAGE").
    """

    def __init__(self, message: str, kind: str):
        super().__init__(message)
        self.kind = kind


class Info(ctypes.Structure):
    """bnn_info of bnneon.h, API 2 (fields are only ever appended)."""

    _fields_ = [("struct_size", ctypes.c_uint32), ("model", ctypes.c_int),
                ("format_version", ctypes.c_int),
                ("sample_rate", ctypes.c_int), ("n_samples", ctypes.c_int),
                ("hop_samples", ctypes.c_int), ("n_classes", ctypes.c_int),
                ("output_kind", ctypes.c_int), ("post", ctypes.c_int),
                ("logit_scale", ctypes.c_float), ("emb_dim", ctypes.c_int),
                ("has_labels_sha", ctypes.c_int),
                ("labels_sha256", ctypes.c_ubyte * 32),
                ("model_id", ctypes.c_char * 64),
                ("path", ctypes.c_char * 32),
                ("precision", ctypes.c_char_p)]

    @property
    def labels_sha256_hex(self) -> str | None:
        """The compiled-in label file's sha256 as hex (None without one)."""
        if not self.has_labels_sha:
            return None
        return bytes(self.labels_sha256).hex()

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={getattr(self, k)!r}"
                           for k, _ in self._fields_ if k != "labels_sha256")
        return f"Info({fields}, labels_sha256_hex={self.labels_sha256_hex!r})"


_FP = ctypes.POINTER(ctypes.c_float)
_SIGNATURES = {
    "bnn_api_version": (ctypes.c_int, []),
    "bnn_cpu_supported": (ctypes.c_int, [ctypes.c_int]),
    "bnn_probe": (ctypes.c_int, [ctypes.c_char_p, ctypes.POINTER(Info)]),
    "bnn_probe_meta": (ctypes.c_int, [ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.c_char_p, ctypes.c_size_t]),
    "bnn_meta": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_char_p, ctypes.c_size_t]),
    "bnn_init": (ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int]),
    "bnn_get_info": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(Info)]),
    "bnn_infer": (ctypes.c_int, [ctypes.c_void_p, _FP, _FP, _FP]),
    "bnn_verify_labels": (ctypes.c_int, [ctypes.c_char_p, ctypes.c_char_p]),
    "bnn_last_error": (ctypes.c_char_p, []),
    "bnn_last_error_kind": (ctypes.c_int, []),
    "bnn_free": (None, [ctypes.c_void_p]),
}


def load_library(lib: str | None = None) -> ctypes.CDLL:
    """Loads libbnneon and checks that it speaks API_VERSION.

    Args:
        lib: the library's path (default: BNNEON_LIB, else "libbnneon.so").

    Returns:
        The ctypes handle, with the argument and result types set.

    Raises:
        OSError: the library cannot be loaded.
        BnnError: (kind USAGE) the library speaks another API version.
    """
    h = ctypes.CDLL(lib or os.environ.get("BNNEON_LIB") or "libbnneon.so")
    for name, (res, args) in _SIGNATURES.items():
        f = getattr(h, name)
        f.restype, f.argtypes = res, args
    v = h.bnn_api_version()
    if v != API_VERSION:
        raise BnnError(f"libbnneon API version {v} vs {API_VERSION} expected "
                       "by this binding; hint: use the "
                       "python/bnneon/binding.py of the same engine checkout",
                       "USAGE")
    return h


def _error(h: ctypes.CDLL) -> BnnError:
    """The library's last error as a BnnError."""
    message = (h.bnn_last_error() or b"").decode("utf-8", "replace")
    return BnnError(message, ERR_KINDS.get(h.bnn_last_error_kind(), "?"))


def _lib(lib: str | ctypes.CDLL | None) -> ctypes.CDLL:
    return lib if isinstance(lib, ctypes.CDLL) else load_library(lib)


def _meta(call, key: str, raw: bool) -> str | bytes | None:
    """Two-call metadata read (the length, then the value).

    Returns:
        The value (bytes if raw), or None if the key is absent.
    """
    k = key.encode()
    n = call(k, None, 0)
    if n < 0:
        return None
    buf = ctypes.create_string_buffer(n + 1)
    call(k, buf, n + 1)
    return buf.raw[:n] if raw else buf.raw[:n].decode("utf-8", "replace")


def probe_meta(path: str, key: str, lib: str | ctypes.CDLL | None = None,
               raw: bool = False) -> str | bytes | None:
    """A metadata value of a file without loading it.

    Returns:
        The value (bytes if raw); None if the key is absent (or the file is
        unreadable).
    """
    h = _lib(lib)
    fpath = os.fsencode(path)
    return _meta(lambda k, b, n: h.bnn_probe_meta(fpath, k, b, n), key, raw)


def verify_labels(path: str, labels_path: str,
                  lib: str | ctypes.CDLL | None = None) -> None:
    """Checks a label file against the one the package was compiled with.

    Raises:
        BnnError: (kind LABELS) labels_path is not byte for byte that file.
    """
    h = _lib(lib)
    if h.bnn_verify_labels(os.fsencode(path), os.fsencode(labels_path)):
        raise _error(h)


def probe(path: str, lib: str | ctypes.CDLL | None = None) -> Info:
    """The file's metadata without loading the weights.

    info.path is the best path this CPU can run ("" if none).

    Raises:
        BnnError: the file cannot be read as a model.
    """
    h = _lib(lib)
    info = Info(struct_size=ctypes.sizeof(Info))
    if h.bnn_probe(os.fsencode(path), ctypes.byref(info)):
        raise _error(h)
    return info


def class_labels(text: str) -> list[str]:
    """One label per class from a label file's text: the compiler's rule.

    A line ends at "\\n" and loses one "\\r" before it; a line of only
    spaces, tabs and "\\r" is blank and is not a label; a first label line
    that is exactly LABEL_HEADER is a header, not a class. Twin of
    bnneon.compiler.prep.class_labels (this module is self-contained);
    tests/test_label_rule.py checks that both agree.
    """
    lines = [ln.removesuffix("\r") for ln in text.split("\n")]
    lines = [ln for ln in lines if ln.strip(" \t\r")]
    if lines and lines[0] == LABEL_HEADER:
        return lines[1:]
    return lines


class Model:
    """One loaded model (a context of the library); use it as a context manager.

    Attributes:
        path: the model file (bytes).
        info: its Info.
    """

    def __init__(self, path: str, mode: int = MODE_DEFAULT,
                 lib: str | ctypes.CDLL | None = None, threads: int = 1):
        """Loads a model.

        Args:
            path: the .bnpk / .bn24 file.
            mode: MODE_DEFAULT (the best path this CPU runs) or a MODE_*.
            lib: the library (a path or a handle from load_library).
            threads: intra-op threads (see set_threads).

        Raises:
            BnnError: the library refused the file.
        """
        self._h = _lib(lib)
        self.path = os.fsencode(path)
        self._ctx = self._h.bnn_init(self.path, int(mode))
        if not self._ctx:
            raise _error(self._h)
        self.info = Info(struct_size=ctypes.sizeof(Info))
        if self._h.bnn_get_info(self._ctx, ctypes.byref(self.info)):
            raise _error(self._h)
        self._out = np.empty(self.info.n_classes, np.float32)
        self._emb = None
        if self.info.emb_dim > 0:
            self._emb = np.empty(self.info.emb_dim, np.float32)
        if threads != 1:
            self.set_threads(threads)

    def set_threads(self, threads: int) -> None:
        """Intra-op threads for a package.

        1 = the calling thread only; outputs are identical for every count.

        Raises:
            BnnError: (kind USAGE) a legacy .bn24 file (1 only), or a
                library without bnn_set_threads.
        """
        self._check()
        try:
            f = self._h.bnn_set_threads
        except AttributeError:
            raise BnnError("this libbnneon has no bnn_set_threads; hint: use "
                           "a newer libbnneon for threads", "USAGE") from None
        f.restype, f.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]
        if f(self._ctx, int(threads)):
            raise _error(self._h)

    def meta(self, key: str, raw: bool = False) -> str | bytes | None:
        """A metadata string by key ("license", "attribution", "labels", ...).

        Returns:
            The value (bytes if raw), None if absent.
        """
        self._check()
        return _meta(lambda k, b, n: self._h.bnn_meta(self._ctx, k, b, n), key,
                     raw)

    def labels(self) -> list[str] | None:
        """One label per class from the label file compiled into the package.

        The file is stored as shipped; class_labels numbers its lines as
        the compiler did (blank lines and Perch v2's header line are not
        classes).

        Returns:
            The labels; None for legacy files (no labels inside).
        """
        t = self.meta("labels")
        if t is None:
            return None
        return class_labels(t)

    def infer(self, pcm) -> tuple[np.ndarray, np.ndarray | None]:
        """Scores one window.

        Args:
            pcm: info.n_samples float32 samples in [-1, 1).

        Returns:
            (scores, embedding): new float32 arrays of n_classes and emb_dim
            (embedding None if emb_dim is 0).

        Raises:
            ValueError: the window has the wrong shape.
            BnnError: (kind WINDOW) this window could not be scored.
        """
        self._check()
        x = np.ascontiguousarray(pcm, dtype=np.float32)
        if x.shape != (self.info.n_samples,):
            raise ValueError(f"pcm has shape {x.shape}; this model takes "
                             f"({self.info.n_samples},) float32 samples at "
                             f"{self.info.sample_rate} Hz")
        emb = None
        if self._emb is not None:
            emb = self._emb.ctypes.data_as(_FP)
        rc = self._h.bnn_infer(self._ctx, x.ctypes.data_as(_FP),
                               self._out.ctypes.data_as(_FP), emb)
        # the only failure for valid arguments: this window was not scored
        if rc:
            if self.info.format_version == 0:
                # legacy engines return nonzero without setting the message
                raise BnnError(f"window not scored (bnn_infer returned {rc}); "
                               f"hint: skip this window", "WINDOW")
            raise _error(self._h)
        emb_copy = None if self._emb is None else self._emb.copy()
        return self._out.copy(), emb_copy

    def confidences(self, scores) -> np.ndarray:
        """Applies logit_scale and the model's post to infer()'s scores.

        Works per row of a [..., n_classes] array; the identity when the
        library already did it (a legacy .bnv3 on an older libbnneon).
        """
        if self.info.output_kind == OUT_SIGMOID:
            # a legacy .bnv3 (older libbnneon): bnn_infer wrote confidences
            return np.asarray(scores, np.float32)
        z = np.asarray(scores, np.float64) * self.info.logit_scale
        if self.info.post == POST_SIGMOID:
            return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        if self.info.post == POST_SOFTMAX:
            e = np.exp(z - z.max(axis=-1, keepdims=True))
            return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
        return z.astype(np.float32)

    def _check(self) -> None:
        if not self._ctx:
            raise BnnError("model is closed", "USAGE")

    def close(self) -> None:
        """Frees the context (also on garbage collection and leaving `with`)."""
        if getattr(self, "_ctx", None):
            self._h.bnn_free(self._ctx)
            self._ctx = None

    __del__ = close

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
