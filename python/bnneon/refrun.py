"""Reference runtimes: ONNX Runtime sessions and LiteRT runners.

The gates and check compare a package against its reference model on the
same windows. This module builds the reference side: ONNX Runtime on the CPU
execution provider (session_options / session / ort_session, and a batch
runner) and LiteRT (or TensorFlow Lite) one window per invoke. Each caller
passes its own thread count: reference results can depend on the number of
intra-op threads, so none is chosen here.

onnxruntime and LiteRT are imported only when a function needs them: the
generic gate's `declare` and a LiteRT-only reference run without
onnxruntime, and an ONNX reference without LiteRT.
"""
from collections.abc import Callable
from typing import Any

import numpy as np

Runner = Callable[[np.ndarray], np.ndarray]


def _ort() -> Any:
    """The onnxruntime module."""
    # optional heavy dependency: only an ONNX reference needs it
    import onnxruntime  # pylint: disable=import-outside-toplevel
    return onnxruntime


def session_options(threads: int = 0, quiet: bool = True) -> Any:
    """ONNX Runtime session options.

    Args:
        threads: intra-op threads (0: ONNX Runtime's default).
        quiet: log errors only (log_severity_level 3).

    Returns:
        An onnxruntime.SessionOptions.
    """
    so = _ort().SessionOptions()
    if quiet:
        so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = threads
    return so


def session(path: str, so: Any) -> Any:
    """An ONNX Runtime session of the model on the CPU execution provider."""
    return _ort().InferenceSession(path, so,
                                   providers=["CPUExecutionProvider"])


def ort_session(path: str, threads: int = 0, quiet: bool = True) -> Any:
    """session(path, session_options(threads, quiet))."""
    return session(path, session_options(threads, quiet))


def onnx_runner(path: str, threads: int = 0, quiet: bool = True) -> Runner:
    """run(batch) -> the model's first output for a [n, samples] batch."""
    sess = ort_session(path, threads, quiet)
    iname = sess.get_inputs()[0].name

    def run(batch: np.ndarray) -> np.ndarray:
        return sess.run(None, {iname: batch})[0]
    return run


def onnx_model(path: str) -> Any:
    """The onnx.ModelProto of an ONNX file."""
    # optional heavy dependency: only a reference read at an internal tensor
    # needs the graph itself
    import onnx  # pylint: disable=import-outside-toplevel
    return onnx.load(path)


def producer(m: Any, tensor: str) -> Any:
    """The node of model m that outputs tensor, or None."""
    for n in m.graph.node:
        if tensor in n.output:
            return n
    return None


def session_at(m: Any, tensor: str, threads: int = 0,
               quiet: bool = True) -> Any:
    """A CPU session of model m that outputs tensor (internal or not).

    The model is changed in place: tensor becomes its only graph output (as
    bnc check reads the logits of a graph that outputs only their sigmoid).
    """
    # the compiler's graph module imports onnx and onnxruntime: only this
    # function needs it here
    from .compiler import onnxg  # pylint: disable=import-outside-toplevel
    onnxg.set_outputs(m.graph, [tensor])
    return session(m.SerializeToString(), session_options(threads, quiet))


def tflite_interpreter() -> Any:
    """The TFLite Interpreter class (LiteRT, else TensorFlow), or None."""
    # optional heavy dependencies, only a .tflite reference needs one of them
    # pylint: disable=import-outside-toplevel
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError:
            return None
    return Interpreter


def tflite_runner(path: str, threads: int | None = None) -> Runner | None:
    """run(windows) -> [n, outputs]: the first output, one window per invoke.

    Args:
        path: the .tflite model.
        threads: the interpreter's num_threads (None: its default).

    Returns:
        The runner, or None when neither ai_edge_litert nor tensorflow is
        installed.
    """
    interpreter = tflite_interpreter()
    if interpreter is None:
        return None
    if threads is None:
        it = interpreter(model_path=path)
    else:
        it = interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    i = it.get_input_details()[0]
    o = it.get_output_details()[0]

    def run(x: np.ndarray) -> np.ndarray:
        ys = []
        for w in x:
            it.set_tensor(i["index"], w.reshape(i["shape"]).astype(np.float32))
            it.invoke()
            ys.append(it.get_tensor(o["index"]).reshape(-1))
        return np.stack(ys)
    return run
