"""Unit tests for bnneon.refrun (the reference runtimes).

  python -P -s tests/test_refrun.py     (bnneon installed, or
                                         PYTHONPATH=<checkout>/python)

Needs onnx and onnxruntime (a two-node model is built in memory); the LiteRT
check runs only when ai_edge_litert or tensorflow is installed. Exit 0 when
every check passes.
"""
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime

from bnneon import gate
from bnneon import refrun
from bnneon import testutil

CHECKS = testutil.Checks()
check = CHECKS.check


def _model(path):
    """y = 2 x and z = x + 1 on [n, 4] float32 inputs, written to path."""
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    g = helper.make_graph(
        [helper.make_node("Mul", ["x", "two"], ["y"]),
         helper.make_node("Add", ["x", "one"], ["z"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 4]),
         helper.make_tensor_value_info("z", TensorProto.FLOAT, ["n", 4])],
        [two, one])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def _sigmoid_model(path):
    """p = Sigmoid(2 x) (graph output), e = x + 1 (graph output)."""
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])
    g = helper.make_graph(
        [helper.make_node("Mul", ["x", "two"], ["a"]),
         helper.make_node("Sigmoid", ["a"], ["p"]),
         helper.make_node("Add", ["x", "one"], ["e"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("p", TensorProto.FLOAT, [1, 4]),
         helper.make_tensor_value_info("e", TensorProto.FLOAT, [1, 4])],
        [two, one])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def test_internal_tensors():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.onnx")
        _sigmoid_model(path)
        m = refrun.onnx_model(path)
        check("producer", refrun.producer(m, "p").op_type == "Sigmoid"
              and refrun.producer(m, "x") is None)
        x = np.array([[0.5, -1.0, 2.0, 0.0]], np.float32)
        s = refrun.session_at(refrun.onnx_model(path), "a")
        check("session_at reads an internal tensor",
              np.allclose(s.run(["a"], {"x": x})[0], 2 * x))
        y, kind = gate.ref_logits(path, x, "a")
        check("gate: an internal --ref-output", kind == "onnxruntime:a"
              and np.allclose(y, 2 * x), (kind, y))
        y, kind = gate.ref_logits(path, x, "p", post="sigmoid")
        check("gate: the Sigmoid output is read before the Sigmoid",
              kind == "onnxruntime:a" and np.allclose(y, 2 * x), (kind, y))
        y, kind = gate.ref_logits(path, x, "p", post="none")
        check("gate: --post none keeps the named output",
              kind == "onnxruntime:p" and y.min() > 0 and y.max() < 1, kind)
        e = testutil.raises(lambda: gate.ref_logits(path, x, "nope"),
                            SystemExit)
        check("gate: an unknown tensor exits 3", e is not None
              and e.code == 3, e)
        hint = gate._logits_hint(path)  # pylint: disable=protected-access
        check("gate: the class-count hint names the logits", "a (before "
              "the Sigmoid that gives 'p')" in hint, hint)


def test_session_options():
    so = refrun.session_options()
    check("quiet options log errors only", so.log_severity_level == 3)
    check("threads 0 keeps ONNX Runtime's default",
          so.intra_op_num_threads == 0)
    so = refrun.session_options(3, quiet=False)
    default = onnxruntime.SessionOptions().log_severity_level
    check("threads, and the log level left at its default",
          so.intra_op_num_threads == 3 and so.log_severity_level == default,
          (so.intra_op_num_threads, so.log_severity_level))


def test_onnx():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.onnx")
        _model(path)
        x = np.arange(8, dtype=np.float32).reshape(2, 4)
        s = refrun.ort_session(path, 1)
        check("ort_session on the CPU provider",
              s.get_providers() == ["CPUExecutionProvider"],
              s.get_providers())
        y = refrun.onnx_runner(path, 1)(x)
        check("onnx_runner returns the first output", np.array_equal(y, 2 * x),
              y)


def test_tflite():
    if refrun.tflite_interpreter() is None:
        check("no LiteRT: tflite_runner is None",
              refrun.tflite_runner("missing.tflite") is None)
        return
    print("skip: a LiteRT run needs a .tflite file (none is built here)")


def main():
    test_session_options()
    test_onnx()
    test_internal_tensors()
    test_tflite()
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
