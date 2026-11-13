"""Convert a BirdNET v2.4 custom-classifier .tflite to ONNX with tf2onnx.

BirdNET's MEL_SPEC front end lowers to TFLite as
RFFT2D -> SQUEEZE -> CAST(complex64->float32), i.e. it keeps only the REAL
part of the STFT (CAST drops the imaginary part). tf2onnx 1.17.0 only
converts RFFT2D when its consumer is ComplexAbs, so the stock CLI fails with
  "Current implementation of RFFT2D only allows ComplexAbs as consumer not
  {'Squeeze'}".
This script registers one custom rewriter that replaces each RFFT2D(x, [1, N])
whose result is only real-cast with an ONNX MatMul(x, C),
C[n, k] = cos(2*pi*n*k/N) (k = 0..N/2), which is exactly Re(rfft(x)). The
downstream Squeeze/Cast then operate on float32. Everything else is stock
tf2onnx.

Memory: run with PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python. With protobuf
7.36.2's default upb backend every tf2onnx optimizer pass (graph.update_proto)
leaves ~38 MB (the weight size) unreleased, and tf2onnx's per-pass deepcopy
adds more; the run is OOM-killed in a 2 GB container. With the pure Python
backend peak RSS stays under ~0.8 GB.

Usage:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -I tflite_to_onnx.py in.tflite out.onnx [opset]
"""
import sys

import numpy as np
from onnx import TensorProto

import tf2onnx
from tf2onnx import utils

PASSTHROUGH_OPS = ("Squeeze", "Reshape", "ExpandDims", "Identity")


def real_part_consumers(g, rfft):
    """The shape ops between an RFFT2D and its real-part Casts.

    Every consumer path must end in a Cast to float32, possibly through
    Squeeze / Reshape / ExpandDims / Identity.

    Returns:
        The pass-through nodes found on the way.

    Raises:
        ValueError: a consumer uses more than the real part.
    """
    frontier = list(g.find_output_consumers(rfft.output[0]))
    passthrough = []
    while frontier:
        c = frontier.pop()
        if c.type in PASSTHROUGH_OPS:
            passthrough.append(c)
            frontier.extend(g.find_output_consumers(c.output[0]))
        elif (c.type == "Cast" and c.get_attr_value("DstT", c.get_attr_value(
                "to")) in (TensorProto.FLOAT, 1)):
            continue
        else:
            raise ValueError(f"RFFT2D {rfft.name} has a non real-part "
                             f"consumer {c.type}; not handled")
    return passthrough


def cos_matrix(n):
    """C[t, k] = cos(2 pi t k / n) for k = 0..n/2, float32 [n, n/2+1]."""
    k = np.arange(n // 2 + 1, dtype=np.int64)
    t = np.arange(n, dtype=np.int64)
    return np.cos(2.0 * np.pi * ((t[:, None] * k[None, :]) % n) / n).astype(
        np.float32)


def rewrite_rfft_real_part(g, ops):
    """tf2onnx rewriter: a real-cast RFFT2D becomes a MatMul with a cos matrix.

    Args:
        g: the tf2onnx graph.
        ops: its nodes.

    Returns:
        The graph's nodes after the rewrite.
    """
    for rfft in [op for op in ops if op.type == "RFFT2D"]:
        length = rfft.inputs[1].get_tensor_value(as_list=True)
        utils.make_sure(len(length) == 2 and length[0] == 1,
                        "unexpected RFFT2D fft_length %r", length)
        n = int(length[1])
        passthrough = real_part_consumers(g, rfft)
        cmat = g.make_const(utils.make_name(rfft.name + "_dft_cos"),
                            cos_matrix(n))
        # an ONNX-domain MatMul (skip_conversion=True: already ONNX, no TF
        # handler needed)
        mm = g.make_node("MatMul", [rfft.input[0], cmat.output[0]],
                         op_name_scope=rfft.name,
                         shapes=[g.get_shape(rfft.output[0])],
                         dtypes=[TensorProto.FLOAT], skip_conversion=True)
        g.replace_all_inputs(rfft.output[0], mm.output[0])
        for p in passthrough:
            g.set_dtype(p.output[0], TensorProto.FLOAT)
        g.remove_node(rfft.name)
    return g.get_nodes()


def dims(value_info):
    """A graph input's or output's dims (values or parameter names)."""
    return [d.dim_value or d.dim_param
            for d in value_info.type.tensor_type.shape.dim]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    opset = int(sys.argv[3]) if len(sys.argv) > 3 else 17
    model_proto, _ = tf2onnx.convert.from_tflite(
        src, opset=opset, custom_rewriter=[rewrite_rfft_real_part],
        output_path=dst)
    print("tf2onnx", tf2onnx.__version__, "opset", opset, "nodes",
          len(model_proto.graph.node))
    print("inputs", [(i.name, dims(i)) for i in model_proto.graph.input])
    print("outputs", [(o.name, dims(o)) for o in model_proto.graph.output])


if __name__ == "__main__":
    main()
