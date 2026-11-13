"""Dump an ONNX graph (BirdNET v2.4, Perch v2, ...) with concrete shapes (all
intermediates exposed and run on one window), conv geometry and MACs.

usage: python -I dump_graph.py <model.onnx> <out.txt> <compact.txt>

The input name and window length are read from the model's first input.
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

# shape plumbing: listed in the compact dump only when the output has rank > 1
SHAPE_OPS = ("Shape", "Gather", "Unsqueeze", "Concat", "Range", "Cast",
             "Squeeze")


def intermediate_names(graph):
    """Every node output, in graph order, once."""
    names = []
    for n in graph.node:
        for o in n.output:
            if o and o not in names:
                names.append(o)
    return names


def run_all(m, names):
    """Runs the model with every intermediate exposed on one fixed-seed window.

    Returns:
        (input name, window length, {name: array}).
    """
    g = m.graph
    del g.output[:]
    for o in names:
        g.output.append(onnx.helper.make_empty_tensor_value_info(o))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so,
                             providers=["CPUExecutionProvider"])
    inp = m.graph.input[0].name
    nsamp = m.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    x = np.random.default_rng(0).standard_normal((1, nsamp)) * 0.05
    x = x.astype(np.float32)
    return inp, nsamp, dict(zip(names, s.run(names, {inp: x})))


def conv_entry(i, n, shp, init, att):
    """(MACs, compact line) of a Conv."""
    w = init[n.input[1]]
    o = shp[n.output[0]]
    mac = int(np.prod(o[1:])) * int(np.prod(w.shape[1:]))
    k = w.shape
    line = (f"{i:3d} Conv  in {shp[n.input[0]]} w {list(k)} "
            f"g {att.get('group',1)} s {att.get('strides')} "
            f"p {att.get('pads')} out {o} MAC {mac/1e6:.1f}M  {n.name}\n")
    return mac, line


def matmul_entry(i, n, shp):
    """(MACs, compact line) of a MatMul or Gemm."""
    a, b = shp[n.input[0]], shp[n.input[1]]
    batch = int(np.prod(b[:-2])) if len(b) > 2 and len(a) < len(b) else 1
    mac = int(np.prod(a)) * int(b[-1]) * batch
    return mac, f"{i:3d} {n.op_type} {a} x {b} MAC {mac/1e6:.1f}M  {n.name}\n"


def compact_entry(i, n, shp, init, att):
    """(MACs, the compact listing line or None) of node i."""
    if n.op_type == "Conv":
        return conv_entry(i, n, shp, init, att)
    if n.op_type in ("MatMul", "Gemm"):
        return matmul_entry(i, n, shp)
    if n.op_type not in SHAPE_OPS or len(shp.get(n.output[0], [])) > 1:
        return 0, (f"{i:3d} {n.op_type:10s} {[shp.get(x) for x in n.input]} "
                   f"-> {shp.get(n.output[0])}  {n.name}\n")
    return 0, None


def full_entry(i, n, shp, att, mac):
    """The full listing of node i: inputs, outputs, attributes, MACs."""
    sa = {k: (v if not isinstance(v, bytes) else v.decode())
          for k, v in att.items() if k != "value"}
    return (f"{i:3d} {n.op_type:18s} {n.name}\n"
            f"     in  {[(x, shp.get(x)) for x in n.input]}\n"
            f"     out {[(x, shp.get(x)) for x in n.output]}\n"
            f"     att {sa} MAC {mac/1e6:.2f}M\n")


def main():
    m = onnx.load(sys.argv[1])
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    names = intermediate_names(m.graph)
    inp, nsamp, res = run_all(m, names)
    shp = {k: list(v.shape) for k, v in res.items()}
    shp.update({k: list(v.shape) for k, v in init.items()})
    shp[inp] = [1, nsamp]
    tot = 0
    with open(sys.argv[2], "w", encoding="utf-8") as out, \
            open(sys.argv[3], "w", encoding="utf-8") as cmp_:
        for i, n in enumerate(m.graph.node):
            att = {a.name: onnx.helper.get_attribute_value(a)
                   for a in n.attribute}
            mac, line = compact_entry(i, n, shp, init, att)
            if line:
                cmp_.write(line)
            tot += mac
            out.write(full_entry(i, n, shp, att, mac))
        out.write(f"total MAC {tot/1e9:.3f} G\n")
        cmp_.write(f"total MAC {tot/1e9:.3f} G\n")
    print("total MAC", tot / 1e9)


if __name__ == "__main__":
    main()
