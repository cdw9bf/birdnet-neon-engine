"""Canonicalization: exact, local rewrites of exporter artifacts.

Rewrites what tf2onnx and friends leave behind into the forms the recognizer
matches; runs before recognition. Every rewrite preserves the graph function
exactly (up to fp32 reduction order for the GlobalAveragePool one); prep.load
verifies the outputs with ONNX Runtime before and after and refuses the model
if they differ.

  1. Unsqueeze -> MatMul(const 2-D) -> Squeeze (a [.., N] x [N, K] product
     wrapped in a unit axis) -> MatMul
  2. Cast float32 -> float32 (left behind when a complex cast loses its
     complex producer) -> removed
  3. ReverseSequence with full-length sequences along time_axis
     -> Slice(step -1)
  4. a constant zero Pad on H/W whose only consumer is a Conv
     -> the Conv's pads
  5. GlobalAveragePool -> ReduceMean(H, W)
  6. a node none of whose outputs is consumed or a graph output (dead code
     left by the exporter) -> removed
"""
import numpy as np
from onnx import helper

from . import onnxg


def canonicalize(g: onnxg.Graph) -> list[tuple[str, int]]:
    """Rewrites g.nodes in place until no rule fires.

    Args:
        g: the onnxg.Graph, with shapes and dtypes (run_shapes) known.

    Returns:
        A list of (rule name, count) for the rules that fired, in RULES order.

    Raises:
        RuntimeError: the rules did not converge in 8 rounds.
    """
    counts = {}
    for _ in range(8):
        fired = 0
        for name, fn in RULES:
            k = fn(g)
            if k:
                counts[name] = counts.get(name, 0) + k
                g.index()
                fired += k
        if not fired:
            break
    else:
        raise RuntimeError("canonicalization did not converge in 8 rounds "
                           "(a rule re-creates what another removes)")
    return [(name, counts[name]) for name, _ in RULES if name in counts]


def _text(v):
    """A string attribute's value (onnx returns bytes) as str."""
    return v.decode() if isinstance(v, bytes) else v


def _drop(g, nodes):
    """Removes the given nodes from g.nodes (by identity)."""
    ids = {id(n) for n in nodes}
    g.nodes = [n for n in g.nodes if id(n) not in ids]


def _rewire(g, old, new):
    """Makes every consumer of tensor `old` read `new` instead.

    Graph outputs keep their names: callers avoid them.
    """
    for n in g.nodes:
        for i, x in enumerate(n.input):
            if x == old:
                n.input[i] = new


def _matmul_unit_axes(g):
    """Rule 1: Unsqueeze -> MatMul(const 2-D) -> Squeeze becomes the MatMul.

    Returns the number of rewrites (as every rule).
    """
    k = 0
    for mm in list(g.nodes):
        if (mm.op_type != "MatMul" or mm.input[1] not in g.const
                or np.asarray(g.const[mm.input[1]]).ndim != 2):
            continue
        un = g.prod.get(mm.input[0])
        sq = g.single_cons(mm.output[0])
        if (not un or un.op_type != "Unsqueeze" or not sq
                or sq.op_type != "Squeeze" or mm.output[0] in g.outputs
                or un.output[0] in g.outputs):
            continue
        si = g.shape.get(un.input[0])
        so = g.shape.get(un.output[0])
        mo = g.shape.get(mm.output[0])
        qo = g.shape.get(sq.output[0])
        if (not (si and so and mo and qo) or len(so) != len(si) + 1
                or so[-1] != si[-1] or so[-2] != 1 or so[:-2] != si[:-1]):
            continue  # the unit axis must sit just before the contracted one
        if (qo != si[:-1] + [mo[-1]]
                or len(g.cons.get(un.output[0], [])) != 1):
            continue
        mm.input[0] = un.input[0]
        mm.output[0] = sq.output[0]
        _drop(g, [un, sq])
        g.index()
        k += 1
    return k


def _identity_cast(g):
    """Rule 2: a Cast float32 -> float32 is removed."""
    k = 0
    for n in list(g.nodes):
        if (n.op_type != "Cast" or g.attrs(n).get("to") != 1
                or n.output[0] in g.outputs):
            continue
        if g.dtype.get(n.input[0]) != np.float32:
            continue
        _rewire(g, n.output[0], n.input[0])
        _drop(g, [n])
        g.index()
        k += 1
    return k


def _reverse_sequence(g):
    """Rule 3: a full-length ReverseSequence becomes a Slice with step -1."""
    k = 0
    for n in list(g.nodes):
        if n.op_type != "ReverseSequence" or n.input[1] not in g.const:
            continue
        a = g.attrs(n)
        ta = a.get("time_axis", 0)
        sh = g.shape.get(n.input[0])
        ln = np.asarray(g.const[n.input[1]]).reshape(-1)
        if not sh or not np.all(ln == sh[ta]):
            continue  # partial reversal: not a plain reverse
        base = f"canon/{n.name or n.output[0]}"
        g.const[base + "/starts"] = np.array([-1], np.int64)
        g.const[base + "/ends"] = np.array([-(1 << 63)], np.int64)
        g.const[base + "/axes"] = np.array([ta], np.int64)
        g.const[base + "/steps"] = np.array([-1], np.int64)
        sl = helper.make_node("Slice",
                              [n.input[0], base + "/starts", base + "/ends",
                               base + "/axes", base + "/steps"],
                              [n.output[0]], name=base + "/Slice")
        g.nodes[g.nodes.index(n)] = sl
        g.index()
        k += 1
    return k


def _pad_into_conv(g):
    """Rule 4: a zero spatial Pad feeding only a Conv joins the Conv's pads."""
    k = 0
    for n in list(g.nodes):
        if (n.op_type != "Pad" or len(n.input) < 2 or n.input[1] not in g.const
                or (len(n.input) > 3 and n.input[3])):
            # (an `axes` input, opset 18+, permutes the pads list: not handled)
            continue
        a = g.attrs(n)
        if _text(a.get("mode", b"constant")) != "constant":
            continue
        if len(n.input) > 2 and n.input[2] and (
                n.input[2] not in g.const
                or float(np.asarray(g.const[n.input[2]]).reshape(-1)[0]) != 0):
            continue
        pv = [int(x) for x in np.asarray(g.const[n.input[1]]).reshape(-1)]
        r = len(pv) // 2
        if r != 4 or any(pv[:2]) or any(pv[4:6]) or min(pv) < 0:
            continue  # NCHW, spatial dims only
        conv = g.single_cons(n.output[0])
        if (not conv or conv.op_type != "Conv" or conv.input[0] != n.output[0]
                or n.output[0] in g.outputs):
            continue
        ca = g.attrs(conv)
        if _text(ca.get("auto_pad", b"NOTSET")) != "NOTSET":
            continue
        pads = list(ca.get("pads", [0, 0, 0, 0]))
        pads = [pads[0] + pv[2], pads[1] + pv[3], pads[2] + pv[6],
                pads[3] + pv[7]]
        for at in list(conv.attribute):
            if at.name == "pads":
                conv.attribute.remove(at)
        conv.attribute.append(helper.make_attribute("pads", pads))
        conv.input[0] = n.input[0]
        _drop(g, [n])
        g.index()
        k += 1
    return k


def _gap(g):
    """Rule 5: a 4-D GlobalAveragePool becomes a ReduceMean over H and W."""
    k = 0
    for n in list(g.nodes):
        if n.op_type != "GlobalAveragePool":
            continue
        sh = g.shape.get(n.input[0])
        if not sh or len(sh) != 4:
            continue
        base = f"canon/{n.name or n.output[0]}"
        if g.opset >= 18:
            g.const[base + "/axes"] = np.array([2, 3], np.int64)
            rm = helper.make_node("ReduceMean", [n.input[0], base + "/axes"],
                                  [n.output[0]], name=base + "/ReduceMean",
                                  keepdims=1)
        else:
            rm = helper.make_node("ReduceMean", [n.input[0]], [n.output[0]],
                                  name=base + "/ReduceMean", axes=[2, 3],
                                  keepdims=1)
        g.nodes[g.nodes.index(n)] = rm
        g.index()
        k += 1
    return k


def _dead(g):
    """Rule 6: a node whose outputs nothing reads is removed."""
    k = 0
    for n in list(g.nodes):
        if any(o and (o in g.cons or o in g.outputs) for o in n.output):
            continue
        _drop(g, [n])
        g.index()
        k += 1
    return k


# (name, rewrite) in the order they run; names are what `bnc report` prints.
RULES = (("Unsqueeze/Squeeze around a MatMul", _matmul_unit_axes),
         ("float->float Cast", _identity_cast),
         ("ReverseSequence -> Slice", _reverse_sequence),
         ("zero Pad folded into a Conv", _pad_into_conv),
         ("GlobalAveragePool -> ReduceMean", _gap),
         ("dead node removed", _dead))
