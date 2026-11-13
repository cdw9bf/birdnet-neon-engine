"""ONNX import for the bnneon compiler.

Nodes, constants (initializers, Constant nodes, folded subgraphs), producer /
consumer maps and concrete shapes (one ONNX Runtime run with every
intermediate exposed, batch 1).
"""
import functools

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx import reference
import onnxruntime as ort

from .errs import ModelError


def _load(path):
    """Loads ONNX file path.

    Raises:
        ModelError: the file is missing or not a loadable ONNX model.
    """
    try:
        return onnx.load(path)
    except FileNotFoundError as e:
        raise ModelError(f"{path}: file not found") from e
    except Exception as e:
        raise ModelError(
            f"{path}: not a loadable ONNX file ({e}); hint: a .tflite "
            f"model must first be converted (tf2onnx, guide section 3); "
            f"check the file is complete (sha256)") from e


def _window_input(path, inputs):
    """The name and sample count of the graph's single PCM input.

    Args:
        path: the ONNX file (for messages).
        inputs: the graph inputs that are not initializers.

    Returns:
        (input name, n_samples).

    Raises:
        ModelError: not exactly one fixed-length float32 [1, n] or [n] input.
    """
    if inputs and inputs[0].type.tensor_type.elem_type not in (
            0, onnx.TensorProto.FLOAT):
        raise ModelError(
            f"{path}: input '{inputs[0].name}' is not float32; "
            f"hint: the engine feeds float32 PCM in [-1, 1)")
    if len(inputs) != 1:
        raise ModelError(
            f"{path}: expected exactly one graph input (the PCM window), "
            f"found {[i.name for i in inputs]}; hint: export the "
            f"model with the audio window as its only input (no state / "
            f"length inputs)")
    inp = inputs[0].name
    dims = inputs[0].type.tensor_type.shape.dim
    nsamp = dims[-1].dim_value if len(dims) else 0
    if nsamp <= 0 or len(dims) not in (1, 2) or (
            len(dims) == 2 and dims[0].dim_value not in (0, 1)):
        raise ModelError(
            f"{path}: input '{inp}' has shape "
            f"{[d.dim_value or d.dim_param or '?' for d in dims]}; "
            f"expected [1, n_samples] (or [n_samples]) float32 PCM with a "
            f"fixed n_samples; hint: export with a fixed window length "
            f"(e.g. 144000 for 3 s at 48 kHz)")
    return inp, nsamp


def _head_name(x, at, head_in, pre):
    """Name in the joined graph of tensor x of an appended head."""
    if x == head_in:
        return at
    return pre + x if x else x


def set_outputs(g: onnx.GraphProto, outputs: list[str]) -> None:
    """Makes these tensors (graph outputs or internal ones) g's outputs.

    ONNX Runtime then returns any of them, e.g. the logits before a final
    Sigmoid that the exported model does not output.
    """
    del g.output[:]
    for o in outputs:
        g.output.append(helper.make_empty_tensor_value_info(o))


class Graph:
    """An ONNX graph as the compiler sees it.

    Attributes:
        path: the ONNX file.
        opset: the default-domain opset version.
        const: tensor name -> numpy array of every constant (initializers,
            Constant nodes, folded nodes).
        nodes: the non-constant nodes in graph order.
        inp: the name of the single graph input (the PCM window).
        nsamp: its number of samples.
        outputs: the graph output names (with any appended head's).
        prod, cons, pos: producer node, consumer nodes and node position
            maps (index()).
        shape, dtype: tensor name -> concrete shape / numpy dtype
            (run_shapes()).
        canon, canon_err: the canonicalization rewrites as (rule, count) and
            their relative max output change (set by prep.load).
    """

    def __init__(self, path: str,
                 append: list[tuple[str, str]] | None = None):
        """Loads and checks the model, appending any heads.

        Args:
            path: the ONNX file.
            append: optional list of (head_model_path, at_tensor): the head's
                single input is wired to at_tensor of this graph and the
                head's outputs become graph outputs (BattyBirdNET: embeddings
                model + dense head).

        Raises:
            ModelError: the file is missing, not ONNX, or its input is not one
                fixed-length float32 window.
        """
        m = _load(path)
        self.path = path
        g = m.graph
        self.opset = max(o.version for o in m.opset_import
                         if o.domain in ("", "ai.onnx"))
        self.const = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        self.nodes = []
        for n in g.node:
            if n.op_type == "Constant":
                self.const[n.output[0]] = numpy_helper.to_array(
                    helper.get_attribute_value(n.attribute[0]))
            else:
                self.nodes.append(n)
        inits = {t.name for t in g.initializer}
        self.inputs = [i for i in g.input if i.name not in inits]
        self.inp, self.nsamp = _window_input(path, self.inputs)
        self.outputs = [o.name for o in g.output]
        self.model = m
        self.appended = []
        self.shape = {}  # tensor name -> concrete shape (run_shapes)
        self.dtype = {}  # tensor name -> numpy dtype (run_shapes)
        # (rule, count) of the canonicalization rewrites (prep.load)
        self.canon = []
        self.canon_err = 0.0  # their relative max output change (prep.load)
        for hp, at in append or []:
            self._append_head(hp, at)
        self.index()

    def _append_head(self, hp, at):
        """Joins head model hp: its single input reads tensor at.

        The head's tensors are renamed "head<n>/..." and its outputs become
        graph outputs.
        """
        hm = onnx.load(hp)
        hg = hm.graph
        hinits = {t.name for t in hg.initializer}
        hin = [i.name for i in hg.input if i.name not in hinits]
        assert len(hin) == 1
        pre = f"head{len(self.appended)}/"
        ren = functools.partial(_head_name, at=at, head_in=hin[0], pre=pre)
        for t in hg.initializer:
            self.const[pre + t.name] = numpy_helper.to_array(t)
        for n in hg.node:
            if n.op_type == "Constant":
                self.const[pre + n.output[0]] = numpy_helper.to_array(
                    helper.get_attribute_value(n.attribute[0]))
                continue
            nn = helper.make_node(n.op_type, [ren(x) for x in n.input],
                                  [ren(x) for x in n.output],
                                  name=pre + (n.name or n.op_type),
                                  domain=n.domain)
            nn.attribute.extend(n.attribute)
            self.nodes.append(nn)
        self.appended.append((hp, at, [ren(o.name) for o in hg.output]))
        self.outputs += [ren(o.name) for o in hg.output]

    def index(self):
        """Rebuilds the producer, consumer and position maps of self.nodes."""
        self.prod = {}
        for n in self.nodes:
            for o in n.output:
                self.prod[o] = n
        self.cons = {}
        for n in self.nodes:
            # a node reading a tensor twice (x * x) is one consumer
            for x in dict.fromkeys(n.input):
                self.cons.setdefault(x, []).append(n)
        self.pos = {id(n): i for i, n in enumerate(self.nodes)}

    def attrs(self, n):
        """Returns node n's attributes as a name -> value dict."""
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    def fold(self) -> int:
        """Constant folding with the onnx reference evaluator.

        Every node whose inputs are all constant is evaluated, and Shape nodes
        of tensors with a known shape become constants.

        Returns:
            The number of nodes folded.
        """
        changed, keep = 0, []
        shp = self.shape
        for n in self.nodes:
            ins = [x for x in n.input if x]
            # static window: every shape is known (batch 1)
            if n.op_type == "Shape" and ins and ins[0] in shp:
                a = self.attrs(n)
                full = np.array(shp[ins[0]], np.int64)
                self.const[n.output[0]] = full[a.get("start", 0):
                                               a.get("end", len(full))]
                changed += 1
                continue
            if ins and all(x in self.const for x in ins):
                g = helper.make_graph(
                    [n], "f",
                    [helper.make_tensor_value_info(
                        x, onnx.TensorProto.UNDEFINED, None) for x in ins],
                    [helper.make_tensor_value_info(
                        o, onnx.TensorProto.UNDEFINED, None)
                     for o in n.output])
                mm = helper.make_model(
                    g, opset_imports=[helper.make_opsetid("", self.opset)])
                try:
                    outs = reference.ReferenceEvaluator(mm).run(
                        None, {x: self.const[x] for x in ins})
                except Exception:  # pylint: disable=broad-exception-caught
                    # leave it; the matcher reports it if it matters
                    keep.append(n)
                    continue
                for o, v in zip(n.output, outs):
                    self.const[o] = np.asarray(v)
                changed += 1
            else:
                keep.append(n)
        self.nodes = keep
        self.index()
        return changed

    def session(self, outputs: list[str]) -> ort.InferenceSession:
        """Builds an ONNX Runtime session of the current graph.

        No graph optimization, CPU provider.

        Args:
            outputs: the tensor names the session outputs.

        Returns:
            The onnxruntime.InferenceSession.

        Raises:
            ModelError: ONNX Runtime cannot load the graph.
        """
        m = onnx.ModelProto()
        m.CopyFrom(self.model)
        g = m.graph
        # always from the current node list (appended heads, canonicalized
        # nodes) and the constant table
        del g.node[:]
        g.node.extend(self.nodes)
        del g.value_info[:]
        known = {t.name for t in g.initializer}
        for k, v in self.const.items():
            if k not in known:
                g.initializer.append(numpy_helper.from_array(np.asarray(v), k))
        set_outputs(g, outputs)
        so = ort.SessionOptions()
        so.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        so.log_severity_level = 3
        try:
            return ort.InferenceSession(m.SerializeToString(), so,
                                        providers=["CPUExecutionProvider"])
        except Exception as e:
            raise ModelError(
                f"{self.path}: ONNX Runtime cannot load the graph "
                f"({str(e).splitlines()[0]}); hint: run "
                f"onnx.checker.check_model and fix the export before "
                f"compiling") from e

    def run_shapes(self, x: np.ndarray | None = None,
                   want_values=()) -> dict[str, np.ndarray]:
        """Runs the graph once and records every tensor's shape and dtype.

        Args:
            x: the input window, [1, nsamp] float32 (default: seeded noise).
            want_values: names of the tensors whose values to return.

        Returns:
            A dict name -> value for the names in want_values that the run
            computed.

        Raises:
            ModelError: ONNX Runtime cannot load or run the graph.
        """
        # every node output once, in node order (a dict keeps the order)
        names = list(dict.fromkeys(o for n in self.nodes for o in n.output
                                   if o and o not in self.const))
        s = self.session(names)
        if x is None:
            x = (np.random.default_rng(0).standard_normal((1, self.nsamp))
                 * 0.05).astype(np.float32)
        try:
            res = dict(zip(names, s.run(names, {self.inp: x})))
        except Exception as e:
            raise ModelError(
                f"{self.path}: ONNX Runtime fails to run the graph on a window "
                f"({str(e).splitlines()[0]})") from e
        self.shape = {k: list(v.shape) for k, v in res.items()}
        self.shape.update({k: list(np.asarray(v).shape)
                           for k, v in self.const.items()})
        self.shape[self.inp] = [1, self.nsamp]
        self.dtype = {k: np.asarray(v).dtype for k, v in res.items()}
        self.dtype.update({k: np.asarray(v).dtype
                           for k, v in self.const.items()})
        self.dtype[self.inp] = np.dtype(np.float32)
        return {k: res[k] for k in want_values if k in res}

    def ancestors(self, tensors) -> set[int]:
        """Returns the ids of every node the given tensors depend on."""
        seen, todo = set(), list(tensors)
        while todo:
            n = self.prod.get(todo.pop())
            if n is None or id(n) in seen:
                continue
            seen.add(id(n))
            todo.extend(x for x in n.input if x)
        return seen

    def single_cons(self, t):
        """Returns the only consumer of tensor t (None if not exactly one)."""
        c = self.cons.get(t, [])
        return c[0] if len(c) == 1 else None

    def cval(self, n):
        """Returns the single constant input of node n (None if not one)."""
        r = [self.const[x] for x in n.input if x in self.const]
        return r[0] if len(r) == 1 else None

    def other_in(self, n, t):
        """Returns node n's input other than t.

        t itself when every input is t, None when there are several others.
        """
        r = [x for x in n.input if x != t]
        if len(r) == 1:
            return r[0]
        if not r:
            return t
        return None

    def where(self, n):
        """Returns "node <position> <op> '<name>'" for messages."""
        return f"node {self.pos.get(id(n), '?')} {n.op_type} '{n.name}'"
