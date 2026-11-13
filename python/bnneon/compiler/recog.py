"""Pattern recognition: folded ONNX graph -> fused IR.

Covers the front ends, convs with folded BN + activation + residual, SE,
pools, means, dense heads and the ProtoPNet head. Everything is matched by
structure (op types, constant shapes, verified values), not by node names.
Any node the walker cannot place is reported as "unsupported op X at node Y
(reason)".

IR values: Val(kind='pcm'|'act'|'vec', H, W, C, layout 'nchw'|'nhwc' for
acts). IR ops: dicts with 'op', 'ins' (value ids), 'out', parameters and
'nodes' (the ONNX node names they cover).
"""
import numpy as np


class Unsupported(Exception):
    """Why a node cannot be placed.

    Attributes:
        node: the ONNX node the matcher stopped at (None: the recognizer
            fills it in).
    """

    def __init__(self, msg, node=None):
        super().__init__(msg)
        self.node = node


# Op types the body recognizer dispatches on or absorbs into a pattern (swish,
# SE). The census after a failed front end lists the others, so a reader sees
# the size of the gap behind it.
BODY_OPS = {
    "Transpose", "Reshape", "Unsqueeze", "Squeeze", "Flatten", "Conv",
    "ReduceMean", "ReduceSum", "Mul", "Add", "Relu", "MaxPool", "AveragePool",
    "Concat", "MatMul", "Gemm", "Sigmoid", "Neg", "Exp", "Div"
}

# The nearest supported pattern per ONNX op type and where to add a new one
# (guide = docs/ADDING_A_MODEL.md).
NEAREST = {
    "Conv": (
        "dense 1x1 conv (any channels, folded BN, swish|relu, residual), "
        "depthwise kxk (k 3|5|7, stride 1|2, SAME or "
        "VALID pads, swish), a 1-channel kxk VALID stem, or the v2.4 4x8 stem",
        "add a recognizer rule in python/bnneon/compiler/recog.py r_conv and "
        "a lowering in python/bnneon/compiler/lower.py l_conv / l_dwconv "
        "(guide section 6); a new geometry also needs a kernel in "
        "src/rt_ops_f32.c / rt_ops_f16.c"
    ),
    "MatMul": (
        "dense head MatMul [1024][N] (+ Add bias) on the pooled 1024 vector; "
        "the SE and prototype-head MatMuls",
        "a head on a vector of another width needs a kernel: see the DENSE "
        "kernel in src/rt_ops_f32.c (guide section 6)"
    ),
    "Gemm": (
        "Gemm with default attributes (alpha 1, beta 1, no transpose, no bias "
        "input) on the pooled vector",
        "export the head as MatMul + Add, or extend rule() in "
        "python/bnneon/compiler/recog.py to fold transB / the bias input"
    ),
    "ReduceMean": (
        "spatial mean over H and W (axes [2,3] in NCHW, [1,2] in NHWC) "
        "producing the embedding vector",
        "other reductions need a recognizer rule "
        "(python/bnneon/compiler/recog.py r_mean) and a kernel"
    ),
    "ReduceSum": (
        "ReduceSum / (H*W) as a spatial mean; the L2 norm inside a prototype "
        "head",
        "other reductions need a recognizer rule "
        "(python/bnneon/compiler/recog.py r_mean) and a kernel"
    ),
    "Mul": (
        "per-channel affine (BN) after a conv, swish (Sigmoid * x), the SE "
        "gate, the window / square of a front end",
        "a Mul by a non-per-channel constant or between two activations has no "
        "kernel: add one (guide section 6)"
    ),
    "Add": (
        "per-channel bias after a conv, residual Add of two activations of "
        "equal shape, the +1 of a sigmoid",
        "an Add between activations of different shapes has no kernel"),
    "Sigmoid": (
        "swish (Sigmoid * x) and the SE gate; a final Sigmoid on the logits is "
        "applied by the host, not the engine",
        "if this is the output activation: compile with --logits <its input "
        "tensor> --post sigmoid"
    ),
    "Softmax": ("none in the engine: a final Softmax is applied by the host",
                "compile with --logits <its input tensor> --post softmax"),
    "Relu": (
        "relu after a conv or a per-channel affine, and on a vector between "
        "two dense layers (a hidden layer, "
        "BirdNET-Analyzer 'hidden units')", "relu elsewhere needs a kernel"),
    "GlobalAveragePool":
        ("spatial mean (canonicalized to ReduceMean over H, W)",
         "a GlobalAveragePool on a non-4-D tensor"),
    "Pad": (
        "a constant zero Pad on H/W is folded into the Conv that consumes it "
        "(canonicalization)",
        "other pads: re-export, or extend python/bnneon/compiler/canon.py"),
    "Cast": ("float -> float casts are removed (canonicalization)",
             "a real dtype change has no kernel: export fp32"),
    "ReverseSequence": (
        "a full-length ReverseSequence is canonicalized to a reverse Slice "
        "(the v2.4 mel reverse)",
        "a partial reversal is not a mel reverse"),
    "Unsqueeze": (
        "layout-only; the Unsqueeze/Squeeze pair around a MatMul is removed by "
        "canonicalization",
        "another unit-axis insertion is not representable"),
    "Squeeze": (
        "layout-only; the Unsqueeze/Squeeze pair around a MatMul is removed by "
        "canonicalization",
        "another unit-axis removal is not representable"),
    "Transpose": ("layout-only transposes NCHW <-> NHWC of an activation",
                  "other permutations are not representable"),
    "Reshape": (
        "layout-only reshapes (spectrogram to image, pooled activation to "
        "vector)",
        "other reshapes are not representable"),
    "Concat": ("channel concat of activations",
               "concat along time/frequency has no kernel"),
    "MaxPool": ("the v2.4 1x2 max|avg pool pair after the stem",
                "other pools need a kernel"),
    "AveragePool": ("the v2.4 1x2 max|avg pool pair after the stem",
                    "other pools need a kernel"),
    "BatchNormalization": (
        "BN folded into the preceding conv (the exporter normally folds it)",
        "export with BN folded (onnxsim / tf2onnx --fold_const), or add a rule "
        "folding the BN node"
    ),
}
NEAREST_DEFAULT = (
    "no similar pattern in the engine",
    "a new recognizer rule (python/bnneon/compiler/recog.py), a lowering "
    "(python/bnneon/compiler/lower.py) and probably a kernel "
    "(src/rt_ops_f32.c, src/rt_ops_f16.c) are needed: guide section 6")

# Scalar / per-channel arithmetic codes of the dB-image front end (the post
# and chan lists of FE_DBIMG).
ARITH_CODE = {"Mul": 1, "Div": 2, "Add": 3, "Sub": 4}

# Layout-only ops a matcher may step over between two pattern nodes.
LAYOUT_OPS = ("Transpose", "Reshape", "Squeeze", "Unsqueeze", "Flatten")


def first_float(v):
    """Returns the first element of a constant as a Python float."""
    return float(np.asarray(v).reshape(-1)[0])


def decode_attr(v):
    """Returns an ONNX string attribute as str (bytes are decoded)."""
    return v.decode() if isinstance(v, bytes) else v


def slice_indices(dim, start, end, step):
    """Indices an ONNX Slice selects along one axis.

    Args:
        dim: length of the axis.
        start: Slice start (may be negative or a huge sentinel).
        end: Slice end (may be negative or a huge sentinel).
        step: Slice step (negative for a reversal).

    Returns:
        The selected indices of range(dim), in output order.
    """
    lo = max(start, -(1 << 62))
    hi = max(min(end, 1 << 62), -(1 << 62))
    return list(range(dim))[slice(lo, hi, step)]


def take(nodes, n):
    """Records node n as matched and returns its first output tensor."""
    nodes.append(n)
    return n.output[0]


def is_windowed_dft(w, win, n_fft, n_bins, rtol):
    """Whether a conv kernel stacks window * cos and window * sin DFT rows.

    Args:
        w: the kernel [2 * n_bins][1][n_fft].
        win: the window (row 0 of the kernel), float64.
        n_fft: DFT length.
        n_bins: n_fft / 2 + 1.
        rtol: tolerance relative to the largest window value.

    Returns:
        True when rows 0..n_bins are window * cos and the rest are
        +/- window * sin, within the tolerance.
    """
    f = np.arange(n_bins)[:, None]
    k = np.arange(n_fft)[None, :]
    tol = rtol * max(1e-12, float(np.abs(win).max()))
    cos_rows = win * np.cos(2 * np.pi * f * k / n_fft)
    if np.abs(w[:n_bins, 0] - cos_rows).max() > tol:
        return False
    sin_rows = np.abs(win * np.sin(2 * np.pi * f * k / n_fft))
    return not np.abs(np.abs(w[n_bins:, 0]) - sin_rows).max() > tol


def einsum_is_matmul(eq):
    """Whether an Einsum equation is a matrix product x[.., F] . M[F, nmel].

    Args:
        eq: the equation without spaces, e.g. "bft,fm->btm".
    """
    lhs, rhs = eq.split("->") if "->" in eq else (eq, "")
    terms = lhs.split(",")
    if len(terms) != 2 or len(terms[1]) != 2:
        return False
    return terms[0][-1] == terms[1][0] and rhs == terms[0][:-1] + terms[1][1]


def stft2_image_layout(cshape, nmel):
    """Image size and layout tag of the dual-STFT Concat output.

    Args:
        cshape: shape of the Concat output, [mel][time][1][C] or
            [1][time][mel][C].
        nmel: number of mel rows.

    Returns:
        (H, W, layout tag 'hw1c' | '1whc').

    Raises:
        Unsupported: any other layout.
    """
    if cshape[0] == nmel and cshape[2] == 1:
        return cshape[0], cshape[1], "hw1c"  # [H=mel][W=time][1][C]
    if cshape[0] == 1 and cshape[2] == nmel:
        return cshape[2], cshape[1], "1whc"  # [1][W=time][H=mel][C]
    raise Unsupported(f"spectrogram image layout {cshape}")


def conv_pads(attrs, in_hw, out_hw, kernel, strides):
    """Explicit pads of a 2-D conv.

    Args:
        attrs: the Conv attributes (auto_pad, pads).
        in_hw: (H, W) of the input.
        out_hw: (H, W) of the output.
        kernel: (kh, kw).
        strides: (sh, sw).

    Returns:
        [top, left, bottom, right].
    """
    auto_pad = decode_attr(attrs.get("auto_pad", b"NOTSET"))
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        ph = max((out_hw[0] - 1) * strides[0] + kernel[0] - in_hw[0], 0)
        pw = max((out_hw[1] - 1) * strides[1] + kernel[1] - in_hw[1], 0)
        if auto_pad == "SAME_UPPER":
            return [ph // 2, pw // 2, ph - ph // 2, pw - pw // 2]
        return [ph - ph // 2, pw - pw // 2, ph // 2, pw // 2]
    if auto_pad == "VALID":
        return [0, 0, 0, 0]
    return list(attrs.get("pads", [0, 0, 0, 0]))


class Val:
    """An IR value: the result of one IR op.

    Attributes:
        id: unique value id (the 'ins' / 'out' entries of the IR ops).
        kind: 'pcm', 'act' (an activation H x W x C) or 'vec' (C values).
        H: rows of an activation (1 for vectors).
        W: columns of an activation (1 for vectors).
        C: channels of an activation, length of a vector.
        layout: how the ONNX tensor stores an activation ('nchw', 'nhwc',
            or a front-end layout); None for vectors.
    """
    _n = 0

    def __init__(self, kind, h=1, w=1, c=1, layout=None):
        Val._n += 1
        self.id = Val._n
        self.kind = kind
        # H, W, C are read by the lowering under these names.
        self.H = h  # pylint: disable=invalid-name
        self.W = w  # pylint: disable=invalid-name
        self.C = c  # pylint: disable=invalid-name
        self.layout = layout

    def __repr__(self):
        if self.kind == "act":
            return f"v{self.id}:{self.kind}[{self.H}x{self.W}x{self.C}]"
        return f"v{self.id}:{self.kind}[{self.C}]"


class Recognizer:
    """Walks a folded ONNX graph and emits the IR ops that cover it.

    Attributes:
        g: the graph (onnxg.Graph).
        fold_bn: fold a per-channel Mul + Add after a conv into the conv
            (Perch); off: they are left as ops.
        val: ONNX tensor name -> Val.
        ops: the IR ops emitted so far.
        covered: ids of the ONNX nodes some IR op covers.
        problems: (node, reason) for every node that could not be placed.
        at: the node a multi-node matcher last looked at (blamed when it
            raises without a node).
        fe_failed: the node the front-end matcher stopped at, when no front
            end matched.
    """

    def __init__(self, g, log=print, fold_bn: bool = True):
        self.g = g
        self.fold_bn = fold_bn
        self.log = log
        self.val = {}
        self.ops = []
        self.covered = set()
        self.problems = []
        self.at = None
        self.fe_failed = None

    # ---- helpers
    def cov(self, *nodes):
        """Marks ONNX nodes as covered by an IR op."""
        for n in nodes:
            self.covered.add(id(n))

    def only_cons(self, t, op=None):
        """The single consumer of tensor t.

        Args:
            t: tensor name.
            op: an op type or a tuple of op types the consumer must have
                (None: any).

        Returns:
            The consuming node, or None when t has no consumer, several, or
            one of another type. The first consumer is remembered in self.at
            either way.
        """
        c = self.g.cons.get(t, [])
        if c:
            self.at = c[0]
        if len(c) != 1:
            return None
        if op and c[0].op_type not in (op if isinstance(op, tuple) else (op,)):
            return None
        return c[0]

    def const_scalar(self, n):
        """The scalar constant operand of node n as a float, or None."""
        v = self.g.cval(n)
        return None if v is None or v.size != 1 else float(v.reshape(-1)[0])

    def chan_vec(self, v, n_ch, layout):
        """A per-channel constant broadcast against an activation.

        Args:
            v: the constant (None allowed).
            n_ch: channels of the activation.
            layout: 'nchw' or 'nhwc', the layout of the activation.

        Returns:
            The constant as [n_ch] values, or None when it is not a
            per-channel constant in that layout.
        """
        if v is None:
            return None
        a = np.asarray(v)
        if a.size == 1:
            return np.full(n_ch, float(a.reshape(-1)[0]))
        if a.size != n_ch:
            return None
        sh = list(a.shape)
        if layout == "nchw" and len(sh) == 4 and sh[1] == n_ch:
            return a.reshape(-1)
        if layout == "nhwc" and (len(sh) in (1, 4)) and sh[-1] == n_ch:
            return a.reshape(-1)
        if (layout == "nchw" and len(sh) == 3 and sh[0] == n_ch and
                sh[1:] == [1, 1]):
            return a.reshape(-1)
        # a 1-D [C] constant broadcasts along W of an NCHW tensor: not a
        # per-channel op
        return None

    def opt_const(self, n, i):
        """Constant value of optional input i of node n, or None."""
        if len(n.input) > i and n.input[i]:
            return self.g.const.get(n.input[i])
        return None

    def bias(self, n, size, dtype):
        """Bias (input 2) of a conv node, or zeros when it has none."""
        if len(n.input) > 2 and n.input[2]:
            return np.asarray(self.g.const[n.input[2]], dtype)
        return np.zeros(size, dtype)

    def _cint(self, n, i, default=None):
        """Integer list of constant input i of node n.

        Args:
            n: the node.
            i: input index.
            default: returned when the input is absent.

        Returns:
            The values as a list of int, or default.

        Raises:
            Unsupported: the input is present but not a constant.
        """
        if len(n.input) <= i or not n.input[i]:
            return default
        v = self.g.const.get(n.input[i])
        if v is None:
            raise Unsupported(f"non-constant input {i} of {self.g.where(n)}", n)
        return [int(x) for x in np.asarray(v).reshape(-1)]

    def _layout(self, t, nodes, ops=LAYOUT_OPS):
        """Steps over single-consumer layout ops after tensor t.

        Args:
            t: tensor name to start at.
            nodes: list the stepped-over nodes are appended to.
            ops: the op types to step over.

        Returns:
            (tensor reached, its single consumer or None).
        """
        n = self.only_cons(t)
        while n is not None and n.op_type in ops:
            t = take(nodes, n)
            n = self.only_cons(t)
        return t, n

    def _slice_axis(self, n, t):
        """Axis and indices of a single-axis Slice of tensor t.

        Returns:
            (axis counted from the end, -1 = last; the index list).

        Raises:
            Unsupported: the Slice covers several axes.
        """
        g = self.g
        sh = g.shape[t]
        st, en = self._cint(n, 1), self._cint(n, 2)
        ax = self._cint(n, 3, [0])
        sp = self._cint(n, 4, [1])
        if len(st) != 1 or len(ax) != 1 or len(sp) != 1:
            raise Unsupported(f"multi-axis Slice at {g.where(n)}", n)
        a = ax[0] % len(sh)
        return a - len(sh), slice_indices(sh[a], st[0], en[0], sp[0])

    # ---- activations
    def match_act(self, t):
        """Activation applied to tensor t.

        relu: Relu(t); swish: Sigmoid(t) * t | t * (1 / (1 + exp(-t)))
        [Neg, Exp, Add 1, Div 1/., Mul]; sigmoid: Sigmoid(t) |
        1 / (1 + exp(-t)).

        Returns:
            (kind, nodes, out_tensor), or None when t feeds no activation.
        """
        g = self.g
        cs = g.cons.get(t, [])
        if len(cs) == 1 and cs[0].op_type == "Relu":
            return "relu", [cs[0]], cs[0].output[0]
        sig = self.match_sigmoid(t)
        if sig:
            _, nodes, s_out = sig
            muls = [
                c for c in cs
                if c.op_type == "Mul" and set(c.input) == {t, s_out}
            ]
            others = [c for c in cs if c not in nodes and c not in muls]
            gate_uses = len(g.cons.get(s_out, []))
            if len(muls) == 1 and not others and gate_uses == 1:
                return "swish", nodes + muls, muls[0].output[0]
            if not muls and len(cs) == 1:
                return "sigmoid", nodes, s_out
        return None

    def match_sigmoid(self, t):
        """Sigmoid of tensor t, as one node or spelled out with Exp.

        Returns:
            ("sigmoid", nodes, out_tensor), or None.
        """
        g = self.g
        for c in g.cons.get(t, []):
            if c.op_type == "Sigmoid":
                return "sigmoid", [c], c.output[0]
            if c.op_type == "Neg":
                e = self.only_cons(c.output[0], "Exp")
                a = e and self.only_cons(e.output[0], "Add")
                if not a or self.const_scalar(a) != 1.0:
                    continue
                d = self.only_cons(a.output[0], "Div")
                if (not d or d.input[1] != a.output[0] or
                        g.const.get(d.input[0]) is None or
                        first_float(g.const[d.input[0]]) != 1.0):
                    continue
                return "sigmoid", [c, e, a, d], d.output[0]
        return None

    def _nonzero_pad_value(self, n):
        """True when Pad node n has a constant pad value other than zero."""
        return (len(n.input) > 2 and n.input[2] and
                first_float(self.g.const[n.input[2]]) != 0)

    # ---- front end: framed log-mel (Perch)
    def fe_logmel(self):
        """Matches the Perch-style framed log-mel front end.

        [Reshape] -> framing Conv with an identity kernel (stride hop, pads)
        -> layout ops -> Mul window -> Pad (zeros at the end) -> [Reshape] ->
        DFT (onesided) -> ReduceL2 over re/im (magnitude) | MatMul by an
        interleaved re/im DFT matrix -> magnitude -> MatMul mel -> Max(floor)
        -> Log -> Mul(scale).

        Returns:
            The FE_LOGMEL op, or None when the graph does not start with a
            framing conv.

        Raises:
            Unsupported: the framing matched but a later node did not.
        """
        g = self.g
        nodes = []
        framing = self._logmel_framing(nodes)
        if framing is None:
            return None
        n_win, hop, pad, t = framing
        win, t = self._logmel_window(t, nodes, n_win)
        nfft, t = self._logmel_pad(t, nodes, n_win)
        mode, t = self._logmel_magnitude(t, nodes, nfft)
        mel, t = self._logmel_mel(t, nodes, nfft)
        floor, scale, t = self._logmel_log(t, nodes)
        nfr = (g.nsamp + 2 * pad - n_win) // hop + 1
        out = Val("act", nfr, mel.shape[1], 1, "nhwc")  # [time][mel][1]
        op = dict(op="FE_LOGMEL",
                  ins=[],
                  out=out.id,
                  frame=n_win,
                  hop=hop,
                  pad=pad,
                  nfft=nfft,
                  window=win.astype(np.float32),
                  mel=np.asarray(mel, np.float32).T.copy(),
                  floor=floor,
                  scale=scale,
                  mode=mode,
                  nodes=[x.name for x in nodes],
                  tap=t)
        self.cov(*nodes)
        self.val[t] = out
        # the [B, T, M] spectrogram becomes an NCHW image [B, 1, T, M] through
        # Unsqueeze/Reshape
        return op

    def _logmel_framing(self, nodes):
        """Framing conv: (frame length, hop, pad, out tensor), or None."""
        g = self.g
        t = g.inp
        n = self.only_cons(t)
        if n and n.op_type == "Reshape":
            t = take(nodes, n)
            n = self.only_cons(t)
        if not n or n.op_type != "Conv":
            return None
        w = g.const.get(n.input[1])
        if (w is None or w.ndim != 4 or w.shape[1] != 1 or w.shape[3] != 1 or
                w.shape[0] != w.shape[2]):
            return None
        n_win = w.shape[0]
        eye = np.eye(n_win, dtype=w.dtype)
        if not np.array_equal(w.reshape(n_win, n_win), eye):
            raise Unsupported(
                f"framing conv at {g.where(n)} is not an identity kernel")
        at = g.attrs(n)
        hop = at["strides"][0]
        pads = at.get("pads", [0, 0, 0, 0])
        if pads[1] or pads[3] or pads[0] != pads[2]:
            raise Unsupported(f"framing conv pads {pads} at {g.where(n)}")
        return n_win, hop, pads[0], take(nodes, n)

    def _logmel_window(self, t, nodes, n_win):
        """Window Mul after the framing: (window float64, out tensor)."""
        g = self.g
        t, n = self._layout(t, nodes,
                            ("Transpose", "Reshape", "Squeeze", "Unsqueeze"))
        if not n or n.op_type != "Mul":
            raise Unsupported(
                "expected the window Mul after the framing at "
                f"{g.where(n) if n else t}")
        win = np.asarray(g.cval(n), np.float64).reshape(-1)
        if win.size != n_win:
            raise Unsupported("window length")
        return win, take(nodes, n)

    def _logmel_pad(self, t, nodes, n_win):
        """Zero Pad of the frame to the FFT length: (nfft, out tensor)."""
        g = self.g
        n = self.only_cons(t, "Pad")
        if not n:
            raise Unsupported("expected Pad to the FFT length after the window")
        pv = np.asarray(g.const[n.input[1]]).reshape(-1)
        rank = len(pv) // 2
        if (any(pv[:rank - 1]) or pv[rank - 1] != 0 or
                any(pv[rank:2 * rank - 1])):
            raise Unsupported(
                f"pad layout {pv} (only zeros appended to the frame)")
        if self._nonzero_pad_value(n):
            raise Unsupported("non-zero pad value")
        nfft = n_win + int(pv[-1])
        return nfft, take(nodes, n)

    def _logmel_magnitude(self, t, nodes, nfft):
        """DFT op + ReduceL2: (mode, out tensor)."""
        g = self.g
        t, n = self._layout(t, nodes, ("Reshape", "Unsqueeze"))
        if n and n.op_type == "MatMul":
            self._refuse_matrix_dft(n, nfft)
        if not n or n.op_type != "DFT":
            raise Unsupported(
                "expected DFT after the frame padding at "
                f"{g.where(n) if n else t}")
        a = g.attrs(n)
        ln = None
        if len(n.input) > 1 and n.input[1]:
            ln = int(np.asarray(g.const[n.input[1]]).reshape(-1)[0])
        if (a.get("onesided") != 1 or a.get("inverse", 0) or
                (ln is not None and ln != nfft)):
            raise Unsupported(f"DFT attributes {a} length {ln}")
        t = take(nodes, n)
        n = self.only_cons(t, "ReduceL2")
        if not n:
            raise Unsupported("expected ReduceL2 (magnitude) after the DFT")
        return "dft-op", take(nodes, n)

    def _refuse_matrix_dft(self, n, nfft):
        """Raises for a MatMul after the frame padding (no kernel for it)."""
        g = self.g
        dft = g.cval(n)
        k = np.arange(nfft // 2 + 1)
        ang = 2 * np.pi * np.outer(np.arange(nfft), k) / nfft
        ok = (dft is not None and dft.shape == (nfft, 2 * len(k)) and
              np.abs(dft[:, 0::2] - np.cos(ang)).max() < 1e-3 and
              np.abs(np.abs(dft[:, 1::2]) - np.abs(np.sin(ang))).max() < 1e-3)
        if not ok:
            raise Unsupported(
                f"MatMul at {g.where(n)} is not an interleaved re/im DFT matrix"
            )
        raise Unsupported(
            "matrix-DFT magnitude chain (no_dft file): use the DFT-op file")

    def _logmel_mel(self, t, nodes, nfft):
        """Mel MatMul after the magnitude: (mel matrix, out tensor)."""
        g = self.g
        n = self.only_cons(t, "MatMul")
        mel = n is not None and g.cval(n)
        if n is None or mel is None or mel.shape[0] != nfft // 2 + 1:
            raise Unsupported("expected the mel MatMul after the magnitude")
        return mel, take(nodes, n)

    def _logmel_log(self, t, nodes):
        """Max(floor) -> Log -> [Mul scale]: (floor, scale, out tensor)."""
        n = self.only_cons(t, "Max")
        floor = self.const_scalar(n) if n else None
        if floor is None:
            raise Unsupported("expected Max(floor) before the log")
        t = take(nodes, n)
        n = self.only_cons(t, "Log")
        if not n:
            raise Unsupported("expected Log")
        t = take(nodes, n)
        scale = 1.0
        n = self.only_cons(t)
        if n and n.op_type == "Mul" and self.const_scalar(n) is not None:
            scale = self.const_scalar(n)
            t = take(nodes, n)
        return floor, scale, t

    # ---- front end: dual STFT (BirdNET v2.4)
    def fe_stft2(self):
        """Matches the BirdNET v2.4-style dual STFT front end.

        min/max normalization -> 2 branches of (framing by grouped Gather ->
        window Mul -> cos-only DFT MatMul (real part) -> [Reshape] -> mel
        MatMul -> square (Mul x x) -> Pow p -> mel-axis reverse) -> channel
        Concat -> per-channel affine (BN).

        Returns:
            The FE_STFT2_POW op, or None when the graph does not start with
            the ReduceMin / Sub normalization.

        Raises:
            Unsupported: the normalization matched but a later node did not.
        """
        g = self.g
        nodes = []
        norm = self._stft2_normalization(nodes)
        if norm is None:
            return None
        eps, y = norm
        branches, cn = self._stft2_branches(y)
        # [96, 511, 1, 2]: H = mel (reversed), W = time, channel last
        cshape = g.shape[cn.output[0]]
        bm, ba, bs, bb = self._stft2_bn(cn.output[0])
        nodes += [cn, bm, ba]
        for b in branches:
            nodes += b["nodes"]
        t = ba.output[0]
        h, w, lay = stft2_image_layout(cshape, branches[0]["mel"].shape[0])
        out = Val("act", h, w, 2, lay)  # a Transpose to NCHW follows
        self.cov(*nodes)
        self.val[t] = out
        keys = ("nfft", "hop", "window", "dft", "mel", "pow")
        return dict(op="FE_STFT2_POW",
                    ins=[],
                    out=out.id,
                    eps=eps,
                    branches=[{k: b[k] for k in keys} for b in branches],
                    bn_s=bs,
                    bn_b=bb,
                    nodes=[x.name for x in nodes],
                    tap=t)

    def _stft2_normalization(self, nodes):
        """(x - min) / (max + eps), (. - 0.5) * 2: (eps, out) or None."""
        g = self.g
        n0 = list(g.cons.get(g.inp, []))
        rmin = [c for c in n0 if c.op_type == "ReduceMin"]
        sub = [c for c in n0 if c.op_type == "Sub"]
        if len(rmin) != 1 or len(sub) != 1 or len(n0) != 2:
            return None
        nodes += rmin + sub
        cs = g.cons.get(sub[0].output[0], [])
        rmax = [c for c in cs if c.op_type == "ReduceMax"]
        div = [c for c in cs if c.op_type == "Div"]
        if len(rmax) != 1 or len(div) != 1:
            raise Unsupported(
                "min/max normalization: expected ReduceMax and Div")
        add = self.only_cons(rmax[0].output[0], "Add")
        eps = self.const_scalar(add) if add else None
        if eps is None or div[0].input[1] != add.output[0]:
            raise Unsupported("min/max normalization epsilon")
        s2 = self.only_cons(div[0].output[0], "Sub")
        m2 = s2 and self.only_cons(s2.output[0], "Mul")
        if (not m2 or self.const_scalar(s2) != 0.5 or
                self.const_scalar(m2) != 2.0):
            raise Unsupported("normalization (x - 0.5) * 2")
        nodes += rmax + div + [add, s2, m2]
        return eps, m2.output[0]

    def _stft2_branches(self, y):
        """The two branches in Concat input order, and the Concat node."""
        branches = []
        for c in self.g.cons.get(y, []):
            branches.append(self.stft_branch(c, y))
        cat = set(b["concat"] for b in branches)
        if len(branches) != 2 or len(cat) != 1:
            raise Unsupported(
                "expected two spectrogram branches joined by one Concat")
        cn = branches[0]["concat_node"]
        order = [list(cn.input).index(b["tap"]) for b in branches]
        ranked = sorted(zip(order, branches), key=lambda z: z[0])
        return [b for _, b in ranked], cn

    def _stft2_bn(self, t):
        """Per-channel Mul, Add after the Concat: (Mul, Add, scale, bias)."""
        g = self.g
        bm = self.only_cons(t, "Mul")
        ba = bm and self.only_cons(bm.output[0], "Add")
        if not ba:
            raise Unsupported(
                "expected the spectrogram BN (Mul, Add) after the Concat")
        bs = np.asarray(g.cval(bm), np.float32).reshape(-1)
        bb = np.asarray(g.cval(ba), np.float32).reshape(-1)
        return bm, ba, bs, bb

    def stft_branch(self, n, y):
        """Matches one spectrogram branch of the dual STFT front end.

        Args:
            n: the first node of the branch (a consumer of y).
            y: the normalized input tensor.

        Returns:
            A dict with the branch parameters (nfft, hop, window, dft, mel,
            pow), its nodes, the tensor entering the Concat (tap) and the
            Concat node.

        Raises:
            Unsupported: a node of the branch did not match.
        """
        nodes = []
        n_fft, hop, t = self._branch_framing(n, y, nodes)
        win, dft, t = self._branch_dft(t, nodes, n_fft)
        ntrunc = dft.shape[1]
        mel, t = self._branch_mel(t, nodes, n_fft, ntrunc)
        p, t = self._branch_power(t, nodes)
        n, t = self._branch_reverse(t, nodes, mel.shape[1])
        melfull = np.zeros((n_fft // 2 + 1, mel.shape[1]), np.float32)
        # bins past the truncation are zero in the graph
        melfull[:ntrunc] = mel[:ntrunc]
        return dict(nodes=nodes,
                    tap=t,
                    concat=n.name,
                    concat_node=n,
                    nfft=n_fft,
                    hop=hop,
                    window=np.asarray(win, np.float32).reshape(-1),
                    dft=np.asarray(dft, np.float32),
                    mel=melfull.T.copy(),
                    pow=p)

    def _branch_framing(self, n, y, nodes):
        """Framing by a grouped Gather: (n_fft, hop, out tensor).

        [Slice] -> Reshape [L/g, g] -> Gather(axis 1) with an index table
        [T][N/g] -> Reshape [T, N].
        """
        g = self.g
        t = y
        if n.op_type == "Slice":
            st = np.asarray(g.const.get(n.input[1], [1])).reshape(-1)
            if st[0] != 0:
                raise Unsupported(f"framing slice start {st} at {g.where(n)}")
            t = take(nodes, n)
            n = self.only_cons(t)
        if not n or n.op_type != "Reshape":
            raise Unsupported(
                f"framing: expected Reshape at {g.where(n) if n else t}")
        grp = g.shape[n.output[0]][-1]
        t = take(nodes, n)
        n = self.only_cons(t, "Gather")
        idx = n is not None and g.const.get(n.input[1])
        if n is None or idx is None or g.attrs(n).get("axis") != 1:
            raise Unsupported(
                "framing: expected Gather(axis=1) with a constant index table")
        ng, hopg = self._gather_table(np.asarray(idx))
        t = take(nodes, n)
        n = self.only_cons(t, "Reshape")
        if not n:
            raise Unsupported("framing: expected Reshape [T, N]")
        return ng * grp, hopg * grp, take(nodes, n)

    def _gather_table(self, idx):
        """Checks a framing index table: (samples per row, hop), in groups."""
        n_fr, ng = idx.shape
        hopg = int(idx[1, 0] - idx[0, 0])
        frames = hopg * np.arange(n_fr)[:, None]
        expected = idx[0, 0] + frames + np.arange(ng)[None, :]
        if not np.array_equal(idx, expected) or idx[0, 0] != 0:
            raise Unsupported("framing: index table is not t*hop + n")
        return ng, hopg

    def _branch_dft(self, t, nodes, n_fft):
        """Window Mul and cos-only DFT MatMul: (window, dft, out tensor).

        The DFT may be truncated ("dfttrunc"): its bins are zero-padded back
        to N/2+1 by the graph.
        """
        g = self.g
        n = self.only_cons(t, "Mul")
        win = n is not None and g.cval(n)
        if n is None or win is None or win.size != n_fft:
            raise Unsupported("expected the window Mul")
        t = take(nodes, n)
        n = self.only_cons(t, "MatMul")
        dft = n is not None and g.cval(n)
        if n is None or dft is None or dft.shape[0] != n_fft:
            raise Unsupported("expected the DFT MatMul")
        kk = np.arange(dft.shape[1])
        cos = np.cos(2 * np.pi * np.outer(np.arange(n_fft), kk) / n_fft)
        if dft.shape[1] > n_fft // 2 + 1 or np.abs(dft - cos).max() > 1e-3:
            raise Unsupported("DFT MatMul is not the cos (real) part")
        return win, dft, take(nodes, n)

    def _branch_mel(self, t, nodes, n_fft, ntrunc):
        """[Reshape | zero-bin Pad]* -> mel MatMul: (mel, out tensor)."""
        g = self.g
        while True:
            t, n = self._layout(t, nodes, ("Reshape",))
            if not n or n.op_type != "Pad":
                break
            pv = np.asarray(g.const[n.input[1]]).reshape(-1)
            r = len(pv) // 2
            if (any(pv[:r]) or any(pv[r:2 * r - 1]) or
                    self._nonzero_pad_value(n)):
                raise Unsupported("front-end Pad other than zero bins appended")
            t = take(nodes, n)
        mel = n is not None and n.op_type == "MatMul" and g.cval(n)
        if (mel is None or mel is False or mel.shape[0] < ntrunc or
                mel.shape[0] > n_fft // 2 + 1):
            raise Unsupported("expected the mel MatMul")
        return mel, take(nodes, n)

    def _branch_power(self, t, nodes):
        """Square (Mul x x) -> Pow p: (p, out tensor)."""
        t, n = self._layout(t, nodes, ("Reshape",))
        if not n or n.op_type != "Mul" or n.input[0] != n.input[1]:
            raise Unsupported("expected the square (Mul x x)")
        t = take(nodes, n)
        n = self.only_cons(t, "Pow")
        p = self.const_scalar(n) if n else None
        if p is None:
            raise Unsupported("expected Pow")
        return p, take(nodes, n)

    def _branch_reverse(self, t, nodes, nmel):
        """Mel-axis reverse into the Concat: (Concat node, tensor into it).

        Transpose + Slice(step -1) (ReverseV2), then Unsqueeze.
        """
        g = self.g
        rev = False
        while True:
            t, n = self._layout(t, nodes, ("Transpose", "Unsqueeze", "Reshape"))
            if not n or n.op_type != "Slice":
                break
            steps = [1]
            if len(n.input) > 4:
                steps = np.asarray(g.const.get(n.input[4], [1])).reshape(-1)
            axes = [0]
            if len(n.input) > 3:
                axes = np.asarray(g.const.get(n.input[3], [0])).reshape(-1)
            if list(steps) != [-1] or g.shape[t][int(axes[0])] != nmel:
                raise Unsupported(
                    "slice other than a reverse of the mel axis in the front "
                    "end")
            rev = True
            t = take(nodes, n)
        if not rev or not n or n.op_type != "Concat":
            raise Unsupported("expected mel reverse and Concat")
        return n, t

    # ---- front end: dB image (BirdNET v3.0 family)
    def fe_dbimg(self):
        """Matches the dB-image front end.

        [Pad n/2 (reflect | zero)] -> [Unsqueeze] -> Conv1d with a [2F, 1, N]
        kernel = window * cos | window * sin of an N-point DFT (F = N/2 + 1),
        stride hop -> Slice re / im -> re^2 + im^2 -> layout -> mel MatMul |
        Einsum [F][nmel] -> layout -> Clip(floor) -> Log -> up to three
        scalar Mul / Div / Add / Sub -> [ReduceMax over the image - top_db ->
        Max] -> [Slices of the mel rows] -> [min/max normalization
        (x - min) / (max - min + eps) or (max - x) / (max - min + eps)] ->
        [Unsqueeze] -> [Resize linear half_pixel] -> [Expand to C channels]
        -> [per-channel Sub / Add / Mul / Div]. Every constant is verified;
        the output is an image NCHW [1][C][Ho][Wo] (H = mel rows, W =
        frames).

        Returns:
            The FE_DBIMG op, or None when the graph does not start with a
            Conv-framed DFT.

        Raises:
            Unsupported: the DFT kernel matched but a later node did not.
        """
        nodes = []
        stft = self._dbimg_stft(nodes)
        if stft is None:
            return None
        fe, t = stft
        nfr = fe["nfr"]
        mel, t, n = self._dbimg_mel(t, nodes, fe["n_bins"], nfr)
        floor, t = self._dbimg_log(t, n, nodes)
        post, t = self._dbimg_scalars(t, nodes)
        top_db, t = self._dbimg_top_db(t, nodes)
        rows, t = self._dbimg_rows(t, nodes, mel.shape[1])
        norm, eps, t = self._dbimg_minmax(t, nodes, len(rows), nfr)
        h_out, w_out, t, n = self._dbimg_resize(t, nodes, len(rows), nfr)
        n_ch, t, n = self._dbimg_expand(t, n, nodes, h_out, w_out)
        chan, t = self._dbimg_chan(t, n, nodes, n_ch)
        out = Val("act", h_out, w_out, n_ch, "nchw")
        self.cov(*nodes)
        self.val[t] = out
        return dict(op="FE_DBIMG", ins=[], out=out.id,
                    nfft=fe["n_fft"], hop=fe["hop"], pad=fe["pad"],
                    pad_mode=fe["pad_mode"], nfr=nfr,
                    window=fe["win"].astype(np.float32), mel=mel.T.copy(),
                    floor=floor, post=post, top_db=top_db,
                    rows=np.array(rows, np.int32), norm=norm, eps=eps,
                    Ho=h_out, Wo=w_out, C=n_ch, chan=chan,
                    nodes=[x.name for x in nodes], tap=t)

    def _dbimg_stft(self, nodes):
        """[Pad] -> [Unsqueeze] -> DFT Conv -> power spectrum.

        Returns:
            (dict with pad, pad_mode, n_fft, n_bins, win, hop, nfr; the power
            spectrum tensor), or None when this is not a Conv-framed DFT.
        """
        g = self.g
        t = g.inp
        n = self.only_cons(t)
        pad, pad_mode = 0, "none"
        if n is not None and n.op_type == "Pad":
            padding = self._dbimg_pad(n, t)
            if padding is None:
                return None
            pad, pad_mode = padding
            t = take(nodes, n)
            n = self.only_cons(t)
        if n is not None and n.op_type in ("Unsqueeze", "Reshape"):
            t = take(nodes, n)
            n = self.only_cons(t)
        kernel = self._dbimg_kernel(n)
        if kernel is None:
            return None
        n_fft, n_bins, win, hop = kernel
        t = take(nodes, n)
        t, ad = self._dbimg_power(t, n, nodes, n_bins)
        nfr = g.shape[t][-1]
        if nfr != (g.nsamp + 2 * pad - n_fft) // hop + 1:
            raise Unsupported(
                f"framing: {nfr} frames, expected (n + 2 pad - n_fft) / hop "
                "+ 1", ad)
        fe = dict(pad=pad,
                  pad_mode=pad_mode,
                  n_fft=n_fft,
                  n_bins=n_bins,
                  win=win,
                  hop=hop,
                  nfr=nfr)
        return fe, t

    def _dbimg_pad(self, n, t):
        """Centering Pad n of the signal t: (pad, pad mode), or None."""
        g = self.g
        mode = decode_attr(g.attrs(n).get("mode", b"constant"))
        pv = self._cint(n, 1)
        if pv is None or len(pv) != 2 * len(g.shape[t]):
            return None
        r = len(pv) // 2
        if (any(pv[:r - 1]) or any(pv[r:2 * r - 1]) or
                pv[r - 1] != pv[2 * r - 1] or pv[r - 1] <= 0):
            return None
        if mode == "constant":
            cv = self.opt_const(n, 2)
            if cv is not None and first_float(cv) != 0.0:
                return None
        elif mode != "reflect":
            return None
        return pv[r - 1], ("reflect" if mode == "reflect" else "zero")

    def _dbimg_kernel(self, n):
        """DFT framing conv n: (n_fft, bins, window, hop), or None.

        Raises:
            Unsupported: a cos | sin DFT kernel the front-end kernel cannot
                take (size, pads, dilation, groups, bias).
        """
        g = self.g
        if n is None or n.op_type != "Conv" or g.const.get(n.input[1]) is None:
            return None
        w = np.asarray(g.const[n.input[1]], np.float64)
        if (w.ndim != 3 or w.shape[1] != 1 or
                w.shape[0] != 2 * (w.shape[2] // 2 + 1)):
            return None
        n_fft, n_bins = int(w.shape[2]), int(w.shape[0]) // 2
        win = w[0, 0].copy()  # row f = 0 is window * cos(0)
        if not is_windowed_dft(w, win, n_fft, n_bins, 1e-5):
            return None  # not a cos | sin DFT kernel: some other 1-D conv
        # from here on the graph IS a Conv-framed STFT: every mismatch is
        # reported with its reason
        if n_fft & (n_fft - 1) or not 64 <= n_fft <= 8192:
            raise Unsupported(
                f"Conv-framed DFT with n_fft {n_fft}: the front-end kernel "
                "takes a power of two in 64..8192", n)
        a = g.attrs(n)
        hop = int(a.get("strides", [1])[0])
        if (any(a.get("pads", [0, 0])) or a.get("dilations", [1])[0] != 1 or
                a.get("group", 1) != 1 or
            (len(n.input) > 2 and n.input[2] and
             np.any(g.const.get(n.input[2], 1)))):
            raise Unsupported(
                f"framing conv attributes {a} / bias at {g.where(n)} "
                "(expected no pads, no dilation, no bias)", n)
        return n_fft, n_bins, win, hop

    def _dbimg_power(self, t, n, nodes, n_bins):
        """Slice re / im -> squares -> Add: (power tensor, the Add node)."""
        g = self.g
        cs = g.cons.get(t, [])
        if len(cs) != 2 or any(c.op_type != "Slice" for c in cs):
            raise Unsupported(
                "expected two Slices (real / imaginary rows) after the "
                f"framing conv {g.where(n)}", n)
        halves = (list(range(n_bins)), list(range(n_bins, 2 * n_bins)))
        sq = []
        for c in cs:
            ax, idx = self._slice_axis(c, t)
            if ax != -2 or idx not in halves:
                raise Unsupported(
                    f"{g.where(c)}: expected a Slice of rows 0..{n_bins} or "
                    f"{n_bins}..{2 * n_bins} of the DFT", c)
            m = self.only_cons(c.output[0], "Mul")
            if m is None or m.input[0] != m.input[1]:
                raise Unsupported(
                    f"expected the square (Mul x x) after {g.where(c)}", c)
            nodes += [c, m]
            sq.append(m)
        ad = self.only_cons(sq[0].output[0], "Add")
        if ad is None or set(ad.input) != {sq[0].output[0], sq[1].output[0]}:
            raise Unsupported("expected re^2 + im^2 (Add of the two squares)",
                              sq[0])
        return take(nodes, ad), ad

    def _dbimg_mel(self, t, nodes, n_bins, nfr):
        """Mel MatMul | Einsum: (mel float32 [F][nmel], out, consumer)."""
        g = self.g
        t, n = self._layout(t, nodes)
        if n is None or n.op_type not in ("MatMul", "Einsum"):
            raise Unsupported(
                "expected the mel MatMul / Einsum after the power spectrum at "
                f"{g.where(n) if n else t}", n)
        if n.op_type == "Einsum":
            eq = decode_attr(g.attrs(n).get("equation", b"")).replace(" ", "")
            if not einsum_is_matmul(eq):
                raise Unsupported(
                    f"Einsum '{eq}' is not a matrix product x[.., F] . M[F, "
                    "nmel]", n)
        mel = g.const.get(n.input[1])
        if (mel is None or np.asarray(mel).ndim != 2 or
                np.asarray(mel).shape[0] != n_bins or n.input[0] in g.const or
                g.shape[n.input[0]][-1] != n_bins):
            raise Unsupported(
                f"mel product at {g.where(n)}: expected power[.., {n_bins}] . "
                f"M[{n_bins}, nmel] with a constant M", n)
        mel = np.asarray(mel, np.float32)
        t = take(nodes, n)
        t, n = self._layout(t, nodes)
        sh = g.shape[t]
        if [d for d in sh if d != 1] != [mel.shape[1], nfr]:
            raise Unsupported(
                f"mel spectrogram layout {sh}: the front-end kernel takes "
                "[mel][frames] (H = mel, W = time)", n)
        return mel, t, n

    def _dbimg_log(self, t, n, nodes):
        """Clip(floor) -> Log: (floor, out tensor)."""
        g = self.g
        if n is None or n.op_type != "Clip":
            raise Unsupported(
                "expected Clip(floor) before the log at "
                f"{g.where(n) if n else t}", n)
        lo = self.opt_const(n, 1)
        if lo is None or (len(n.input) > 2 and n.input[2]):
            raise Unsupported(
                f"{g.where(n)}: expected a Clip with a constant minimum (the "
                "floor) and no maximum", n)
        floor = first_float(lo)
        t = take(nodes, n)
        n = self.only_cons(t, "Log")
        if n is None:
            raise Unsupported("expected Log after the Clip", self.at)
        return floor, take(nodes, n)

    def _dbimg_scalars(self, t, nodes):
        """Up to three scalar ops after the log: (post list, out tensor).

        post holds (code, value) pairs, applied in order after the log; the
        codes are ARITH_CODE.
        """
        g = self.g
        post = []
        while len(post) < 3:
            n = self.only_cons(t)
            if n is None or n.op_type not in ("Mul", "Div", "Add", "Sub"):
                break
            cv = g.cval(n)
            if (cv is None or np.asarray(cv).size != 1 or
                (n.op_type in ("Div", "Sub") and n.input[1] not in g.const)):
                break
            post.append((ARITH_CODE[n.op_type], first_float(cv)))
            t = take(nodes, n)
        return post, t

    def _dbimg_top_db(self, t, nodes):
        """Optional Max(x, ReduceMax(x) - top_db): (top_db | None, out)."""
        g = self.g
        cs = g.cons.get(t, [])
        if len(cs) != 2 or {c.op_type for c in cs} != {"ReduceMax", "Max"}:
            return None, t
        rm = [c for c in cs if c.op_type == "ReduceMax"][0]
        mx = [c for c in cs if c.op_type == "Max"][0]
        if int(np.prod(g.shape[rm.output[0]])) != 1:
            raise Unsupported(
                f"{g.where(rm)}: expected a maximum over the whole "
                "spectrogram (top-db floor)", rm)
        sb = self.only_cons(rm.output[0], "Sub")
        if (sb is None or sb.input[0] != rm.output[0] or g.cval(sb) is None or
                set(mx.input) != {t, sb.output[0]}):
            raise Unsupported(
                "top-db floor: expected Max(x, ReduceMax(x) - top_db)", rm)
        top_db = first_float(g.cval(sb))
        nodes += [rm, sb, mx]
        return top_db, mx.output[0]

    def _dbimg_rows(self, t, nodes, nmel):
        """Mel-row selection / reversal: (row indices, out tensor).

        Any composition of Slices on the mel axis.
        """
        g = self.g
        rows = list(range(nmel))
        while True:
            n = self.only_cons(t)
            if n is None or n.op_type != "Slice":
                break
            ax, idx = self._slice_axis(n, t)
            if ax != -2:
                raise Unsupported(
                    f"{g.where(n)}: Slice along the time axis of the "
                    "spectrogram (only mel rows can be selected)", n)
            rows = [rows[i] for i in idx]
            t = take(nodes, n)
        return rows, t

    def _dbimg_minmax(self, t, nodes, n_rows, nfr):
        """Optional min/max normalization: (norm code, eps, out tensor).

        norm: 0 none, 1 (x - min) / (max - min + eps), 2 (max - x) / (...).
        """
        g = self.g
        cs = g.cons.get(t, [])
        rs = [c for c in cs if c.op_type == "Reshape"]
        t2 = rs[0].output[0] if len(rs) == 1 else t
        c2 = g.cons.get(t2, [])
        kinds = {c.op_type for c in c2}
        if len(c2) != 2 or kinds != {"ReduceMin", "ReduceMax"}:
            return 0, 0.0, t
        rmin = [c for c in c2 if c.op_type == "ReduceMin"][0]
        rmax = [c for c in c2 if c.op_type == "ReduceMax"][0]
        if (int(np.prod(g.shape[t2])) != n_rows * nfr or
                int(np.prod(g.shape[rmin.output[0]])) != 1 or
                int(np.prod(g.shape[rmax.output[0]])) != 1):
            raise Unsupported(
                "min/max normalization over part of the image (expected all "
                "rows x frames)", rmin)
        if t2 != t:
            nodes += rs[:1]
        nodes += [rmin, rmax]
        mn_t, _ = self._layout(rmin.output[0], nodes, ("Reshape",))
        mx_t, _ = self._layout(rmax.output[0], nodes, ("Reshape",))
        num, den, norm = self._minmax_terms(t, mn_t, mx_t, rmin)
        ae = self.only_cons(den.output[0], "Add")
        dv = ae and self.only_cons(ae.output[0], "Div")
        if (not dv or g.cval(ae) is None or
                list(dv.input) != [num.output[0], ae.output[0]]):
            raise Unsupported(
                "min/max normalization: expected Div(numerator, (max - min) + "
                "eps)", den)
        if set(id(c) for c in cs) - {id(num)} - set(id(c) for c in rs[:1]):
            raise Unsupported(
                "the spectrogram before the min/max normalization has other "
                "consumers", num)
        eps = first_float(g.cval(ae))
        nodes += [num, den, ae, dv]
        return norm, eps, dv.output[0]

    def _minmax_terms(self, t, mn_t, mx_t, rmin):
        """Numerator and denominator Subs of the min/max normalization.

        Returns:
            (numerator node, denominator node, norm code 1 | 2).
        """
        g = self.g
        subs_by_id = {}  # each Sub once, in consumer order
        for c in g.cons.get(mx_t, []) + g.cons.get(mn_t, []):
            if c.op_type == "Sub":
                subs_by_id[id(c)] = c
        subs = list(subs_by_id.values())
        num = [s for s in subs if t in s.input]
        den = [s for s in subs if list(s.input) == [mx_t, mn_t]]
        if len(num) != 1 or len(den) != 1:
            raise Unsupported(
                "min/max normalization: expected (max - x) or (x - min) over "
                "(max - min + eps)", rmin)
        if list(num[0].input) == [mx_t, t]:
            norm = 2
        elif list(num[0].input) == [t, mn_t]:
            norm = 1
        else:
            raise Unsupported(
                f"min/max normalization numerator {list(num[0].input)}", num[0])
        return num[0], den[0], norm

    def _dbimg_resize(self, t, nodes, n_rows, nfr):
        """Optional Resize to the image size: (Ho, Wo, out, consumer)."""
        g = self.g
        t, n = self._layout(t, nodes, ("Unsqueeze", "Reshape"))
        h_out, w_out = n_rows, nfr
        if n is not None and n.op_type == "Resize":
            a = {k: decode_attr(v) for k, v in g.attrs(n).items()}
            if (a.get("mode") != "linear" or
                    a.get("coordinate_transformation_mode",
                          "half_pixel") != "half_pixel" or
                    a.get("antialias", 0) or a.get("exclude_outside", 0)):
                raise Unsupported(
                    f"{g.where(n)}: Resize {a}: the front-end kernel "
                    "implements linear, half_pixel, no antialias", n)
            osh = g.shape[n.output[0]]
            if len(osh) != 4 or osh[:2] != [1, 1]:
                raise Unsupported(
                    f"{g.where(n)}: Resize to {osh} (expected [1, 1, Ho, Wo])",
                    n)
            h_out, w_out = osh[2], osh[3]
            t = take(nodes, n)
            n = self.only_cons(t)
        if len(g.shape[t]) != 4:
            raise Unsupported(
                f"front end output {g.shape[t]}: expected an image [1, C, H, "
                "W]", n)
        return h_out, w_out, t, n

    def _dbimg_expand(self, t, n, nodes, h_out, w_out):
        """Optional Expand to C channels: (channels, out, consumer)."""
        g = self.g
        if n is None or n.op_type != "Expand":
            return 1, t, n
        self._cint(n, 1)  # raises when the Expand shape is not a constant
        osh = g.shape[n.output[0]]
        if len(osh) != 4 or osh[0] != 1 or osh[2:] != [h_out, w_out]:
            raise Unsupported(
                f"{g.where(n)}: Expand to {osh} (expected [1, C, Ho, Wo])", n)
        t = take(nodes, n)
        return osh[1], t, self.only_cons(t)

    def _dbimg_chan(self, t, n, nodes, n_ch):
        """Up to two per-channel ops: (chan list, out tensor).

        chan holds (code, values[C]) pairs; the codes are ARITH_CODE.
        """
        g = self.g
        chan = []
        while (len(chan) < 2 and n is not None and
               n.op_type in ("Mul", "Div", "Add", "Sub") and
               g.cval(n) is not None and n.input[1] in g.const):
            v = self.chan_vec(g.cval(n), n_ch, "nchw")
            if v is None:
                break
            chan.append((ARITH_CODE[n.op_type], np.asarray(v, np.float32)))
            t = take(nodes, n)
            n = self.only_cons(t)
        return chan, t

    def conv_stft(self, first):
        """Structure check only, for the refusal message.

        An optional reflect/constant Pad and Unsqueeze, then a 1-D Conv whose
        constant kernel [2F, 1, N] with F = N/2 + 1 stacks the cos and sin
        rows of an N-point DFT.

        Args:
            first: the consumers of the graph input.

        Returns:
            A description of the Conv-framed DFT, or None.
        """
        g = self.g
        n = first[0] if len(first) == 1 else None
        if n is not None and n.op_type == "Pad":
            c = g.cons.get(n.output[0], [])
            n = c[0] if len(c) == 1 else None
        if n is not None and n.op_type == "Unsqueeze":
            c = g.cons.get(n.output[0], [])
            n = c[0] if len(c) == 1 else None
        if n is None or n.op_type != "Conv" or g.const.get(n.input[1]) is None:
            return None
        w = np.asarray(g.const[n.input[1]])
        if (w.ndim != 3 or w.shape[1] != 1 or
                w.shape[0] != 2 * (w.shape[2] // 2 + 1)):
            return None
        n_fft, n_bins = int(w.shape[2]), int(w.shape[0]) // 2
        win = w[0, 0].astype(np.float64)  # row f = 0 is window * cos(0)
        if not is_windowed_dft(w, win, n_fft, n_bins, 1e-3):
            return None
        hop = int(g.attrs(n).get("strides", [0])[0])
        return (f"a Conv-framed DFT ('{n.name}': kernel {list(w.shape)}, "
                f"n_fft {n_fft}, hop {hop})")

    # ---- body walker
    def walk(self) -> list:
        """Recognizes the whole graph.

        Returns:
            The IR ops in graph order. Nodes that could not be placed are in
            self.problems.
        """
        g = self.g
        fe = self._front_end(g.cons.get(g.inp, []))
        if fe:
            self.ops.append(fe)
        else:
            # the rest of the graph is reported too, as downstream of the
            # front end
            self.fe_failed = self.problems[-1][0]
            self.cov(*(n for n, _ in self.problems if n is not None))
        for n in g.nodes:
            if id(n) not in self.covered:
                self._place(n)
        return self.ops

    def _front_end(self, first):
        """Tries the front-end rules on the consumers of the graph input.

        Returns:
            The front-end op, or None with the reason in self.problems.
        """
        fe = None
        for f in (self.fe_stft2, self.fe_logmel, self.fe_dbimg):
            try:
                self.at = first[0] if first else None
                fe = f()  # raises Unsupported with the node it stopped at
            except Unsupported as e:
                self.problems.append((e.node or self.at, f"front end: {e}"))
                break
            if fe:
                break
        stft = None if fe or self.problems else self.conv_stft(first)
        if stft:
            self.problems.append((
                first[0],
                f"front end: {stft} is recognized as a short-time Fourier "
                "transform but no front-end kernel lowers it: the dB-image "
                "front end (the BirdNET v3.0 form: power spectrum, mel, dB "
                "with a top-db floor, min/max normalization, Resize to an "
                "image, per-channel affine; guide section 2.1) did not match "
                "what follows the transform"))
        if not fe and not self.problems:
            self.problems.append((
                first[0] if first else None,
                "no supported front end at the graph input: expected the "
                "BirdNET v2.4 dual STFT (ReduceMin/Sub normalization first) "
                "or a framed log-mel (Reshape/Conv framing first), found "
                f"{[c.op_type for c in first]}"))
        return fe

    def _place(self, n):
        """Applies the body rules to one uncovered node."""
        g = self.g
        if not any(x in self.val for x in n.input):
            if all((x in g.const) or not x for x in n.input):
                return  # constant plumbing left after folding
            self.problems.append((n, "input not produced by a supported op"))
            return
        try:
            ok = self.rule(n)
        except Unsupported as e:
            self.problems.append((n, str(e)))
            return
        if not ok:
            self.problems.append((n, "no rule"))

    def alias(self, n, v, layout=None, kind=None):
        """Maps the output of a layout-only op to the value of its input.

        The same IR value seen through a Transpose / Reshape: same id, new
        layout tag.
        """
        nv = Val.__new__(Val)
        nv.__dict__.update(v.__dict__)
        nv.layout = layout or v.layout
        if kind:
            nv.kind = kind
        self.val[n.output[0]] = nv
        self.cov(n)
        return True

    def _emit(self, op, t, out, nodes):
        """Appends an IR op whose result `out` is ONNX tensor t."""
        self.ops.append(op)
        self.val[t] = out
        self.cov(*nodes)
        return True

    def rule(self, n):
        """Places one body node (and the nodes its pattern absorbs).

        Args:
            n: an ONNX node with at least one input that is an IR value.

        Returns:
            True when an IR op (or an alias) covers the node, False when no
            rule applies.

        Raises:
            Unsupported: a rule applies but the node does not fit it.
        """
        op = n.op_type
        ins = [self.val.get(x) for x in n.input]
        v = ins[0]
        is_act = v is not None and v.kind == "act"
        done = None
        if op == "Transpose" and is_act:
            done = self._r_transpose(n, v)
        elif op in ("Reshape", "Unsqueeze", "Squeeze",
                    "Flatten") and v is not None:
            done = self._r_reshape(n, v)
        elif op == "Conv" and is_act:
            done = self.r_conv(n, v)
        elif op in ("ReduceMean", "ReduceSum") and is_act:
            done = self._r_reduce(n, v)
        elif op == "Mul":
            done = self._r_mul(n, v, ins)
        elif op == "Add":
            done = self._r_add(n, ins)
        elif op == "Relu" and v is not None and v.kind in ("act", "vec"):
            done = self._r_relu(n, v)
        elif (op in ("MaxPool", "AveragePool") and v is not None and
              v.layout == "nchw"):
            done = self._r_pool(n, v)
        elif op == "Concat" and all(x is not None and x.kind == "act"
                                    for x in ins):
            done = self._r_concat(n, v, ins)
        elif op in ("MatMul", "Gemm") and v is not None and v.kind == "vec":
            done = self._r_matmul(n, v)
        elif op == "Clip" and is_act:
            done = self.r_gem(n, v)
        if done is not None:
            return done
        if v is not None and v.kind == "vec" or (
                len(ins) == 2 and ins[1] is not None and ins[1].kind == "vec"):
            return self.r_vec(n, ins)
        return False

    # ---- layout, element-wise and pooling rules
    def _r_transpose(self, n, v):
        """Transpose of an activation between NCHW and NHWC -> alias."""
        perm = list(self.g.attrs(n)["perm"])
        if v.layout == "nchw" and perm == [0, 2, 3, 1]:
            return self.alias(n, v, "nhwc")
        if v.layout == "nhwc" and perm == [0, 3, 1, 2]:
            return self.alias(n, v, "nchw")
        if (v.layout == "hw1c" and perm == [2, 3, 0, 1]) or (
                v.layout == "1whc" and perm == [0, 3, 2, 1]):
            return self.alias(n, v, "nchw")
        raise Unsupported(f"transpose {perm} of a {v.layout} activation")

    def _r_reshape(self, n, v):
        """Layout-only Reshape / Unsqueeze / Squeeze / Flatten -> alias."""
        g = self.g
        shp = g.shape.get(n.output[0])
        if (v.kind == "act" and v.layout == "nhwc" and v.C == 1 and
                shp is not None and len(shp) == 4 and shp[1] == 1 and
                shp[2:] == [v.H, v.W]):
            # [B][T][M] spectrogram -> NCHW image with C = 1
            return self.alias(n, v, "nchw")
        if v.kind == "vec" and shp is not None and int(np.prod(shp)) == v.C:
            return self.alias(n, v)
        if (v.kind == "act" and shp is not None and len(shp) == 2 and
                v.H * v.W == 1):
            return self.alias(n, v, kind="vec")
        raise Unsupported(
            f"reshape {g.shape.get(n.input[0])} -> {shp} of {v}")

    def _r_reduce(self, n, v):
        """ReduceMean / ReduceSum of an activation: SED, SE or a mean."""
        if n.op_type == "ReduceMean" and self.r_sed(n, v):
            return True
        if self.r_se(n, v):
            return True
        return self.r_mean(n, v)

    def _r_mul(self, n, v, ins):
        """Mul: the square of a prototype head or a per-channel affine.

        Returns:
            True, or None when neither applies.
        """
        if v is not None and n.input[0] == n.input[1] and v.kind == "act":
            return self.r_proto(n, v)
        if sum(x is not None for x in ins) == 1:
            a = [x for x in ins if x is not None][0]
            if a.kind == "act":
                return self.r_affine(n, a)
        return None

    def _r_add(self, n, ins):
        """Add of two activations of equal shape -> ADD (None: no match)."""
        if len(ins) != 2 or ins[0] is None or ins[1] is None:
            return None
        a, b = ins
        if not (a.kind == "act" and b.kind == "act" and
                (a.H, a.W, a.C) == (b.H, b.W, b.C) and a.layout == b.layout):
            return None
        out = Val("act", a.H, a.W, a.C, a.layout)
        op = dict(op="ADD", ins=[a.id, b.id], out=out.id, nodes=[n.name])
        return self._emit(op, n.output[0], out, [n])

    def _r_relu(self, n, v):
        """Relu on an activation, or on the vector a dense layer made."""
        if v.kind == "vec":
            prod = next((o for o in reversed(self.ops) if o["out"] == v.id),
                        None)
            if not prod or prod["op"] != "DENSE":
                raise Unsupported(
                    "relu on a vector produced by "
                    f"{prod['op'] if prod else 'the front end'}: on vectors "
                    "the engine fuses relu into a dense layer only (a hidden "
                    "layer: Dense -> ReLU -> Dense)")
        if v.kind == "act":
            out = Val("act", v.H, v.W, v.C, v.layout)
        else:
            out = Val("vec", c=v.C)
        op = dict(op="ACT",
                  act="relu",
                  ins=[v.id],
                  out=out.id,
                  nodes=[n.name])
        return self._emit(op, n.output[0], out, [n])

    def _r_pool(self, n, v):
        """Unpadded MaxPool / AveragePool of an NCHW activation -> POOL."""
        g = self.g
        a = g.attrs(n)
        sh = g.shape[n.output[0]]
        if (a.get("pads", [0, 0, 0, 0]) != [0, 0, 0, 0] and
                a.get("pads") is not None):
            raise Unsupported("padded pooling")
        out = Val("act", sh[2], sh[3], sh[1], "nchw")
        op = dict(op="POOL",
                  kind="max" if n.op_type == "MaxPool" else "avg",
                  k=list(a["kernel_shape"]),
                  s=list(a.get("strides", [1, 1])),
                  ins=[v.id],
                  out=out.id,
                  nodes=[n.name])
        return self._emit(op, n.output[0], out, [n])

    def _r_concat(self, n, v, ins):
        """Channel Concat of activations -> CONCAT."""
        ax = self.g.attrs(n)["axis"]
        if (v.layout == "nchw" and ax != 1) or (v.layout == "nhwc" and
                                                ax not in (3, -1)):
            raise Unsupported("concat along a non-channel axis")
        out = Val("act", v.H, v.W, sum(x.C for x in ins), v.layout)
        op = dict(op="CONCAT",
                  ins=[x.id for x in ins],
                  out=out.id,
                  nodes=[n.name])
        return self._emit(op, n.output[0], out, [n])

    def _r_matmul(self, n, v):
        """MatMul / Gemm on a vector -> DENSE."""
        if n.op_type == "Gemm":
            a = self.g.attrs(n)
            has_bias = len(n.input) > 2 and n.input[2]
            if (a.get("alpha", 1.0) != 1.0 or a.get("transA", 0) or
                (has_bias and a.get("beta", 1.0) != 1.0)):
                raise Unsupported(
                    f"Gemm with attributes {a} (alpha and beta 1, no transA "
                    "are lowered)")
            if a.get("transB", 0) or has_bias:
                return self.r_gemm(n, v, a)
        return self.r_dense(n, v)

    # ---- vector algebra, GeM, SED attention heads
    def r_gemm(self, n, v, a):
        """Gemm on a vector with transB and / or a bias input -> DENSE.

        Args:
            n: the Gemm node.
            v: its input vector.
            a: the node attributes.

        Returns:
            True (the DENSE op holds the weights as [N][K]).

        Raises:
            Unsupported: B or the bias is not a constant of the right shape.
        """
        g = self.g
        bmat = g.const.get(n.input[1])
        if bmat is None or np.asarray(bmat).ndim != 2:
            raise Unsupported("Gemm with a non-constant or non-2-D B")
        wmat = np.asarray(bmat, np.float32)
        wmat = wmat if a.get("transB", 0) else wmat.T  # -> [N][K]
        if wmat.shape[1] != v.C:
            raise Unsupported(
                f"Gemm with B {list(np.asarray(bmat).shape)} (transB "
                f"{a.get('transB', 0)}) on a {v.C}-wide vector")
        b = np.zeros(wmat.shape[0], np.float32)
        if len(n.input) > 2 and n.input[2]:
            cb = g.const.get(n.input[2])
            if cb is None or np.asarray(cb).size not in (1, wmat.shape[0]):
                raise Unsupported(
                    "Gemm bias that is not a constant of N or 1 values")
            flat = np.asarray(cb, np.float32).reshape(-1)
            b = np.broadcast_to(flat, (wmat.shape[0],)).copy()
        out = Val("vec", c=wmat.shape[0])
        op = dict(op="DENSE",
                  ins=[v.id],
                  out=out.id,
                  w=np.ascontiguousarray(wmat),
                  b=b,
                  nodes=[n.name])
        return self._emit(op, n.output[0], out, [n])

    def r_vec(self, n, ins):
        """Element-wise ops on [N] vectors -> VADD / VAFFINE.

        a + b, a + c, a - c, c + a, a * s, a / s (c a constant [N] or scalar,
        s a scalar).

        Args:
            n: the node.
            ins: the IR values of its inputs (None for constants).

        Returns:
            True when an op was emitted, False when the node is none of the
            forms above.

        Raises:
            Unsupported: an Add of two values that are not equal vectors.
        """
        g = self.g
        op = n.op_type
        vs = [x for x in ins if x is not None]
        if op == "Add" and len(vs) == 2:
            return self._r_vadd(n, vs[0], vs[1])
        if len(vs) != 1 or len(n.input) != 2:
            return False
        a = vs[0]
        ci = [x for x in n.input if x in g.const]
        if len(ci) != 1:
            return False
        c = np.asarray(g.const[ci[0]], np.float64).reshape(-1)
        first = n.input[0] in g.const
        if op in ("Mul", "Div") and c.size == 1 and not (op == "Div" and first):
            coef = float(c[0])
        elif (op in ("Add", "Sub") and c.size in (1, a.C) and
              not (op == "Sub" and first)):
            coef = np.broadcast_to(c, (a.C,)).astype(np.float32)
        else:
            return False
        out = Val("vec", c=a.C)
        vaffine = dict(op="VAFFINE",
                       ins=[a.id],
                       out=out.id,
                       kind=op.lower(),
                       c=coef,
                       nodes=[n.name])
        return self._emit(vaffine, n.output[0], out, [n])

    def _r_vadd(self, n, a, b):
        """Add of two vectors of equal length -> VADD."""
        if a.kind != "vec" or b.kind != "vec" or a.C != b.C:
            raise Unsupported(
                f"Add of {a} and {b}: only vectors of equal length")
        out = Val("vec", c=a.C)
        op = dict(op="VADD", ins=[a.id, b.id], out=out.id, nodes=[n.name])
        return self._emit(op, n.output[0], out, [n])

    def _pow_exponent(self, n):
        """Constant scalar exponent of Pow node n (None: no such Pow)."""
        if n is None:
            return None
        e = self.g.const.get(n.input[1])
        if e is None or np.asarray(e).size != 1:
            return None
        return first_float(e)

    def r_gem(self, n, v):
        """GeM pooling -> GEM.

        Clip(min eps) -> Pow p -> spatial mean -> Pow q -> vector [C].

        Returns:
            True.

        Raises:
            Unsupported: the Clip does not start a GeM pooling.
        """
        lo = self.opt_const(n, 1)
        if lo is None or (len(n.input) > 2 and n.input[2]):
            raise Unsupported(
                "Clip on an activation other than the GeM floor (constant "
                "minimum, no maximum)")
        eps = first_float(lo)
        p1 = self.only_cons(n.output[0], "Pow")
        p = self._pow_exponent(p1)
        if p is None:
            raise Unsupported(
                "Clip not followed by Pow(constant p): not a GeM pooling")
        rm = self.only_cons(p1.output[0], ("ReduceMean",))
        if rm is None:
            raise Unsupported("GeM: expected the spatial mean after Pow")
        ok, _ = self.spatial_axes(rm, v)
        if not ok:
            raise Unsupported("GeM: the mean is not over the spatial axes")
        p2 = self.only_cons(rm.output[0], "Pow")
        q = self._pow_exponent(p2)
        if q is None:
            raise Unsupported(
                "GeM: expected Pow(constant 1/p) after the mean")
        if not (eps > 0 and p > 0 and abs(p * q - 1) < 1e-5):
            raise Unsupported(
                f"GeM with eps {eps}, p {p}, q {q} (expected eps > 0, p > 0, "
                "q = 1/p)")
        out = Val("vec", c=v.C)
        op = dict(op="GEM",
                  ins=[v.id],
                  out=out.id,
                  eps=eps,
                  p=p,
                  q=q,
                  nodes=[n.name, p1.name, rm.name, p2.name])
        return self._emit(op, p2.output[0], out, [n, p1, rm, p2])

    def r_sed(self, n, v):
        """SED attention branch on an NCHW activation -> SED.

        ReduceMean over ONE spatial axis -> {MaxPool1d | depthwise Conv1d}
        (summed when both) -> Transpose -> MatMul + bias -> Relu -> Transpose
        -> {att Conv1d k1 -> [Clip] -> Softmax over time | cla Conv1d k1} ->
        Mul -> ReduceSum over time -> [Mul scale] -> vector [n_classes].

        Returns:
            True, or False when the mean is not over one spatial axis.

        Raises:
            Unsupported: the axis mean matched but a later node did not.
        """
        g = self.g
        a = g.attrs(n)
        axes = a.get("axes") or self._cint(n, 1)
        if (v.layout != "nchw" or len(axes) != 1 or
                int(axes[0]) % 4 not in (2, 3) or a.get("keepdims", 1) != 0):
            return False
        # 3: mean over W (time steps = H rows), 2: mean over H
        red = int(axes[0]) % 4
        length, n_ch = (v.H, v.C) if red == 3 else (v.W, v.C)
        nodes = [n]
        pool, dconv, t = self._sed_smooth(n.output[0], nodes, n_ch)
        w1, b1, h = self._sed_hidden(t, nodes, n_ch)
        heads, x, cla = self._sed_heads(h, nodes, np.asarray(w1).shape[1])
        scale, t = self._sed_sum(x, cla, nodes)
        out = Val("vec", c=heads["wa"].shape[0])
        op = dict(op="SED",
                  ins=[v.id],
                  out=out.id,
                  red=red,
                  L=length,
                  C=n_ch,
                  pool=pool,
                  dconv=dconv,
                  w1=np.asarray(w1, np.float32).T.copy(),
                  b1=np.asarray(b1, np.float32).reshape(-1),
                  wa=heads["wa"],
                  ba=heads["ba"],
                  wc=heads["wc"],
                  bc=heads["bc"],
                  clip=heads["clip"],
                  scale=scale,
                  nodes=[x_.name for x_ in nodes])
        return self._emit(op, t, out, nodes)

    def _sed_smooth(self, s, nodes, n_ch):
        """MaxPool1d and / or depthwise Conv1d on the axis mean s.

        Returns:
            (pool (k, pad) | None, dconv (w, pad, bias) | None, out tensor).
        """
        g = self.g
        cs = g.cons.get(s, [])
        mp = [c for c in cs if c.op_type == "MaxPool"]
        dc = [c for c in cs if c.op_type == "Conv"]
        if (len(cs) not in (1, 2) or len(mp) > 1 or len(dc) > 1 or
                len(mp) + len(dc) != len(cs)):
            raise Unsupported(
                "SED branch: expected MaxPool1d and / or a depthwise Conv1d "
                f"on the axis mean, found {[c.op_type for c in cs]}")
        pool = None
        if mp:
            pool = self._sed_pool(mp[0])
            nodes.append(mp[0])
        dconv = None
        if dc:
            dconv = self._sed_dconv(dc[0], n_ch)
            nodes.append(dc[0])
        t = (mp or dc)[0].output[0]
        if mp and dc:
            ad = self.only_cons(mp[0].output[0], "Add")
            if ad is None or set(
                    ad.input) != {mp[0].output[0], dc[0].output[0]}:
                raise Unsupported(
                    "SED branch: expected MaxPool1d + Conv1d summed")
            t = take(nodes, ad)
        return pool, dconv, t

    def _sed_pool(self, mp):
        """MaxPool1d of the SED branch: (kernel, pad)."""
        pa = self.g.attrs(mp)
        k = list(pa.get("kernel_shape", []))
        if (len(k) != 1 or list(pa.get("strides", [1])) != [1] or
                pa.get("ceil_mode", 0) or
                list(pa.get("dilations", [1])) != [1] or
                len(set(pa.get("pads", [0, 0]))) != 1):
            raise Unsupported(
                f"SED MaxPool1d {pa}: stride 1, symmetric pads, no dilation")
        return k[0], int(pa.get("pads", [0, 0])[0])

    def _sed_dconv(self, dc, n_ch):
        """Depthwise Conv1d of the SED branch: (weights [C][k], pad, bias)."""
        g = self.g
        w = g.const.get(dc.input[1])
        ca = g.attrs(dc)
        if (w is None or np.asarray(w).shape[:2] != (n_ch, 1) or
                np.asarray(w).ndim != 3 or ca.get("group", 1) != n_ch or
                list(ca.get("strides", [1])) != [1] or
                len(set(ca.get("pads", [0, 0]))) != 1):
            raise Unsupported(
                "SED Conv1d: expected a depthwise [C, 1, k] kernel, stride 1, "
                "symmetric pads")
        bb = self.bias(dc, n_ch, np.float32)
        weights = np.asarray(w, np.float32)[:, 0, :].copy()
        return weights, int(ca.get("pads", [0, 0])[0]), bb

    def _sed_hidden(self, t, nodes, n_ch):
        """Transpose -> MatMul + bias -> Relu -> Transpose.

        Returns:
            (hidden weights [C][hidden], hidden bias, the hidden sequence).
        """
        g = self.g
        tr = self.only_cons(t, "Transpose")
        mm = tr and self.only_cons(tr.output[0], "MatMul")
        w1 = g.const.get(mm.input[1]) if mm else None
        if (w1 is None or np.asarray(w1).ndim != 2 or
                np.asarray(w1).shape[0] != n_ch or
                list(g.attrs(tr)["perm"]) != [0, 2, 1]):
            raise Unsupported(
                "SED branch: expected Transpose [0, 2, 1] -> MatMul by a "
                "constant [C, hidden]")
        hid = np.asarray(w1).shape[1]
        ab = self.only_cons(mm.output[0], "Add")
        b1 = g.cval(ab) if ab else None
        if b1 is None or np.asarray(b1).size != hid:
            raise Unsupported("SED branch: expected the hidden bias Add")
        rl = self.only_cons(ab.output[0], "Relu")
        tr2 = rl and self.only_cons(rl.output[0], "Transpose")
        if tr2 is None or list(g.attrs(tr2)["perm"]) != [0, 2, 1]:
            raise Unsupported(
                "SED branch: expected Relu -> Transpose [0, 2, 1] after the "
                "hidden layer")
        nodes += [tr, mm, ab, rl, tr2]
        return w1, b1, tr2.output[0]

    def _sed_heads(self, h, nodes, hid):
        """Attention and class Conv1d on the hidden sequence h.

        Returns:
            (dict with wa, ba, wc, bc, clip; the Softmax node; the class
            conv node).
        """
        g = self.g
        cv = g.cons.get(h, [])
        if len(cv) != 2 or any(c.op_type != "Conv" for c in cv):
            raise Unsupported(
                "SED branch: expected two 1x1 Conv1d (attention, classes) on "
                "the hidden sequence")
        att = cla = None
        for c in cv:
            nx = self.only_cons(c.output[0])
            if nx is not None and nx.op_type in ("Clip", "Softmax"):
                att = c
            else:
                cla = c
        if att is None or cla is None:
            raise Unsupported(
                "SED branch: expected one Conv1d followed by [Clip ->] "
                "Softmax (attention) and one not")
        wa, ba = self._sed_k1(att, hid)
        wc, bc = self._sed_k1(cla, hid)
        if wa.shape[0] != wc.shape[0]:
            raise Unsupported(
                "SED branch: attention and class convs of different widths")
        clip, x = self._sed_softmax(att, nodes)
        nodes += [att, cla, x]
        return dict(wa=wa, ba=ba, wc=wc, bc=bc, clip=clip), x, cla

    def _sed_k1(self, c, hid):
        """1x1 Conv1d c on the hidden sequence: (weights [N][hid], bias)."""
        g = self.g
        w = g.const.get(c.input[1])
        if (w is None or np.asarray(w).ndim != 3 or
                np.asarray(w).shape[1:] != (hid, 1)):
            raise Unsupported(
                f"SED branch: {g.where(c)} is not a 1x1 Conv1d on the "
                f"{hid}-wide hidden sequence")
        b = self.bias(c, np.asarray(w).shape[0], np.float32)
        return np.asarray(w, np.float32)[:, :, 0].copy(), b

    def _sed_softmax(self, att, nodes):
        """[Clip] -> Softmax over time after the attention conv.

        Returns:
            (clip bounds (lo, hi) | None, the Softmax node).
        """
        g = self.g
        clip = None
        x = self.only_cons(att.output[0])
        if x.op_type == "Clip":
            lo = self.opt_const(x, 1)
            hi = self.opt_const(x, 2)
            if lo is None or hi is None:
                raise Unsupported("SED attention Clip without constant bounds")
            clip = (first_float(lo), first_float(hi))
            nodes.append(x)
            x = self.only_cons(x.output[0])
        if (x is None or x.op_type != "Softmax" or
                int(g.attrs(x).get("axis", -1)) % 3 != 2):
            raise Unsupported(
                "SED branch: expected Softmax over the time axis of the "
                "attention")
        return clip, x

    def _sed_sum(self, x, cla, nodes):
        """Mul(softmax, classes) -> ReduceSum over time -> [Mul scale].

        Returns:
            (scale, out tensor).
        """
        g = self.g
        mu = self.only_cons(x.output[0], "Mul")
        if mu is None or set(mu.input) != {x.output[0], cla.output[0]}:
            raise Unsupported("SED branch: expected Mul(softmax, classes)")
        rs = self.only_cons(mu.output[0], "ReduceSum")
        rax = (g.attrs(rs).get("axes") or self._cint(rs, 1)) if rs else None
        if (rs is None or len(rax) != 1 or int(rax[0]) % 3 != 2 or
                g.attrs(rs).get("keepdims", 1) != 0):
            raise Unsupported(
                "SED branch: expected ReduceSum over time (keepdims 0)")
        nodes += [mu, rs]
        t = rs.output[0]
        scale = 1.0
        sc = self.only_cons(t)
        if (sc is not None and sc.op_type == "Mul" and
                g.cval(sc) is not None and np.asarray(g.cval(sc)).size == 1):
            scale = first_float(g.cval(sc))
            t = take(nodes, sc)
        return scale, t

    # ---- convs, means, SE, dense and prototype heads
    def r_conv(self, n, v):
        """Conv with its folded BN, activation -> CONV / DWCONV.

        Conv -> [Transpose to NHWC] -> [per-channel Mul, Add (BN) | Add] ->
        [swish | relu].

        Returns:
            True.

        Raises:
            Unsupported: a conv the engine has no IR op for.
        """
        g = self.g
        if v.layout != "nchw":
            raise Unsupported("conv on a non-NCHW activation")
        w = g.const.get(n.input[1])
        if w is None:
            raise Unsupported("conv with non-constant weights")
        w = np.asarray(w, np.float64)
        b = self.bias(n, w.shape[0], np.float64)
        a = g.attrs(n)
        grp = a.get("group", 1)
        if a.get("dilations", [1, 1]) != [1, 1]:
            raise Unsupported("dilated conv")
        sh = g.shape[n.output[0]]
        c_out = sh[1]
        k = [w.shape[2], w.shape[3]]
        s = list(a.get("strides", [1, 1]))
        pads = conv_pads(a, (v.H, v.W), (sh[2], sh[3]), k, s)
        nodes = [n]
        w, b, bn, layout, t = self._conv_affine(n, w, b, c_out, nodes)
        act = None
        m = self.match_act(t)
        # a plain sigmoid stays a node of its own (an output activation:
        # --logits)
        if m and m[0] in ("swish", "relu"):
            act, an, t = m
            nodes += an
        dw = grp > 1
        if dw and not (grp == v.C == c_out and w.shape[1] == 1):
            raise Unsupported(
                f"grouped conv (group {grp}) other than depthwise")
        out = Val("act", sh[2], sh[3], c_out, layout)
        op = dict(op="DWCONV" if dw else "CONV",
                  ins=[v.id],
                  out=out.id,
                  w=w.astype(np.float32),
                  b=b.astype(np.float32),
                  k=k,
                  s=s,
                  pads=pads,
                  act=act,
                  bn_folded=bn,
                  nodes=[x.name for x in nodes])
        return self._emit(op, t, out, nodes)

    def _conv_affine(self, n, w, b, c_out, nodes):
        """Folds what follows conv n into its float64 weights and bias.

        An optional Transpose back to NHWC, then a per-channel Mul + Add (BN,
        when fold_bn) or a per-channel Add.

        Returns:
            (weights, bias, BN folded?, layout, out tensor).
        """
        g = self.g
        t = n.output[0]
        layout = "nchw"
        c = self.only_cons(t)
        if (c and c.op_type == "Transpose" and
                list(g.attrs(c)["perm"]) == [0, 2, 3, 1]):
            t = take(nodes, c)
            layout = "nhwc"
            c = self.only_cons(t)
        bn = False
        if c and c.op_type == "Mul" and self.fold_bn:
            sc = self.chan_vec(g.cval(c), c_out, layout)
            ad = self.only_cons(c.output[0], "Add")
            shift = self.chan_vec(g.cval(ad), c_out, layout) if ad else None
            if sc is not None and shift is not None:
                w = w * sc[:, None, None, None]
                b = b * sc + shift
                nodes += [c, ad]
                t = ad.output[0]
                bn = True
        elif (c and c.op_type == "Add" and
              self.chan_vec(g.cval(c), c_out, layout) is not None):
            b = b + self.chan_vec(g.cval(c), c_out, layout)
            t = take(nodes, c)
        return w, b, bn, layout, t

    def spatial_axes(self, n, v):
        """Whether reduction n runs over the spatial axes of v.

        Returns:
            (axes are exactly H and W, the keepdims attribute).
        """
        a = self.g.attrs(n)
        axes = a.get("axes")
        if axes is None and len(n.input) > 1:
            axes = list(np.asarray(self.g.const[n.input[1]]).reshape(-1))
        axes = sorted(int(x) % 4 for x in axes)
        spatial = [2, 3] if v.layout == "nchw" else [1, 2]
        return axes == spatial, a.get("keepdims", 1)

    def _mean_tail(self, n, v, nodes):
        """Output of a spatial mean n: ReduceMean, or ReduceSum / (H*W).

        Returns:
            The mean tensor, or None when a ReduceSum is not divided by H*W.
        """
        t = n.output[0]
        if n.op_type != "ReduceSum":
            return t
        d = self.only_cons(t, "Div")
        if not d or self.const_scalar(d) != float(v.H * v.W):
            return None
        return take(nodes, d)

    def r_mean(self, n, v):
        """Spatial mean of an activation -> MEAN (a vector [C]).

        Returns:
            True.

        Raises:
            Unsupported: not a mean over H and W.
        """
        ok, _ = self.spatial_axes(n, v)
        if not ok:
            raise Unsupported(f"reduction over non-spatial axes of {v}")
        nodes = [n]
        t = self._mean_tail(n, v, nodes)
        if t is None:
            raise Unsupported("ReduceSum not followed by / (H*W)")
        out = Val("vec", c=v.C)
        op = dict(op="MEAN",
                  ins=[v.id],
                  out=out.id,
                  nodes=[x.name for x in nodes])
        return self._emit(op, t, out, nodes)

    def _bias_add(self, t, size, nodes):
        """Optional constant Add of `size` values after t: (bias, out)."""
        g = self.g
        b = np.zeros(size, np.float32)
        a = self.only_cons(t, "Add")
        if a is not None and g.cval(a) is not None and g.cval(a).size == size:
            b = np.asarray(g.cval(a), np.float32).reshape(-1)
            t = take(nodes, a)
        return b, t

    def fc(self, t, n_in):
        """Fully connected layer on a pooled vector.

        MatMul const [C][R] (+ Add) or Conv 1x1 const on [1, C, 1, 1] (with
        bias), with layout ops in between.

        Args:
            t: the tensor holding the vector.
            n_in: its width.

        Returns:
            (W [R][C], b [R], nodes, out tensor), or None.
        """
        g = self.g
        nodes = []
        t, n = self._layout(t, nodes, ("Reshape", "Unsqueeze", "Squeeze"))
        if not n:
            return None
        if (n.op_type == "MatMul" and g.cval(n) is not None and
                g.cval(n).shape[0] == n_in):
            wmat = np.asarray(g.cval(n), np.float32).T
            t = take(nodes, n)
            b, t = self._bias_add(t, wmat.shape[0], nodes)
            return wmat, b, nodes, t
        if n.op_type == "Conv" and g.const.get(n.input[1]) is not None:
            w = np.asarray(g.const[n.input[1]], np.float32)
            if w.shape[2:] != (1, 1) or w.shape[1] != n_in:
                return None
            b = self.bias(n, w.shape[0], np.float32)
            nodes.append(n)
            return w[:, :, 0, 0], b, nodes, n.output[0]
        return None

    def r_se(self, n, v):
        """Squeeze-and-excitation -> SE.

        Spatial mean of v -> FC -> act -> FC -> sigmoid -> Mul(v, gate).

        Returns:
            True, or False when the mean does not start an SE block.
        """
        ok, _ = self.spatial_axes(n, v)
        if not ok:
            return False
        nodes = [n]
        t = self._mean_tail(n, v, nodes)
        if t is None:
            return False
        f1 = self.fc(t, v.C)
        if not f1:
            return False
        w1, b1, n1, t = f1
        m = self.match_act(t)
        if not m or m[0] not in ("swish", "relu"):
            return False
        act1, an, t = m
        f2 = self.fc(t, w1.shape[0])
        if not f2:
            return False
        w2, b2, n2, t = f2
        m = self.match_act(t)
        if not m or m[0] != "sigmoid":
            return False
        _, sn, t = m
        lay = []
        t, c = self._layout(t, lay, ("Reshape", "Unsqueeze"))
        xv = None
        if c and c.op_type == "Mul":
            xv = self.val.get(self.g.other_in(c, t))
        # the gate multiplies the same value (possibly seen in another layout)
        if xv is None or xv.id != v.id:
            return False
        nodes += n1 + an + n2 + sn + lay + [c]
        out = Val("act", v.H, v.W, v.C, xv.layout)
        op = dict(op="SE",
                  ins=[v.id],
                  out=out.id,
                  w1=w1,
                  b1=b1,
                  act1=act1,
                  w2=w2,
                  b2=b2,
                  nodes=[x.name for x in nodes])
        return self._emit(op, c.output[0], out, nodes)

    def r_affine(self, n, v):
        """Per-channel Mul + Add on an activation -> AFFINE.

        Returns:
            True.

        Raises:
            Unsupported: the Mul or the Add is not per channel.
        """
        g = self.g
        s = self.chan_vec(g.cval(n), v.C, v.layout)
        a = self.only_cons(n.output[0], "Add")
        t = self.chan_vec(g.cval(a), v.C, v.layout) if a else None
        if s is None or t is None:
            raise Unsupported("per-channel Mul without a per-channel Add")
        out = Val("act", v.H, v.W, v.C, v.layout)
        op = dict(op="AFFINE",
                  ins=[v.id],
                  out=out.id,
                  s=s.astype(np.float32),
                  t=t.astype(np.float32),
                  nodes=[n.name, a.name])
        return self._emit(op, a.output[0], out, [n, a])

    def r_dense(self, n, v):
        """MatMul by a constant [C][N] (+ bias Add) on a vector -> DENSE.

        Returns:
            True.

        Raises:
            Unsupported: the matrix is not a constant of the right shape.
        """
        wmat = self.g.cval(n)
        if wmat is None or wmat.ndim != 2 or wmat.shape[0] != v.C:
            raise Unsupported("MatMul with a non-constant or mismatched matrix")
        nodes = [n]
        b, t = self._bias_add(n.output[0], wmat.shape[1], nodes)
        out = Val("vec", c=wmat.shape[1])
        op = dict(op="DENSE",
                  ins=[v.id],
                  out=out.id,
                  w=np.asarray(wmat, np.float32).T.copy(),
                  b=b,
                  nodes=[x.name for x in nodes])
        return self._emit(op, t, out, nodes)

    def r_proto(self, n, v):
        """ProtoPNet head -> PROTO.

        x / (sqrt(sum_c x^2) + eps) -> MatMul P [C][NP] -> Reshape
        [.., H, W, NC, ppc] -> ReduceMax over the cells -> Mul w [NC][ppc] ->
        ReduceSum(ppc) -> Add b.

        Args:
            n: the Mul x x node that starts the L2 norm.
            v: the NHWC activation x.

        Returns:
            True.

        Raises:
            Unsupported: a node of the head did not match.
        """
        g = self.g
        norm = self._proto_l2(n, v)
        eps = self.const_scalar(norm[2])
        mm = self.only_cons(norm[3].output[0], "MatMul")
        protos = mm is not None and g.cval(mm)
        if mm is None or protos is None or protos.shape[0] != v.C:
            raise Unsupported("prototype MatMul")
        rh = self.only_cons(mm.output[0], "Reshape")
        rm = rh and self.only_cons(rh.output[0], "ReduceMax")
        if not rm:
            raise Unsupported("prototype Reshape + ReduceMax")
        rsh = g.shape[rh.output[0]]
        n_cls, ppc = rsh[-2], rsh[-1]
        mw = self.only_cons(rm.output[0], "Mul")
        cw = mw is not None and g.cval(mw)
        rs2 = mw and self.only_cons(mw.output[0], "ReduceSum")
        ab = rs2 and self.only_cons(rs2.output[0], "Add")
        cb = ab is not None and ab and g.cval(ab)
        if (not ab or cw is None or cb is None or cw.size != n_cls * ppc or
                cb.size != n_cls):
            raise Unsupported("class weights / bias of the prototype head")
        nodes = [n] + norm + [mm, rh, rm, mw, rs2, ab]
        out = Val("vec", c=n_cls)
        op = dict(op="PROTO",
                  ins=[v.id],
                  out=out.id,
                  eps=eps,
                  P=np.asarray(protos, np.float32).T.copy(),
                  cw=np.asarray(cw, np.float32).reshape(n_cls, ppc),
                  cb=np.asarray(cb, np.float32).reshape(-1),
                  ppc=ppc,
                  nodes=[x.name for x in nodes])
        return self._emit(op, ab.output[0], out, nodes)

    def _proto_l2(self, n, v):
        """L2 normalization after the square n: [ReduceSum, Sqrt, Add, Div]."""
        rs = self.only_cons(n.output[0], "ReduceSum")
        sq = rs and self.only_cons(rs.output[0], "Sqrt")
        ad = sq and self.only_cons(sq.output[0], "Add")
        dv = ad and self.only_cons(ad.output[0], "Div")
        if not dv or dv.input[0] != n.input[0] or v.layout != "nhwc":
            raise Unsupported("L2 normalization pattern")
        return [rs, sq, ad, dv]


def json_attrs(attrs):
    """Node attributes as values a JSON report can hold.

    Args:
        attrs: attribute name -> value (bytes, arrays, lists, numbers).

    Returns:
        A dict with bytes decoded, arrays as lists, integer lists as int and
        anything else that is not int / float / str / list as its str().
    """
    out = {}
    for k, v in attrs.items():
        if isinstance(v, bytes):
            v = v.decode(errors="replace")
        elif hasattr(v, "tolist"):
            v = v.tolist()
        elif isinstance(v, (list, tuple)):
            v = [int(x) if isinstance(x, (int, np.integer)) else x for x in v]
        out[k] = v if isinstance(v, (int, float, str, list)) else str(v)
    return out


def guidance(op_type, why, down):
    """Nearest supported pattern and next step for an unsupported node.

    Args:
        op_type: ONNX op type of the node.
        why: the reason the recognizer gave.
        down: the tensor of an earlier unsupported node this one depends on,
            or None.

    Returns:
        (nearest supported pattern, next step), both user-facing text.
    """
    near, step = NEAREST.get(op_type, NEAREST_DEFAULT)
    if (why.startswith("front end:") or
            why.startswith("no supported front end")):
        near = ("the BirdNET v2.4 dual STFT or a framed log-mel with the ONNX "
                "DFT op (guide section 2), matched node by node")
        step = (
            "the message names the node the matcher stopped at: export the "
            "spectrogram in one of the two supported forms, or extend "
            "fe_stft2 / fe_logmel in python/bnneon/compiler/recog.py (guide "
            "section 6)")
    if why.startswith("front end: a Conv-framed DFT"):
        step = (
            "stock BirdNET v3.0 compiles (guide section 2.1); a modified "
            "v3.0-style front end: compare it with fe_dbimg in "
            "python/bnneon/compiler/recog.py, or add a front-end rule and "
            "kernel (guide section 6)")
    if down:
        step = (f"fix the unsupported node producing '{down}' first; this "
                "node may then match")
    return near, step


def problem_report(g, r: Recognizer) -> list:
    """Describes every unsupported node of a recognized graph.

    Args:
        g: the graph (onnxg.Graph).
        r: the Recognizer after walk().

    Returns:
        One dict per problem: index, op type, name, reason, attributes,
        input / output shapes, whether it is only downstream of another
        unsupported node, the nearest supported pattern and the next step.
    """
    bad_out = set()
    out = []
    for n, why in r.problems:
        if n is None:
            out.append(
                dict(index=None,
                     op_type="?",
                     name="",
                     reason=why,
                     attrs={},
                     inputs=[],
                     outputs=[],
                     downstream_of=None,
                     nearest=NEAREST_DEFAULT[0],
                     next_step=NEAREST_DEFAULT[1]))
            continue
        attrs = json_attrs(g.attrs(n))
        ins = [
            dict(name=x, shape=g.shape.get(x), const=x in g.const)
            for x in n.input if x
        ]
        outs = [dict(name=x, shape=g.shape.get(x)) for x in n.output if x]
        up = [x for x in n.input if x in bad_out]
        down = up[0] if up else None
        if down is None and r.fe_failed is not None and n is not r.fe_failed:
            # no front end: nothing downstream can match
            down = r.fe_failed.output[0] if r.fe_failed.output else "?"
        near, step = guidance(n.op_type, why, down)
        out.append(
            dict(index=g.pos.get(id(n)),
                 op_type=n.op_type,
                 name=n.name,
                 reason=why,
                 attrs=attrs,
                 inputs=ins,
                 outputs=outs,
                 downstream_of=down,
                 nearest=near,
                 next_step=step))
        bad_out.update(x for x in n.output if x)
    return out


def format_problems(probs: list, verbose: bool = False) -> list:
    """Formats a problem report as text lines.

    Args:
        probs: the dicts of problem_report().
        verbose: list the downstream problems too (default: as a count).

    Returns:
        The lines: primary problems in full, then one line counting the
        downstream ones.
    """
    lines = []
    down = [p for p in probs if p["downstream_of"]]
    for p in probs:
        if p["downstream_of"] and not verbose:
            continue
        lines += format_problem(p)
    if down and not verbose:
        names = ", ".join(f"{p['op_type']} '{p['name']}'" for p in down[:3])
        lines.append(
            f"({len(down)} downstream node(s) not listed: "
            f"{names}{', ...' if len(down) > 3 else ''}; --verbose lists them)")
    return lines


def format_problem(p: dict) -> list:
    """Formats one problem of problem_report() as text lines."""
    if p["index"] is not None:
        where = f"node {p['index']} {p['op_type']} '{p['name']}'"
    else:
        where = "graph input"
    lines = [f"unsupported op {p['op_type']} at {where}: {p['reason']}"]
    if p["downstream_of"]:
        lines.append("    (downstream of the unsupported node producing "
                     f"'{p['downstream_of']}')")
        return lines
    if p["attrs"]:
        lines.append(f"    attributes: {p['attrs']}")
    if p["inputs"]:
        lines.append("    inputs: " + ", ".join(
            f"{i['name']} {i['shape']}{' (const)' if i['const'] else ''}"
            for i in p["inputs"]))
    if p["outputs"]:
        lines.append("    outputs: " +
                     ", ".join(f"{o['name']} {o['shape']}"
                               for o in p["outputs"]))
    lines.append(f"    nearest supported pattern: {p['nearest']}")
    lines.append(f"    next step: {p['next_step']}")
    return lines


def _retarget(uses, o, a):
    """Makes the consumers of op a read op o instead; returns them."""
    for x in uses.get(a["out"], []):
        x["ins"] = [o["out"] if i == a["out"] else i for i in x["ins"]]
    uses[o["out"]] = uses.get(a["out"], [])
    return uses[o["out"]]


def _fuse_act(o, u, uses, by_out, outputs_vids, dead):
    """Fuses the ACT that alone consumes op o into o; returns o's users."""
    if not (o["op"] in ("CONV", "AFFINE", "DENSE") and o.get("act") is None and
            len(u) == 1 and u[0]["op"] == "ACT" and
            o["out"] not in outputs_vids):
        return u
    a = u[0]
    o["act"] = a["act"]
    o["nodes"] += a["nodes"]
    dead.add(id(a))
    if a["out"] in outputs_vids:
        # the activated tensor is requested (--logits / --emb): the fused op
        # takes its id
        del by_out[o["out"]]
        o["out"] = a["out"]
    u = _retarget(uses, o, a)
    by_out[o["out"]] = o
    return u


def _fuse_residual(o, u, ops, uses, by_out, outputs_vids, dead):
    """Fuses the residual ADD that alone consumes conv o; returns its users.

    The other ADD input must come from an earlier op or from outside the op
    list.
    """
    if not (o["op"] == "CONV" and len(u) == 1 and u[0]["op"] == "ADD" and
            o["out"] not in outputs_vids):
        return u
    a = u[0]
    other = [i for i in a["ins"] if i != o["out"]]
    if not (len(other) == 1 and other[0] in by_out and
            ops.index(by_out[other[0]]) < ops.index(o) or
            (len(other) == 1 and other[0] not in by_out)):
        return u
    o["res"] = other[0]
    o["nodes"] += a["nodes"]
    dead.add(id(a))
    return _retarget(uses, o, a)


def _fuse_pool(o, u, uses, dead):
    """Fuses conv + relu + (1x2 max, avg pool) + concat into the conv."""
    if not (o["op"] == "CONV" and o.get("act") == "relu" and len(u) == 2 and
            sorted(x["op"] for x in u) == ["POOL", "POOL"]):
        return
    pm = [x for x in u if x.get("kind") == "max"]
    pa = [x for x in u if x.get("kind") == "avg"]
    cu = [uses.get(x["out"], []) for x in u]
    if not (pm and pa and
            all(len(c) == 1 and c[0]["op"] == "CONCAT" for c in cu) and
            cu[0][0] is cu[1][0] and pm[0]["k"] == pa[0]["k"] == [1, 2] and
            pm[0]["s"] == pa[0]["s"] == [1, 2] and
            cu[0][0]["ins"] == [pm[0]["out"], pa[0]["out"]]):
        return
    cat = cu[0][0]
    o["pool"] = "maxavg_w2"
    o["nodes"] += pm[0]["nodes"] + pa[0]["nodes"] + cat["nodes"]
    dead.update({id(pm[0]), id(pa[0]), id(cat)})
    o["out"] = cat["out"]


def fuse(ops: list, outputs_vids) -> list:
    """Fuses neighboring IR ops the kernels run as one.

    conv + act, conv + residual ADD, conv + relu + (max, avg pool) + concat,
    affine + relu. The op dicts are updated in place.

    Args:
        ops: the IR ops in graph order.
        outputs_vids: value ids that must survive (requested outputs).

    Returns:
        The remaining ops, in order.
    """
    by_out = {o["out"]: o for o in ops}
    uses = {}
    for o in ops:
        for i in o["ins"]:
            uses.setdefault(i, []).append(o)
    keep = []
    dead = set()
    for o in ops:
        if id(o) in dead:
            continue
        u = uses.get(o["out"], [])
        u = _fuse_act(o, u, uses, by_out, outputs_vids, dead)
        u = _fuse_residual(o, u, ops, uses, by_out, outputs_vids, dead)
        _fuse_pool(o, u, uses, dead)
        keep.append(o)
    return [o for o in keep if id(o) not in dead]


def dce(ops: list, want) -> list:
    """Drops the IR ops no wanted value depends on.

    Args:
        ops: the IR ops in graph order.
        want: the value ids to keep.

    Returns:
        The ops needed for the wanted values, in order.
    """
    need = set(want)
    keep = []
    for o in reversed(ops):
        if o["out"] in need:
            keep.append(o)
            need.update(o["ins"])
            if o.get("res"):
                need.add(o["res"])
    return list(reversed(keep))


def _scaled(form, kind, c):
    """Affine form after a VAFFINE op (mul, div, add or sub by c)."""
    x, cf, b = form
    if kind == "div":
        # keep the exact division of the bias; coefficients are exact
        # products in float64
        return x, {j: v / c for j, v in cf.items()}, b / c
    if kind == "mul":
        return x, {j: v * c for j, v in cf.items()}, b * c
    d = np.asarray(c, np.float64)
    return x, dict(cf), b + d if kind == "add" else b - d


def _summed(form_a, form_b):
    """Affine form of the sum of two forms of the same input vector."""
    x, ca, ba = form_a
    _, cb, bb = form_b
    cf = dict(ca)
    for j, v in cb.items():
        cf[j] = cf.get(j, 0.0) + v
    return x, cf, ba + bb


def _affine_forms(ops):
    """Affine forms of the vector values made from dense layers.

    Returns:
        (value id -> (input value id, {dense op index: coefficient}, bias
        float64 [N]); the original DENSE ops by index).
    """
    aff = {}
    dense = []
    for o in ops:
        if o["op"] == "DENSE" and o.get("act") is None:
            dense.append(o)
            bias = np.asarray(o["b"], np.float64)
            aff[o["out"]] = (o["ins"][0], {len(dense) - 1: 1.0}, bias)
        elif o["op"] == "VAFFINE" and o["ins"][0] in aff:
            aff[o["out"]] = _scaled(aff[o["ins"][0]], o["kind"], o["c"])
        elif (o["op"] == "VADD" and all(i in aff for i in o["ins"]) and
              aff[o["ins"][0]][0] == aff[o["ins"][1]][0]):
            aff[o["out"]] = _summed(aff[o["ins"][0]], aff[o["ins"][1]])
    return aff, dense


def _folded_dense(o, form, dense):
    """The single DENSE op that computes the affine form of op o's value."""
    x, cf, b = form
    wsum = None
    for j, c in sorted(cf.items()):
        t = np.asarray(dense[j]["w"], np.float64) * c
        wsum = t if wsum is None else wsum + t
    nodes = []
    for j in sorted(cf):
        nodes += dense[j]["nodes"]
    nodes += o["nodes"]
    return dict(op="DENSE",
                ins=[x],
                out=o["out"],
                w=wsum.astype(np.float32),
                b=b.astype(np.float32),
                nodes=nodes,
                folded=len(cf))


def fold_vec(ops: list, want) -> list:
    """Folds every affine function of ONE input vector into one DENSE op.

    Dense layers, their sums, scalings and constant offsets are combined in
    float64: sum_i c_i (W_i x + b_i) + d = (sum_i c_i W_i) x +
    (sum_i c_i b_i + d). Values whose consumers are all foldable are never
    materialized (the 5-head sums of BirdNET v3.0 become one dense layer).

    Args:
        ops: the IR ops in graph order.
        want: value ids that must be materialized (requested outputs).

    Returns:
        The ops with each folded chain replaced by one DENSE op (the input
        list itself when there is nothing to fold).
    """
    uses = {}
    for o in ops:
        for i in o["ins"] + ([o["res"]] if o.get("res") else []):
            uses.setdefault(i, []).append(o)
    aff, dense = _affine_forms(ops)
    if not aff:
        return ops

    def foldable(o):
        return o["op"] in ("VAFFINE", "VADD") and o["out"] in aff

    out = []
    for o in ops:
        v = o["out"]
        standalone = o["op"] == "DENSE" and o.get("act") is None
        needed = v in want or any(not foldable(u) for u in uses.get(v, []))
        if v in aff and not standalone and needed:
            out.append(_folded_dense(o, aff[v], dense))
        else:
            out.append(o)
    return out
