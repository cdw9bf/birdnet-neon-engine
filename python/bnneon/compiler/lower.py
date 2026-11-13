"""Lowering: fused IR -> registry ops + buffers + pre-packed weights per path.

Paths: "neon-fp16" (fp16 activations / GEMMs, Cortex-A76 class: hwcap fp16),
"neon-fp32" (fp32 everything, any AArch64) and "neon-int8" (fp32 with int8
GEMMs, no dotprod, on the layer classes the policy names). The op sequences
reproduce the hand-written engines exactly where they exist (BirdNET v2.4:
v24_fast.c and v24_f32.c), so a compiled v2.4 package is bit-identical to the
legacy .bn24 engine.
"""
# Shapes are named in lower case after the kernels' notation (N, K, Kp, Np,
# H, W, C, Co, Ci as in src/gemm_*.c and the package format): n, k, k_pad,
# n_pad, h, w, c (h_in, w_in, c_in where w is the weights or h a halo), co,
# ci. The l_<op> methods are named after the IR ops they lower, in lower
# case (run() dispatches on o["op"].lower()).
import dataclasses

import numpy as np

from . import pkgfmt
from . import recog

# Registry opcodes (src/rt_ops_*.c); a record's opcode is base | variant << 8.
OP = dict(FE_STFT2=1, C0POOL=2, PW=3, DW3=4, SE=5, POSTPC=6, DENSE=7,
          FE_LOGMEL=8, STEM=9, DWK=10, PAD=11, MEAN=12, PROTO=13,
          PROTO_F32H=14, GEMV=15, FE_DBIMG=16, CONVK=17, GEM=18, SED=19,
          VADD=20, CONVW=21, PW_S8=22, GEMV_S8=23, SED_S8=24, SE_GATE=25,
          PROTO_S8=35)
# Fused activation codes the kernels take.
ACT = {None: 0, "swish": 1, "relu": 2}


def act_code(o: dict) -> int:
    """Returns the kernel code of IR op o's fused activation.

    Raises:
        recog.Unsupported: no kernel fuses that activation.
    """
    a = o.get("act")
    if a not in ACT:
        raise recog.Unsupported(
            f"activation '{a}' fused after a conv has no kernel (swish, relu "
            f"or none); hint: if it is the output activation, compile with "
            f"--logits <the conv's output tensor> and --post")
    return ACT[a]
F32, F16, I8 = 0, 1, 2  # buffer dtypes


@dataclasses.dataclass(frozen=True)
class PathInfo:
    """What a CPU path is, as the compiler and check use it.

    Attributes:
        act: the activation dtype (F16 or F32).
        var: the kernel variant of its records (opcode | var << 8).
        prefix: the prefix of its weight tensors in the package.
        hwcap: the hwcap bits the path needs before any family adds its own
            (1: fp16).
        cli_mode: the engine CLI's mode for it (2 fp32, 3 fp16, 4 int8).
        need_db: the SQNR check asks of it when its GEMMs are not int8.
    """
    act: int
    var: int
    prefix: str
    hwcap: int
    cli_mode: int
    need_db: float


PATHS = {
    "neon-fp16": PathInfo(act=F16, var=1, prefix="f16/", hwcap=1, cli_mode=3,
                          need_db=50.0),
    "neon-fp32": PathInfo(act=F32, var=0, prefix="f32/", hwcap=0, cli_mode=2,
                          need_db=60.0),
    "neon-int8": PathInfo(act=F32, var=0, prefix="s8n/", hwcap=0, cli_mode=4,
                          need_db=60.0),
}


def path_info(path: str) -> PathInfo:
    """PATHS[path]; another name lowers (and runs) as neon-fp32 does."""
    return PATHS.get(path, PATHS["neon-fp32"])


# 2: records the source ONNX node of every op (metadata src_nodes/<path>);
# 3: the license metadata key is "license" (before: "licence")
COMPILER_VERSION = "bnc 3"


def r8(x: int) -> int:
    """Returns x rounded up to a multiple of 8."""
    return (x + 7) & ~7


def r16(x: int) -> int:
    """Returns x rounded up to a multiple of 16."""
    return (x + 15) & ~15


def pack_h16(w: np.ndarray, b: np.ndarray, k_pad: int) -> tuple:
    """hp_pack (gemm_f16.c): fp16 GEMM panels.

    Args:
        w: [N][K] fp32 weights.
        b: [N] bias.
        k_pad: the padded K (row stride of a panel).

    Returns:
        (panels, b32, b16, (N, K, Kp, Np)): fp16 panels [Np/16][Kp][16] + tail
        [Kp][8], and the bias as fp32 and fp16 [Np + 16].
    """
    n, k = w.shape
    n_pad = r8(n)
    n16 = n_pad // 16
    nbytes = ((n_pad * k_pad * 2) + 127) & ~63
    panels = np.zeros(nbytes // 2, np.float16)
    w16 = np.zeros((n_pad, k), np.float16)
    w16[:n] = w
    # 16-row panels: element (row o, column c) at (o // 16, c, o % 16)
    full = panels[:n16 * k_pad * 16].reshape((n16, k_pad, 16))
    full[:, :k, :] = w16[:n16 * 16].reshape((n16, 16, k)).transpose(0, 2, 1)
    if n_pad > n16 * 16:  # the last 8 rows: a [Kp][8] panel
        at = n16 * k_pad * 16
        tail = panels[at:at + k_pad * 8].reshape((k_pad, 8))
        tail[:k, :] = w16[n16 * 16:].T
    b32 = np.zeros(n_pad + 16, np.float32)
    b32[:n] = b
    return panels, b32, b32.astype(np.float16), (n, k, k_pad, n_pad)


def pack_f32(w: np.ndarray, b: np.ndarray, k_pad: int) -> tuple:
    """fp_pack (gemm_f32.c): fp32 GEMM panels.

    Args:
        w: [N][K] weights.
        b: [N] bias.
        k_pad: the padded K (row stride of a panel).

    Returns:
        (panels, bb, (N, K, Kp, Np)): panels [Np/8][Kp][8] and the bias
        [Np + 8].
    """
    n, k = w.shape
    n_pad = r8(n)
    nbytes = ((n_pad * k_pad * 4) + 127) & ~63
    panels = np.zeros(nbytes // 4, np.float32)
    w32 = np.zeros((n_pad, k), np.float32)
    w32[:n] = w
    # 8-row panels: element (row o, column c) at (o // 8, c, o % 8)
    view = panels[:n_pad * k_pad].reshape((n_pad // 8, k_pad, 8))
    view[:, :k, :] = w32.reshape((n_pad // 8, 8, k)).transpose(0, 2, 1)
    bb = np.zeros(n_pad + 8, np.float32)
    bb[:n] = b
    return panels, bb, (n, k, k_pad, n_pad)


# the sum of |w| an int16 lane can take against activations in [-127, 127]
N4_LANE_LIMIT = 32767 // 127  # 258


def quant_rows_s8(w):
    """Quantizes each row of w to symmetric int8 with its own scale.

    Args:
        w: float32 [N][K].

    Returns:
        (q, sw): q int8 [N][K] = round half away from zero of w / sw, clipped
        to [-127, 127]; sw float32 [N] = max |row| / 127 (0 for an all-zero
        row, whose q is 0).
    """
    am = np.abs(w).max(1)
    sw = np.where(am > 0, am / 127.0, 0.0).astype(np.float32)
    sw_safe = np.where(sw > 0, sw, 1)
    q = np.where(sw[:, None] > 0, w / sw_safe[:, None], 0)
    q = np.clip(np.sign(q) * np.floor(np.abs(q) + 0.5), -127,
                127).astype(np.int8)
    return q, sw


def n4_panel_flags(q, lane_limit=N4_LANE_LIMIT):
    """Flags the 4-row panels of int8 weights that may take 4 products a lane.

    The kernel's n=4 K loop adds, into each of its 8 int16 lanes, the products
    of 4 weights per 32-byte step: bytes l, l + 8, l + 16 and l + 24 of the
    step. With activations in [-127, 127] the lane cannot overflow when those
    4 weights have sum |w| <= lane_limit.

    Args:
        q: int8 [Np][Kp], Np a multiple of 4.
        lane_limit: the largest allowed sum of |w| per lane and step.

    Returns:
        int8 [Np / 4]: 1 where every lane of every step of the panel's 4 rows
        is within the limit; all 0 when Kp is not a multiple of 32 (the n=4
        loop is not used then).
    """
    rows, kp = q.shape
    if kp % 32:
        return np.zeros(rows // 4, np.int8)
    lane_sums = np.abs(q.astype(np.int32)).reshape(rows, kp // 32, 4, 8).sum(2)
    row_worst = lane_sums.max(axis=(1, 2))
    safe = row_worst.reshape(rows // 4, 4).max(1) <= lane_limit
    return safe.astype(np.int8)


class Lowering:
    """Lowers a fused IR op list to one path's buffers and op records.

    Weights go into the package as pre-packed tensors. Buffers are tuples
    (dtype, H, W, C, ld, halo, elems); buffer 0 is the input window.

    Attributes:
        policy: per-model choices set by the caller before run(): "proto"
            ("s8"), "proto32" ("h16"), "int8" (layer classes), "wino"
            (minimum input channels) and "dense" ("f16" / "f32").
        vbuf: IR value id -> buffer id.
        src: per emitted op, the ONNX node(s) it was lowered from.
        INT8_GEMMS: the family runs every GEMM in int8 with dynamic
            activations, whatever the path (bnc check then judges the path
            by the int8 floor); False here, where int8 is the neon-int8
            path's policy.
    """
    INT8_GEMMS = False

    def __init__(self, pkg: pkgfmt.Package, ops: list[dict], path: str,
                 nsamp: int):
        """Prepares the lowering of ops for one path.

        Args:
            pkg: the pkgfmt.Package that receives the weight tensors.
            ops: the fused IR op list.
            path: "neon-fp16", "neon-fp32" or "neon-int8".
            nsamp: the input window's number of samples.
        """
        self.pkg = pkg
        self.ops = ops
        self.path = path
        info = path_info(path)
        self.f16 = info.act == F16
        # fp32 path with int8 GEMMs (no dotprod) on the classes in
        # policy["int8"]
        self.int8 = path == "neon-int8"
        self.adt = info.act
        self.var = info.var
        # buffer 0: the input window
        self.bufs = [(F32, 1, 1, nsamp, nsamp, 0, nsamp)]
        self.vbuf = {}      # IR value id -> buffer id
        # IR value id of an SE output -> its gate vector buffer (applied by
        # the 1x1 consumer)
        self.gate = {}
        self.out = []
        self.uses = {}
        for o in ops:
            res = [o["res"]] if o.get("res") else []
            for i in o["ins"] + res:
                self.uses.setdefault(i, []).append(o)
        self.pre = info.prefix
        self.k = 0
        self.policy = {}
        # per emitted op: the ONNX node(s) it was lowered from (debug
        # metadata in the package)
        self.src = []
        self.cur = None
        self.vshape = None  # IR value id -> (H, W) of a spatial value (run)
        # (output buffer, channel-sum buffer) of the last depthwise conv
        self.chsum = (None, None)

    def buf(self, dtype, h, w, c, ld=None, halo=0, elems=None):
        """Adds a buffer and returns its id.

        Args:
            dtype: 0 fp32, 1 fp16, 2 int8, 3 int32.
            h, w, c: the activation's shape.
            ld: the channel stride (default c).
            halo: zero border width in pixels.
            elems: number of elements to allocate (default from the shape).
        """
        ld = ld or c
        if elems is None:
            # unpadded activations: rows rounded up to 8 (the GEMM kernels
            # read whole 8-row tiles)
            elems = ((h + 2 * halo) * (w + 2 * halo) * ld if halo
                     else r8(h * w) * ld)
        self.bufs.append((dtype, h, w, c, ld, halo, int(elems)))
        return len(self.bufs) - 1

    def tensor(self, name, a):
        """Adds tensor a to the package; returns its index."""
        return self.pkg.tensor(name, a)

    def emit(self, base, bl, tl, il, var=None):
        """Appends an op record (opcode base, buffers, tensors, ints).

        var: the kernel variant (default: the path's).
        """
        v = self.var if var is None else var
        self.out.append((base | (v << 8), [int(x) for x in bl],
                         [int(x) for x in tl], [int(x) for x in il]))
        self.src.append(src_name(self.cur))

    def gemm_tensors(self, tag, w, b, k_pad):
        """Packs a GEMM's weights for the path.

        Returns:
            (tensor indices, (N, K, Kp, Np)).
        """
        if self.f16:
            panels, b32, b16, shp = pack_h16(w, b, k_pad)
            return [self.tensor(self.pre + tag + ".w", panels),
                    self.tensor(self.pre + tag + ".b32", b32),
                    self.tensor(self.pre + tag + ".b16", b16)], shp
        panels, bb, shp = pack_f32(w, b, k_pad)
        return [self.tensor(self.pre + tag + ".w", panels),
                self.tensor(self.pre + tag + ".b", bb)], shp

    def consumers(self, vid):
        """Returns the IR ops that read value vid (as input or residual)."""
        return self.uses.get(vid, [])

    def run(self, vshape: dict) -> tuple[list[tuple], list[tuple]]:
        """Lowers every op.

        Args:
            vshape: IR value id -> (H, W) of the spatial values.

        Returns:
            (bufs, ops): the buffer list and the op records
            (opcode, buffers, tensors, ints).

        Raises:
            recog.Unsupported: an op has no kernel on this path.
        """
        self.vshape = vshape
        i = 0
        ops = self.ops
        while i < len(ops):
            o = ops[i]
            self.cur = o
            fn = getattr(self, "l_" + o["op"].lower(), None)
            if fn is None:
                raise recog.Unsupported(
                    f"no {self.path} lowering for IR op {o['op']} (from "
                    f"{src_name(o)}); hint: the recognizer accepted this op "
                    f"but no kernel implements it on this path: add "
                    f"l_{o['op'].lower()} in python/bnneon/compiler/lower.py "
                    f"and a kernel in src/rt_ops_*.c (guide section 6)")
            try:
                i = fn(i, o)
            except recog.Unsupported as e:
                raise recog.Unsupported(
                    f"{e} [IR op {o['op']} from {src_name(o)}, path "
                    f"{self.path}]") from e
            except KeyError as e:
                if not isinstance(e.args[0], int):
                    raise
                raise recog.Unsupported(
                    f"IR value {e} has no buffer on the {self.path} path [IR "
                    f"op {o['op']} from {src_name(o)}]; hint: an op the "
                    f"recognizer kept standalone (see the report) feeds this "
                    f"one") from e
            self.k += 1
        return self.bufs, self.out

    # ------------------------------------------------------------ v2.4 family
    def l_fe_stft2_pow(self, i, o):
        """BirdNET v2.4's two-branch STFT power + mel front end."""
        b1, b2 = o["branches"]
        if ((b1["nfft"], b1["hop"], b2["nfft"], b2["hop"])
                != (2048, 278, 1024, 280) or self.bufs[0][3] != 144000
                or np.float32(o["eps"]) != np.float32(1e-6)
                or b1["mel"].shape != (96, 1025)
                or b2["mel"].shape != (96, 513)):
            raise recog.Unsupported(
                "FE_STFT2 kernel supports the BirdNET v2.4 geometry only "
                "(2048/278 + 1024/280, 144000 samples, 96 mel); got "
                f"{b1['nfft']}/{b1['hop']} {b2['nfft']}/{b2['hop']}")
        tl = [self.tensor("fe/win1", b1["window"]),
              self.tensor("fe/win2", b2["window"]),
              self.tensor("fe/mel1", b1["mel"]),
              self.tensor("fe/mel2", b2["mel"]),
              self.tensor("fe/pow1", np.array([b1["pow"]], np.float32)),
              self.tensor("fe/pow2", np.array([b2["pow"]], np.float32)),
              self.tensor("fe/bn_s", o["bn_s"]),
              self.tensor("fe/bn_b", o["bn_b"])]
        ob = self.buf(F32, 96, 511, 2)
        self.emit(OP["FE_STFT2"], [0, ob], tl, [], var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    def l_fe_logmel(self, i, o):
        """An exact-DFT log-mel front end (FFT kernel)."""
        if o["mode"] != "dft-op":
            raise recog.Unsupported(
                "log-mel front end: only the exact-DFT form is lowered")
        nfr = (self.bufs[0][3] + 2 * o["pad"] - o["frame"]) // o["hop"] + 1
        nmel = o["mel"].shape[0]
        nfft = o["nfft"]
        if (nfft & (nfft - 1) or not 16 <= nfft <= 2048 or o["frame"] > nfft
                or nmel > 512):
            raise recog.Unsupported(
                f"log-mel front end: nfft {nfft} (frame {o['frame']}, {nmel} "
                f"mel bins): the FFT kernel takes nfft a power of two in "
                f"16..2048, frame <= nfft and <= 512 mel bins")
        tl = [self.tensor("fe/window", o["window"]),
              self.tensor("fe/mel", o["mel"]),
              self.tensor("fe/consts",
                     np.array([o["floor"], o["scale"]], np.float32))]
        ob = self.buf(F32, nfr, nmel, 1)
        self.emit(OP["FE_LOGMEL"], [0, ob], tl,
                  [self.bufs[0][3], o["frame"], o["hop"], o["pad"], o["nfft"],
                   nmel], var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    def l_mean(self, i, o):
        """A spatial mean (global average pool) to a vector."""
        ib = self.vbuf[o["ins"][0]]
        _, h, w, c, ld, halo, _ = self.bufs[ib]
        if halo or c % 4:
            raise recog.Unsupported(
                f"spatial mean of a buffer with halo {halo} and {c} channels: "
                f"the MEAN kernel takes an unpadded input with channels a "
                f"multiple of 4")
        ob = self.buf(F32, 1, 1, c)
        self.emit(OP["MEAN"], [ib, ob], [], [h * w, c, ld])
        self.vbuf[o["out"]] = ob
        return i + 1

    def l_proto(self, i, o):
        """A prototype head (Perch): prototype GEMM, max over cells, linear."""
        ib = self.vbuf[o["ins"][0]]
        _, h, w, c, ld, halo, _ = self.bufs[ib]
        n_proto = o["P"].shape[0]
        nc, ppc = o["cw"].shape
        if halo or (h * w) % 8 or c % 8:
            raise recog.Unsupported(
                f"prototype head on {h}x{w} cells x {c} channels (halo "
                f"{halo}): the PROTO kernels take an unpadded input with cells "
                f"and channels multiples of 8")
        if self.f16 and self.policy.get("proto") == "s8":
            return self.proto_s8(i, o, ib, h * w, c, ld, n_proto, nc, ppc)
        if self.int8 and "proto" in self.int8_classes():
            if self.policy.get("proto32") == "h16":
                raise recog.Unsupported(
                    "--proto32-h16 (fp16-stored prototypes, fp32 math) with "
                    "the int8 prototype head of the neon-int8 path: choose "
                    "one; hint: drop --proto32-h16, or keep the fp32 head on "
                    "this path with --int8 naming the other classes only (e.g. "
                    "--int8 pw,head)")
            return self.proto_s8n(i, o, ib, h * w, c, ld, n_proto, nc, ppc)
        if not self.f16 and self.policy.get("proto32") == "h16":
            return self.proto_f32h(i, o, ib, h * w, c, ld, n_proto, nc, ppc)
        tl, shp = self.gemm_tensors(f"proto.{self.k}", o["P"],
                                    np.zeros(n_proto, np.float32), r8(c))
        tl += self.cls_tensors(o)
        return self.emit_proto(i, o, OP["PROTO"], ib, (self.adt, h * w, c),
                               shp[3], tl, list(shp) + [h * w, c, ld, nc, ppc])

    def emit_proto(self, i, o, op, ib, a_buf, cm_n, tl, ints, var=None):
        """The tail every prototype head shares: its buffers, the record.

        Args:
            i: the op index (the next one is returned).
            o: the PROTO IR op.
            op: the opcode.
            ib: the input buffer.
            a_buf: (dtype, rows, row length) of the activation scratch.
            cm_n: the length of the column-maximum buffer.
            tl: the tensor ids.
            ints: the record's ints.
            var: the kernel variant (default: the path's).

        The buffers are made in this order: logits [nc], activation scratch,
        column maxima.
        """
        nc = o["cw"].shape[0]
        ob = self.buf(F32, 1, 1, nc)
        a = self.buf(a_buf[0], a_buf[1], 1, a_buf[2])
        cm = self.buf(F32, 1, 1, cm_n)
        self.emit(op, [ib, ob, a, cm], tl, ints, var=var)
        self.vbuf[o["out"]] = ob
        return i + 1

    def cls_tensors(self, o):
        """Adds a prototype head's class weights, bias and eps tensors."""
        return [self.tensor(f"cls.{self.k}.w", o["cw"]),
                self.tensor(f"cls.{self.k}.b", o["cb"]),
                self.tensor(f"cls.{self.k}.eps",
                            np.array([o["eps"]], np.float32))]

    def proto_f32h(self, i, o, ib, n_cells, c, ld, n_proto, nc, ppc):
        """fp32 prototype head with fp16-stored prototypes [Np/8][K][8]."""
        n_pad = r8(n_proto)
        w16p = np.zeros((n_pad, c), np.float16)
        w16p[:n_proto] = o["P"].astype(np.float16)
        # [Np/8][K][8]
        # pylint misreads ndarray.transpose's varargs:
        # pylint: disable-next=too-many-function-args
        wp = w16p.reshape(n_pad // 8, 8, c).transpose(0, 2, 1).copy()
        tl = [self.tensor(f"f32h/proto.{self.k}.w", wp)] + self.cls_tensors(o)
        return self.emit_proto(i, o, OP["PROTO_F32H"], ib, (F32, n_cells, c),
                               n_pad, tl,
                               [n_proto, c, n_pad, n_cells, c, ld, nc, ppc],
                               var=0)

    def proto_s8(self, i, o, ib, n_cells, c, ld, n_proto, nc, ppc):
        """int8 SDOT prototype head.

        Per-prototype symmetric int8 weights (round half away from zero, as the
        kernel's activation rounding), packed [Np/12][K/4][12][4], scales sw
        [Np].
        """
        n_pad = (n_proto + 11) // 12 * 12
        wq = np.zeros((n_pad, c), np.float32)
        wq[:n_proto] = o["P"]
        q, sw = quant_rows_s8(wq)
        # [Np/12][K/4][12][4]
        qp = q.reshape(n_pad // 12, 12, c // 4, 4).transpose(0, 2, 1, 3).copy()
        tl = [self.tensor(f"s8/proto.{self.k}.q", qp),
              self.tensor(f"s8/proto.{self.k}.s", sw)] + self.cls_tensors(o)
        return self.emit_proto(i, o, OP["PROTO"], ib, (I8, n_cells, c), n_pad,
                               tl, [n_proto, c, n_pad, n_cells, c, ld, nc, ppc],
                               var=2)

    def proto_s8n(self, i, o, ib, n_cells, c, ld, n_proto, nc, ppc):
        """int8 prototype head without dotprod.

        The SDOT head's bytes (quant_rows_s8) packed [Np/4][Kp/16][4][16],
        scales sw [Np], and one flag per 4-prototype panel for the kernel's
        n=4 K loop (n4_panel_flags).
        """
        n_pad, k_pad = (n_proto + 3) & ~3, r16(c)
        wq = np.zeros((n_pad, k_pad), np.float32)
        wq[:n_proto, :c] = o["P"]
        q, sw = quant_rows_s8(wq)
        # [Np/4][Kp/16][4][16]
        qp = q.reshape(n_pad // 4, 4, k_pad // 16, 16).transpose(
            0, 2, 1, 3).copy()
        n4 = n4_panel_flags(q)
        tl = [self.tensor(f"s8n/proto.{self.k}.q", qp),
              self.tensor(f"s8n/proto.{self.k}.s", sw),
              self.tensor(f"s8n/proto.{self.k}.n4", n4)] + self.cls_tensors(o)
        return self.emit_proto(
            i, o, OP["PROTO_S8"], ib, (I8, n_cells, k_pad), n_pad, tl,
            [n_proto, c, k_pad, n_pad, n_cells, c, ld, nc, ppc], var=0)

    def l_conv(self, i, o):
        """A group-1 conv: dispatches to the kernel its geometry fits."""
        k, s, pads = o["k"], o["s"], o["pads"]
        w, b = o["w"], o["b"]
        co, ci = w.shape[0], w.shape[1]
        ib = self.vbuf[o["ins"][0]]
        _, h_in, w_in, c_in, _, _, _ = self.bufs[ib]
        if (k == [4, 8] and s == [2, 2] and pads == [1, 3, 1, 4] and ci == 2
                and o.get("act") == "relu" and o.get("pool") == "maxavg_w2"
                and (h_in, w_in) == (96, 511) and co == 24):
            tl, shp = self.gemm_tensors(
                f"c0.{self.k}", w.transpose(0, 2, 3, 1).reshape(co, -1), b, 64)
            ob = self.buf(self.adt, 48, 128, 48)
            self.emit(OP["C0POOL"], [ib, ob], tl, list(shp))
            self.vbuf[o["out"]] = ob
            return i + 1
        if (k == [1, 1] and s == [1, 1] and pads == [0, 0, 0, 0]
                and not o.get("pool")):
            return self.pw(i, o, w[:, :, 0, 0], b, ib)
        if (ci == 1 and c_in == 1 and self.bufs[ib][0] == F32
                and pads == [0, 0, 0, 0] and k[0] == k[1] and s[0] == s[1]
                and not o.get("pool")
                and o.get("act") in (None, "swish", "relu")):
            return self.stem(i, o, ib)
        if (k == [3, 3] and pads == [0, 0, 0, 0] and s == [1, 1]
                and o.get("act") == "relu" and o.get("res") is None
                and not self.dense_ok(o, ib)):
            raise recog.Unsupported(
                "3x3 valid conv outside the v2.4 post/pc pattern")
        if dense_kxk(o) and not o.get("pool") and not self.f16:
            return self.convk(i, o, ib)
        raise recog.Unsupported(
            f"conv k{k} s{s} pads {pads} cin {ci} act {o.get('act')} has no "
            f"{self.path} kernel"
            + (" (dense k x k convs run on the neon-fp32 path only)"
               if dense_kxk(o) and self.f16 else ""))

    def dense_ok(self, o, ib):
        """Whether conv o on buffer ib can run as a dense k x k conv."""
        return dense_kxk(o) and not self.f16 and self.bufs[ib][0] == F32

    def convk(self, i, o, ib):
        """Dense k x k conv (group 1, any stride, asymmetric pads).

        Runs on a zero-haloed NHWC input: per output row one GEMM over the k
        patch rows (segments of k * ld floats), no im2col; epilogue bias + act
        + residual + store.
        """
        w, b = o["w"], o["b"]
        co, ci, kh, kw = w.shape
        sh, sw = o["s"]
        pt, pl, _, _ = o["pads"]
        dt, h_in, w_in, c_in, ld, halo, _ = self.bufs[ib]
        ho, wo = self.vshape[o["out"]]
        if dt != F32:
            raise recog.Unsupported("dense k x k conv on a non-fp32 buffer")
        ib, halo = self.convk_input(ib, o["pads"])
        self.convk_check(o, (h_in, w_in, c_in, ld), (ho, wo))
        wmin = int(self.policy.get("wino", 0))
        if (wmin and [kh, kw] == [3, 3] and [sh, sw] == [1, 1]
                and o["pads"] == [1, 1, 1, 1] and ld == ci and ci >= wmin
                and halo >= 1):
            return self.convw(i, o, ib)
        wk = np.zeros((co, kh, kw, ld), np.float32)
        wk[:, :, :, :ci] = w.transpose(0, 2, 3, 1)
        k = kh * kw * ld
        tl, shp = self.gemm_tensors(f"convk.{self.k}", wk.reshape(co, k), b, k)
        ob, _, opad, ldo = self.out_for_dw(o, ho, wo, co)
        rb = self.residual_buf(o, (ho, wo, co), "conv output")
        self.emit(OP["CONVK"], [ib, ob, rb], tl,
                  list(shp) + [h_in, w_in, ho, wo, kh, kw, sh, halo - pt,
                               halo - pl, act_code(o), ldo, opad], var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    def convk_input(self, ib, pads):
        """A dense conv's input with a zero halo of max(pads).

        Emits a PAD into a new buffer when buffer ib's halo is too small.

        Returns:
            (buffer id, its halo).

        Raises:
            recog.Unsupported: ib has a halo too small or a channel stride
                that is not a multiple of 4.
        """
        _, h, w, c, ld, halo, _ = self.bufs[ib]
        hp = max(pads)
        if halo < hp or ld % 4:
            if halo:
                raise recog.Unsupported(
                    f"dense k x k conv with pads {pads} on a buffer with "
                    f"halo {halo}")
            ldp = ld if ld % 4 == 0 else (ld + 3) & ~3
            if ldp != ld:
                raise recog.Unsupported(
                    f"dense k x k conv on a buffer with channel stride {ld} "
                    f"(not a multiple of 4)")
            pb_ = self.buf(F32, h, w, c, ld=ld, halo=hp,
                           elems=self.conv_elems(h, w, ld, hp))
            self.emit(OP["PAD"], [ib, pb_], [], [h, w, c, ld, hp], var=0)
            ib, halo = pb_, hp
        return ib, halo

    @staticmethod
    def convk_check(o, in_shape, out_hw):
        """Raises recog.Unsupported unless the CONVK kernel takes conv o.

        Args:
            o: the CONV IR op.
            in_shape: (H, W, C, ld) of its input buffer.
            out_hw: (Ho, Wo) of its output.
        """
        _, ci, kh, kw = o["w"].shape
        sh, sw = o["s"]
        pt, pl, pb, pr = o["pads"]
        h, w, c, ld = in_shape
        ho, wo = out_hw
        if c != ci or ld < c:
            raise recog.Unsupported(
                f"dense conv with {ci} input channels on a {c}-channel buffer")
        if (ho - 1) * sh - pt + kh > h + pb or (wo - 1) * sw - pl + kw > w + pr:
            raise recog.Unsupported(
                f"dense conv geometry {h}x{w} -> {ho}x{wo} k{kh}x{kw} "
                f"s{sh}x{sw} pads {o['pads']}")
        if sh != sw or sh > 2 or kh > 7 or kw > 7:
            raise recog.Unsupported(
                f"dense conv k{kh}x{kw} s{sh}x{sw}: the CONVK kernel takes "
                f"k <= 7, equal strides <= 2")

    def residual_buf(self, o, out_shape, what):
        """The fp32 residual buffer conv o adds (-1 when it has none).

        Args:
            o: the conv IR op.
            out_shape: (H, W, C) of the conv output the residual must match.
            what: the conv's description in the error message.

        Raises:
            recog.Unsupported: the residual has another dtype or shape.
        """
        if not o.get("res"):
            return -1
        rb = self.vbuf[o["res"]]
        r_ = self.bufs[rb]
        h, w, co = out_shape
        if r_[0] != F32 or (r_[1], r_[2], r_[3]) != (h, w, co):
            raise recog.Unsupported(
                f"residual {r_[1]}x{r_[2]}x{r_[3]} for a {h}x{w}x{co} {what}")
        return rb

    def halo_need(self, vid):
        """The zero halo value vid's consumers can read.

        DWCONV k -> (k-1)/2; a dense k x k CONV -> max(pads) (any ld); a
        residual of a CONV (1x1 or k x k) -> any halo; everything else (1x1
        input, heads, means) -> unpadded.

        Returns:
            (halo, dw): the halo the consumers can use (None if one of them
            needs an unpadded buffer) and whether a depthwise consumer needs
            ld == C.
        """
        h, dw = 0, False
        for c in self.consumers(vid):
            if c["op"] == "DWCONV" and c["ins"][0] == vid:
                h, dw = max(h, (c["k"][0] - 1) // 2), True
            elif (c["op"] == "CONV" and c["ins"][0] == vid
                  and c["k"] != [1, 1] and dense_kxk(c)):
                widest = max(c["pads"])
                h = max(h, widest)
            elif (c["op"] == "CONV" and c.get("res") == vid
                  and c["ins"][0] != vid
                  and (c["k"] == [1, 1] or dense_kxk(c))):
                pass
            else:
                return None, dw
        return h, dw

    def out_for_dw(self, o, ho, wo, co):
        """Allocates the output buffer of conv o.

        Zero-haloed when every consumer can read a halo and one needs it (a
        depthwise or dense k x k conv; residual reads take any halo), else
        unpadded with channel stride R8.

        Returns:
            (buffer id, padded output width W or 0, halo, channel stride).
        """
        cons = self.consumers(o["out"])
        if len(cons) == 1 and cons[0]["op"] == "DWCONV" and co % 8 == 0:
            hk = (cons[0]["k"][0] - 1) // 2
            return self.buf(self.adt, ho, wo, co, ld=co, halo=hk), wo, hk, co
        h, dw = self.halo_need(o["out"])
        if (h and not self.f16 and (not dw or co % 8 == 0)
                and any(c["op"] == "CONV" and dense_kxk(c) for c in cons)):
            ldo = co if dw else r8(co)
            return self.buf(self.adt, ho, wo, co, ld=ldo, halo=h,
                            elems=self.conv_elems(ho, wo, ldo, h)), wo, h, ldo
        ldo = r8(co)
        return self.buf(self.adt, ho, wo, co, ld=ldo), 0, 0, ldo

    @staticmethod
    def conv_elems(h_in, w_in, ld, h):
        """Elements of a zero-haloed conv input buffer.

        The 8-pixel tiles of the last output row read up to 8 strides + k
        pixels past the row end (their values are ignored): slack for stride
        <= 2, k <= 7.
        """
        return (h_in + 2 * h) * (w_in + 2 * h) * ld + (2 * 8 + 8) * ld + 64

    def stem(self, i, o, ib):
        """Dense k x k VALID conv on a 1-channel fp32 image (Perch stem).

        im2col + fp32 GEMM in both paths (the precision screen, N003, shows
        this layer must not run in fp16), output in the path's activation
        dtype.
        """
        w, b = o["w"], o["b"]
        co, k, st = w.shape[0], o["k"][0], o["s"][0]
        _, h_in, w_in, _, _, _, _ = self.bufs[ib]
        ho, wo = self.vshape[o["out"]]
        taps = k * k
        k_pad = (taps + 3) & ~3
        panels, bb, shp = pack_f32(w.reshape(co, taps), b, k_pad)
        tl = [self.tensor(f"f32/stem.{self.k}.w", panels),
              self.tensor(f"f32/stem.{self.k}.b", bb)]
        ob, _, opad, ldo = self.out_for_dw(o, ho, wo, co)
        col = self.buf(F32, (ho * wo + 7) & ~7, 1, k_pad)
        bl = [ib, ob, col]
        if self.f16:
            bl.append(self.buf(F32, ho * wo, 1, shp[3]))
        self.emit(OP["STEM"], bl, tl,
                  list(shp) + [h_in, w_in, ho, wo, k, st, act_code(o), ldo,
                               opad])
        self.vbuf[o["out"]] = ob
        return i + 1

    def pw(self, i, o, w, b, ib):
        """A 1x1 conv (GEMM; int8 on the neon-int8 path for its classes)."""
        co, k = w.shape
        _, h_in, w_in, _, ld, halo, _ = self.bufs[ib]
        k_pad = r8(k)
        self.pw_check(k, ld, halo)
        s8 = self.int8 and self.int8_class(o) in self.int8_classes()
        if s8:
            tl, shp = self.s8_tensors(f"pw.{self.k}", w, b)
        else:
            tl, shp = self.gemm_tensors(f"pw.{self.k}", w, b, k_pad)
        ob, ow, opad, ldo = self.out_for_dw(o, h_in, w_in, co)
        rb, ldr = self.pw_residual(o, h_in, w_in)
        # the SE gate this conv applies to its input, if any
        gate = [self.gate[o["ins"][0]]] if o["ins"][0] in self.gate else []
        if s8:
            m4 = (h_in * w_in + 3) & ~3
            qb = self.buf(2, m4, 1, shp[2])
            sb = self.buf(F32, m4, 1, 1)
            zb = self.buf(3, m4, 1, 1)
            code = OP["PW_S8"]
            bl = [ib, ob, rb, qb, sb, zb] + gate
            var = 0
        else:
            if gate and shp[1] != shp[2]:
                raise recog.Unsupported(
                    f"SE-gated 1x1 conv with K {shp[1]} != Kp {shp[2]}")
            code = OP["PW"]
            bl = [ib, ob, rb] + gate
            var = None  # the path's
        self.emit(code, bl, tl,
                  list(shp) + [h_in * w_in, ld, act_code(o), ldo, ow, opad,
                               ldr],
                  var=var)
        self.vbuf[o["out"]] = ob
        return i + 1

    @staticmethod
    def pw_check(k, ld, halo):
        """Raises recog.Unsupported unless the PW kernel reads the input.

        Args:
            k: the conv's input channels.
            ld: the input buffer's channel stride.
            halo: the input buffer's halo.
        """
        if halo:
            raise recog.Unsupported(
                f"1x1 conv on a buffer with halo {halo}: the PW kernel reads "
                f"unpadded rows (only a depthwise conv consumes padded "
                f"buffers)")
        k_pad = r8(k)
        if ld < k_pad:
            raise recog.Unsupported(
                f"1x1 conv with {k} input channels on a buffer with channel "
                f"stride {ld}: the GEMM needs the stride >= {k_pad} (K rounded "
                f"up to 8); a 1x1 conv straight after a 1-channel front end "
                f"is not representable")

    def pw_residual(self, o, h, w):
        """(buffer, channel stride) of 1x1 conv o's residual; (-1, 0) if none.

        Raises:
            recog.Unsupported: the residual is a padded buffer the kernel
                cannot read.
        """
        rb, ldr = -1, 0
        if o.get("res"):
            rb = self.vbuf[o["res"]]
            ldr = self.bufs[rb][4]
            if self.bufs[rb][5] and (
                    self.f16
                    or (self.bufs[rb][1], self.bufs[rb][2]) != (h, w)):
                raise recog.Unsupported("residual from a padded buffer")
        return rb, ldr

    # ------------------------------ int8 (no dotprod) policy and packing
    INT8_DEFAULT = "pw,head,sed,lin,proto"

    def int8_classes(self):
        """Returns the layer classes the int8 policy puts on int8."""
        return set(x for x in str(self.policy.get("int8", self.INT8_DEFAULT))
                   .split(",") if x)

    def int8_class(self, o):
        """Layer class of 1x1 conv o for the int8 policy.

        'head' feeds a pooling head (GeM / SED / mean), 'pwl' has no
        activation (projection), 'pw' the rest (expansions with an
        activation).
        """
        if any(c["op"] in ("GEM", "SED", "MEAN")
               for c in self.consumers(o["out"])):
            return "head"
        return "pwl" if o.get("act") is None else "pw"

    def s8_tensors(self, tag, w, b):
        """Packs an int8 GEMM's weights.

        Per-output-channel symmetric int8 weights (round half away from zero),
        packed [Np/4][Kp/16][4][16], scales, column sums (zero-point
        correction) and the fp32 bias.

        Returns:
            (tensor indices, (N, K, Kp, Np)).
        """
        n, k = w.shape
        n_pad, k_pad = r8(n), r16(k)
        wq = np.zeros((n_pad, k_pad), np.float64)
        wq[:n, :k] = w
        am = np.abs(wq).max(1)
        sw = np.where(am > 0, am / 127.0, 1.0)
        ws = wq / sw[:, None]
        q = np.clip(np.sign(ws) * np.floor(np.abs(ws) + 0.5), -127,
                    127).astype(np.int8)
        panels = q.reshape(n_pad // 4, 4, k_pad // 16, 16)
        panels = panels.transpose(0, 2, 1, 3).copy()  # [Np/4][Kp/16][4][16]
        cs = q.astype(np.int32).sum(1).astype(np.int32)
        bb = np.zeros(n_pad + 8, np.float32)
        bb[:n] = b
        swf = np.zeros(n_pad + 8, np.float32)
        swf[:n_pad] = sw
        csp = np.zeros(n_pad + 8, np.int32)
        csp[:n_pad] = cs
        t = [self.tensor(f"s8n/{tag}.q", panels),
             self.tensor(f"s8n/{tag}.sw", swf),
             self.tensor(f"s8n/{tag}.cs", csp), self.tensor(f"s8n/{tag}.b", bb)]
        return t, (n, k, k_pad, n_pad)

    def l_dwconv(self, i, o):
        """A depthwise conv with fused swish (DW3 for 3x3, else DWK)."""
        k, s, pads = o["k"], o["s"], o["pads"]
        ib = self.vbuf[o["ins"][0]]
        _, h, w, c, ld, halo, _ = self.bufs[ib]
        pt, pl, pb, pr = pads
        self.dw_check(o)
        ho, wo = self.vshape[o["out"]]
        self.bufs.append((F32, 1, 1, c, c, 0, c))
        cs = len(self.bufs) - 1
        ob = self.buf(self.adt, ho, wo, c)
        w9 = o["w"][:, 0].transpose(1, 2, 0).reshape(k[0] * k[1], c)
        dt_ = np.float16 if self.f16 else np.float32
        dw3 = (k == [3, 3] and halo == 1 and ld == c and pt <= 1 and pl <= 1
               and pb <= 1 and pr <= 1
               and (self.f16 and c <= 2048
                    or (not self.f16 and pt == 1 and pl == 1)))
        if dw3:
            tl = self.dw_tensors(w9, o["b"], dt_)
            il = [h, w, c, s[0], ho, wo]
            if self.f16:
                il += [pt, pl]
            self.emit(OP["DW3"], [ib, ob, cs], tl, il)
        else:
            hk = (k[0] - 1) // 2
            if k[0] != k[1] or s[0] != s[1] or max(pads) > hk or c % 8:
                raise recog.Unsupported(
                    f"depthwise k{k} s{s} pads {pads} C {c}: no kernel")
            ib = self.dw_input(ib, hk)
            tl = self.dw_tensors(w9, o["b"], dt_)
            self.emit(OP["DWK"], [ib, ob, cs], tl,
                      [h, w, c, k[0], s[0], pt, pl, ho, wo, 1])
        self.vbuf[o["out"]] = ob
        self.chsum = (ob, cs)
        return i + 1

    @staticmethod
    def dw_check(o):
        """Raises recog.Unsupported unless a depthwise kernel takes conv o."""
        k, s, pads = o["k"], o["s"], o["pads"]
        if o.get("act") != "swish":
            raise recog.Unsupported(
                f"depthwise conv with activation {o.get('act')}: the "
                f"depthwise kernels fuse swish only")
        if (k[0] != k[1] or k[0] % 2 == 0 or k[0] > 7 or s[0] != s[1]
                or s[0] > 2 or max(pads) > (k[0] - 1) // 2
                or (k[0] - 1) // 2 > 3):
            raise recog.Unsupported(
                f"depthwise conv k{k} s{s} pads {pads}: the kernels take "
                f"square odd k <= 7, equal strides <= 2 and pads <= (k-1)/2")

    def dw_input(self, ib, hk):
        """Buffer ib, or a PAD of it with halo hk and ld == C if it differs."""
        _, h, w, c, ld, halo, _ = self.bufs[ib]
        if halo != hk or ld != c:
            pb_ = self.buf(self.adt, h, w, c, ld=c, halo=hk)
            self.emit(OP["PAD"], [ib, pb_], [], [h, w, c, ld, hk])
            ib = pb_
        return ib

    def dw_tensors(self, w9, b, dt_):
        """Adds a depthwise conv's [k*k][C] weights and bias as dtype dt_."""
        return [self.tensor(self.pre + f"dw.{self.k}.w", w9.astype(dt_)),
                self.tensor(self.pre + f"dw.{self.k}.b", b.astype(dt_))]

    def l_se(self, i, o):
        """A squeeze-excite block on the preceding depthwise conv's output."""
        ib = self.vbuf[o["ins"][0]]
        if len(self.consumers(o["ins"][0])) != 1:
            raise recog.Unsupported(
                "SE input has other consumers (the SE is applied in place)")
        if self.chsum[0] != ib:
            raise recog.Unsupported(
                "SE block whose input is not the preceding depthwise conv's "
                "output: the SE kernel reuses that conv's channel sums")
        if o["act1"] != "swish":
            raise recog.Unsupported(
                f"SE block with activation {o['act1']} between its two FCs: "
                f"the SE kernel implements swish")
        cs = self.chsum[1]
        _, h, w, c, _, _, _ = self.bufs[ib]
        rd = o["w1"].shape[0]
        tl = [self.tensor(self.pre + f"se.{self.k}.w1", o["w1"]),
              self.tensor(self.pre + f"se.{self.k}.b1", o["b1"]),
              self.tensor(self.pre + f"se.{self.k}.w2t",
                     np.ascontiguousarray(o["w2"].T)),
              self.tensor(self.pre + f"se.{self.k}.b2", o["b2"])]
        if self.gate_in_next(o, c):
            # the SE output feeds only a 1x1 conv: emit the gate vector and
            # let that conv apply it while it reads its input (no in-place
            # pass over the activation; the same products, so bit-identical
            # to SE + PW)
            gb = self.buf(F32, 1, 1, c, elems=c)
            self.emit(OP["SE_GATE"], [cs, gb], tl, [c, rd, h * w])
            self.gate[o["out"]] = gb
        else:
            self.emit(OP["SE"], [ib, cs], tl, [c, rd, h * w])
        # in place (or ungated, with the gate pending for its 1x1 consumer)
        self.vbuf[o["out"]] = ib
        return i + 1

    def gate_in_next(self, o, c):
        """Whether SE o's only consumer is a 1x1 conv that can apply its gate.

        Args:
            o: the SE IR op.
            c: its channels.
        """
        cons = self.consumers(o["out"])
        c0 = cons[0] if len(cons) == 1 else None
        return (not self.f16 and c0 is not None and c0["op"] == "CONV"
                and c0["ins"][0] == o["out"] and c0["k"] == [1, 1]
                and c0["s"] == [1, 1] and c0["pads"] == [0, 0, 0, 0]
                and not c0.get("pool") and c0["w"].shape[1] == c
                and c % 8 == 0)

    def l_affine(self, i, o):
        """v2.4's post BN + ReLU + 3x3 conv + mean, as one POSTPC op."""
        ops = self.ops
        nx = ops[i + 1] if i + 1 < len(ops) else None
        mn = ops[i + 2] if i + 2 < len(ops) else None
        ib = self.vbuf[o["ins"][0]]
        _, h_in, w_in, c_in, ld, _, _ = self.bufs[ib]
        ok = (o.get("act") == "relu" and nx and nx["op"] == "CONV"
              and nx["k"] == [3, 3] and nx["pads"] == [0, 0, 0, 0]
              and nx["s"] == [1, 1] and nx.get("act") == "relu"
              and nx["ins"] == [o["out"]] and mn and mn["op"] == "MEAN"
              and mn["ins"] == [nx["out"]]
              and (h_in, w_in, c_in, ld) == (3, 8, 192, 192)
              and nx["w"].shape[0] == 1024)
        if not ok:
            raise recog.Unsupported(
                "per-channel affine outside the v2.4 post BN + ReLU + 3x3 "
                "conv + mean pattern")
        w = nx["w"].transpose(0, 2, 3, 1).reshape(1024, 1728)
        tl, shp = self.gemm_tensors(f"pc.{self.k}", w, nx["b"], 1728)
        tl = [self.tensor(self.pre + f"post.{self.k}.s", o["s"]),
              self.tensor(self.pre + f"post.{self.k}.t", o["t"])] + tl
        fb = self.buf(F32, 1, 1, 1024)
        pin = self.buf(self.adt, 8, 1, 1728)
        pout = self.buf(self.adt, 8, 1, 1024)
        self.emit(OP["POSTPC"], [ib, fb, pin, pout], tl, list(shp))
        self.vbuf[mn["out"]] = fb
        return i + 3

    def l_dense(self, i, o):
        """A dense layer on an fp32 vector (DENSE, GEMV or GEMV_S8)."""
        ib = self.vbuf[o["ins"][0]]
        w, b = o["w"], o["b"]
        n, k = w.shape
        act = self.dense_check(o, ib)
        # (the GEMV_S8 path allocates an output of its own: this one stays
        # unused there)
        ob = self.buf(F32, 1, 1, n)
        f16w = self.f16 and self.policy.get(
            "dense", "f16" if n > 64 else "f32") == "f16"
        if self.int8 and "lin" in self.int8_classes():
            return self.gemv_s8(i, o, ib, act)
        if k != 1024 or act is not None:
            # general dense layer (custom-classifier heads, hidden layers):
            # fp32 math, fp16 weights on the fp16 path
            tl = [self.tensor(self.pre + f"gemv.{self.k}.w",
                         w.astype(np.float16 if f16w else np.float32)),
                  self.tensor(self.pre + f"gemv.{self.k}.b",
                              b.astype(np.float32))]
            self.emit(OP["GEMV"], [ib, ob], tl, [n, k, ACT[act]],
                      var=1 if f16w else 0)
            self.vbuf[o["out"]] = ob
            return i + 1
        if f16w:
            tl = [self.tensor(self.pre + f"dense.{self.k}.w",
                              w.astype(np.float16)),
                  self.tensor(self.pre + f"dense.{self.k}.b", b)]
            self.emit(OP["DENSE"], [ib, ob], tl, [n])
        else:  # small heads (bats): exact fp32 GEMV in both paths
            tl = [self.tensor(f"f32/dense.{self.k}.w", w.astype(np.float32)),
                  self.tensor(f"f32/dense.{self.k}.b", b)]
            self.emit(OP["DENSE"], [ib, ob], tl, [n], var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    def dense_check(self, o, ib):
        """Returns dense layer o's activation.

        Raises:
            recog.Unsupported: buffer ib is not an fp32 vector of the layer's
                input width, or the activation is not relu or none.
        """
        _, k = o["w"].shape
        dt, _, _, c, _, halo, _ = self.bufs[ib]
        if dt != F32 or halo:
            raise recog.Unsupported(
                f"dense layer on a {('f32', 'f16')[dt]} buffer with halo "
                f"{halo}: dense layers take an fp32 vector (a pooled "
                f"embedding or another dense layer's output)")
        if k != c:
            raise recog.Unsupported(
                f"dense layer with {k} inputs on a {c}-wide vector")
        act = o.get("act")
        if act not in (None, "relu"):
            raise recog.Unsupported(
                f"dense layer with activation {act}: dense layers fuse relu "
                f"only (a final sigmoid/softmax is the host's: --logits <this "
                f"layer's output> --post)")
        return act

    def gemv_s8(self, i, o, ib, act):
        """Dense layer o as an int8 GEMV (neon-int8 path, class 'lin')."""
        tl, shp = self.s8_tensors(f"gemv.{self.k}", o["w"], o["b"])
        ob = self.buf(F32, 1, 1, shp[3])
        qb = self.buf(2, 4, 1, shp[2])
        sb = self.buf(F32, 4, 1, 1)
        zb = self.buf(3, 4, 1, 1)
        self.emit(OP["GEMV_S8"], [ib, ob, qb, sb, zb], tl,
                  list(shp) + [ACT[act]], var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    # ------------ dB-image front end, GeM, SED heads, vectors
    def l_fe_dbimg(self, i, o):
        """A dB spectrogram image front end (STFT, mel, dB, resize)."""
        # fp32 math on every path (fp16 too): the power spectrum reaches ~1e6
        nfft, nbins = o["nfft"], o["nfft"] // 2 + 1
        nmel, nrows, c = o["mel"].shape[0], len(o["rows"]), o["C"]
        if o["mel"].shape[1] != nbins:
            raise recog.Unsupported("mel matrix width != n_fft / 2 + 1")
        if len(o["post"]) > 3 or len(o["chan"]) > 2:
            raise recog.Unsupported(
                "front end: more than 3 scalar ops after the log or 2 "
                "per-channel ops")
        ld, h = self.fe_layout(o)
        ob = self.buf(F32, o["Ho"], o["Wo"], c, ld=ld, halo=h,
                      elems=self.conv_elems(o["Ho"], o["Wo"], ld, h)
                      if h else None)
        post = list(o["post"]) + [(0, 0.0)] * (3 - len(o["post"]))
        chan = (list(o["chan"])
                + [(0, np.zeros(c, np.float32))] * (2 - len(o["chan"])))
        scalars = np.array([o["floor"], o["top_db"] or 0.0, o["eps"]]
                           + [v for _, v in post], np.float32)
        chan0 = np.asarray(chan[0][1], np.float32).reshape(-1)
        chan1 = np.asarray(chan[1][1], np.float32).reshape(-1)
        consts = np.concatenate([scalars, chan0, chan1])
        tl = [self.tensor(f"fe/dbimg.{self.k}.window", o["window"]),
              self.tensor(f"fe/dbimg.{self.k}.mel", o["mel"]),
              self.tensor(f"fe/dbimg.{self.k}.rows",
                          o["rows"].astype(np.int32)),
              self.tensor(f"fe/dbimg.{self.k}.consts", consts)]
        pm = {"none": 0, "reflect": 1, "zero": 2}[o["pad_mode"]]
        resize = 1  # half-pixel linear (an identity when the sizes match)
        il = [self.bufs[0][3], nfft, o["hop"], o["pad"], pm, o["nfr"], nmel,
              nrows, int(o["top_db"] is not None), o["norm"], post[0][0],
              post[1][0], post[2][0], resize, o["Ho"], o["Wo"], c, chan[0][0],
              chan[1][0], ld, h]
        self.emit(OP["FE_DBIMG"], [0, ob], tl, il, var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    def fe_layout(self, o):
        """The front-end image's layout: what its consumers read.

        A dense conv takes a zero halo and a stride that is a multiple of 4.

        Returns:
            (channel stride, halo).
        """
        c = o["C"]
        h, dw = self.halo_need(o["out"])
        h = h or 0
        ld = c if c % 4 == 0 else (c + 3) & ~3
        if dw and c % 8:
            h = 0
        return ld, h

    def l_gem(self, i, o):
        """Generalized-mean (GeM) pooling to a vector."""
        ib = self.vbuf[o["ins"][0]]
        dt, h, w, c, ld, halo, _ = self.bufs[ib]
        if dt != F32 or halo or c % 4:
            raise recog.Unsupported(
                f"GeM on a {('f32', 'f16')[dt]} buffer with halo {halo} and "
                f"{c} channels: the GEM kernel takes an unpadded fp32 input "
                f"with channels a multiple of 4")
        ob = self.buf(F32, 1, 1, c)
        tl = [self.tensor(f"gem.{self.k}.consts",
                     np.array([o["eps"], o["p"], o["q"]], np.float32))]
        self.emit(OP["GEM"], [ib, ob], tl, [h * w, c, ld], var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    def l_sed(self, i, o):
        """A sound-event-detection head (attention pooling over time)."""
        ib = self.vbuf[o["ins"][0]]
        dt, h, w, c, ld, halo, _ = self.bufs[ib]
        if dt != F32 or halo or c != o["C"]:
            raise recog.Unsupported(
                f"SED head on a buffer {h}x{w}x{c} halo {halo}: the SED "
                f"kernel takes an unpadded fp32 input")
        steps = o["L"]
        if steps != (h if o["red"] == 3 else w):
            raise recog.Unsupported(
                "SED head: time steps do not match the activation")
        nc = o["wa"].shape[0]
        c_pad = r8(c)
        s8 = self.int8 and "sed" in self.int8_classes()
        gemms, (sf, sa, sc) = self.sed_gemms(o, s8, c_pad)
        fc, at, cl = gemms
        pk, dw_, db, dk = self.sed_smoothing(o, c)
        lo, hi = o["clip"] if o["clip"] else (0.0, 0.0)
        tl = ([self.tensor(f"sed.{self.k}.dw", np.ascontiguousarray(dw_.T)),
               self.tensor(f"sed.{self.k}.db", db)] + fc + at + cl
              + [self.tensor(f"sed.{self.k}.consts",
                        np.array([lo, hi, o["scale"]], np.float32))])
        ob = self.buf(F32, 1, 1, nc)
        steps_pad = r8(steps)
        sb = self.buf(F32, steps_pad, 1, c_pad)  # axis mean [L][C]
        pb = self.buf(F32, steps_pad, 1, c_pad)  # pooled [L][C]
        hb = self.buf(F32, steps_pad, 1, sf[3])  # hidden [L][hid]
        ab = self.buf(F32, steps_pad, 1, sa[3])  # attention logits [L][NCp]
        cb = self.buf(F32, steps_pad, 1, sc[3])  # class logits [L][NCp]
        il = list(sf) + list(sa) + list(sc) + [o["red"], h, w, c, ld, steps, pk,
                                               dk, int(o["clip"] is not None)]
        if s8:
            k_pad_max = max(sf[2], sa[2])
            qb = self.buf(2, steps_pad, 1, k_pad_max)
            qs = self.buf(F32, steps_pad, 1, 1)
            qz = self.buf(3, steps_pad, 1, 1)
            bl = [ib, ob, sb, pb, hb, ab, cb, qb, qs, qz]
            self.emit(OP["SED_S8"], bl, tl, il, var=0)
        else:
            self.emit(OP["SED"], [ib, ob, sb, pb, hb, ab, cb], tl, il, var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    def sed_gemms(self, o, s8, c_pad):
        """Packs a SED head's three GEMMs (fc, attention, classes).

        Args:
            o: the SED IR op.
            s8: int8 GEMMs (neon-int8 path) instead of the path's.
            c_pad: the channels rounded up to 8 (the fc GEMM's Kp).

        Returns:
            ((fc, att, cla) tensor index lists, (their (N, K, Kp, Np)
            shapes)).
        """
        hid = o["w1"].shape[0]
        if s8:
            fc, sf = self.s8_tensors(f"sed.{self.k}.fc", o["w1"], o["b1"])
            at, sa = self.s8_tensors(f"sed.{self.k}.att", o["wa"], o["ba"])
            cl, sc = self.s8_tensors(f"sed.{self.k}.cla", o["wc"], o["bc"])
        else:
            fc, sf = self.gemm_tensors(f"sed.{self.k}.fc", o["w1"], o["b1"],
                                       c_pad)
            at, sa = self.gemm_tensors(f"sed.{self.k}.att", o["wa"], o["ba"],
                                       r8(hid))
            cl, sc = self.gemm_tensors(f"sed.{self.k}.cla", o["wc"], o["bc"],
                                       r8(hid))
        return (fc, at, cl), (sf, sa, sc)

    @staticmethod
    def sed_smoothing(o, c):
        """A SED head's max-pool and conv1d over time (0: none).

        Args:
            o: the SED IR op.
            c: its channels.

        Returns:
            (pool k, conv1d weights [C][k], conv1d bias [C], conv1d k).

        Raises:
            recog.Unsupported: either one changes the sequence length.
        """
        pk, pp = o["pool"] if o["pool"] else (0, 0)
        if o["dconv"] is not None:
            dw_, dp, db = o["dconv"]
            dk = dw_.shape[1]
        else:
            dw_ = np.zeros((c, 1), np.float32)
            dp = 0
            db = np.zeros(c, np.float32)
            dk = 0
        if (pk and (pk % 2 == 0 or pp != pk // 2)
                or dk and (dk % 2 == 0 or dp != dk // 2)):
            raise recog.Unsupported(
                "SED head: the pooling / conv1d must keep the length (odd k, "
                "pad k/2)")
        return pk, dw_, db, dk

    def convw(self, i, o, ib):
        """Dense 3x3 / stride 1 / pad 1 conv by Winograd F(4x4, 3x3).

        U = G g G^T in float64, 36 fp32 panel sets.
        """
        w, b = np.asarray(o["w"], np.float64), o["b"]
        co, ci = w.shape[:2]
        _, h_in, w_in, _, _, _, _ = self.bufs[ib]
        g_mat = np.array([[1 / 4, 0, 0], [-1 / 6, -1 / 6, -1 / 6],
                      [-1 / 6, 1 / 6, -1 / 6], [1 / 24, 1 / 12, 1 / 6],
                      [1 / 24, -1 / 12, 1 / 6], [0, 0, 1]])
        u_mat = np.einsum("ia,ncab,jb->ijnc", g_mat, w, g_mat).reshape(
            36, co, ci)
        n_pad = r8(co)
        panels = np.zeros((36, n_pad // 8, ci, 8), np.float32)
        u_pad = np.zeros((36, n_pad, ci))
        u_pad[:, :co] = u_mat
        # pylint misreads ndarray.transpose's varargs:
        # pylint: disable-next=too-many-function-args
        panels[:] = u_pad.reshape(36, n_pad // 8, 8, ci).transpose(0, 1, 3, 2)
        bb = np.zeros(n_pad + 8, np.float32)
        bb[:co] = b
        tl = [self.tensor(f"wino.{self.k}.U", panels),
              self.tensor(f"wino.{self.k}.b", bb)]
        ob, _, opad, ldo = self.out_for_dw(o, h_in, w_in, co)
        # the runtime picks the strip and block itself (K060); recorded for
        # the package reader
        strip, block = 8, 40
        rb = self.residual_buf(o, (h_in, w_in, co), "Winograd conv")
        self.emit(OP["CONVW"], [ib, ob, rb], tl,
                  [ci, co, n_pad, h_in, w_in, act_code(o), ldo, opad, strip,
                   min(block, n_pad)],
                  var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    def l_vadd(self, i, o):
        """An elementwise Add of two fp32 vectors."""
        a, b = (self.vbuf[x] for x in o["ins"])
        n = self.bufs[a][3]
        if (self.bufs[b][3] != n or self.bufs[a][0] != F32
                or self.bufs[b][0] != F32):
            raise recog.Unsupported(
                "vector Add of different lengths or dtypes")
        ob = self.buf(F32, 1, 1, n)
        self.emit(OP["VADD"], [a, b, ob], [], [n], var=0)
        self.vbuf[o["out"]] = ob
        return i + 1

    def l_vaffine(self, i, o):
        """A vector affine no dense layer absorbed: always unsupported."""
        raise recog.Unsupported(
            "a scaled / offset vector that is not an affine function of one "
            "dense layer's input (no kernel); hint: the compiler folds "
            "scalings and offsets of dense heads into the dense layer")


def dense_kxk(o: dict) -> bool:
    """Whether IR op o is a group-1 conv with a kernel larger than 1x1.

    The CONVK kernel's domain, outside the special stems.
    """
    return o["op"] == "CONV" and o["k"] != [1, 1]


def src_name(o: dict | None) -> str:
    """Names the first ONNX node an IR op covers and how many more.

    Returns:
        For example "'model/block_1/conv/Conv' +5"; "?" without nodes.
    """
    if not o or not o.get("nodes"):
        return "?"
    n = [x for x in o["nodes"] if x] or ["?"]
    more = f" +{len(n) - 1}" if len(n) > 1 else ""
    return f"'{n[0]}'" + more


def plan_slots(bufs: list[tuple], ops: list[tuple],
               pinned: list[int | None]) -> tuple[list[tuple], int]:
    """Assigns memory slots to buffers.

    A buffer's lifetime runs from the first to the last op that names it;
    buffers with disjoint lifetimes and the same dtype share a slot (best fit
    by size). Buffer 0 (the input window) and the pinned output buffers get
    slots of their own.

    Args:
        bufs: the buffer tuples (dtype, H, W, C, ld, halo, elems).
        ops: the op records (opcode, buffers, tensors, ints).
        pinned: buffer ids (or None) that must not share a slot.

    Returns:
        (bufs with the slot appended to each tuple, total bytes of the shared
        slots).
    """
    first, last = _lifetimes(ops)
    esz = {0: 4, 1: 2, 2: 1, 3: 4}
    slot = {0: 0}
    slots = [None]          # slot -> [dtype, bytes, busy_until]
    for b in pinned:
        if b is not None and b not in slot:
            slot[b] = len(slots)
            slots.append([bufs[b][0], 0, 1 << 30])
    for i, (_, bl, _, _) in enumerate(ops):
        for b in bl:
            if b < 0 or b in slot or first[b] != i:
                continue
            need = bufs[b][6] * esz[bufs[b][0]]
            j = _pick_slot(slots, bufs[b][0], need, i)
            slots[j][1] = max(slots[j][1], need)
            slots[j][2] = last[b]
            slot[b] = j
    for b in range(len(bufs)):  # buffers no op names: an empty slot each
        if b not in slot:
            slot[b] = len(slots)
            slots.append([bufs[b][0], 0, 0])
    return ([tuple(bufs[b]) + (slot[b],) for b in range(len(bufs))],
            sum(x[1] for x in slots[1:] if x))


def _lifetimes(ops):
    """(first, last): buffer id -> index of the first / last op naming it."""
    first, last = {}, {}
    for i, (_, bl, _, _) in enumerate(ops):
        for b in bl:
            if b >= 0:
                first.setdefault(b, i)
                last[b] = i
    return first, last


def _pick_slot(slots, dtype, need, i):
    """The slot for a buffer of dtype and need bytes first used by op i.

    Among the slots of that dtype free before op i: the smallest that holds
    need bytes, else the largest; with none free, a new slot (appended).

    Args:
        slots: [dtype, bytes, busy_until] per slot (slot 0: None).
        dtype: the buffer's dtype code.
        need: its size in bytes.
        i: the index of the first op that names it.

    Returns:
        The slot index.
    """
    free = [j for j in range(1, len(slots))
            if slots[j][0] == dtype and slots[j][2] < i]
    if not free:
        slots.append([dtype, 0, 0])
        return len(slots) - 1
    fit = [j for j in free if slots[j][1] >= need]
    if fit:
        return min(fit, key=lambda j: slots[j][1])
    return max(free, key=lambda j: slots[j][1])
