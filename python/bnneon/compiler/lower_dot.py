"""Lowering for the dotprod + fp16 kernel family (src/rt_ops_dot.c).

The "neon-fp16" path of dense-conv models. The family is the mixed-precision
plan of the BirdNET v3.0 engine as registry ops: fp32 stem, fp16 24-channel
3x3 convs, int8 SDOT dense 3x3 and 1x1 convs with dynamic quantization, fp16
depthwise, fp32 SE / GeM / attention heads. Weights go into the package as
fp32 tensors in the layouts the kernels' packers take (they pack at load).

A layer that has no kernel in the family raises Unsupported with the reason,
so such a model simply has no "neon-fp16" package; its "neon-fp32" and
"neon-int8" paths are unaffected.
"""
import numpy as np

from . import lower
from . import recog

OP = dict(STEM3=26, CONV3C24=27, UNPAD=28, WINO_S8=29, QCONV3=30, QPW=31,
          SE_GATE16=32, GEM_A=33, HEADS_S8=34)
DOT = 2  # kernel variant: needs fp16 + dotprod
F32, F16, I8 = 0, 1, 2
# the SED heads' average-pool weight the HEADS_S8 kernel applies
AVG3 = np.float32(0.33333334)
SED_CLIP = (-10.0, 10.0)


def tf_same_pads(size: int, k: int, stride: int) -> tuple[int, int]:
    """(low, high) padding of TensorFlow 'same' for one axis."""
    out = -(-size // stride)
    total = max((out - 1) * stride + k - size, 0)
    return total // 2, total - total // 2


def is_tf_same(o: dict, h: int, w: int) -> bool:
    """Whether conv o's pads [top, left, bottom, right] are TF-same.

    Args:
        o: the CONV IR op (its kernel and stride decide).
        h, w: the input's height and width.
    """
    top, bottom = tf_same_pads(h, o["k"][0], o["s"][0])
    left, right = tf_same_pads(w, o["k"][1], o["s"][1])
    return list(o["pads"]) == [top, left, bottom, right]


def is_1x1(o):
    """Whether conv o is a plain 1x1 conv (stride 1, no pads, no pool)."""
    return (o["k"] == [1, 1] and o["s"] == [1, 1]
            and o["pads"] == [0, 0, 0, 0] and not o.get("pool"))


def ohwi(w):
    """Conv weights OIHW -> OHWI, contiguous fp32."""
    return np.ascontiguousarray(w.transpose(0, 2, 3, 1), dtype=np.float32)


def applies(path: str, ops: list[dict]) -> bool:
    """Whether the family lowers this path of the model.

    It takes the fp16 path of models with a dB-image front end (dense-conv
    bodies).
    """
    return (lower.path_info(path).act == lower.F16
            and any(o["op"] == "FE_DBIMG" for o in ops))


def sed_reason(o: dict, c: int) -> str | None:
    """Why the HEADS_S8 kernel cannot run SED head o (None if it can).

    Args:
        o: the SED IR op.
        c: its channels.
    """
    if o["pool"] != (3, 1):
        return f"max-pool {o['pool']} (the kernel pools 3 with pad 1)"
    if o["dconv"] is None:
        return "no average-pool conv"
    weights, pad, bias = o["dconv"]
    if (weights.shape != (c, 3) or pad != 1 or np.any(weights != AVG3)
            or np.any(bias)):
        return ("an average-pool conv other than 3 taps of 0.33333334 with pad "
                "1 and no bias")
    if o["clip"] != SED_CLIP:
        return f"attention clip {o['clip']} (the kernel clips to {SED_CLIP})"
    return None


class LoweringDot(lower.Lowering):
    """Lowering onto the dotprod + fp16 family.

    Activations are zero-haloed fp16 images in the 24-channel stage, fp32 rows
    on the residual stream, and fp16 rows between an expansion (or depthwise)
    conv and its projection.
    """
    INT8_GEMMS = True  # 1x1 convs and heads as int8 SDOT products

    def __init__(self, *args):
        """Takes lower.Lowering's arguments."""
        super().__init__(*args)
        # IR value of a dense / SED head or of a partial sum -> the terms in it
        self.head_terms = {}

    def tensor(self, name, a):
        """Adds tensor a to the package under "dot/"; returns its index."""
        return self.pkg.tensor("dot/" + name, a)

    def emit_dot(self, name, bufs, tensors, ints):
        """Appends a record of family op `name` (variant DOT)."""
        self.emit(OP[name], bufs, tensors, ints, var=DOT)

    def fe_layout(self, o):
        """(channel stride, halo): the stem reads a plain [H][W][C] image."""
        return o["C"], 0

    def rows32(self, vid):
        """The fp32 rows buffer of value vid.

        Converts a zero-haloed fp16 image once (UNPAD).

        Raises:
            recog.Unsupported: the buffer is neither fp32 rows nor such an
                image.
        """
        ib = self.vbuf[vid]
        dtype, h, w, c, ld, halo, _ = self.bufs[ib]
        if dtype == F32 and halo == 0:
            return ib
        if dtype != F16 or halo != 1 or ld != c or (w * c) % 8:
            raise recog.Unsupported(
                f"no conversion to fp32 rows for a {('fp32', 'fp16')[dtype]} "
                f"buffer {h}x{w}x{c} with halo {halo} and channel stride {ld}")
        ob = self.buf(F32, h, w, c, ld=c)
        self.emit_dot("UNPAD", [ib, ob], [], [h, w, c])
        self.vbuf[vid] = ob
        return ob

    def l_conv(self, i, o):
        """A 1x1 or dense 3x3 conv with swish."""
        if is_1x1(o):
            return self.qpw(i, o)
        if (o["k"] != [3, 3] or o.get("pool") or o.get("act") != "swish"
                or not lower.dense_kxk(o)):
            raise recog.Unsupported(
                f"conv k{o['k']} s{o['s']} act {o.get('act')}: the dotprod "
                f"family has 3x3 dense convs with swish and 1x1 convs")
        co, ci = o["w"].shape[:2]
        if (co, ci) == (24, 3):
            return self.stem3(i, o)
        if (co, ci) == (24, 24) and o.get("res") == o["ins"][0]:
            return self.conv3c24(i, o)
        return self.conv3_s8(i, o)

    def stem3(self, i, o):
        """The 3 -> 24 channel stride-2 stem conv on the image."""
        ib = self.vbuf[o["ins"][0]]
        dtype, h, w, c, ld, halo, _ = self.bufs[ib]
        if ((dtype, c, ld, halo) != (F32, 3, 3, 0) or o["s"] != [2, 2]
                or not is_tf_same(o, h, w) or o.get("res")):
            raise recog.Unsupported(
                "3 -> 24 channel 3x3 conv outside the stem pattern (stride 2, "
                "TF-same pads, on the image)")
        if h % 2 or w % 2:
            raise recog.Unsupported(
                f"stem conv on a {h}x{w} image: the stem kernel pads bottom / "
                f"right only, which is TF-same padding for an even height and "
                f"width")
        ho, wo = self.vshape[o["out"]]
        ob = self.buf(F16, ho, wo, 24, ld=24, halo=1)
        tensors = [self.tensor(f"stem.{self.k}.w", ohwi(o["w"])),
                   self.tensor(f"stem.{self.k}.b", o["b"])]
        self.emit_dot("STEM3", [ib, ob], tensors, [h, w, ho, wo])
        self.vbuf[o["out"]] = ob
        return i + 1

    def conv3c24(self, i, o):
        """A stage-0 24 -> 24 channel 3x3 conv with its residual, in fp16."""
        ib = self.vbuf[o["ins"][0]]
        dtype, h, w, c, ld, halo, _ = self.bufs[ib]
        if ((dtype, c, ld, halo) != (F16, 24, 24, 1) or o["s"] != [1, 1]
                or o["pads"] != [1, 1, 1, 1] or h % 2 or w % 2):
            raise recog.Unsupported(
                "24 -> 24 channel 3x3 conv with a residual outside the stage-0 "
                "pattern (stride 1, pad 1, even height and width, on a "
                "zero-haloed fp16 image)")
        ob = self.buf(F16, h, w, 24, ld=24, halo=1)
        tensors = [self.tensor(f"c24.{self.k}.w", ohwi(o["w"])),
                   self.tensor(f"c24.{self.k}.b", o["b"])]
        self.emit_dot("CONV3C24", [ib, ob], tensors, [h, w])
        self.vbuf[o["out"]] = ob
        return i + 1

    def conv3_s8(self, i, o):
        """Dense 3x3 expansion conv in int8.

        Winograd F(2,3) at stride 1, im2col otherwise; fp16 rows out.
        """
        co, ci = o["w"].shape[:2]
        ib = self.rows32(o["ins"][0])
        _, h, w, c, ld, _, _ = self.bufs[ib]
        stride = o["s"][0]
        if (o.get("res") or o["s"][0] != o["s"][1] or stride > 2
                or not is_tf_same(o, h, w) or c != ci or ld != ci):
            raise recog.Unsupported(
                f"dense 3x3 conv s{o['s']} pads {o['pads']} res "
                f"{o.get('res')}: the int8 kernels take TF-same pads, stride 1 "
                f"or 2 and no residual")
        if ci % 8 or co % 16 or ci > 2048 or co > 2048:
            raise recog.Unsupported(
                f"dense 3x3 conv {ci} -> {co} channels: the int8 kernels take "
                f"inputs in multiples of 8 and outputs in multiples of 16, up "
                f"to 2048")
        ho, wo = self.vshape[o["out"]]
        ob = self.buf(F16, ho, wo, co, ld=co)
        tensors = [self.tensor(f"e.{self.k}.w", ohwi(o["w"])),
                   self.tensor(f"e.{self.k}.b", o["b"])]
        if stride == 1:
            self.emit_dot("WINO_S8", [ib, ob], tensors, [h, w, ci, co])
        else:
            self.emit_dot("QCONV3", [ib, ob], tensors,
                          [h, w, ci, co, stride, ho, wo])
        self.vbuf[o["out"]] = ob
        return i + 1

    def qpw_input(self, o):
        """(buffer, gate buffer or -1) 1x1 conv o reads.

        fp16 rows (with a pending SE gate) or fp32 rows.
        """
        vid = o["ins"][0]
        ib = self.vbuf[vid]
        dtype, _, _, _, _, halo, _ = self.bufs[ib]
        if dtype == F16 and halo == 0:
            return ib, self.gate.get(vid, -1)
        return self.rows32(vid), -1

    def qpw_output(self, o, h, w, n):
        """(buffer, dtype, image width, halo) 1x1 conv o writes.

        A zero-haloed fp16 image for a depthwise 3x3 consumer, fp32 rows
        otherwise.
        """
        cons = self.consumers(o["out"])
        if (len(cons) == 1 and cons[0]["op"] == "DWCONV"
                and cons[0]["ins"][0] == o["out"] and cons[0]["k"] == [3, 3]):
            return self.buf(F16, h, w, n, ld=n, halo=1), F16, w, 1
        return self.buf(F32, h, w, n, ld=n), F32, 0, 0

    def qpw(self, i, o):
        """A 1x1 conv as an int8 GEMM with dynamic activation quantization."""
        wt = np.ascontiguousarray(o["w"][:, :, 0, 0], dtype=np.float32)
        n, k = wt.shape
        ib, gate = self.qpw_input(o)
        idt, h, w, c, ldx, _, _ = self.bufs[ib]
        if c != k or n % 16 or k % 8:
            raise recog.Unsupported(
                f"1x1 conv {k} -> {n} channels on a {c}-channel buffer: the "
                f"int8 kernel takes inputs in multiples of 8 and outputs in "
                f"multiples of 16")
        m = h * w
        ob, odt, ow, opad = self.qpw_output(o, h, w, n)
        if odt == F16 and o.get("act") != "swish":
            raise recog.Unsupported(
                "1x1 conv without swish feeding a depthwise conv")
        rb = self.qpw_residual(o, odt, m, n)
        a8 = self.buf(I8, m + 4, 1, lower.r16(k))
        rq = self.buf(F32, m + 4, 1, 2)
        tensors = [self.tensor(f"pw.{self.k}.w", wt),
                   self.tensor(f"pw.{self.k}.b", o["b"])]
        ints = [n, k, m, idt, ldx, lower.act_code(o), odt, n, ow, opad]
        self.emit_dot("QPW", [ib, ob, a8, rq, rb, gate], tensors, ints)
        self.vbuf[o["out"]] = ob
        return i + 1

    def qpw_residual(self, o, odt, m, n):
        """The fp32 rows residual of 1x1 conv o (-1 when it has none).

        Args:
            o: the CONV IR op.
            odt: its output dtype.
            m, n: its output rows and channels.

        Raises:
            recog.Unsupported: the residual does not match an fp32 output.
        """
        rb = -1
        if o.get("res"):
            rb = self.rows32(o["res"])
            _, rh, rw, rc, rld, _, _ = self.bufs[rb]
            if odt != F32 or (rh * rw, rc, rld) != (m, n, n):
                raise recog.Unsupported(
                    f"residual {rh}x{rw}x{rc} for a 1x1 conv output of {m} "
                    f"rows x {n}")
        return rb

    def l_dwconv(self, i, o):
        """A depthwise conv on fp16 rows or on a swish conv's haloed image.

        Raises:
            recog.Unsupported: the input is neither, so no family kernel
                reads it.
        """
        dtype, _, _, c, ld, halo, _ = self.bufs[self.vbuf[o["ins"][0]]]
        haloed_3x3 = o["k"] == [3, 3] and (halo, ld) == (1, c)
        if dtype != F16 or not (halo == 0 or haloed_3x3):
            kind = ("fp32", "fp16")[dtype]
            raise recog.Unsupported(
                f"depthwise conv k{o['k']} on a {kind} buffer with halo "
                f"{halo}: the dotprod family's depthwise kernels read fp16, "
                f"either unpadded rows (any odd k up to 7) or the zero-haloed "
                f"image a swish 1x1 conv writes for a single 3x3 depthwise "
                f"consumer")
        return super().l_dwconv(i, o)

    def l_se(self, i, o):
        """An SE block as an fp16 gate its 1x1 consumer applies."""
        vid = o["ins"][0]
        ib = self.vbuf[vid]
        cons = self.consumers(o["out"])
        if self.chsum[0] != ib or len(self.consumers(vid)) != 1:
            raise recog.Unsupported(
                "SE block whose input is not the preceding depthwise conv's "
                "only output: the gate kernel reuses that conv's channel sums")
        if (o["act1"] != "swish" or len(cons) != 1 or cons[0]["op"] != "CONV"
                or not is_1x1(cons[0]) or cons[0]["ins"][0] != o["out"]):
            raise recog.Unsupported(
                "SE block that is not swish-activated and followed by exactly "
                "one 1x1 conv (which applies the gate)")
        _, h, w, c, _, _, _ = self.bufs[ib]
        if c % 8:
            raise recog.Unsupported(
                f"SE block on {c} channels (the gate kernel takes multiples "
                f"of 8)")
        rd = o["w1"].shape[0]
        gb = self.buf(F16, 1, 1, c, elems=c + 16)
        tensors = [self.tensor(f"se.{self.k}.wr", o["w1"]),
                   self.tensor(f"se.{self.k}.br", o["b1"]),
                   self.tensor(f"se.{self.k}.we", o["w2"]),
                   self.tensor(f"se.{self.k}.be", o["b2"])]
        self.emit_dot("SE_GATE16", [self.chsum[1], gb], tensors,
                      [c, rd, h * w])
        self.gate[o["out"]] = gb
        self.vbuf[o["out"]] = ib
        return i + 1

    def l_gem(self, i, o):
        """GeM pooling of fp32 rows (outer exponent 1 / p)."""
        ib = self.rows32(o["ins"][0])
        _, h, w, c, ld, _, _ = self.bufs[ib]
        if c % 4 or ld != c:
            raise recog.Unsupported(
                f"GeM on {c} channels with stride {ld}: the kernel takes a "
                f"multiple of 4 with stride = channels")
        if np.float32(1.0) / np.float32(o["p"]) != np.float32(o["q"]):
            raise recog.Unsupported(
                f"GeM whose outer exponent {o['q']} is not 1 / p (p = "
                f"{o['p']})")
        ob = self.buf(F32, 1, 1, c)
        consts = np.array([o["eps"], o["p"]], np.float32)
        self.emit_dot("GEM_A", [ib, ob],
                      [self.tensor(f"gem.{self.k}.consts", consts)],
                      [h * w, c])
        self.vbuf[o["out"]] = ob
        return i + 1

    def sed_tensors(self, tag, o):
        """Adds the fc, attention and class weights of one SED head.

        The head's output scale is folded into the class layer.
        """
        scale = np.float32(o["scale"])
        wc = (o["wc"] * scale).astype(np.float32)
        bc = (o["bc"] * scale).astype(np.float32)
        return [self.tensor(f"{tag}.fc.w", o["w1"]),
                self.tensor(f"{tag}.fc.b", o["b1"]),
                self.tensor(f"{tag}.att.w", o["wa"]),
                self.tensor(f"{tag}.att.b", o["ba"]),
                self.tensor(f"{tag}.cla.w", wc),
                self.tensor(f"{tag}.cla.b", bc)]

    def head_part(self, i, o, key):
        """Records a dense or SED head as one term of the summed logits.

        It waits there for the Adds that sum the terms.
        """
        cons = self.consumers(o["out"])
        if len(cons) != 1 or cons[0]["op"] != "VADD":
            raise recog.Unsupported(
                "a dense or SED head that is not a term of dense(pooled) + SED "
                "over height + SED over width (the dotprod family's head "
                "kernel computes that sum)")
        self.head_terms[o["out"]] = {key: o}
        return i + 1

    def l_dense(self, i, o):
        """The dense head: a term of the summed logits."""
        return self.head_part(i, o, "dense")

    def l_sed(self, i, o):
        """A SED head: a term of the summed logits."""
        key = {3: "sed_h", 2: "sed_w"}.get(o["red"], "sed_other")
        return self.head_part(i, o, key)

    def l_vadd(self, i, o):
        """logits = (dense + sed_h) + sed_w, the order the head kernel adds."""
        terms = [self.head_terms.get(v) for v in o["ins"]]
        if None in terms:
            raise recog.Unsupported(
                "vector Add of something other than dense / SED head outputs")
        merged = {**terms[0], **terms[1]}
        first_sum = set(merged) == {"dense", "sed_h"}
        second_sum = (set(merged) == {"dense", "sed_h", "sed_w"}
                      and {"dense", "sed_h"} in [set(t) for t in terms])
        if not first_sum and not second_sum:
            raise recog.Unsupported(
                f"head sum of {sorted(merged)} in this order: the head kernel "
                f"computes (dense + SED over height) + SED over width")
        self.head_terms[o["out"]] = merged
        if first_sum:
            self._check_second_sum(o)
        if second_sum:
            self.heads(merged["dense"], merged["sed_h"], merged["sed_w"],
                       o["out"])
        return i + 1

    def _check_second_sum(self, o):
        """dense + SED over height must then be added to SED over width."""
        cons = self.consumers(o["out"])
        if len(cons) != 1 or cons[0]["op"] != "VADD":
            raise recog.Unsupported(
                "dense + SED over height that is not then added to an SED "
                "over width (the head kernel computes the sum of all three)")

    def heads(self, dense, sed_h, sed_w, out):
        """Emits the HEADS_S8 op: dense + SED over height + SED over width.

        Args:
            dense, sed_h, sed_w: the three head IR ops.
            out: the IR value of their sum.

        Raises:
            recog.Unsupported: the heads do not fit the kernel.
        """
        if sed_h["ins"] != sed_w["ins"]:
            raise recog.Unsupported("SED heads on different feature maps")
        fb = self.rows32(sed_h["ins"][0])
        eb = self.vbuf[dense["ins"][0]]
        _, h, w, c, ld, _, _ = self.bufs[fb]
        hid, nc = sed_h["w1"].shape[0], dense["w"].shape[0]
        for sed in (sed_h, sed_w):
            reason = sed_reason(sed, c)
            if reason:
                raise recog.Unsupported(f"SED head with {reason}")
        shapes_ok = (dense["w"].shape == (nc, c)
                     and sed_w["w1"].shape == (hid, c)
                     and sed_h["wa"].shape == (nc, hid)
                     and sed_w["wa"].shape == (nc, hid)
                     and (sed_h["L"], sed_w["L"]) == (h, w))
        if (not shapes_ok or c % 16 or hid % 16 or ld != c
                or dense.get("act")):
            raise recog.Unsupported(
                f"head shapes: {c} features, {hid} hidden, {nc} classes over "
                f"a {h}x{w} map")
        steps = max(h, w)
        ob = self.buf(F32, 1, 1, nc, elems=lower.r16(nc) + 16)
        a8 = self.buf(I8, steps + 4, 1, max(lower.r16(c), lower.r16(hid)))
        rq = self.buf(F32, steps + 4, 1, 2)
        tensors = [self.tensor(f"lin.{self.k}.w", dense["w"]),
                   self.tensor(f"lin.{self.k}.b", dense["b"])]
        tensors += (self.sed_tensors(f"sed0.{self.k}", sed_h)
                    + self.sed_tensors(f"sed1.{self.k}", sed_w))
        self.emit_dot("HEADS_S8", [fb, eb, ob, a8, rq], tensors,
                      [h, w, c, hid, nc])
        self.vbuf[out] = ob
