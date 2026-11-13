"""Export Google Perch v2 (the published perch_v2_fp32.onnx or
perch_v2_no_dft_fp32.onnx) to a flat, mmap-able engine file (.bnp2).

usage:
  python -I export_perch.py <perch_v2[_no_dft]_fp32.onnx> <labels.txt> \
      <out.bnp2> [--classes idx.txt]
  --classes: keep only these classes (0-based indices into the model's
             14,795, one per line, in output order; the regional slices of
             the BirdNET-Go catalog are such column slices); labels.txt is
             then the sliced label file.

File layout (little endian), same conventions as .bn24:
  char magic[8] = "BNP2ENG1"; u32 version=1; u32 n_tensors; u32 n_classes;
  u32 n_blocks;
  u32 license_len; char license[license_len] (padded to 8);
  u8 labels_sha256[32] (sha256 of the label file as shipped)
  block table: n_blocks x 8 x i32 {k, stride, cin, cexp (0 = no expansion
      conv), cout, se_rd, skip, pads = top | left << 8 | bottom << 16 |
      right << 24}
  tensor dir: n_tensors x {char name[48]; u32 dtype(0=f32); u32 ndim;
      u32 dims[4]; u64 offset; u64 nbytes}
  data: each tensor 64-byte aligned, offsets from file start.
Tensors: fe.window [640], fe.mel [128][513], fe.consts [8] = {frame, hop,
  pad, n_fft, log floor, log scale, norm eps, prototypes per class};
  stem.w OHWI [40][3][3][1], stem.b; m<i>.ex.w [cexp][cin] (+ .b),
  m<i>.dw.w [k][k][C] (+ .b), m<i>.se1.w [rd][C], m<i>.se2.w [C][rd] (+ .b),
  m<i>.pj.w [cout][C] (+ .b); top.w [1536][384] (+ .b);
  proto [n_classes * 4][1536] (prototype rows); cls.w [n_classes][4];
  cls.b [n_classes].
The BatchNorm Mul/Add after every conv is folded into the conv (w * s,
b * s + t); every other value is copied.
License: Perch v2 (Google Research) is Apache-2.0; the file is a re-layout of
the published weights.
"""
import hashlib
import re
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

LICENSE = ("Perch v2 (bird-vocalization-classifier, perch_v2) by Google "
           "Research, https://www.kaggle.com/models/google/"
           "bird-vocalization-classifier ; ONNX conversion by justinchuby "
           "(huggingface.co/justinchuby/Perch-onnx), catalog "
           "tphakala/Perch-v2-Models. Licensed under the Apache License, "
           "Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). "
           "Changes: re-laid-out for a standalone engine, BatchNorm folded "
           "into the convolutions; no retraining.")
BN_RE = r"/(Stem_0|Head_0|MBConv_(\d+))/(\w*?)BatchNorm"
SE_RE = r"MBConv_(\d+)/SqueezeAndExcitation_0/(Reduce|Expand)/dot_general"


class Model:
    """The loaded ONNX graph: constants, nodes, indices and consumers.

    Attributes:
        values: the initializers by name (numpy arrays).
        nodes: the nodes in graph order.
        nidx: each node's index by name.
        cons: the nodes consuming each tensor, by tensor name.
    """

    def __init__(self, path):
        graph = onnx.load(path).graph
        self.values = {t.name: numpy_helper.to_array(t)
                       for t in graph.initializer}
        self.nodes = list(graph.node)
        self.nidx = {n.name: i for i, n in enumerate(self.nodes)}
        self.cons = {}
        for n in self.nodes:
            for x in n.input:
                self.cons.setdefault(x, []).append(n)

    @staticmethod
    def attrs(n):
        """A node's attributes by name."""
        return {a.name: onnx.helper.get_attribute_value(a)
                for a in n.attribute}

    def const_in(self, n):
        """The one constant input of a node."""
        r = [self.values[x] for x in n.input if x in self.values]
        assert len(r) == 1, (n.name, n.op_type)
        return r[0]


def only(xs, what):
    """The one node of a list (what names it in the assertion)."""
    assert len(xs) == 1, (what, [x.name for x in xs])
    return xs[0]


class Export:
    """What the file gets: the tensors in order, the block table, MAC rows.

    Attributes:
        tensors: (name, float32 array) in file order.
        blocks: block table rows (k, stride, cin, cexp, cout, se_rd, skip,
            pads).
        macs: (layer, MACs) rows.
    """

    def __init__(self):
        self.tensors = []
        self.blocks = []
        self.macs = []

    def add(self, name, a):
        """Appends a tensor (contiguous float32); names fit 48 bytes."""
        a = np.ascontiguousarray(a, dtype=np.float32)
        assert len(name) < 48, name
        self.tensors.append((name, a))


# ---- front end ----------------------------------------------------------
def framing(model):
    """(frame, hop, pad) of the identity framing conv (node 1)."""
    fc = model.nodes[1]
    assert fc.op_type == "Conv"
    fa = model.attrs(fc)
    fw = model.values[fc.input[1]]
    frame, hop, pad = fw.shape[0], fa["strides"][0], fa["pads"][0]
    assert fw.shape == (frame, 1, frame, 1) and np.array_equal(
        fw.reshape(frame, frame), np.eye(frame, dtype=np.float32))
    assert fa["pads"] == [pad, 0, pad, 0]
    return frame, hop, pad


def window_and_nfft(model, frame):
    """(the analysis window, n_fft) from the window Mul and the Pad."""
    win = model.const_in(model.nodes[4]).reshape(-1)
    assert win.shape == (frame,)
    padn = model.nodes[5]
    assert padn.op_type == "Pad"
    pv = model.values[padn.input[1]]
    assert list(pv[:5]) == [0, 0, 0, 0, 0]
    return win, frame + int(pv[5])


def has_bins_input(model, n, bins):
    """Does a node read a constant with bins rows?"""
    return any(model.values.get(x) is not None
               and model.values[x].shape[0] == bins for x in n.input)


def mel_matmul(model, nfft):
    """The MatMul applying the mel matrix (after a DFT, or the no_dft form)."""
    nodes = model.nodes
    if nodes[7].op_type == "DFT":
        assert (model.attrs(nodes[7]).get("onesided") == 1
                and int(model.values[nodes[7].input[1]]) == nfft)
        assert nodes[8].op_type == "ReduceL2"
        return nodes[9]
    # no_dft: MatMul with a [n_fft][2 * bins] matrix (re / im interleaved),
    # then the magnitude
    return [n for n in nodes[:20] if n.op_type == "MatMul"
            and has_bins_input(model, n, nfft // 2 + 1)][0]


def log_constants(model, melmm):
    """(log floor, log scale): the Max -> Log -> Mul after the mel MatMul."""
    mx = only([n for n in model.cons[melmm.output[0]] if n.op_type == "Max"],
              "max")
    floor = float(model.const_in(mx))
    lg = only(model.cons[mx.output[0]], "log")
    assert lg.op_type == "Log"
    sc = only(model.cons[lg.output[0]], "scale")
    assert sc.op_type == "Mul"
    return floor, float(model.const_in(sc))


def export_front_end(model, out):
    """The window and mel matrix; returns the front-end constants.

    Returns:
        (frame, hop, pad, nfft, log floor, log scale).
    """
    frame, hop, pad = framing(model)
    win, nfft = window_and_nfft(model, frame)
    melmm = mel_matmul(model, nfft)
    mel = model.const_in(melmm)
    assert mel.shape == (nfft // 2 + 1, 128), mel.shape
    floor, lscale = log_constants(model, melmm)
    out.add("fe.window", win)
    out.add("fe.mel", mel.T)
    return frame, hop, pad, nfft, floor, lscale


# ---- body: every Conv, classified by the BatchNorm node that follows it --
def conv_key(bn_name):
    """("stem", None), ("head", None) or (block index, kind) of a BN name."""
    mb = re.search(BN_RE, bn_name)
    if mb.group(1) == "Stem_0":
        return ("stem", None)
    if mb.group(1) == "Head_0":
        return ("head", None)
    return (int(mb.group(2)), mb.group(3))


def conv_followers(model, addn):
    """(swish, residual) of a conv from what consumes its BatchNorm Add.

    swish: Neg/Exp/Add/Div/Mul follows. A residual Add of THIS block comes
    before the next conv (the next block's residual Add also consumes it,
    later).
    """
    nxt = model.cons[addn.output[0]]
    swish = any(x.op_type == "Neg" for x in nxt)
    nconv = min([j for j in range(model.nidx[addn.name], len(model.nodes))
                 if model.nodes[j].op_type == "Conv"] + [len(model.nodes)])
    res = [x for x in nxt if x.op_type == "Add" and model.nidx[x.name] < nconv]
    return swish, bool(res)


def fold_bn(model, n):
    """A Conv with the BatchNorm after its Transpose folded in.

    Returns:
        (key, record): the key from conv_key and a dict with the folded
        float64 weights w and bias b, the attributes at, swish, res and the
        BatchNorm's name.
    """
    w = model.values[n.input[1]].astype(np.float64)
    if len(n.input) > 2 and n.input[2]:
        b = model.values[n.input[2]].astype(np.float64)
    else:
        b = np.zeros(w.shape[0])
    t = only(model.cons[n.output[0]], "transpose")
    assert t.op_type == "Transpose" and model.attrs(t)["perm"] == [0, 2, 3, 1]
    mul = only(model.cons[t.output[0]], "bn mul")
    addn = only(model.cons[mul.output[0]], "bn add")
    assert (mul.op_type == "Mul" and addn.op_type == "Add"
            and "BatchNorm" in mul.name), mul.name
    s = model.const_in(mul).reshape(-1).astype(np.float64)
    tt = model.const_in(addn).reshape(-1).astype(np.float64)
    wf = w * s[:, None, None, None]
    bf = b * s + tt
    key = conv_key(mul.name)
    swish, res = conv_followers(model, addn)
    return key, dict(w=wf, b=bf, at=model.attrs(n), swish=swish, res=res,
                     name=mul.name)


def folded_convs(model):
    """Every Conv after the framing conv, BN folded; returns (convs, order)."""
    convs = {}
    order = []
    for n in model.nodes[2:]:
        if n.op_type != "Conv":
            continue
        key, rec = fold_bn(model, n)
        convs[key] = rec
        order.append(key)
    return convs, order


def export_stem(convs, out):
    """The 3x3 stride-2 stem conv."""
    st = convs[("stem", None)]
    assert (st["w"].shape == (40, 1, 3, 3) and st["at"]["strides"] == [2, 2]
            and not st["at"].get("pads") and st["swish"])
    out.add("stem.w", st["w"].transpose(0, 2, 3, 1))
    out.add("stem.b", st["b"])
    out.macs.append(("stem 3x3 s2 1->40 +swish", 249 * 63 * 40 * 9))


def se_matmuls(model):
    """{(block, "Reduce" | "Expand"): (weights, bias, the bias Add's
    consumers)} of the squeeze-excite MatMuls."""
    se_mm = {}
    for n in model.nodes:
        mm = re.search(SE_RE, n.name)
        if mm and n.op_type == "MatMul":
            ba = only(model.cons[n.output[0]], "se bias")
            se_mm[(int(mm.group(1)), mm.group(2))] = (
                model.const_in(n), model.const_in(ba).reshape(-1),
                model.cons[ba.output[0]])
    return se_mm


def export_expand(i, ex, out, h, w, cin):
    """The optional 1x1 expansion conv; returns the expanded width."""
    if ex is None:
        return cin
    assert (ex["w"].shape[1] == cin and ex["w"].shape[2:] == (1, 1)
            and ex["swish"])
    cexp = ex["w"].shape[0]
    out.add(f"m{i}.ex.w", ex["w"][:, :, 0, 0])
    out.add(f"m{i}.ex.b", ex["b"])
    out.macs.append((f"m{i}.ex 1x1 {cin}->{cexp} +swish", h * w * cin * cexp))
    return cexp


def export_depthwise(i, dw, out, h, w, cexp):
    """The depthwise conv; returns (k, stride, pads, h_out, w_out)."""
    k = dw["w"].shape[2]
    a = dw["at"]
    assert a["group"] == cexp and dw["w"].shape == (cexp, 1, k, k) \
        and dw["swish"]
    s = a["strides"][0]
    pads = a.get("pads", [0, 0, 0, 0])
    pt, pl, pb, pr = pads
    ho, wo = (h + pt + pb - k) // s + 1, (w + pl + pr - k) // s + 1
    out.add(f"m{i}.dw.w", dw["w"][:, 0].transpose(1, 2, 0))
    out.add(f"m{i}.dw.b", dw["b"])
    out.macs.append((f"m{i}.dw {k}x{k} s{s} +swish", ho * wo * cexp * k * k))
    return k, s, pads, ho, wo


def export_se(i, se_mm, out, cexp):
    """The squeeze-excite pair; returns the reduced width."""
    r1, b1, c1 = se_mm[(i, "Reduce")]
    r2, b2, _ = se_mm[(i, "Expand")]
    rd = r1.shape[1]
    assert (r1.shape == (cexp, rd) and r2.shape == (rd, cexp)
            and any(x.op_type == "Neg" for x in c1))
    out.add(f"m{i}.se1.w", r1.T)
    out.add(f"m{i}.se1.b", b1)
    out.add(f"m{i}.se2.w", r2.T)
    out.add(f"m{i}.se2.b", b2)
    return rd


def export_project(i, pj, out, ho, wo, cexp):
    """The 1x1 projection conv; returns the output width."""
    cout = pj["w"].shape[0]
    assert pj["w"].shape == (cout, cexp, 1, 1) and not pj["swish"]
    out.add(f"m{i}.pj.w", pj["w"][:, :, 0, 0])
    out.add(f"m{i}.pj.b", pj["b"])
    out.macs.append((f"m{i}.pj 1x1 {cexp}->{cout}", ho * wo * cexp * cout))
    return cout


def export_block(i, convs, se_mm, out, h, w, c):
    """Block i: expansion (optional), depthwise, squeeze-excite, projection.

    Returns:
        The output geometry (height, width, channels).
    """
    ex = convs.get((i, "Expand"))
    dw = convs[(i, "Depthwise")]
    pj = convs[(i, "Project")]
    cin = c
    cexp = export_expand(i, ex, out, h, w, cin)
    k, s, pads, ho, wo = export_depthwise(i, dw, out, h, w, cexp)
    rd = export_se(i, se_mm, out, cexp)
    cout = export_project(i, pj, out, ho, wo, cexp)
    skip = int(pj["res"])
    assert not skip or (s == 1 and cin == cout)
    pt, pl, pb, pr = pads
    out.blocks.append((k, s, cin, cexp if ex is not None else 0, cout, rd,
                       skip, pt | pl << 8 | pb << 16 | pr << 24))
    return ho, wo, cout


def export_body(convs, order, se_mm, out):
    """The MBConv blocks in order; asserts the final geometry."""
    nblk = 1 + max(k[0] for k in order if isinstance(k[0], int))
    h, w, c = 249, 63, 40
    for i in range(nblk):
        h, w, c = export_block(i, convs, se_mm, out, h, w, c)
    assert (h, w, c) == (16, 4, 384), (h, w, c)


def export_top(convs, out):
    """The 1x1 head conv (384 -> 1536)."""
    hd = convs[("head", None)]
    assert hd["w"].shape == (1536, 384, 1, 1) and hd["swish"]
    out.add("top.w", hd["w"][:, :, 0, 0])
    out.add("top.b", hd["b"])
    out.macs.append(("top 1x1 384->1536 +swish", 64 * 384 * 1536))


# ---- ProtoPNet head -----------------------------------------------------
def proto_head(model):
    """(prototypes [1536][NP], class weights [C][4], class bias [C], norm
    eps) read from the ProtoPNet head."""
    nodes = model.nodes
    cons = model.cons
    pm = only([n for n in nodes if n.op_type == "MatMul"
               and "protopnet" in n.name], "proto matmul")
    p = model.const_in(pm)  # [1536][NP]
    dv = [n for n in nodes
          if n.op_type == "Div" and n.output[0] == pm.input[0]][0]
    eps_add = [n for n in nodes
               if n.op_type == "Add" and n.output[0] == dv.input[1]][0]
    neps = float(model.const_in(eps_add))
    rmax = only(cons[only(cons[pm.output[0]], "reshape").output[0]],
                "reducemax")
    assert rmax.op_type == "ReduceMax"
    mulw = only(cons[rmax.output[0]], "class weight mul")
    cw = model.const_in(mulw)[0]  # [C][4]
    rs = only(cons[mulw.output[0]], "reducesum")
    badd = only(cons[rs.output[0]], "bias add")
    cb = model.const_in(badd).reshape(-1)
    return p, cw, cb, neps


def export_proto_head(model, out, cls_sel):
    """The ProtoPNet head, sliced to --classes when given.

    Returns:
        (n_classes, prototypes per class, norm eps).
    """
    p, cw, cb, neps = proto_head(model)
    nc, ppc = cw.shape
    assert p.shape == (1536, nc * ppc)
    if cls_sel is not None:
        cols = (cls_sel[:, None] * ppc + np.arange(ppc)[None, :]).reshape(-1)
        p, cw, cb = p[:, cols], cw[cls_sel], cb[cls_sel]
        nc = len(cls_sel)
    out.add("proto", p.T)
    out.add("cls.w", cw)
    out.add("cls.b", cb)
    out.macs.append((f"proto {nc * ppc} x 1536 x 64 cells",
                     64 * 1536 * nc * ppc))
    return nc, ppc, neps


def read_labels(path, nc):
    """The sha256 of the label file; asserts it has nc labels.

    A leading "inat2024_fsd50k" line (the catalog's header) is skipped.
    """
    with open(path, "rb") as f:
        labels_raw = f.read()
    lab = [line for line in labels_raw.decode().splitlines() if line.strip()]
    if lab and lab[0] == "inat2024_fsd50k":
        lab = lab[1:]
    assert len(lab) == nc, (len(lab), nc)
    return hashlib.sha256(labels_raw).digest()


# ---- write --------------------------------------------------------------
def file_header(out, nc, sha):
    """The header: magic, counts, license, label hash and the block table."""
    lic = LICENSE.encode()
    lic += b"\0" * ((8 - len(lic) % 8) % 8)
    hdr = b"BNP2ENG1" + struct.pack("<5I", 1, len(out.tensors), nc,
                                    len(out.blocks), len(lic)) + lic + sha
    hdr += b"".join(struct.pack("<8i", *b) for b in out.blocks)
    return hdr


def tensor_directory(tensors, start):
    """(directory entries, (offset, array) pairs, end offset).

    Each tensor's data starts 64-byte aligned, the first one at start.
    """
    off = start
    entries = []
    blobs = []
    for name, a in tensors:
        dims = list(a.shape) + [0] * (4 - a.ndim)
        entries.append(name.encode().ljust(48, b"\0")
                       + struct.pack("<II4IQQ", 0, a.ndim, *dims, off,
                                     a.nbytes))
        blobs.append((off, a))
        off = (off + a.nbytes + 63) & ~63
    return entries, blobs, off


def write_file(dst, out, nc, sha):
    """Writes the .bnp2 file; returns the end offset (its size)."""
    hdr = file_header(out, nc, sha)
    dir_size = len(out.tensors) * (48 + 4 + 4 + 16 + 8 + 8)
    off = (len(hdr) + dir_size + 63) & ~63
    entries, blobs, off = tensor_directory(out.tensors, off)
    with open(dst, "wb") as f:
        f.write(hdr + b"".join(entries))
        for o, a in blobs:
            f.write(b"\0" * (o - f.tell()))
            f.write(a.tobytes())
    return off


def report(dst, out, nc, fe, neps, off):
    """Prints the file summary and the block table."""
    tot = sum(x[1] for x in out.macs)
    frame, hop, pad, nfft, floor, lscale = fe
    print(f"{dst}: {len(out.tensors)} tensors, {nc} classes, "
          f"{len(out.blocks)} blocks, {off / 1e6:.1f} MB; front end frame "
          f"{frame} hop {hop} pad {pad} nfft {nfft} floor {floor} scale "
          f"{lscale}; norm eps {neps}; MACs {tot / 1e6:.0f} M")
    for i, b in enumerate(out.blocks):
        print(f"  m{i}: k{b[0]} s{b[1]} {b[2]}->{b[3]}->{b[4]} se{b[5]} "
              f"skip{b[6]} pads t{b[7] & 255} l{b[7] >> 8 & 255} "
              f"b{b[7] >> 16 & 255} r{b[7] >> 24 & 255}")


def parse_args(argv):
    """(model, labels, destination, class indices or None) from argv."""
    args = [a for a in argv[1:] if not a.startswith("--")]
    src, labels_path, dst = args[:3]
    cls_sel = None
    if "--classes" in argv:
        with open(argv[argv.index("--classes") + 1], encoding="utf-8") as f:
            cls_sel = np.array([int(line) for line in f if line.strip()],
                               np.int64)
    return src, labels_path, dst, cls_sel


def main():
    src, labels_path, dst, cls_sel = parse_args(sys.argv)
    model = Model(src)
    out = Export()
    fe = export_front_end(model, out)
    convs, order = folded_convs(model)
    export_stem(convs, out)
    se_mm = se_matmuls(model)
    export_body(convs, order, se_mm, out)
    export_top(convs, out)
    nc, ppc, neps = export_proto_head(model, out, cls_sel)
    frame, hop, pad, nfft, floor, lscale = fe
    out.add("fe.consts", np.array([frame, hop, pad, nfft, floor, lscale,
                                   neps, ppc], np.float32))
    sha = read_labels(labels_path, nc)
    off = write_file(dst, out, nc, sha)
    report(dst, out, nc, fe, neps, off)


if __name__ == "__main__":
    main()
