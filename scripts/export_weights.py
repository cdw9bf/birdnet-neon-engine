"""Export BirdNET v2.4 (the published BirdNET_v2.4_fp32.onnx) to a flat,
mmap-able engine file (.bn24) and print the per-layer MAC / shape table.

usage:
  python -I export_weights.py <BirdNET_v2.4_fp32.onnx> <labels.txt> \
      <out.bn24> [mac_table.txt]

File layout (little endian):
  char magic[8] = "BN24ENG1"; u32 version=1; u32 n_tensors; u32 n_classes;
  u32 n_blocks;
  u32 license_len; char license[license_len] (padded to 8);
  u8 labels_sha256[32]
  block table: n_blocks x 8 x i32 {stride, cin, cmid, cout, se_rd, skip, 0, 0}
  tensor dir: n_tensors x {char name[48]; u32 dtype(0=f32); u32 ndim;
      u32 dims[4]; u64 offset; u64 nbytes}
  data: each tensor 64-byte aligned, offsets from file start.
Layouts: dense conv OHWI [cout][kh][kw][cin]; depthwise [kh][kw][C];
1x1 / linear [out][in]; mel [96][bins].
Every value is copied from the ONNX file (BN is already folded there); no
retraining, no re-fitting.
License: derived from BirdNET v2.4 and therefore CC BY-NC-SA 4.0.
"""
import hashlib
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

LICENSE = ("BirdNET v2.4 (BirdNET_GLOBAL_6K_V2.4) weights, K. Lisa Yang Center "
           "for Conservation Bioacoustics, Cornell Lab of Ornithology, and "
           "Chemnitz University of Technology (Kahl et al. 2021, "
           "https://github.com/kahst/BirdNET-Analyzer), re-laid-out for a "
           "standalone engine. Licensed CC BY-NC-SA 4.0 "
           "(https://creativecommons.org/licenses/by-nc-sa/4.0/). "
           "Non-commercial use only; derivatives keep this licence. Changes: "
           "layout conversion only; no retraining.")
N_CLASSES = 6522
# (spectrogram, n_fft, hop, DFT matrix, mel MatMul, power node) of the two
# front-end branches
SPECTROGRAMS = (("MEL_SPEC1", 2048, 278, "rfft3", "MatMul2", "Pow_1"),
                ("MEL_SPEC2", 1024, 280, "rfft2", "MatMul1", "Pow_1"))
# (stage, blocks, expanded channels, output channels) of the body
STAGES = ((1, 3, 72, 36), (2, 4, 288, 72), (3, 5, 864, 108),
          (4, 4, 1536, 192))


class Model:
    """The loaded ONNX graph: its constants and nodes.

    Attributes:
        values: the initializers by name (numpy arrays).
        nodes: the nodes in graph order.
    """

    def __init__(self, path):
        graph = onnx.load(path).graph
        self.values = {t.name: numpy_helper.to_array(t)
                       for t in graph.initializer}
        self.nodes = list(graph.node)

    @staticmethod
    def attrs(n):
        """A node's attributes by name."""
        return {a.name: onnx.helper.get_attribute_value(a)
                for a in n.attribute}

    def scalar(self, name):
        """An initializer's first element as a Python float."""
        return float(self.values[name].reshape(-1)[0])

    def find(self, op, sub):
        """The one node of type op whose name contains sub."""
        r = [n for n in self.nodes if n.op_type == op and sub in n.name]
        assert len(r) == 1, (op, sub, [n.name for n in r])
        return r[0]

    def first(self, op, sub):
        """The first node of type op whose name contains sub."""
        return [n for n in self.nodes if n.op_type == op and sub in n.name][0]


class ConvStream:
    """The graph's Conv nodes in order, consumed one at a time."""

    def __init__(self, model):
        self.model = model
        self._convs = iter([n for n in model.nodes if n.op_type == "Conv"])

    def take(self):
        """(node, float32 weights, float32 bias, attributes) of the next Conv.

        A Conv without a bias gets zeros.
        """
        n = next(self._convs)
        w = self.model.values[n.input[1]].astype(np.float32)
        if len(n.input) > 2 and n.input[2]:
            b = self.model.values[n.input[2]].astype(np.float32)
        else:
            b = np.zeros(w.shape[0], np.float32)
        return n, w, b, self.model.attrs(n)

    def check_done(self):
        """Exits with "unconsumed conv" when a Conv was left unread."""
        try:
            next(self._convs)
            raise SystemExit("unconsumed conv")
        except StopIteration:
            pass


class Export:
    """What the file gets: the tensors in order, the block table, MAC rows.

    Attributes:
        tensors: (name, float32 array) in file order.
        blocks: block table rows (stride, cin, cmid, cout, se_rd, skip, 0, 0).
        macs: (layer, pixels, N, K) rows; MACs = pixels * N * K.
        fe_meta: (tag, n_fft, hop, first mel bin, last mel bin + 1) per
            front-end branch.
    """

    def __init__(self):
        self.tensors = []
        self.blocks = []
        self.macs = []
        self.fe_meta = []

    def add(self, name, a):
        """Appends a tensor (contiguous float32); names fit 48 bytes."""
        a = np.ascontiguousarray(a, dtype=np.float32)
        assert len(name) < 48, name
        self.tensors.append((name, a))


def check_front_end(model):
    """The normalization constants.

    (x - min) / (max(x - min) + eps) -> (.. - 0.5) * 2
    """
    assert abs(model.scalar("model/MEL_SPEC1/add/y") - 1e-6) < 1e-12
    assert (model.scalar("model/MEL_SPEC1/Sub_1/y") == 0.5
            and model.scalar("model/MEL_SPEC1/Mul/y") == 2.0)


def export_spectrogram(model, out, spec, n_fft, hop, dft, melmm, pw):
    """One mel spectrogram branch: window, DFT (cos) matrix, mel matrix, power.

    Returns:
        (tag, n_fft, hop, first mel bin, last mel bin + 1).
    """
    assert int(model.scalar(f"model/{spec}/stft/frame_length")) == n_fft
    assert int(model.scalar(f"model/{spec}/stft/frame_step")) == hop
    win = model.values[f"model/{spec}/stft/hann_window/sub_2"]
    d = model.values[f"model/{spec}/stft/{dft}_dft_matrix"]
    k = np.arange(d.shape[1])
    nn = np.arange(n_fft)
    assert np.abs(d - np.cos(2 * np.pi * np.outer(nn, k) / n_fft)).max() \
        < 1e-3, "DFT matrix is not the cos part"
    mm = model.find("MatMul", f"{spec}/Tensordot/{melmm}")
    mel = model.values[mm.input[1]]  # [bins][96]
    p = model.find("Pow", f"{spec}/{pw}")
    tag = "fe1" if spec == "MEL_SPEC1" else "fe2"
    out.add(f"{tag}.window", win)
    # the dense cos matrix as in the graph: kept for the bit-faithful oracle
    out.add(f"{tag}.dft", d)
    out.add(f"{tag}.mel", mel.T)  # [96][bins]
    out.add(f"{tag}.pow", np.array([model.scalar(p.input[1])]))
    nz = np.nonzero(np.abs(mel).sum(1))[0]
    return tag, n_fft, hop, int(nz[0]), int(nz[-1]) + 1


def export_front_end(model, out):
    """Both spectrogram branches, then the per-channel BatchNorm."""
    for spec in SPECTROGRAMS:
        out.fe_meta.append(export_spectrogram(model, out, *spec))
    # channel order: concat(MEL_SPEC1, MEL_SPEC2); mel axis reversed
    # (ReverseV2 on the mel axis); BN per channel
    cc = model.find("Concat", "concatenate/concat")
    assert "MEL_SPEC1" in cc.input[0] and "MEL_SPEC2" in cc.input[1]
    bm = model.find("Mul", "BNORM_SPEC_NOQUANT/FusedBatchNormV32")
    ba = model.first("Add", "BNORM_SPEC_NOQUANT")
    out.add("fe.bn_s", model.values[bm.input[1]].reshape(2))
    out.add("fe.bn_b", model.values[ba.input[1]].reshape(2))


def export_stem(out, convs):
    """The 4x8 stem conv, the max/avg pool concat and the 1x1 after it."""
    model = convs.model
    _, w, b, at = convs.take()
    assert (list(w.shape) == [24, 2, 4, 8] and at["strides"] == [2, 2]
            and at.get("auto_pad") == b"SAME_UPPER")
    out.add("c0.w", w.transpose(0, 2, 3, 1))
    out.add("c0.b", b)
    out.macs.append(("c0 4x8 s2 2->24 +ReLU", 48 * 256, 24, 64))
    pool_ops = [x.op_type for x in model.nodes if "POOL_0" in x.name]
    assert pool_ops[:3] == ["MaxPool", "AveragePool", "Concat"]
    pc = model.find("Concat", "POOL_0_CONCAT")
    assert "MAX" in pc.input[0] and "AVG" in pc.input[1]
    _, w, b, _ = convs.take()
    assert list(w.shape) == [24, 48, 1, 1]
    out.add("p0.w", w[:, :, 0, 0])
    out.add("p0.b", b)
    out.macs.append(("p0 1x1 48->24", 48 * 128, 24, 48))


class Block:
    """One MBConv block of the body: stage s, index j.

    Attributes:
        s, j: the stage and the index within it.
        tag: "b<s>.<j>", the tensor name prefix.
        h, w, c: the input geometry (height, width, channels).
        cm, co: the expanded and the output channel counts.
        st, ho, wo: the depthwise stride and the output size (set by
            export_dw).
    """

    def __init__(self, s, j, h, w, c, cm, co):
        self.s = s
        self.j = j
        self.tag = f"b{s}.{j}"
        self.h = h
        self.w = w
        self.c = c
        self.cm = cm
        self.co = co
        self.st = None
        self.ho = None
        self.wo = None

    def export(self, model, out, convs):
        """Reads the block's convs in graph order.

        Returns:
            The output geometry (height, width, channels).
        """
        self.export_pw(out, convs)
        self.export_dw(out, convs)
        rd = 0
        if self.s > 1:
            rd = self.export_se(out, convs)
        skip = self.export_pwl(model, out, convs)
        out.blocks.append((self.st, self.c, self.cm, self.co, rd, skip, 0, 0))
        return self.ho, self.wo, self.co

    def export_pw(self, out, convs):
        """The 1x1 expansion conv (+SiLU)."""
        _, w, b, _ = convs.take()
        assert (w.shape[2:] == (1, 1) and w.shape[0] == self.cm
                and w.shape[1] == self.c), (self.tag, w.shape)
        out.add(f"{self.tag}.pw.w", w[:, :, 0, 0])
        out.add(f"{self.tag}.pw.b", b)
        out.macs.append((f"{self.tag}.pw {self.c}->{self.cm} +SiLU",
                         self.h * self.w, self.cm, self.c))

    def export_dw(self, out, convs):
        """The depthwise 3x3 conv (+SiLU); sets the stride and output size."""
        _, w, b, at = convs.take()
        st = at["strides"][0]
        assert at["group"] == self.cm and w.shape[2:] == (3, 3)
        if st == 2:
            assert at.get("pads") == [1, 1, 1, 1]
        else:
            assert at.get("auto_pad") == b"SAME_UPPER"
        out.add(f"{self.tag}.dw.w", w[:, 0].transpose(1, 2, 0))
        out.add(f"{self.tag}.dw.b", b)
        self.st = st
        self.ho = (self.h + 2 - 3) // st + 1
        self.wo = (self.w + 2 - 3) // st + 1
        out.macs.append((f"{self.tag}.dw 3x3 s{st} +SiLU", self.ho * self.wo,
                         self.cm, 9))

    def export_se(self, out, convs):
        """The squeeze-excite pair; returns the reduced width."""
        _, w1, b1, _ = convs.take()
        _, w2, b2, _ = convs.take()
        rd = w1.shape[0]
        assert (w1.shape[1] == self.cm and w2.shape[0] == self.cm
                and w2.shape[1] == rd)
        out.add(f"{self.tag}.se1.w", w1[:, :, 0, 0])
        out.add(f"{self.tag}.se1.b", b1)
        out.add(f"{self.tag}.se2.w", w2[:, :, 0, 0])
        out.add(f"{self.tag}.se2.b", b2)
        return rd

    def export_pwl(self, model, out, convs):
        """The 1x1 projection conv; returns the residual flag (0 or 1).

        The flag is checked against the graph's BLOCK_<s>-<j>_ADD node.
        """
        _, w, b, _ = convs.take()
        assert w.shape[0] == self.co and w.shape[1] == self.cm
        out.add(f"{self.tag}.pwl.w", w[:, :, 0, 0])
        out.add(f"{self.tag}.pwl.b", b)
        skip = int(self.st == 1 and self.c == self.co)
        add_name = f"BLOCK_{self.s}-{self.j}_ADD"
        assert skip == int(any(x.op_type == "Add" and add_name in x.name
                               for x in model.nodes)), self.tag
        res = " +res" if skip else ""
        out.macs.append((f"{self.tag}.pwl {self.cm}->{self.co}" + res,
                         self.ho * self.wo, self.co, self.cm))
        return skip


def export_body(model, out, convs):
    """The four stages of MBConv blocks; asserts the final geometry."""
    h, w, c = 48, 128, 24
    for s, nb, cm, co in STAGES:
        for j in range(1, nb + 1):
            h, w, c = Block(s, j, h, w, c, cm, co).export(model, out, convs)
    assert (h, w, c) == (3, 8, 192)


def export_post(model, out, convs):
    """The post BatchNorm and the 3x3 valid conv (+ReLU)."""
    pm = model.find("Mul", "BNORM_POST_NOQUANT/FusedBatchNormV32")
    pa = model.first("Add", "BNORM_POST_NOQUANT")
    out.add("post.bn_s", model.values[pm.input[1]].reshape(192))
    out.add("post.bn_b", model.values[pa.input[1]].reshape(192))
    _, w, b, at = convs.take()
    assert (list(w.shape) == [1024, 192, 3, 3] and "pads" not in at
            and "auto_pad" not in at)
    out.add("pc.w", w.transpose(0, 2, 3, 1))
    out.add("pc.b", b)
    out.macs.append(("pc 3x3 valid 192->1024 +ReLU", 6, 1024, 9 * 192))


def export_head(model, out, convs):
    """The 1024->6522 head (MatMul + Add); every Conv must be consumed."""
    hm = [x for x in model.nodes
          if x.op_type == "MatMul" and x.output[0].startswith("Identity")][0]
    hw = model.values[hm.input[1]]
    ha = [x for x in model.nodes
          if x.op_type == "Add" and x.output[0] == "output"][0]
    hb = model.values[ha.input[1]]
    assert hw.shape == (1024, 6522)
    out.add("head.w", hw.T)
    out.add("head.b", hb)
    out.macs.append(("head 1024->6522", 1, 6522, 1024))
    convs.check_done()


def read_labels(path):
    """The label file's bytes; asserts 6522 non-empty lines."""
    with open(path, "rb") as f:
        labels = f.read()
    n_lab = len([line for line in labels.decode().splitlines()
                 if line.strip()])
    assert n_lab == 6522, n_lab
    return labels


def file_header(out, labels):
    """The header: magic, counts, license, label hash and the block table."""
    lic = LICENSE.encode()
    lic_pad = lic + b"\0" * ((-len(lic)) % 8)
    hdr = b"BN24ENG1" + struct.pack("<5I", 1, len(out.tensors), 6522,
                                    len(out.blocks), len(lic_pad)) + lic_pad
    hdr += hashlib.sha256(labels).digest()
    hdr += b"".join(struct.pack("<8i", *bk) for bk in out.blocks)
    return hdr


def tensor_directory(tensors, start):
    """(directory entries, (offset, bytes) pairs, end offset).

    Each tensor's data starts 64-byte aligned, the first one at start.
    """
    off = start
    ents = []
    datas = []
    for name, a in tensors:
        dims = list(a.shape) + [1] * (4 - a.ndim)
        ents.append(name.encode().ljust(48, b"\0")
                    + struct.pack("<6I", 0, a.ndim, *dims)
                    + struct.pack("<2Q", off, a.nbytes))
        datas.append((off, a.tobytes()))
        off = (off + a.nbytes + 63) & ~63
    return ents, datas, off


def write_file(dst, out, labels):
    """Writes the .bn24 file; returns the end offset (its size)."""
    hdr = file_header(out, labels)
    dir_size = len(out.tensors) * (48 + 4 * 6 + 16)
    off = (len(hdr) + dir_size + 63) & ~63
    ents, datas, off = tensor_directory(out.tensors, off)
    with open(dst, "wb") as f:
        f.write(hdr + b"".join(ents))
        for o, d in datas:
            f.seek(o)
            f.write(d)
    return off


def write_mac_table(path, macs):
    """Writes the per-layer MAC table and prints it."""
    tot = sum(px * co * k for _, px, co, k in macs)
    with open(path, "w", encoding="utf-8") as f:
        f.write("BirdNET v2.4 body+head MACs (front end excluded: dense DFT "
                "in the graph = 1341 MMAC, FFT ~ 25 MFLOP)\n")
        f.write(f"{'layer':34s} {'pixels':>7} {'N':>6} {'K':>7} {'MMAC':>8} "
                f"{'%':>6} {'out elems':>10}\n")
        for nm, px, co, k in macs:
            mac = px * co * k
            f.write(f"{nm:34s} {px:7d} {co:6d} {k:7.1f} {mac/1e6:8.2f} "
                    f"{100*mac/tot:6.2f} {px*co:10d}\n")
        f.write(f"total {tot/1e6:.1f} MMAC\n")
    with open(path, encoding="utf-8") as f:
        print(f.read())


def main():
    if len(sys.argv) < 4:
        print("usage: python -I export_weights.py <BirdNET_v2.4_fp32.onnx> "
              "<labels.txt> <out.bn24> [mac_table.txt]", file=sys.stderr)
        sys.exit(2)
    src, labels_path, dst = sys.argv[1:4]
    mac_path = sys.argv[4] if len(sys.argv) > 4 else None
    model = Model(src)
    check_front_end(model)
    out = Export()
    export_front_end(model, out)
    convs = ConvStream(model)
    export_stem(out, convs)
    export_body(model, out, convs)
    export_post(model, out, convs)
    export_head(model, out, convs)
    labels = read_labels(labels_path)
    off = write_file(dst, out, labels)
    print(f"wrote {dst}: {len(out.tensors)} tensors, {len(out.blocks)} "
          f"blocks, {off/1e6:.1f} MB; front end {out.fe_meta}")
    if mac_path:
        write_mac_table(mac_path, out.macs)


if __name__ == "__main__":
    main()
