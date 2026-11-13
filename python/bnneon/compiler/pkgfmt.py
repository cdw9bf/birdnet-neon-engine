"""The .bnpk model package format (writer + reader).

Little endian, every table 8-byte aligned, every tensor blob 64-byte aligned
and followed by >= 64 zero bytes (the kernels' K loops read ahead).

  0   char magic[8] "BNNPKG01"; u32 bom 0x01020304; u16 format_version;
      u16 min_reader_version
  16  u32 n_meta, n_tensors, n_paths, reserved
  32  u64 meta_off, tensor_off, path_off, file_size
  metadata records: u16 key_len, u16 type (0 str, 1 i64, 2 f64, 3 bytes),
      u32 val_len, key, value, pad to 8
  tensor table: n x {char name[48]; u32 dtype (0 f32, 1 f16, 2 i8, 3 i32);
      u32 ndim; u32 dims[4]; u64 offset, nbytes, reserved}  (96 bytes)
  path table: n x {char name[24]; u32 hwcap (1 fp16, 2 dotprod), n_buf,
      n_ops, ops_bytes, in_buf, out_buf, emb_buf (0xffffffff = none),
      reserved; u64 buf_off, ops_off}  (72 bytes)
  per path: buffers n_buf x {u32 dtype, H, W, C, ld, halo, slot, 0;
      u64 elems} (40 bytes; buffers with the same slot share memory: the
      compiler gives them disjoint lifetimes); ops: records {u16 opcode
      (base | variant << 8: 0 fp32 kernel, 1 fp16, 2 fp16 + dotprod),
      u16 rec_bytes, u16 n_buf, u16 n_ten, u16 n_int, u16 0, i32 bufs[],
      i32 tensors[], i32 ints[]} padded to 8
"""
import struct

import numpy as np

FORMAT_VERSION = 1
MIN_READER = 1
DT = {np.dtype(np.float32): 0, np.dtype(np.float16): 1, np.dtype(np.int8): 2,
      np.dtype(np.int32): 3}


def pad8(b: bytes) -> bytes:
    """Returns bytes b zero-padded to a multiple of 8 bytes."""
    return b + b"\0" * ((8 - len(b) % 8) % 8)


class Package:
    """A .bnpk package being built: metadata, tensors and CPU paths.

    Attributes:
        meta: (key, value) metadata records in insertion order.
        tensors: (name, array, nbytes) per tensor, in table order.
        tindex: tensor name -> its index in tensors.
        paths: (name, hwcap, bufs, ops, in_buf, out_buf, emb_buf) per path.
    """

    def __init__(self):
        self.meta = []
        self.tensors = []  # (name, array, nbytes)
        self.tindex = {}
        self.paths = []

    def set(self, key: str, val) -> None:
        """Adds a metadata record (str, int / bool, float or bytes value)."""
        self.meta.append((key, val))

    def tensor(self, name: str, a: np.ndarray, min_bytes: int = 0) -> int:
        """Adds a tensor, deduplicated by name.

        Args:
            name: table name (< 48 bytes).
            a: the array (fp32, fp16, int8 or int32; at most 4-D).
            min_bytes: allocate at least this many bytes (zero tail).

        Returns:
            The tensor's index in the table (the existing one for a name
            added before, which must hold the same data).
        """
        if name in self.tindex:
            old = self.tensors[self.tindex[name]][1]
            assert old.dtype == a.dtype and np.array_equal(old, a), name
            return self.tindex[name]
        a = np.ascontiguousarray(a)
        assert a.dtype in DT, (name, a.dtype)
        assert len(name) < 48, name
        self.tindex[name] = len(self.tensors)
        self.tensors.append((name, a, max(min_bytes, a.nbytes)))
        return self.tindex[name]

    def add_path(self, name: str, hwcap: int, bufs: list, ops: list,
                 in_buf: int, out_buf: int, emb_buf: int | None) -> None:
        """Adds a CPU path: its buffers (with slots) and op records."""
        self.paths.append((name, hwcap, bufs, ops, in_buf, out_buf, emb_buf))

    def write(self, dst: str) -> int:
        """Writes the package to file dst; returns its size in bytes."""
        meta = b"".join(_meta_record(k, v) for k, v in self.meta)
        hdr_len = 64
        meta_off = hdr_len
        tensor_off = meta_off + len(meta)
        path_off = tensor_off + 96 * len(self.tensors)
        off = path_off + 72 * len(self.paths)
        pdesc, pblobs, off = self._path_tables(off)
        off = (off + 63) & ~63
        tdesc, tblobs, off = self._tensor_tables(off)
        size = off + 64
        with open(dst, "wb") as f:
            f.write(b"BNNPKG01" +
                    struct.pack("<IHH4I4Q", 0x01020304, FORMAT_VERSION,
                                MIN_READER, len(self.meta), len(self.tensors),
                                len(self.paths), 0, meta_off, tensor_off,
                                path_off, size))
            f.write(meta)
            f.write(b"".join(tdesc))
            f.write(b"".join(pdesc))
            for b in pblobs:
                f.write(b)
            for o, a, nbytes in tblobs:
                f.write(b"\0" * (o - f.tell()))
                f.write(a)  # C-contiguous (tensor()): its buffer, no copy
                f.write(b"\0" * (nbytes - a.nbytes))
            f.write(b"\0" * (size - f.tell()))
        return size

    def _path_tables(self, off):
        """The path table and the paths' buffer / op blobs placed from off.

        Returns:
            (path descriptors, blobs in file order, the offset after them).
        """
        pblobs = []
        pdesc = []
        for name, hw, bufs, ops, ib, ob, eb in self.paths:
            bb = b"".join(struct.pack("<8IQ", *b[:6], b[7], 0, b[6])
                          for b in bufs)
            ob_ = b"".join(_op_record(*op) for op in ops)
            boff = off
            off += len(bb)
            ooff = off
            off += len(ob_)
            pblobs += [bb, ob_]
            eb_ = 0xffffffff if eb is None else eb
            pdesc.append(name.encode().ljust(24, b"\0") +
                         struct.pack("<8I2Q", hw, len(bufs), len(ops),
                                     len(ob_), ib, ob, eb_, 0, boff, ooff))
        return pdesc, pblobs, off

    def _tensor_tables(self, off):
        """The tensor table with the blobs placed from off (64-aligned).

        Returns:
            (tensor descriptors, (offset, array, nbytes) per blob, the offset
            after the last blob).
        """
        tdesc, tblobs = [], []
        for name, a, nbytes in self.tensors:
            dims = list(a.shape) + [0] * (4 - a.ndim)
            assert a.ndim <= 4, name
            tdesc.append(name.encode().ljust(48, b"\0") +
                         struct.pack("<II4IQQQ", DT[a.dtype], a.ndim, *dims,
                                     off, nbytes, 0))
            tblobs.append((off, a, nbytes))
            # >= 64 zero bytes after every blob
            off = (off + nbytes + 64 + 63) & ~63
        return tdesc, tblobs, off


def _meta_record(k, v):
    """One metadata record: str, int / bool, float or bytes value v."""
    kb = k.encode()
    if isinstance(v, (bytes, bytearray)):
        ty, vb = 3, bytes(v)
    elif isinstance(v, bool) or isinstance(v, (int, np.integer)):
        ty, vb = 1, struct.pack("<q", int(v))
    elif isinstance(v, float):
        ty, vb = 2, struct.pack("<d", v)
    else:
        ty, vb = 0, str(v).encode()
    return pad8(struct.pack("<HHI", len(kb), ty, len(vb)) + kb + vb)


def _op_record(code, bl, tl, il):
    """One op record: header, buffer / tensor / int lists, padded to 8."""
    body = struct.pack(f"<{len(bl) + len(tl) + len(il)}i",
                       *(list(bl) + list(tl) + list(il)))
    rec = struct.pack("<6H", code, 0, len(bl), len(tl), len(il), 0) + body
    rec = pad8(rec)
    return rec[:2] + struct.pack("<H", len(rec)) + rec[4:]
