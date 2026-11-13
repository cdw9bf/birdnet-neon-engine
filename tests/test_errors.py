"""Checks that every failure path says what failed, what was expected, what
was found and what to do next.

  python -I tests/test_errors.py [--cli build/check/cli]
      [--model MODEL.onnx --labels LABELS.txt [--logits TENSOR]]
      [--run-path neon-fp16|neon-fp32] [--proto-package PKG.bnpk] [--keep]
      [--workdir DIR]

(-I with bnneon installed in the interpreter; for a checkout that is not
installed: PYTHONPATH=<checkout>/python python -P -s tests/...)

The scratch files go into a new bnc_test_* directory in --workdir DIR (default:
the system's temporary directory, e.g. /tmp), which is removed at the end, also
after a failure, unless --keep. Each file is deleted as soon as its checks are
done; with --model, the directory peaks at about 300 MB. Its path may be of
any length: the runtime's messages are checked with a 100-byte path.

Part A (always): deliberately broken ONNX graphs through `bnc.py report` /
`check`, and the exit codes.
Part B (with --model, a BirdNET v2.4-family graph the compiler supports, and
--labels; --logits names the logits tensor, default: the model's last graph
output): the model is compiled into a scratch package which is then mutated
byte by byte (bad magic, truncation, reader version, unknown opcode, wrong
buffer dtype, short tensor, bad int parameter, hwcap, metadata, label file)
and fed to the runtime CLI; each message must name the op, the kernel, the
source ONNX node, the argument and the values. The CLI is built natively
unless --cli is given. The package is a neon-fp16 one when this CPU can run
that path, else a neon-fp32 one (--run-path picks it); the hwcap and
missing-path cases always mutate the neon-fp16 package, which the loader
refuses on any CPU.
Part C (with --model): the general head and the canonicalization pass: a
hidden layer (Dense -> ReLU -> Dense) spliced into the model must compile on
both paths and agree with ONNX Runtime (the neon-fp16 run is skipped on a CPU
without fp16 arithmetic); tf2onnx-style artifacts
(Unsqueeze/Squeeze around the DFT MatMul, a float->float Cast,
ReverseSequence, a Pad before a depthwise conv, GlobalAveragePool) must be
canonicalized away; `compile` on an unsupported graph exits 2; an unsupported
node after --logits is ignored.
Part D (with --model): the round-3 tools: the relabel script, the label-form
warning, compile refusals and defaults, the CLI's --labels, the generic gate,
cmp_suite without the gated reference data.
Part E (with --proto-package PKG.bnpk, repeatable: packages with a prototype
head, e.g. the Perch v2 neon-int8, neon-fp32 and neon-fp16 packages): corrupt
head shapes (negative, zero or huge K / C / N, Kp != K, K too large for int32
sums in the int8 heads), a stem output stride below Np and forged n4 flags are
refused by the loader with the op and the failed check.
Exit 0 when every check passes.
"""
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from bnneon import testutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BNC = os.path.join(ROOT, "compiler", "bnc.py")
RELABEL = os.path.join(ROOT, "scripts", "relabel_for_birdnetgo.py")
GATE = os.path.join(ROOT, "scripts", "gate_generic.py")
PY = sys.executable
# -P keeps the current directory off sys.path; the bnneon package comes from
# the install or PYTHONPATH
PYFLAGS = ("-P", "-s")
OPSET = [helper.make_opsetid("", 17)]
CHECKS = testutil.Checks()
check = CHECKS.check
opt = testutil.opt


def run(args, **kw):
    """(exit code, stdout + stderr) of a command."""
    r = subprocess.run(args, capture_output=True, text=True, check=False,
                       **kw)
    return r.returncode, (r.stdout + r.stderr)


def run_in(d, cli, *args):
    """(exit code, output) of the CLI run in directory d (files of d can then
    be named by sized_name)."""
    return run([os.path.abspath(cli), *args], cwd=d)


def bnc(*args):
    """Runs the compiler shim with the test interpreter."""
    return run([PY, *PYFLAGS, BNC, *args])


def script(path, *args):
    """Runs a script of scripts/ with the test interpreter."""
    return run([PY, *PYFLAGS, path, *args])


def tmpdir():
    """A fresh scratch directory in --workdir, else in the temp directory."""
    parent = opt("--workdir")
    if parent:
        os.makedirs(parent, exist_ok=True)
    return tempfile.mkdtemp(prefix="bnc_test_", dir=parent)


def drop(*paths):
    """Deletes scratch files and directories once their checks are done
    (a missing one is fine); with --keep, keeps them."""
    if "--keep" in sys.argv:
        return
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.exists(p):
            os.remove(p)


# the host's message buffer holds this many bytes (the terminating NUL aside)
MESSAGE_MAX = 511
# the length of the paths the runtime's messages are tested with
PATH_ROOM = 100


def sized_name(base, size=PATH_ROOM):
    """The file base of the current directory as a path of about size bytes
    ("./" repeated before it): a message that carries it is as long as with
    a typical absolute path, whatever --workdir is."""
    return "./" * max(0, (size - len(base)) // 2) + base


BEST_PATH = "best path for this CPU: "
# what `--info` names when this CPU can run none of the package's paths
NO_PATH = "(none)"


def parse_best_path(info):
    """The path `cli PKG --info` names for this CPU: a path name, NO_PATH
    when this CPU runs none of the package's paths, or None when the output
    has no best-path line."""
    for line in info.splitlines():
        if line.startswith(BEST_PATH):
            return line[len(BEST_PATH):].strip()
    return None


def best_path(cli, package):
    """The path the library would run package on with this CPU, or NO_PATH.

    A failing `--info` or one without a best-path line is a recorded
    failure (not "this CPU cannot run it"); returns None then.
    """
    rc, out = run([cli, package, "--info"])
    name = parse_best_path(out) if rc == 0 else None
    if name is None:
        check(f"--info {os.path.basename(package)} names the best path for "
              f"this CPU", False, f"exit {rc}: {out[-400:]}")
    return name


def read_text(path):
    """A text file's content."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_bytes(path):
    """A file's bytes."""
    with open(path, "rb") as f:
        return f.read()


def write_text(path, text):
    """Writes a text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_json(path):
    """A JSON file's content."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Env:
    """What parts B, C and D share.

    Attributes:
        d: the scratch directory.
        cli: the runtime CLI binary.
        model: the ONNX model (a BirdNET v2.4-family graph).
        labels: its label file.
    """

    def __init__(self, d, cli, model, labels):
        self.d = d
        self.cli = cli
        self.model = model
        self.labels = labels

    def path(self, name):
        """A file in the scratch directory."""
        return os.path.join(self.d, name)


# ------------------------------------------------------ part A: broken graphs
def save(m, path):
    """Saves a model readable by every ONNX Runtime since 1.16."""
    m.ir_version = 9
    onnx.save(m, path)


def vi(name, shape):
    """A float tensor value info."""
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def model_no_frontend(path, nsamp=144000, nc=5):
    """input -> Tanh -> MatMul [nsamp][nc] -> Sigmoid.

    No spectrogram front end, so nothing can be placed.
    """
    w = numpy_helper.from_array(np.zeros((nsamp, nc), np.float32), "W")
    nodes = [helper.make_node("Tanh", ["pcm"], ["t"], name="act/Tanh"),
             helper.make_node("MatMul", ["t", "W"], ["logits"],
                              name="head/MatMul"),
             helper.make_node("Sigmoid", ["logits"], ["conf"],
                              name="head/Sigmoid")]
    g = helper.make_graph(nodes, "g", [vi("pcm", [1, nsamp])],
                          [vi("conf", [1, nc])], [w])
    save(helper.make_model(g, opset_imports=OPSET), path)


def model_two_inputs(path):
    """a + b -> y: two graph inputs."""
    nodes = [helper.make_node("Add", ["a", "b"], ["y"], name="Add")]
    g = helper.make_graph(nodes, "g", [vi("a", [1, 100]), vi("b", [1, 100])],
                          [vi("y", [1, 100])])
    save(helper.make_model(g, opset_imports=OPSET), path)


def model_dynamic_input(path):
    """Relu(a) -> y with a symbolic input shape."""
    nodes = [helper.make_node("Relu", ["a"], ["y"], name="Relu")]
    g = helper.make_graph(nodes, "g", [vi("a", ["batch", "n"])],
                          [vi("y", ["batch", "n"])])
    save(helper.make_model(g, opset_imports=OPSET), path)


def part_a_no_frontend(d):
    """A1, A2: the graph without a front end through report and check."""
    p = os.path.join(d, "nofe.onnx")
    model_no_frontend(p)
    rc, out = bnc("report", p, "--json", os.path.join(d, "nofe.json"))
    check("A1 no front end: exit 2", rc == 2, f"exit {rc}: {out}")
    check("A1 report names the stop and the expected front ends",
          "no supported front end" in out and "ReduceMin" in out
          and "Reshape/Conv" in out, out)
    j = load_json(os.path.join(d, "nofe.json"))
    u = j["unsupported"]
    check("A1 json lists every unsupported node",
          j["status"] == "unsupported" and len(u) >= 2, json.dumps(u)[:400])
    tanh = [x for x in u if x["op_type"] == "Tanh"]
    mm = [x for x in u if x["op_type"] == "MatMul"]
    check("A1 json entries carry op type, name, shapes and the nearest pattern",
          tanh and tanh[0]["name"] == "act/Tanh"
          and tanh[0]["inputs"][0]["shape"] == [1, 144000]
          and tanh[0]["nearest"] and tanh[0]["next_step"],
          json.dumps(tanh)[:400])
    check("A1 downstream nodes are marked as such, not as new problems",
          mm and mm[0]["downstream_of"] == "t"
          and "fix the unsupported node" in mm[0]["next_step"],
          json.dumps(mm)[:400])
    check("A1 primary count excludes downstream nodes", len(j["primary"]) == 1,
          str(len(j["primary"])))
    rc, out = bnc("check", p, "--json", os.path.join(d, "check.json"), "--cli",
                  "/nonexistent")
    check("A2 check on the broken graph: exit 2 with a next step",
          rc == 2 and "FAIL:" in out and "next step" in out,
          f"exit {rc}: {out[-500:]}")
    js = load_json(os.path.join(d, "check.json"))
    check("A2 check --json holds the status, exit code and the next step",
          js["status"] == "FAIL" and js["exit"] == 2
          and "unsupported op" in js["next_step"], json.dumps(js)[:300])
    rc, out = bnc("report", p, "--json", "-")
    check("A2 --json - prints JSON on stdout (text on stderr)",
          (rc == 2 and out.lstrip().startswith("{"))
          or '"status": "unsupported"' in out, out[:200])


def part_a_bad_inputs(d):
    """A3-A7: two inputs, a dynamic shape, a missing file, a bad command,
    a file that is not ONNX."""
    p = os.path.join(d, "two.onnx")
    model_two_inputs(p)
    rc, out = bnc("report", p)
    check("A3 two inputs: exit 4 and a hint",
          rc == 4 and "exactly one graph input" in out and "hint" in out,
          f"exit {rc}: {out}")
    p = os.path.join(d, "dyn.onnx")
    model_dynamic_input(p)
    rc, out = bnc("report", p)
    check("A4 dynamic input shape: exit 4 and a hint",
          rc == 4 and "fixed n_samples" in out, f"exit {rc}: {out}")
    rc, out = bnc("report", os.path.join(d, "missing.onnx"))
    check("A5 missing file: exit 3 or 4 with the path",
          rc in (3, 4) and "missing.onnx" in out, f"exit {rc}: {out}")
    rc, out = bnc("frobnicate", p)
    check("A6 unknown command: usage, exit 3",
          rc == 3 and "exit codes" in out, f"exit {rc}")
    with open(os.path.join(d, "notonnx.onnx"), "wb") as f:
        f.write(b"not an onnx file")
    rc, out = bnc("report", os.path.join(d, "notonnx.onnx"))
    check("A7 garbage file: exit 4, mentions tf2onnx for .tflite",
          rc == 4 and "tf2onnx" in out, f"exit {rc}: {out}")


def part_a_options(d):
    """A8: option parsing on the graph without a front end."""
    p = os.path.join(d, "nofe.onnx")
    rc, out = bnc("report", p, "--json")
    check("A8 --json as the last argument prints JSON, no crash",
          rc == 2 and '"status": "unsupported"' in out,
          f"exit {rc}: {out[-200:]}")
    rc, out = bnc("check", p, "--post", "Softmax", "--cli", "/nonexistent")
    check("A8 --post Softmax: exit 3 naming the choices",
          rc == 3 and "sigmoid or softmax" in out, f"exit {rc}: {out[-200:]}")
    rc, out = bnc("check", p, "--windows", "two", "--cli", "/nonexistent")
    check("A8 --windows two: exit 3", rc == 3 and "integer" in out,
          f"exit {rc}: {out[-200:]}")
    rc, out = bnc("check", p, "--classes", "x.txt", "--cli", "/nonexistent",
                  "--json", os.path.join(d, "cl.json"))
    check("A8 --classes refused by check with the JSON still written",
          rc == 3 and os.path.exists(os.path.join(d, "cl.json")),
          f"exit {rc}: {out[-200:]}")


def part_a(d):
    """Part A: broken graphs through report and check, and the exit codes."""
    part_a_no_frontend(d)
    part_a_bad_inputs(d)
    part_a_options(d)


# ---------------------------------------------------- part B: broken packages
BUF_FIELDS = ("dtype", "H", "W", "C", "ld", "halo", "slot")


def _read_buf(b, at):
    """One buffer record of a path's buffer table."""
    d = dict(at=at)
    d.update(zip(BUF_FIELDS, struct.unpack_from("<7I", b, at)))
    d["elems"] = struct.unpack_from("<Q", b, at + 32)[0]
    return d


def _read_op(b, q):
    """(one op record at q, its record length)."""
    code, rec, nb, nt, ni, _ = struct.unpack_from("<6H", b, q)
    op = dict(at=q, code=code, nb=nb, nt=nt, ni=ni, b_at=q + 12,
              t_at=q + 12 + 4 * nb, p_at=q + 12 + 4 * (nb + nt),
              b=list(struct.unpack_from(f"<{nb}i", b, q + 12)),
              t=list(struct.unpack_from(f"<{nt}i", b, q + 12 + 4 * nb)),
              p=list(struct.unpack_from(f"<{ni}i", b,
                                        q + 12 + 4 * (nb + nt))))
    return op, rec


class Pkg:
    """A minimal reader of the .bnpk tables (compiler/pkgfmt.py is the
    writer); mutations edit a bytearray copy.

    Attributes:
        b: the file's bytes (a bytearray).
        tensors: the tensor directory (name, dtype, off, nbytes, at).
        paths: the paths (name, at, hwcap, in_buf, out_buf, bufs, ops).
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            self.b = bytearray(f.read())
        b = self.b
        self.n_meta, self.n_tensors, self.n_paths = struct.unpack_from(
            "<3I", b, 16)
        self.meta_off, self.tensor_off, self.path_off, self.size = \
            struct.unpack_from("<4Q", b, 32)
        self.tensors = [self._read_tensor(i) for i in range(self.n_tensors)]
        self.paths = [self._read_path(i) for i in range(self.n_paths)]

    def _read_tensor(self, i):
        """Tensor directory entry i."""
        o = self.tensor_off + 96 * i
        name = self.b[o:o + 48].split(b"\0")[0].decode()
        dtype, _, _, _, _, _, off, nbytes, _ = struct.unpack_from(
            "<6I3Q", self.b, o + 48)
        return dict(name=name, dtype=dtype, off=off, nbytes=nbytes, at=o + 48)

    def _read_path(self, i):
        """Path i with its buffer table and op list."""
        b = self.b
        o = self.path_off + 72 * i
        name = b[o:o + 24].split(b"\0")[0].decode()
        hwcap, n_buf, n_ops, _, in_buf, out_buf, _, _ = struct.unpack_from(
            "<8I", b, o + 24)
        buf_off, ops_off = struct.unpack_from("<2Q", b, o + 56)
        bufs = [_read_buf(b, buf_off + 40 * k) for k in range(n_buf)]
        ops = []
        q = ops_off
        for _ in range(n_ops):
            op, rec = _read_op(b, q)
            ops.append(op)
            q += rec
        return dict(name=name, at=o, hwcap=hwcap, in_buf=in_buf,
                    out_buf=out_buf, bufs=bufs, ops=ops)

    def meta(self, key):
        """(offset of the value, type, length) of a metadata record."""
        q = self.meta_off
        for _ in range(self.n_meta):
            kl, ty, vl = struct.unpack_from("<HHI", self.b, q)
            if self.b[q + 8:q + 8 + kl] == key.encode():
                return q + 8 + kl, ty, vl
            q += (8 + kl + vl + 7) & ~7
        return None

    def write(self, path):
        """Writes the (mutated) bytes."""
        with open(path, "wb") as f:
            f.write(self.b)


def logits_name(env):
    """--logits, else the logits_default the compiler's report names."""
    if opt("--logits"):
        return opt("--logits")
    f = env.path("logits.json")
    bnc("report", env.model, "--json", f)
    name = load_json(f)["logits_default"]
    drop(f)
    return name


def write_window(pkg, win):
    """One synthetic window: n_samples from the package metadata."""
    o, _, _ = pkg.meta("n_samples")
    nsamp = struct.unpack_from("<q", pkg.b, o)[0]
    x = np.random.default_rng(1).standard_normal(nsamp) * 0.05
    x.astype(np.float32).tofile(win)


class Mutations:
    """Runs mutated copies of the test package through the CLI."""

    def __init__(self, env, src, win, out_f):
        self.env = env
        self.src = src
        self.win = win
        self.out_f = out_f

    def load(self, name, mutate, mode=0):
        """Writes a mutated copy and runs it; returns (exit code, output).

        Every runtime message must fit the host's 512-byte buffer with the
        hint intact, for a PATH_ROOM-byte package path (the CLI runs in the
        scratch directory on a sized_name, so the length of --workdir does
        not count).
        """
        p = Pkg(self.src)
        mutate(p)
        base = f"{name}.bnpk"
        f = self.env.path(base)
        p.write(f)
        rc, out = run_in(self.env.d, self.env.cli, sized_name(base),
                         str(mode), os.path.abspath(self.win),
                         os.path.abspath(self.out_f), "1")
        drop(f)
        out = out.strip()
        if rc:
            line = out.splitlines()[0]
            check(f"   {name}: message fits (len {len(line)}, a "
                  f"{PATH_ROOM}-byte path) and keeps its hint",
                  len(line) <= MESSAGE_MAX and "hint:" in line
                  and not line.endswith("'"), line[-120:])
        return rc, out


def set_magic(p):
    """Mutation: a wrong magic."""
    p.b[0:8] = b"BNNPKG99"


def set_bnv3_magic(p):
    """Mutation: the magic of a legacy BirdNET v3.0 file (engine retired)."""
    p.b[0:8] = b"BNV3ENG1"


def truncate(p):
    """Mutation: the last 1000 bytes cut off."""
    del p.b[len(p.b) - 1000:]


def check_file_errors(mut, pw, pw_i):
    """B1-B4: bad magic, a .bnv3, truncation, reader version, unknown opcode."""
    rc, out = mut.load("magic", set_magic)
    check("B1 bad magic names the file, the magic found and the expected one",
          rc and "not a bnneon weights file" in out and "BNNPKG99" in out
          and "BNNPKG01" in out, out)
    rc, out = mut.load("bnv3", set_bnv3_magic)
    check("B1b a legacy .bnv3 is refused with the packages to use instead",
          rc and "legacy BirdNET v3.0 .bnv3" in out and "hint" in out
          and "neon-fp16.bnpk" in out and "neon-int8" in out, out)
    rc, out = mut.load("trunc", truncate)
    check("B2 truncated file says truncated, with both sizes",
          rc and "truncated" in out and "header says" in out, out)
    rc, out = mut.load("reader", lambda p: struct.pack_into("<H", p.b, 14, 99))
    check("B3 newer reader version needed: both versions and a hint",
          rc and "reader version 99" in out and "hint" in out, out)
    # an unknown op in the pw's kernel family (fp16: 0x17f, fp32: 0x7f)
    code = (pw["code"] & 0xff00) | 0x7f
    rc, out = mut.load("opcode",
                       lambda p: struct.pack_into("<H", p.b, pw["at"], code))
    check("B4 unknown opcode: op index, opcode, source node, compiler and hint",
          rc and f"op {pw_i}" in out and f"{code:#x}" in out
          and "ONNX node '" in out and "compiler 'bnc" in out
          and "update libbnneon" in out, out)


DTYPES = {"f32": 0, "f16": 1}
# bytes per element of each buffer dtype code (f32, f16, i8, i32)
DTYPE_BYTES = {0: 4, 1: 2, 2: 1, 3: 4}
# the activations' dtype of a path, and the other one (B5 writes that)
PATH_DTYPES = {"neon-fp16": ("f16", "f32"), "neon-fp32": ("f32", "f16")}


def check_argument_errors(mut, pkg, path, pw, pw_i):
    """B5-B7, B9: buffer dtype, short tensor, int parameter, metadata."""
    # B5 wrong buffer dtype on the pw output (buffer arg 1)
    ob = path["bufs"][pw["b"][1]]
    want, wrong = PATH_DTYPES[path["name"]]
    rc, out = mut.load("dtype", lambda p: struct.pack_into(
        "<I", p.b, ob["at"], DTYPES[wrong]))
    check("B5 wrong buffer dtype: op, kernel, source node, argument, expected "
          "vs found", rc and f"op {pw_i} pw_{want} (ONNX node '" in out
          and "buffer argument" in out and f"expected dtype {want}" in out
          and f"found dtype {wrong}" in out and "hint" in out, out)
    # B6 short tensor (the pw weights)
    tw = pkg.tensors[pw["t"][0]]
    rc, out = mut.load("tensor", lambda p: struct.pack_into(
        "<Q", p.b, tw["at"] + 32, 64))
    check("B6 short tensor: tensor name, expected and found bytes",
          rc and "tensor argument 0" in out and tw["name"] in out
          and "expected >=" in out and "found 64 bytes" in out, out)
    # B7 bad int parameter (M = -5 on the pw)
    rc, out = mut.load("param", lambda p: struct.pack_into(
        "<i", p.b, pw["p_at"] + 4 * 4, -5))
    check("B7 bad int parameter: the condition and the parameter list",
          rc and "parameter check failed: M > 0" in out
          and "int params: [" in out and "-5" in out, out)
    # B9 metadata n_classes
    o9, _, _ = pkg.meta("n_classes")
    rc, out = mut.load("meta",
                       lambda p: struct.pack_into("<q", p.b, o9, 999999))
    check("B9 metadata vs buffers: says which field and the buffer's size",
          rc and "n_classes = 999999" in out and "output buffer is f32" in out,
          out)


def check_hwcap(mut16, path16):
    """B8: fp16 kernels in a path that does not declare fp16 (the neon-fp16
    package with its requirement cleared: refused on any CPU)."""
    rc, out = mut16.load("hwcap", lambda p: struct.pack_into(
        "<I", p.b, path16["at"] + 24, 0))
    check("B8 hwcap mismatch: names the kernel variant and the path",
          rc and "fp16 kernel" in out and "does not declare" in out, out)


def check_cli_errors(env, src, src16, win, out_f):
    """B10-B12: a bad mode, a missing path (in the neon-fp16 package src16),
    a label file mismatch.

    Returns:
        The path of the other (three-line) label file.
    """
    rc, out = run([env.cli, src, "7", win, out_f, "1"])
    check("B10 bad mode lists the valid modes",
          rc and "mode 7 is not available" in out and "mode 0" in out, out)
    rc, out = run([env.cli, src16, "2", win, out_f, "1"])
    check("B11 missing path: lists the paths and their CPU needs, names the "
          "other package", rc and "neon-fp32 is not in this package" in out
          and "needs fp16 arithmetic" in out and "-neon-fp32.bnpk" in out, out)
    other = env.path("other_labels.txt")
    write_text(other, "a\nb\nc\n")
    # sized names: the message carries both paths, and its fix comes last
    rc, out = run_in(env.d, env.cli, sized_name(os.path.basename(src)),
                     "--verify-labels", sized_name(os.path.basename(other)))
    check("B12 label mismatch: both hashes, line counts and the fix",
          rc and "sha256" in out and "3 lines" in out and "classes" in out
          and "bnn_probe_meta" in out, out)
    rc, out = run([env.cli, src, "--verify-labels", env.labels])
    check("B12 the right label file matches", rc == 0 and "match" in out, out)
    return other


def check_out_of_memory(env, src, win, out_f, big):
    """B13: a 900 MB buffer (under the 1 GB guard) under a 400 MB address
    space limit is reported as out of memory (Linux ulimit -v only)."""
    if not sys.platform.startswith("linux"):
        print("skip B13 out-of-memory (needs ulimit -v: Linux); the 1 GB "
              "guard is tested instead")
        return
    p = Pkg(src)
    # the record holds elements: 900 MB of the buffer's dtype
    struct.pack_into("<Q", p.b, big["at"] + 32,
                     (900 << 20) // DTYPE_BYTES[big["dtype"]])
    f = env.path("oom.bnpk")
    p.write(f)
    rc, out = run(["sh", "-c", f"ulimit -v 400000; {shlex.quote(env.cli)} "
                   f"{shlex.quote(f)} 0 {shlex.quote(win)} "
                   f"{shlex.quote(out_f)} 1"])
    drop(f)
    check("B13 out of memory is reported as such, not as bad parameters",
          rc and "out of memory" in out and "parameter" not in out, out)


def check_memory_guard(mut, env, src, win, out_f, path, pw):
    """B13, B13b: out of memory vs bad parameters, the 1 GB guard."""
    big = path["bufs"][pw["b"][1]]
    check_out_of_memory(env, src, win, out_f, big)
    rc, out = mut.load("huge", lambda p: struct.pack_into(
        "<Q", p.b, big["at"] + 32, 1 << 30))
    check("B13b the 1 GB activation guard names the path",
          rc and "more than 1 GB" in out, out)


def check_compile_refusals(env, lt, other):
    """B14, B14b, B15: label refusals at compile time, a missing argument."""
    rc, out = bnc("compile", env.model, "--labels", other, "--model-id", "t",
                  "--family", "0", "--post", "sigmoid", "--force", "--paths",
                  "neon-fp16", "--logits", lt, "--force", "-o",
                  env.path("x.bnpk"))
    check("B14 compile with the wrong label file: exit 7, counts both",
          rc == 7 and "3 non-empty lines" in out and "classes" in out,
          f"exit {rc}: {out[-300:]}")
    cr_only = env.path("cr_labels.txt")
    write_text(cr_only, "a\rb\rc\r")
    rc, out = bnc("compile", env.model, "--labels", cr_only, "--model-id", "t",
                  "--family", "0", "--post", "sigmoid", "--force", "--paths",
                  "neon-fp16", "--logits", lt, "-o", env.path("x.bnpk"))
    check("B14b compile with CR-only line ends: exit 7, names the rule",
          rc == 7 and "line break other than" in out and "hint:" in out,
          f"exit {rc}: {out[-300:]}")
    rc, out = bnc("compile", env.model, "--labels", env.labels, "--paths",
                  "neon-fp16", "--logits", lt)
    check("B15 missing -o / --model-id: exit 3 naming the argument",
          rc == 3 and ("-o is required" in out or "--model-id" in out),
          f"exit {rc}: {out[-300:]}")


def dilate_conv(m, add_missing=True):
    """Sets dilations [2, 2] on a model's fourth Conv; returns the node.

    Args:
        m: the model (edited in place).
        add_missing: add the attribute when the conv has none (B16 does; C4
            only changes an existing one).
    """
    conv = [n for n in m.graph.node if n.op_type == "Conv"][3]
    for a in conv.attribute:
        if a.name == "dilations":
            a.ints[:] = [2, 2]
            break
    else:
        if add_missing:
            conv.attribute.append(helper.make_attribute("dilations", [2, 2]))
    return conv


def check_dilated_report(env):
    """B16: an unsupported node injected into a real graph (dilations on a
    conv): the report shows attributes, shapes and the downstream nodes."""
    m = onnx.load(env.model)
    conv = dilate_conv(m)
    bad = env.path("dilated.onnx")
    onnx.save(m, bad)
    rc, out = bnc("report", bad, "--json", env.path("dilated.json"))
    drop(bad)
    j = load_json(env.path("dilated.json"))
    prim = j["primary"]
    check("B16 injected dilated conv: exit 2, one primary problem with "
          "attributes, shapes and the Conv patterns",
          rc == 2 and len(prim) == 1 and prim[0]["name"] == conv.name
          and prim[0]["attrs"].get("dilations") == [2, 2]
          and prim[0]["inputs"][0]["shape"]
          and "depthwise" in prim[0]["nearest"]
          and "dilated conv" in prim[0]["reason"],
          json.dumps(prim)[:500] + out[-300:])
    check("B16 the downstream nodes are listed but marked",
          all(x["downstream_of"] for x in j["unsupported"]
              if x["name"] != conv.name) and len(j["unsupported"]) > 1,
          str(len(j["unsupported"])))


def model_with_tail(env, lt, op, name, out_name, filename, only=False,
                    **attrs):
    """The model with op(logits) -> out_name added as a graph output.

    Args:
        env: the test environment (its model is the base).
        lt: the logits tensor.
        op: the ONNX op type to append.
        name: the new node's name.
        out_name: its output tensor, added to the graph outputs.
        filename: where to save it (in the scratch directory).
        only: drop the existing graph outputs first.
        **attrs: the node's attributes.

    Returns:
        The saved model's path.
    """
    m = onnx.load(env.model)
    m.graph.node.append(helper.make_node(op, [lt], [out_name], name=name,
                                         **attrs))
    if only:
        del m.graph.output[:]
    m.graph.output.append(vi(out_name, None))
    p = env.path(filename)
    onnx.save(m, p)
    return p


def check_final_softmax(env, lt):
    """B17: a Softmax appended to the logits: check picks --post softmax."""
    sm = model_with_tail(env, lt, "Softmax", "head/Softmax", "probs",
                         "softmax.onnx", axis=-1)
    rc, out = bnc("check", sm, "--cli", env.cli, "--labels", env.labels,
                  "--paths", "neon-fp32", "--workdir", env.path("ck_sm"))
    drop(sm, env.path("ck_sm"))
    check("B17 final Softmax: check compiles the logits before it and picks "
          "--post softmax", rc == 0 and "post softmax" in out
          and "ignored" in out, out[-500:])


def compile_test_package(env, lt, path, out_pkg):
    """Compiles the model for one path into a scratch package; returns
    (exit code, output)."""
    return bnc("compile", env.model, "--labels", env.labels, "--model-id",
               "t", "--family", "0", "--post", "sigmoid", "--force",
               "--sample-rate", "48000", "--paths", path, "--logits", lt,
               "--force", "-o", out_pkg)


def run_package(env, lt, src16):
    """The package part B runs and mutates: src16 when this CPU runs
    neon-fp16 (or --run-path neon-fp16), else a neon-fp32 one; None when that
    fails to compile or `--info` gives no answer (a recorded failure)."""
    run_path = opt("--run-path")
    if not run_path:
        best = best_path(env.cli, src16)
        if best is None:
            return None
        run_path = "neon-fp16" if best == "neon-fp16" else "neon-fp32"
    if run_path == "neon-fp16":
        return src16
    print(f"part B runs a {run_path} package (this CPU cannot run neon-fp16, "
          f"or --run-path); B8 and B11 still mutate the neon-fp16 one")
    src = env.path("t32.bnpk")
    rc, out = compile_test_package(env, lt, run_path, src)
    check(f"B0 compile the {run_path} test package", rc == 0, out[-600:])
    return None if rc else src


def part_b_runtime(env, src, src16):
    """B0-B13b on the package src (B8 and B11 on the neon-fp16 one).

    Returns:
        The path of the other (three-line) label file.
    """
    win = env.path("w.f32")
    pkg = Pkg(src)
    write_window(pkg, win)
    out_f = env.path("o.f32")
    rc, out = run([env.cli, src, "0", win, out_f, "1"])
    check("B0 the intact package runs", rc == 0, out)
    mut = Mutations(env, src, win, out_f)
    path = pkg.paths[0]
    ops = path["ops"]
    # an op with a non-input, non-output activation buffer argument and a
    # source name: the first pw
    pw = next(op for op in ops if (op["code"] & 0xff) == 3)
    pw_i = ops.index(pw)
    check_file_errors(mut, pw, pw_i)
    check_argument_errors(mut, pkg, path, pw, pw_i)
    check_hwcap(Mutations(env, src16, win, out_f), Pkg(src16).paths[0])
    other = check_cli_errors(env, src, src16, win, out_f)
    check_memory_guard(mut, env, src, win, out_f, path, pw)
    drop(win, out_f)
    return other


def part_b(env):
    """Broken packages through the runtime CLI, and compile refusals."""
    lt = logits_name(env)
    src16 = env.path("t.bnpk")
    rc, out = compile_test_package(env, lt, "neon-fp16", src16)
    check("B0 compile the test package", rc == 0, out[-600:])
    if rc:
        return
    src = run_package(env, lt, src16)
    if src is None:
        return
    other = part_b_runtime(env, src, src16)
    drop(src, src16)
    check_compile_refusals(env, lt, other)
    check_dilated_report(env)
    check_final_softmax(env, lt)


# -------------------------------------------------- part C: general head, canon
def ort_logits(path, out_name, x):
    """ONNX Runtime's out_name for each window of x, stacked."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0].name
    return np.stack([s.run([out_name], {i: w.reshape(1, -1)})[0].reshape(-1)
                     for w in x])


def sqnr(a, b):
    """Signal-to-quantization-noise ratio of b against a, in dB."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return 10 * np.log10((a ** 2).sum() / max(((a - b) ** 2).sum(), 1e-30))


def find_head(g, lt):
    """The MatMul nodes feeding the logits (directly or through an Add)."""
    head = [n for n in g.node if n.op_type == "MatMul" and n.output[0] == lt]
    if not head:
        head = [n for n in g.node if n.op_type == "MatMul"
                and any(c.op_type == "Add" and c.output[0] == lt
                        for c in g.node if n.output[0] in c.input)]
    return head


def splice_hidden_layer(g, mm, rng):
    """feat -> MatMul W1 -> Add b1 -> Relu -> MatMul W2 before the head's
    MatMul (the original Add bias stays). Returns the number of classes."""
    w = numpy_helper.to_array(next(t for t in g.initializer
                                   if t.name == mm.input[1]))  # [1024][N]
    k, n = w.shape
    hidden = 96
    w1 = (rng.standard_normal((k, hidden)) * 0.05).astype(np.float32)
    b1 = (rng.standard_normal(hidden) * 0.1).astype(np.float32)
    w2 = (rng.standard_normal((hidden, n)) * 0.1).astype(np.float32)
    feat = mm.input[0]
    g.initializer.extend([numpy_helper.from_array(w1, "hid/W1"),
                          numpy_helper.from_array(b1, "hid/b1"),
                          numpy_helper.from_array(w2, "hid/W2")])
    new = [helper.make_node("MatMul", [feat, "hid/W1"], ["hid/mm1"],
                            name="hid/MatMul1"),
           helper.make_node("Add", ["hid/mm1", "hid/b1"], ["hid/pre"],
                            name="hid/Add1"),
           helper.make_node("Relu", ["hid/pre"], ["hid/act"],
                            name="hid/Relu")]
    mm.input[0] = "hid/act"
    mm.input[1] = "hid/W2"
    idx = list(g.node).index(mm)
    for i, node in enumerate(new):
        g.node.insert(idx + i, node)
    return n


def chirp_windows(rng):
    """Three noise windows with a chirp each (the logits are not all noise)."""
    x = (rng.standard_normal((3, 144000)) * 0.05).astype(np.float32)
    for i in range(3):
        x[i] += 0.2 * np.sin(2 * np.pi * (500 + 300 * i) * np.arange(144000)
                             / 48000)
    return x


def runs_here(env, pkg, pth):
    """Whether this CPU runs the one-path package pkg (path pth) for C2.

    Only a neon-fp16 package that this CPU cannot run is a skip (a CPU
    without fp16 arithmetic); no answer from `--info`, or a package that
    names no path or another one, is a recorded failure.
    """
    best = best_path(env.cli, pkg)
    if best is None:
        return False
    if best == NO_PATH and pth == "neon-fp16":
        print(f"skip C2 hidden-layer package runs on {pth} and agrees with "
              f"ONNX Runtime (needs fp16: this CPU cannot run {pth})")
        return False
    if best != pth:
        check(f"C2 this CPU runs the {pth} hidden-layer package", False,
              f"best path for this CPU: {best}")
        return False
    return True


def check_hidden_path(env, hp, lt, win, ref, n, pth, mode):
    """C2 on one path: the hidden-layer package compiles, runs and agrees
    with ONNX Runtime."""
    pkg = env.path(f"hidden-{pth}.bnpk")
    rc, out = bnc("compile", hp, "--labels", env.labels, "--model-id", "hid",
                  "--family", "0", "--post", "sigmoid", "--force",
                  "--sample-rate", "48000", "--paths", pth, "--logits", lt,
                  "-o", pkg)
    check(f"C2 hidden layer compiles on {pth}", rc == 0, out[-400:])
    if rc:
        return
    if not runs_here(env, pkg, pth):
        drop(pkg)
        return
    rc, out = run([env.cli, pkg, str(mode), win, env.path("ho.f32"), "1"])
    drop(pkg)
    check(f"C2 hidden-layer package runs on {pth}", rc == 0, out[-400:])
    if rc:
        return
    y = np.fromfile(env.path("ho.f32"), np.float32).reshape(3, n)
    drop(env.path("ho.f32"))
    s = sqnr(ref, y)
    need = 60 if pth == "neon-fp32" else 40
    check(f"C2 {pth} hidden-layer logits agree with ONNX Runtime ({s:.1f} dB "
          f">= {need})", s >= need and np.isfinite(y).all(), f"{s:.1f} dB")


def check_hidden_layer(env, hp, lt, win, ref, n):
    """C1, C2: the report places the hidden layer; both paths agree."""
    rc, out = bnc("report", hp)
    check("C1 hidden layer: report places every node (Dense+ReLU fused, GEMV "
          "head)", rc == 0 and "0 unsupported" in out
          and "'act': 'relu'" in out, out[-600:])
    for pth, mode in (("neon-fp32", 2), ("neon-fp16", 3)):
        check_hidden_path(env, hp, lt, win, ref, n, pth, mode)


def is_dft_matmul(g, n):
    """A MatMul whose second input is a [1024|2048][*] initializer."""
    return n.op_type == "MatMul" and any(
        t.name == n.input[1] and len(t.dims) == 2
        and t.dims[0] in (1024, 2048) for t in g.initializer)


def splice_dft_artifacts(g):
    """Unsqueeze/Squeeze around the first DFT MatMul and a float->float
    Cast after it. Returns the names of the artifacts added."""
    dft = [n for n in list(g.node) if is_dft_matmul(g, n)]
    check("C3 found the DFT MatMuls", len(dft) >= 1, str(len(dft)))
    if not dft:
        return []
    n = dft[0]
    src = n.input[0]
    g.initializer.append(numpy_helper.from_array(np.array([-2], np.int64),
                                                 "canon_t/axes"))
    un = helper.make_node("Unsqueeze", [src, "canon_t/axes"], ["canon_t/un"],
                          name="canon_t/Unsqueeze")
    n.input[0] = "canon_t/un"
    old_out = n.output[0]
    n.output[0] = "canon_t/mm"
    sq = helper.make_node("Squeeze", ["canon_t/mm", "canon_t/axes"],
                          ["canon_t/sq"], name="canon_t/Squeeze")
    ca = helper.make_node("Cast", ["canon_t/sq"], [old_out],
                          name="canon_t/Cast", to=1)
    i = list(g.node).index(n)
    g.node.insert(i, un)
    g.node.insert(i + 2, sq)
    g.node.insert(i + 3, ca)
    return ["Unsqueeze/Squeeze", "Cast"]


def reverse_slice_axis(g, n):
    """The axis of a step -1 Slice, else None."""
    if not (n.op_type == "Slice" and len(n.input) > 4
            and any(t.name == n.input[4]
                    and numpy_helper.to_array(t).tolist() == [-1]
                    for t in g.initializer)):
        return None
    axes = next(t for t in g.initializer if t.name == n.input[3])
    return int(numpy_helper.to_array(axes)[0])


def splice_reverse_sequence(m, g):
    """The first reverse Slice whose input shape is known becomes a
    ReverseSequence. Returns the names of the artifacts added."""
    shapes = {vi_.name: [d.dim_value for d in vi_.type.tensor_type.shape.dim]
              for vi_ in onnx.shape_inference.infer_shapes(m).graph.value_info}
    for n in list(g.node):
        ax = reverse_slice_axis(g, n)
        if ax is None:
            continue
        sh = shapes.get(n.input[0])
        if not sh or len(sh) < 2:
            continue
        # the batch dim is dynamic in the export; the check runs batch 1
        sh = [x if x > 0 else 1 for x in sh]
        ax %= len(sh)
        if ax > 1:
            # ONNX ReverseSequence takes time_axis in {0, 1} (tf2onnx
            # transposes first; the NFC run covers it)
            continue
        bx = 0 if ax != 0 else 1
        g.initializer.append(numpy_helper.from_array(
            np.full(sh[bx], sh[ax], np.int64), "canon_t/seqlen"))
        rs = helper.make_node("ReverseSequence",
                              [n.input[0], "canon_t/seqlen"], [n.output[0]],
                              name="canon_t/ReverseSequence", time_axis=ax,
                              batch_axis=bx)
        g.node[list(g.node).index(n)].CopyFrom(rs)
        return ["ReverseSequence"]
    return []


def splice_pad(g):
    """A Pad before the first depthwise conv (its pads moved out of it).
    Returns the names of the artifacts added."""
    dw = [n for n in g.node if n.op_type == "Conv"
          and any(a.name == "group" and a.i > 1 for a in n.attribute)]
    if not dw:
        return []
    n = dw[0]
    pads = next((list(a.ints) for a in n.attribute if a.name == "pads"),
                [0, 0, 0, 0])
    if pads != [1, 1, 1, 1]:
        return []
    for a in list(n.attribute):
        if a.name == "pads":
            n.attribute.remove(a)
    n.attribute.append(helper.make_attribute("pads", [0, 0, 0, 0]))
    g.initializer.append(numpy_helper.from_array(
        np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "canon_t/pads"))
    pd = helper.make_node("Pad", [n.input[0], "canon_t/pads"],
                          ["canon_t/padded"], name="canon_t/Pad",
                          mode="constant")
    n.input[0] = "canon_t/padded"
    g.node.insert(list(g.node).index(n), pd)
    return ["Pad"]


def reduce_mean_axes(n, inits):
    """The (sorted, mod 4) axes of a keepdims ReduceMean, else None.

    GlobalAveragePool keeps the spatial dims: only a keepdims mean is
    equivalent.
    """
    if n.op_type != "ReduceMean":
        return None
    if next((a.i for a in n.attribute if a.name == "keepdims"), 1) != 1:
        return None
    axes = next((sorted(int(v) % 4 for v in a.ints)
                 for a in n.attribute if a.name == "axes"), None)
    if axes is None and len(n.input) > 1 and n.input[1] in inits:
        axes = sorted(int(v) % 4 for v in inits[n.input[1]].reshape(-1))
    return axes


def splice_gap(g):
    """The first spatial ReduceMean becomes a GlobalAveragePool.
    Returns the names of the artifacts added."""
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    for n in list(g.node):
        if reduce_mean_axes(n, inits) == [2, 3]:
            gap = helper.make_node("GlobalAveragePool", [n.input[0]],
                                   [n.output[0]], name="canon_t/GAP")
            g.node[list(g.node).index(n)].CopyFrom(gap)
            return ["GlobalAveragePool"]
    return []


def check_canonicalization(env):
    """C3: tf2onnx-style artifacts spliced into the stock graph are
    canonicalized away and the result is verified."""
    m = onnx.load(env.model)
    g = m.graph
    ops_added = splice_dft_artifacts(g)
    ops_added += splice_reverse_sequence(m, g)
    ops_added += splice_pad(g)
    ops_added += splice_gap(g)
    cp = env.path("canon.onnx")
    onnx.save(m, cp)
    rc, out = bnc("report", cp, "--json", env.path("canon.json"))
    j = load_json(env.path("canon.json"))
    done = dict(j.get("canonicalised", []))
    check(f"C3 canonicalization removes the spliced artifacts {ops_added} and "
          "the report is clean",
          rc == 0 and j["status"] == "ok"
          and done.get("Unsqueeze/Squeeze around a MatMul", 0) >= 1
          and done.get("float->float Cast", 0) >= 1
          and done.get("zero Pad folded into a Conv", 0) >= 1
          and (done.get("ReverseSequence -> Slice", 0) >= 1
               or "ReverseSequence" not in ops_added)
          and (done.get("GlobalAveragePool -> ReduceMean", 0) >= 1
               or "GlobalAveragePool" not in ops_added),
          f"exit {rc} {j.get('canonicalised')} {out[-300:]}")
    check("C3 the canonicalized graph is verified against ONNX Runtime",
          j.get("canon_err", 1) < 1e-4, str(j.get("canon_err")))
    rc, out = bnc("report", cp, "--no-canon")
    drop(cp)
    check("C3 --no-canon shows the artifacts as unsupported", rc == 2,
          f"exit {rc}")


def check_compile_unsupported(env, lt):
    """C4: compile on an unsupported graph exits 2, not an internal error.

    Returns:
        The unsupported graph's path.
    """
    bad = env.path("dilated2.onnx")
    m = onnx.load(env.model)
    dilate_conv(m, add_missing=False)
    onnx.save(m, bad)
    rc, out = bnc("compile", bad, "--labels", env.labels, "--model-id", "x",
                  "--family", "0", "--post", "sigmoid", "--force", "--paths",
                  "neon-fp32", "--logits", lt, "-o", env.path("x.bnpk"))
    drop(env.path("x.bnpk"))
    check("C4 compile on an unsupported graph: exit 2 with the problem report, "
          "no internal error", rc == 2 and "dilated conv" in out
          and "internal error" not in out
          and "downstream node(s) not listed" in out,
          f"exit {rc}: {out[-400:]}")
    return bad


def check_tail_compile(env, lt):
    """C5: an unsupported op after the logits tensor does not block compile.

    Returns:
        The model with the tail.
    """
    tp = model_with_tail(env, lt, "Tanh", "tail/Tanh", "tail", "tail.onnx")
    rc, out = bnc("compile", tp, "--labels", env.labels, "--model-id", "x",
                  "--family", "0", "--post", "sigmoid", "--force", "--paths",
                  "neon-fp32", "--logits", lt, "-o", env.path("tail.bnpk"))
    drop(env.path("tail.bnpk"))
    check("C5 an unsupported node after --logits does not block compile",
          rc == 0, f"exit {rc}: {out[-300:]}")
    return tp


def check_emb(env, hp, lt):
    """C7: --emb on the hidden layer's activated, missing and pre-ReLU
    tensors."""
    m = onnx.load(hp)
    m.graph.output.append(vi("hid/act", None))
    hp2 = env.path("hidden2.onnx")
    onnx.save(m, hp2)
    base = ["compile", hp2, "--labels", env.labels, "--model-id", "x",
            "--family", "0", "--post", "sigmoid", "--force", "--paths",
            "neon-fp32", "--logits", lt]
    rc, out = bnc(*base, "--emb", "hid/act", "-o", env.path("hid2.bnpk"))
    drop(env.path("hid2.bnpk"))
    check("C7 --emb on the activated hidden tensor compiles with emb 96",
          rc == 0 and "emb 96" in out, f"exit {rc}: {out[-300:]}")
    rc, out = bnc(*base, "--emb", "no/such/tensor", "-o",
                  env.path("hid3.bnpk"))
    check("C7 a wrong --emb name is refused by name (exit 3), not silently "
          "emb 0", rc == 3 and "no tensor of that name" in out,
          f"exit {rc}: {out[-300:]}")
    rc, out = bnc(*base, "--emb", "hid/pre", "-o", env.path("hid4.bnpk"))
    drop(hp2, env.path("hid3.bnpk"), env.path("hid4.bnpk"))
    check("C7 --emb on the pre-ReLU dense output says to name the ReLU's "
          "output", rc == 2 and "name the ReLU's output" in out,
          f"exit {rc}: {out[-300:]}")


def check_softmax_only(env, lt):
    """C8: a graph whose only output is a Softmax."""
    sp = model_with_tail(env, lt, "Softmax", "only/Softmax", "probs",
                         "softmax_only.onnx", only=True, axis=-1)
    rc, out = bnc("compile", sp, "--labels", env.labels, "--model-id", "x",
                  "--family", "0", "--force", "--paths", "neon-fp32", "-o",
                  env.path("sm.bnpk"))
    drop(sp, env.path("sm.bnpk"))
    check("C8 a graph whose only output is a Softmax: logits default to its "
          "input, --post softmax inferred", rc == 0 and "using softmax" in out,
          f"exit {rc}: {out[-300:]}")


def check_tail_check(env, tp, lt, win):
    """C5: check with --logits before the tail and --windows-file."""
    rc, out = bnc("check", tp, "--labels", env.labels, "--logits", lt,
                  "--sample-rate", "48000", "--post", "sigmoid", "--cli",
                  env.cli, "--windows-file", win, "--paths", "neon-fp32",
                  "--workdir", env.path("ck"))
    drop(env.path("ck"))
    check("C5 check with --logits before the tail and --windows-file: PASS",
          rc == 0 and "ignored" in out and "PASS" in out,
          f"exit {rc}: {out[-400:]}")


def check_downstream_report(bad):
    """C6: downstream nodes are collapsed unless --verbose."""
    _, out = bnc("report", bad)
    check("C6 downstream nodes are collapsed to a count in the text report",
          "downstream node(s) not listed" in out
          and out.count("unsupported op") == 1, out[-300:])
    _, out = bnc("report", bad, "--verbose")
    check("C6 --verbose lists them", out.count("unsupported op") > 5,
          str(out.count("unsupported op")))


def part_c(env):
    """The general head, canonicalization and the output cone."""
    lt = logits_name(env)
    m = onnx.load(env.model)
    g = m.graph
    head = find_head(g, lt)
    check("C0 found the head MatMul feeding the logits", bool(head), lt)
    if not head:
        return
    rng = np.random.default_rng(3)
    n = splice_hidden_layer(g, head[0], rng)
    hp = env.path("hidden.onnx")
    onnx.save(m, hp)
    x = chirp_windows(rng)
    ref = ort_logits(hp, lt, x)
    win = env.path("hw.f32")
    x.tofile(win)
    check_hidden_layer(env, hp, lt, win, ref, n)
    check_canonicalization(env)
    bad = check_compile_unsupported(env, lt)
    tp = check_tail_compile(env, lt)
    check_emb(env, hp, lt)
    drop(hp)
    check_softmax_only(env, lt)
    check_tail_check(env, tp, lt, win)
    drop(tp, win)
    check_downstream_report(bad)
    drop(bad)


# ------------------------------------------------------ part D: round-3 tools
def check_relabel(d):
    """D1: the relabel script's conversions, --check, --sci, --allow-doubled
    and its usage error."""
    pub = os.path.join(d, "published.txt")
    write_text(pub, "Barn_Swallow_Hirundo_rustica\nAnthropogenic\n"
               "Background_birds\nHirundo rustica_Barn Swallow\n"
               "Eurasian_Golden_Oriole\n")
    go = os.path.join(d, "go.txt")
    rc, out = script(RELABEL, pub, go, "--sci",
                     "Eurasian_Golden_Oriole=Oriolus oriolus")
    lines = read_text(go).splitlines()
    check("D1 relabel: Common_Name_Genus_species -> Scientific_Common, "
          "Name_Name, --sci, kept lines", rc == 0 and lines == [
              "Hirundo rustica_Barn Swallow", "Anthropogenic_Anthropogenic",
              "Background birds_Background birds",
              "Hirundo rustica_Barn Swallow",
              "Oriolus oriolus_Eurasian Golden Oriole"], str(lines))
    rc, out = script(RELABEL, go, os.path.join(d, "go2.txt"), "--check")
    check("D1 relabel --check on a converted file: nothing to change, exit 0",
          rc == 0 and "0 of 5" in out, out[-200:])
    rc, out = script(RELABEL, "--check", pub, os.path.join(d, "go3.txt"))
    check("D1 relabel: flags before the files work; the species-like line is "
          "flagged loudly, exit 1",
          rc == 1 and "NO SCIENTIFIC NAME FOUND for 1 line" in out
          and "Eurasian_Golden_Oriole" in out and "need --sci" in out,
          f"exit {rc}: {out[-300:]}")
    rc, out = script(RELABEL, pub, os.path.join(d, "go4.txt"))
    check("D1 relabel refuses to write a doubled species name without --sci / "
          "--allow-doubled (exit 2, no file)",
          rc == 2 and not os.path.exists(os.path.join(d, "go4.txt"))
          and "not written" in out, f"exit {rc}: {out[-300:]}")
    rc, out = script(RELABEL, pub, os.path.join(d, "go4.txt"),
                     "--allow-doubled")
    check("D1 relabel --allow-doubled writes it and says so",
          rc == 0 and "1 doubled by --allow-doubled" in out,
          f"exit {rc}: {out[-200:]}")
    rc, out = script(RELABEL, "--check")
    check("D1 relabel without files: usage, exit 3, no traceback",
          rc == 3 and "Traceback" not in out and "infile" in out,
          f"exit {rc}: {out[-200:]}")


def check_label_form(env, lt):
    """D2: a published-form label file: check warns and names the first
    offender; --quiet keeps the verdict."""
    n = len([line for line in read_text(env.labels).splitlines()
             if line.strip()])
    bad = env.path("bad_form.txt")
    write_text(bad, "".join(f"Common_Name_{i}_Genus_species\n"
                            for i in range(n)))
    _, out = bnc("check", env.model, "--labels", bad, "--logits", lt,
                 "--sample-rate", "48000", "--post", "sigmoid", "--cli",
                 env.cli, "--paths", "neon-fp32", "--quiet", "--workdir",
                 env.path("ck_form"))
    drop(env.path("ck_form"))
    check("D2 check warns about labels not in BirdNET-Go form, naming the "
          "first line", f"{n} of {n} label lines" in out
          and "Common_Name_0_Genus_species" in out
          and "relabel_for_birdnetgo" in out, out[-400:])
    check("D2 check --quiet drops the IR listing but keeps the verdict",
          "  CONV " not in out and ("PASS" in out or "WARN" in out),
          out[-300:])


def check_compile_defaults(env, lt):
    """D3, D4: the license refusal, the inferred family and post, the
    package's metadata and embedded labels.

    Returns:
        (the compile argument list, the license file).
    """
    base = ["compile", env.model, "--labels", env.labels, "--model-id", "x",
            "--paths", "neon-fp32", "--logits", lt, "-o", env.path("r.bnpk")]
    rc, out = bnc(*base)
    check("D3 compile without license/attribution is refused (exit 3) with "
          "the template pointer",
          rc == 3 and "--license-file and --attribution" in out
          and "section 5" in out, f"exit {rc}: {out[-300:]}")
    lic = env.path("LIC.txt")
    write_text(lic, "test license\n")
    rc, out = bnc(*base, "--license-file", lic, "--attribution", "t")
    check("D3 v2.4 backbone: --family 24 and --post sigmoid inferred and "
          "printed", rc == 0 and "using 24" in out and "using sigmoid" in out,
          f"exit {rc}: {out[-300:]}")
    rc, out = run([env.cli, env.path("r.bnpk"), "--info"])
    check("D3 the package carries family 24, post 1, sample_rate 48000",
          "family 24" in out and "post 1" in out
          and "sample_rate 48000" in out, out[:300])
    rc, out = run([env.cli, env.path("r.bnpk"), "--labels"])
    check("D4 cli --labels prints the embedded label file byte for byte",
          rc == 0 and out == read_text(env.labels), out[:120])
    check_license_key(env, base, lic)
    return base, lic


def old_key_copy(src, dst):
    """Writes a copy of package src whose license key is spelled "licence"
    (as compilers before bnc 3 wrote it); True if the key was found."""
    with open(src, "rb") as f:
        b = bytearray(f.read())
    meta_off = struct.unpack_from("<Q", b, 32)[0]
    # the record header: u16 key length 7, u16 type 0 (string), u32 length
    i = b.find(struct.pack("<HH", 7, 0), meta_off)
    while i >= 0 and b[i + 8:i + 15] != b"license":
        i = b.find(struct.pack("<HH", 7, 0), i + 1)
    if i < 0:
        return False
    b[i + 8:i + 15] = b"licence"
    with open(dst, "wb") as f:
        f.write(b)
    return True


def check_license_key(env, base, lic):
    """D4b-D4d: the license key, the old option and key spelling."""
    rc, out = run([env.cli, env.path("r.bnpk"), "--info"])
    check("D4b the package carries the license under the key 'license'",
          rc == 0 and "license: test license" in out, out[-400:])
    rc, out = bnc(*base[:-1], env.path("r_old.bnpk"), "--licence-file", lic,
                  "--attribution", "t")
    same = rc == 0 and read_bytes(env.path("r.bnpk")) == read_bytes(
        env.path("r_old.bnpk"))
    check("D4c the old spelling --licence-file is still accepted (same "
          "package)", same, f"exit {rc}: {out[-200:]}")
    drop(env.path("r_old.bnpk"))
    found = old_key_copy(env.path("r.bnpk"), env.path("r_key.bnpk"))
    rc, out = run([env.cli, env.path("r_key.bnpk"), "--info"])
    drop(env.path("r_key.bnpk"))
    check("D4d a package with the old key 'licence' still shows its license",
          found and rc == 0 and "license: test license" in out, out[-400:])


def check_sample_rate(env, lt, base, lic):
    """D5, D6: --sample-rate refusals and --force; both paths by default."""
    rc, out = bnc(*base, "--license-file", lic, "--attribution", "t",
                  "--sample-rate", "32000")
    check("D5 --sample-rate 32000 on a v2.4 backbone is refused (exit 3) "
          "unless --force", rc == 3 and "48000 Hz" in out,
          f"exit {rc}: {out[-300:]}")
    rc, out = bnc(*base, "--license-file", lic, "--attribution", "t",
                  "--sample-rate", "32000", "--force")
    check("D5 --force writes it", rc == 0, f"exit {rc}: {out[-200:]}")
    rc, out = bnc(*base, "--license-file", lic, "--attribution", "t",
                  "--sample-rate", "0")
    check("D5 --sample-rate 0 is refused", rc == 3 and "positive" in out,
          f"exit {rc}")
    rc, out = bnc("compile", env.model, "--labels", env.labels, "--model-id",
                  "x", "--logits", lt, "--license-file", lic, "--attribution",
                  "t", "-o", env.path("both.bnpk"))
    drop(env.path("both.bnpk"))
    check("D6 no --paths: both paths in one file, said so",
          rc == 0 and "both paths go into one file" in out,
          f"exit {rc}: {out[-200:]}")


def check_gate(env, lt):
    """D7: the generic gate: declare once, score the package against the
    ONNX reference on three windows."""
    crit = env.path("crit.json")
    rc, out = script(GATE, "declare", crit, "--thresholds", "0.1",
                     "--sqnr-db", "40")
    check("D7 gate declare writes the criteria with a timestamp and sha",
          rc == 0 and os.path.exists(crit)
          and "declared_utc" in read_text(crit), out[-200:])
    rc, out = script(GATE, "declare", crit)
    check("D7 declaring twice is refused", rc == 3 and "declared once" in out,
          out[-200:])
    win = env.path("hw.f32")
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((3, 144000)) * 0.05).astype(np.float32)
    x.tofile(win)
    tr = env.path("truth.csv")
    write_text(tr, "0,0\n1,3\n")
    gate_json = env.path("gate.json")
    rc, out = script(GATE, "score", crit, "--ref", env.model, "--ref-output",
                     lt, "--cand", env.path("r.bnpk") + ":2", "--cli",
                     env.cli, "--post", "sigmoid", "--windows-file", win,
                     "--truth", tr, "--json", gate_json)
    j = load_json(gate_json) if os.path.exists(gate_json) else {}
    check("D7 gate score: statistics, P/R with CIs from truth, verdict and "
          "JSON", rc == 0 and "hit-set Jaccard" in out and "dP " in out
          and j.get("overall") == "PASS"
          and j["candidates"][0]["thresholds"]["0.1"]["truth"]["dP"],
          f"exit {rc}: {out[-500:]}")
    rc, out = script(GATE, "score", env.path("nocrit.json"), "--ref",
                     env.model, "--cli", env.cli, "--post", "sigmoid")
    check("D7 scoring without declared criteria is refused",
          rc == 3 and "declare" in out, out[-200:])


def check_cmp_suite():
    """D8: cmp_suite without the gated reference data fails clearly."""
    if os.path.exists(os.path.join(ROOT, "data", "bitref", "a17.f32")):
        print("skip D8 (this checkout has data/bitref)")
        return
    rc, out = run([os.path.join(ROOT, "scripts", "cmp_suite.sh")])
    check("D8 cmp_suite without the reference data fails clearly (exit 2)",
          rc == 2 and "missing" in out and "appendix A" in out,
          f"exit {rc}: {out[-300:]}")


def part_d(env):
    """The round-3 tools: relabel, label form, compile defaults, the gate."""
    lt = logits_name(env)
    check_relabel(env.d)
    check_label_form(env, lt)
    base, lic = check_compile_defaults(env, lt)
    check_sample_rate(env, lt, base, lic)
    check_gate(env, lt)
    drop(env.path("r.bnpk"), env.path("hw.f32"))
    check_cmp_suite()


OP_STEM = 9
OP_PROTO = 13
OP_PROTO_F32H = 14
OP_PROTO_S8 = 35
INT8_HEAD_K_LIMIT = 1 << 17
PATH_MODES = {"neon-fp32": 2, "neon-fp16": 3, "neon-int8": 4}
HEAD_OPS = (OP_PROTO, OP_PROTO_F32H, OP_PROTO_S8)


def is_head(op):
    """True for a prototype head op."""
    return op["code"] & 0xff in HEAD_OPS


def head_layout(op):
    """The int-parameter indices (n, k, kp, np, c, nc, ppc) of a head op.

    kp is None for a head without one.
    """
    if op["ni"] == 9:
        # PROTO fp32 / fp16 and PROTO_S8: [N K Kp Np, P C ld nc ppc]
        return dict(n=0, k=1, kp=2, np=3, c=5, nc=7, ppc=8)
    # SDOT head and PROTO_F32H: [N K Np, P C ld nc ppc]
    return dict(n=0, k=1, kp=None, np=2, c=4, nc=6, ppc=7)


def head_cases(op):
    """(name, {int index: value}) head mutations the loader must refuse."""
    ix = head_layout(op)
    k = op["p"][ix["k"]]
    ppc = op["p"][ix["ppc"]]
    int8_head = (op["code"] & 0xff) == OP_PROTO_S8 or op["code"] >> 8 == 2

    def kc(value, kp=None):
        ints = {ix["k"]: value, ix["c"]: value}
        if ix["kp"] is not None:
            ints[ix["kp"]] = value if kp is None else kp
        return ints

    cases = [
        ("negative K and C", kc(-8, 0)),
        ("K and C zero", kc(0)),
        ("K beyond 2^20", kc(1 << 21)),
        ("N near INT_MAX (rounding it up would overflow)",
         {ix["n"]: 0x7ffffffe, ix["nc"]: 0x7ffffffe // ppc}),
        ("negative N", {ix["n"]: -4, ix["np"]: -4, ix["nc"]: -4 // ppc}),
        ("K not matching the cells' channels", {ix["k"]: k - 8}),
    ]
    if int8_head:
        cases.append(("K above 2^17 (the int32 sums could wrap)",
                      kc(INT8_HEAD_K_LIMIT + 16)))
    if ix["kp"] is not None and not int8_head:
        cases.append(("Kp different from K", {ix["kp"]: k + 8}))
    return cases


def refused(d, cli, package, mode, op, op_index, ints):
    """Writes the package with the op's ints changed and runs the CLI.

    Returns:
        (refused cleanly, the CLI's output).
    """
    mutated = Pkg(package)
    for index, value in ints.items():
        struct.pack_into("<i", mutated.b, op["p_at"] + 4 * index, value)
    f = os.path.join(d, "bad_shape.bnpk")
    mutated.write(f)
    rc, out = run([cli, f, str(mode), os.devnull, os.devnull, "1"])
    drop(f)
    text = out.strip()
    named = f"op {op_index} " in text
    failed = "parameter check failed" in text or "expected" in text
    return rc > 0 and named and failed, text


def check_forged_n4(d, cli, package, mode, pkg, head, head_i, tag):
    """E: every n4 flag set; the loader names the first bad panel."""
    flags = pkg.tensors[head["t"][2]]
    mutated = Pkg(package)
    mutated.b[flags["off"]:flags["off"] + flags["nbytes"]] = (
        b"\x01" * flags["nbytes"])
    f = os.path.join(d, "bad_flags.bnpk")
    mutated.write(f)
    rc, out = run([cli, f, str(mode), os.devnull, os.devnull, "1"])
    drop(f)
    check(f"E {tag} every n4 flag set: the loader names the first panel "
          f"that breaks the lane bound",
          rc > 0 and f"op {head_i} proto_s8n" in out
          and "flagged for 4 products" in out and "258" in out,
          out.strip()[-300:])


def part_e_path(d, cli, package, pkg, path, mode):
    """E: corrupt head, stem and n4 flags of one path are refused."""
    tag = os.path.basename(package)
    head = next(op for op in path["ops"] if is_head(op))
    head_i = path["ops"].index(head)
    for name, ints in head_cases(head):
        ok, text = refused(d, cli, package, mode, head, head_i, ints)
        check(f"E {tag} {path['name']} head, {name}: refused with the op "
              f"and the failed check", ok, text[-300:])
    stem = next((op for op in path["ops"]
                 if (op["code"] & 0xff) == OP_STEM and op["code"] >> 8 == 0),
                None)
    if stem is not None:
        # p = [N K Kp Np, H W Ho Wo k s act ldo opad]: the GEMM stores whole
        # 8-channel panels
        ok, text = refused(d, cli, package, mode, stem,
                           path["ops"].index(stem), {11: stem["p"][3] - 8})
        check(f"E {tag} {path['name']} stem, output stride below Np: refused "
              f"with the op and the failed check", ok, text[-300:])
    if (head["code"] & 0xff) == OP_PROTO_S8:
        check_forged_n4(d, cli, package, mode, pkg, head, head_i, tag)


def part_e(d, cli, package):
    """Part E: packages with a prototype head: corrupt head and stem shapes
    are refused by the loader, not run."""
    pkg = Pkg(package)
    found = False
    for path in pkg.paths:
        mode = PATH_MODES.get(path["name"])
        if mode is None or not any(is_head(op) for op in path["ops"]):
            continue
        found = True
        part_e_path(d, cli, package, pkg, path, mode)
    check(f"E {os.path.basename(package)}: a path with a prototype head was "
          f"found", found, package)


def cli_path():
    """--cli, else the CLI built natively into build/check."""
    cli = opt("--cli")
    if not cli:
        rc, out = run(["make", "-s", f"OUT={ROOT}/build/check", "-j4"],
                      cwd=ROOT)
        check("B build the CLI natively", rc == 0, out[-500:])
        cli = f"{ROOT}/build/check/cli"
    return cli


def run_parts(d):
    """Parts A-E with the scratch directory d."""
    part_a(d)
    model = opt("--model")
    if model:
        env = Env(d, cli_path(), model, opt("--labels"))
        part_b(env)
        part_c(env)
        part_d(env)
    else:
        print("part B skipped: pass --model MODEL.onnx --labels LABELS.txt "
              "(a graph the compiler supports)")
    proto_packages = testutil.opt_all("--proto-package")
    for package in proto_packages:
        part_e(d, opt("--cli") or f"{ROOT}/build/check/cli", package)
    if not proto_packages:
        print("part E skipped: pass --proto-package PKG.bnpk (a package with a "
              "prototype head, e.g. Perch v2; repeatable)")


def main():
    """Runs parts A-E in a scratch directory; returns the exit code."""
    d = tmpdir()
    print(f"scratch: {d}")
    try:
        run_parts(d)
    finally:  # also after an error (a full disk): the scratch goes
        drop(d)
    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())
