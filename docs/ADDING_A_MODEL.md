# Adding a model to libbnneon

libbnneon runs compiled model packages (`.bnpk`) on AArch64 CPUs with hand-written NEON kernels. A package holds a
model's metadata, its weights pre-packed for each CPU path, and an op list the runtime executes on registry kernels.
New models come in through the compiler (`bnc`, the `bnneon.compiler` package in `python/`), never through model-specific C code. Sections 1-11 are
generic; appendix A says which checks run from a clean clone and which need files that are not in this repository.

Contents: 1 what you need, 2 what the engine can run (limits; 2.1 BirdNET v3.0), 3 get the model as ONNX, 4 `bnc check`, 5 compile
the shipping packages, 6 when the recognizer says no (extending the compiler; 6.1 a new model family), 7 precision, 8 debugging parity,
9 gate, time, ship, 10 troubleshooting (every message and its fix), 11 scripted use (exit codes and JSON), A what runs from a clean clone, B reference.

## 1. What you need

- An AArch64 machine with a C compiler (clang, or gcc >= 10): Apple silicon Mac, a Raspberry Pi 4/5, an arm64
  container. The runtime builds with `make` (`make OUT=build`); nothing else is linked.
  On a Mac the build prints one warning (macOS's deprecated `cblas_sgemm` in the v2.4 fp32 reference oracle); it is
  harmless, the Linux gcc 14 build is the reference build.
- Two Python environments (3.11+, as numpy 2.5 needs; 3.12 and 3.13 tested), made like this from the checkout's
  root (the commands below run there too; `cenv` can live anywhere, then substitute its path):

  ```
  python3 -m venv cenv && cenv/bin/pip install onnx==1.23.2 onnxruntime==1.31.1 numpy==2.5.4 ai-edge-litert==2.3.0 soundfile==0.14.0
  cenv/bin/pip install -e .    # this checkout's Python package (bnneon) and the `bnc` command
  python3 -m venv tfenv && tfenv/bin/pip install tensorflow==2.21.0 tf2onnx==1.17.0 onnx==1.23.2 protobuf==7.36.2
  ```

  `cenv` runs everything in this repository (`bnc`, `gate_generic.py`, the tests, `relabel_for_birdnetgo.py`):
  `ai-edge-litert` is what `--tflite` and the gate's `.tflite` reference need, in the SAME interpreter, and
  `soundfile` what `--audio` needs; both optional. `tfenv` runs only `scripts/tflite_to_onnx.py` (section 3.1):
  TensorFlow is not needed anywhere else, so for a model you already have as ONNX (BirdNET v3.0, Perch v2) skip
  `tfenv`. Below, `python` means `cenv/bin/python`.
- Run the compiler as `bnc ...` (`cenv/bin/bnc`, or `bnc` with cenv activated). `python -I compiler/bnc.py ...` runs
  the same installed program and keeps working; `-I` ignores the current directory and the environment, so a
  downloaded file cannot inject code. To run a checkout you have not installed (another worktree), use
  `PYTHONPATH=<checkout>/python python -P -s <checkout>/compiler/bnc.py ...` (`-I` would ignore PYTHONPATH; `-P`
  still keeps the current directory off the import path).
  The editable install and `make` write into the checkout (`python/bnneon.egg-info`, `build/`); `bnc check` writes
  its scratch packages to `eval_tmp/` unless you pass `--workdir DIR`.
- The model as an ONNX file with ONE input, `float32 [1, n_samples]`, a fixed window length. Section 3 covers TFLite.

Sixty-second version:

```
make OUT=build
bnc check model.onnx --labels labels.txt --sample-rate 48000 --post sigmoid \
    --tflite model.tflite --windows-file windows.npy
```

prints a coverage report, compiles one package per CPU path, runs them through the engine on your windows, compares
with ONNX Runtime (and the TFLite file) and ends with `PASS`, `WARN` or `FAIL` plus the concrete next step. Without
`--windows-file`/`--audio` it uses four synthetic windows and without `--tflite` only ONNX Runtime: that is a smoke
test (it can PASS an fp16 package that real audio puts at WARN); give it real windows and the exact reference before
believing a PASS. Everything else in this guide is what to do when that line does not say PASS.

## 2. What the engine can run, and what it cannot

The compiler recognizes graph STRUCTURE (op types, constant shapes, verified constant values), never node names, and
lowers it to a fixed set of kernels. Today's coverage:

Front ends (lowered to an exact real FFT, not to the graph's fp32 DFT matrices):

- The BirdNET v2.4 dual STFT: min/max normalization, two branches of Gather framing, Hann window, cos-only DFT
  MatMul (also the truncated "dfttrunc" form), mel MatMul, square, Pow, mel reverse, channel Concat, channel BN.
  **Fixed geometry**: 144000 samples, n_fft/hop 2048/278 and 1024/280, 96 mel bins. Other geometries are
  recognized but refused by the lowering (`FE_STFT2 kernel supports the BirdNET v2.4 geometry only`).
- A framed log-mel (Perch style): identity-conv framing, stored window, zero pad to n_fft, the ONNX `DFT` op,
  magnitude, mel MatMul, floor, log, scale. n_fft a power of two in 16..2048, frame <= n_fft, <= 512 mel bins, any
  hop. The matrix-DFT form of the same graph ("no_dft" exports) is refused: use the DFT-op export.
- A dB image (BirdNET v3.0 style, section 2.1): reflect center padding, a Conv-framed DFT (a [2F,1,N] cos|sin
  kernel), power, mel MatMul/Einsum, Clip/Log/scale to dB, an optional top-db floor, optional mel-row Slice/reverse,
  min/max normalization, a bilinear half-pixel Resize, Expand to C channels and a per-channel (x - mean) / std.
  n_fft a power of two, any hop, every constant read from the graph. fp32 on every path.

Body:

- Dense 1x1 convs (stride 1, no pads) with folded BatchNorm, bias, swish or ReLU, any channel count (output channels
  are padded to a multiple of 8; the input must come from a conv, not straight from a 1-channel front end).
- A residual add ONLY fused into the 1x1 conv that produces one of its operands (MBConv style); a standalone Add
  between two activations has no kernel.
- Depthwise kxk convs, k in {3, 5, 7} square, equal strides 1 or 2, pads <= (k-1)/2 (SAME or VALID), **swish
  activation only**, channels a multiple of 8.
- A kxk VALID stem on a 1-channel image (Perch): square kernel, equal strides, swish|relu|none, fp32 math on both
  paths; its output halo (for a following depthwise) <= 3.
- Dense k x k convs (group 1, k <= 7, equal strides 1 or 2, any pads incl. TF-style asymmetric SAME), with bias,
  swish|relu|none and an optional fused residual: fp32 direct on the `neon-fp32` and `neon-int8` paths, or fp32
  Winograd F(4,3) for 3x3 stride-1 pad-1 layers with `--wino MIN_C`; no `neon-fp16` lowering.
- The v2.4 4x8 stride-2 stem with its 1x2 max|avg pool pair (that exact shape only; no other pooling has a kernel).
- Squeeze-excitation (mean, FC, **swish**, FC, sigmoid, gate; rd <= 512, C <= 4096), applied in place; the SE must
  consume the preceding depthwise conv's output and that output may have no other consumer.
- Layout-only Transpose/Reshape. Channel Concat is recognized but has no lowering (the v2.4 pool concat is fused
  into the stem).

Heads:

- Spatial mean over H,W to a vector (the embedding): unpadded input, channels a multiple of 4.
- Dense heads: a chain of MatMul (+ Add bias) or Gemm (default attributes) layers on the pooled fp32 vector, any
  width, with ReLU between them: a BirdNET-Analyzer head with "hidden units" (Dense N -> ReLU -> Dense classes)
  compiles end to end. The 1024 -> N linear head keeps the fixed-width kernel the gated packages use; every other
  shape runs on the general GEMV kernel (fp32 math; fp16-stored weights on the fp16 path above 64 outputs). No
  other activation on a vector (a final Sigmoid/Softmax is the host's: `--post`). Perch's 1536-wide embedding goes
  through the prototype head.
- The v2.4 post BN + ReLU + 3x3 VALID conv + ReLU + mean, only in exactly that shape ([3][8][192] -> 1024).
- GeM pooling (Clip eps, Pow p, spatial mean, Pow q) to a vector.
- SED attention heads (BirdNET v3.0): mean over one spatial axis, MaxPool1d + depthwise Conv1d, a per-time-step
  MatMul + ReLU, attention Conv1d -> optional Clip -> Softmax over time, class Conv1d, weighted sum, scale.
- Vector algebra after the pooling: Gemm, vector + vector, + - * / by constants. Every affine function of ONE pooled
  vector (e.g. a sum of several linear heads, scaled) is folded into one dense layer in float64 at compile time.
- ProtoPNet heads (L2-normalized cells, prototype MatMul, max over cells, class weights): cell count (H x W) and
  width multiples of 8.
- The package returns LOGITS. A final Sigmoid or Softmax in the graph is not compiled: name the tensor before it with
  `--logits` and tell the host what to apply with `--post`.

Not covered (each needs a recognizer rule, a lowering and usually a kernel, section 6): dense k x k convs on the
`neon-fp16` path, dense convs with k > 7 or stride > 2, depthwise convs with ReLU or no activation, dilated or
grouped-but-not-depthwise convs, pooling other than the v2.4 pair (GlobalAveragePool included), standalone
Add/Concat/Relu, Clip/ReLU6, HardSwish, LeakyRelu, Pad, Slice, an unfolded BatchNormalization node, LSTM/attention/
any sequence op, dynamic shapes, batch > 1, int8-quantized graphs (QDQ or QLinear ops: compile the fp32 export
instead), any op the report lists. BirdNET v3.0 as a whole: section 2.1.

`bnc report` answers "does the recognizer place every node"; a few recognized IR ops (standalone ADD, CONCAT,
ACT, POOL, a conv shape without a kernel) are refused one step later, by the lowering, with
`no <path> lowering for IR op X` or `conv k.. s.. pads .. has no <path> kernel`. `bnc check` runs both steps, so
use it rather than `report` to decide.

Before recognition the compiler canonicalizes exporter artifacts (python/bnneon/compiler/canon.py), exactly and verified against
ONNX Runtime on a window: an Unsqueeze/Squeeze pair around a MatMul, a float -> float Cast, a full-length
ReverseSequence (-> a reverse Slice), a constant zero Pad on H/W in front of a Conv (-> the Conv's pads), and
GlobalAveragePool (-> ReduceMean). The report's first line says what it rewrote; `--no-canon` turns it off. Only the
nodes the requested outputs (`--logits`, `--emb`) depend on count: a Sigmoid, Softmax or anything else after the
logits tensor is ignored (and a trailing Sigmoid/Softmax sets the default `--post`).

Numerics: the fp32 path (`neon-fp32`) runs on any AArch64 and reproduces ONNX Runtime to ~100 dB when the reference
computes its DFT exactly. The fp16 path (`neon-fp16`) needs ARMv8.2 fp16 arithmetic (Cortex-A76 class, Apple
silicon) and loses precision that depends on the model: ~55-60 dB for bird embeddings, ~45 dB for ultrasonic bat
embeddings, which is why the bat packages ship fp32 only. The int8 path (`neon-int8`, any AArch64) keeps fp32
activations and runs the 1x1 convs and heads in int8 (the classes `--int8` names): 50-57 dB on BirdNET v3.0 audio. Section 7.

### 2.1 BirdNET v3.0: three compiled packages

BirdNET v3.0 (the published fp32 ONNX files: the global model, 11,560 classes, and the regional slices such as
north-america-east, 800) compiles like any other model. The compiler recognizes its parts by structure: the
dB-image front end (`FE_DBIMG`: reflect padding, a Conv-framed DFT, power, mel, dB with a top-db floor, min/max
normalization, a bilinear Resize to 224 x 468, three normalized channels), dense k x k convolutions (the stem, stage 0
and every FusedMBConv; 47 % of the 6.2 GMAC), GeM pooling, the two SED attention heads and the sum of the linear
heads (folded into one dense layer at compile time). Three paths run it:

- `neon-fp16`: the dotprod + fp16 kernel family (`src/rt_ops_dot.c`, `python/bnneon/compiler/lower_dot.py`), for CPUs with fp16
  arithmetic and dotprod (Cortex-A76 class: Raspberry Pi 5). fp32 stem, the 24-channel 3x3 convolutions in fp16
  Winograd F(2,3), the dense 3x3 expansions in int8 (Winograd F(2,3) with per-tile dynamic quantization at stride 1,
  im2col with a per-tensor quantization at stride 2), every 1x1 convolution and the heads as int8 SDOT products with
  per-row dynamic activations, depthwise in fp16, SE / GeM / attention pooling in fp32. It takes no `--int8` or
  `--wino` option: the precision plan is the family's. This is the fast path on a Raspberry Pi 5.

The other two run on any AArch64 CPU:

- `neon-int8`: fp32 activations with the 1x1 convolutions and the heads as int8 matrix products (per-output-channel
  int8 weights, per-row dynamic int8 activations, SMULL/SMLAL/SADALP: no dotprod needed). `--int8` picks the layer
  classes (`pw` expansions, `pwl` projections, `head`, `sed`, `lin`, and `proto` for a prototype head, which
  BirdNET v3.0 does not have); the dense 3x3 convolutions, depthwise, SE, GeM, front end and softmax stay fp32. This
  is the fast path on a Raspberry Pi 4.
- `neon-fp32`: everything fp32; reproduces ONNX Runtime to ~99 dB.

`--wino 24` runs the dense 3x3 stride-1 convolutions with at least 24 input channels as fp32 Winograd F(4,3) on the
`neon-int8` and `neon-fp32` paths (without it they run direct, the parity form); on `neon-fp16` it has no effect, and
compile says so in a note. Check, then compile, one package per path (the shipped commands are in
`scripts/compile_models.sh`). The stock regional model is already the 800-class slice: its
`...-indices-b1.txt` file (the regional classes' indices in the global label list) is not needed for check or
compile; `--classes` is for cutting a slice yourself. For the Pi 4's `neon-int8` package:

```
bnc check birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx \
    --labels birdnet-v3.0-preview3.1-north-america-east-labels-b1.txt --sample-rate 32000 --post sigmoid \
    --emb embeddings --paths neon-fp32,neon-int8 --wino 24 --int8 pw,pwl,head,sed,lin
bnc compile birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx \
    --labels birdnet-v3.0-preview3.1-north-america-east-labels-b1.txt --model-id birdnet_v3.0_north-america-east \
    --family 30 --post sigmoid --sample-rate 32000 --emb embeddings --paths neon-int8 --wino 24 \
    --int8 pw,pwl,head,sed,lin --license-file weights/LICENSE-birdnet-v3.txt \
    --attribution "BirdNET v3.0, K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology, and Chemnitz University of Technology" \
    -o birdnet_v3.0_north-america-east-neon-int8.bnpk
BNN_THREADS=4 build/cli birdnet_v3.0_north-america-east-neon-int8.bnpk 0 windows.f32 out.f32   # 160000-sample windows at 32 kHz
```

For the Pi 5's `neon-fp16` package (no `--int8`, no `--wino`; note the fp16 license file, which states this path's
changes: with `LICENSE-birdnet-v3.txt` the package says the wrong thing and has another sha256 than the shipped one):

```
bnc check birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx \
    --labels birdnet-v3.0-preview3.1-north-america-east-labels-b1.txt --sample-rate 32000 --post sigmoid \
    --emb embeddings --paths neon-fp16,neon-fp32
bnc compile birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx \
    --labels birdnet-v3.0-preview3.1-north-america-east-labels-b1.txt --model-id birdnet_v3.0_north-america-east \
    --family 30 --post sigmoid --sample-rate 32000 --emb embeddings --paths neon-fp16 \
    --license-file weights/LICENSE-birdnet-v3-fp16.txt \
    --attribution "BirdNET v3.0, K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology, and Chemnitz University of Technology" \
    -o birdnet_v3.0_north-america-east-neon-fp16.bnpk
```

(Older command lines with `--wino 24` on `neon-fp16` still work and give the same bytes.)

`windows.f32` is n x 160000 float32 samples at 32 kHz in [-1, 1). The engine does not resample: audio at another
rate (44.1 or 48 kHz) must be resampled to 32 kHz first (for example `ffmpeg -i in.flac -ac 1 -ar 32000 rec.wav`).
From a 32 kHz mono recording, back-to-back 5 s windows (a host such as BirdNET-Go chooses its own overlap) of the
part from `start` to `stop` (in samples; a 1-hour file in full is about 460 MB of windows), and the top 3 classes of
each window through the Python binding:

```
python -I -c "import soundfile as sf; x, sr = sf.read('rec.wav', dtype='float32', start=1545 * 32000, stop=1565 * 32000); \
  assert sr == 32000 and x.ndim == 1; x[:len(x) // 160000 * 160000].tofile('windows.f32')"
python -I -c "import bnneon, numpy as np
with bnneon.Model('birdnet_v3.0_north-america-east-neon-int8.bnpk', lib='build/libbnneon.so', threads=4) as m:
    lab, w = m.labels(), np.fromfile('windows.f32', np.float32).reshape(-1, m.info.n_samples)
    for i, x in enumerate(w):
        c = m.confidences(m.infer(x)[0]); print(i * 5, [(lab[k], round(float(c[k]), 3)) for k in np.argsort(-c)[:3]])"
```

`check` judges the fp32 path at 60 dB (it measures ~99 dB) and the int8 path at 40 dB (50-57 dB on real audio, ~45 dB
on check's synthetic windows). On real audio the int8 path may end at WARN: a few windows change their top class or move a
confidence by more than 0.01 (dynamic per-row int8 activations; the legacy int8 engine moved one by 0.19 over the
gate's 92,829 windows and still passed). That WARN is expected, not a fault; the gate (section 9) decides. The package writes LOGITS (the host applies the sigmoid, `--post sigmoid`), in label-file
order, with the 1280-wide embedding when `--emb embeddings` is given. The four v3.0 packages compile in about 15 s on an
M-series Mac with a peak of ~5.4 GB of memory (the global fp32 package is 549 MB, most of it the 11,560-class head). Speed on a Raspberry Pi 4 (Cortex-A72, 1.5 GHz;
ms per window at 1 / 2 / 4 threads; `BNN_THREADS` in the CLI, `bnn_set_threads` in the API, outputs bit-identical at
every thread count timed): NA-east int8 950 / 525 / 330, fp32 1234 / 682 / 440; global int8 1110 / 598 / 382, fp32 1532 /
861 / 616 (stock ONNX Runtime fp32 NA-east: 2311 / 1406 / 927). Peak memory: NA-east int8 ~78 MB, global int8
~180 MB, global fp32 ~560 MB. Accuracy against fp32 ONNX Runtime on the 34.4 h gate set (gate V002, `results/MENU.txt`): every
package above passes for both models, with detections equal to fp32 in total; the int8 packages are the ones to ship
for a Pi 4 (the fp32 packages are the exact, slower alternative).

On a Cortex-A76-class CPU (Raspberry Pi 5) the `neon-fp16` package is the path: 128 ms (NA-east) and 148 ms (global)
per window on one thread, 54 and 72 ms on four, against ~360 ms for the `neon-int8` package there. `check` judges it
by the int8 floor (40 dB; it measures ~47 dB on real windows), as `neon-int8`: its line says `(int8 GEMMs: judged by
the int8 floor, 40 dB)`, and a WARN there (typically `max |dconf| 0.020 > 0.01` on real windows) is expected, as for
`neon-int8`: the gate (section 9) decides. Peak memory on a Pi 5: NA-east 81 MB,
global 336 MB. The package holds the weights in fp32 (132 MB for NA-east, 540 MB for global) and packs them when it
loads, then releases the file's pages. A model whose layers the family has no kernel for (for example a stem on an
odd-sized image, a depthwise conv other than 3x3, a head other than dense + two SED heads) is refused at compile time
with the reason, and still has its `neon-int8` and `neon-fp32` packages.

The `neon-fp16` family was taken from the hand-written v3 engine (the `.bnv3` file), which had the same speed on one
thread and no threads, and has been retired. The package reproduced its numbers exactly from the front end's image
onwards; the two front ends differ in the last bits of the image (2.4e-6 at most), which is why the package has its
own gate result. This library refuses a `.bnv3` (`BNN_ERR_PACKAGE`, naming the package to use); a libbnneon from
before the retirement still runs one.

License: CC BY-SA 4.0 plus the BirdNET Terms of Use (never NonCommercial; ShareAlike), carried
inside the file (`weights/LICENSE-birdnet-v3.txt` for the `neon-int8` and `neon-fp32` packages,
`weights/LICENSE-birdnet-v3-fp16.txt` for `neon-fp16`). In a BirdNET-Go build with the engine binding (not published
yet) a package is selected with `birdnetv3.backend: neon` and `birdnetv3.neonweights: <file.bnpk>` (ONNX Runtime is the fallback on
any engine failure) or as `birdnetv3.modelpath` itself (no fallback); it runs with `birdnet.threads` threads and its
own label file. The `bnn_info` family code is 30.

## 3. Get the model as ONNX

The compiler reads ONNX only. Exports it has been tested with: tf2onnx from TFLite and SavedModel (BirdNET v2.4,
BattyBirdNET, BirdNET-Go custom classifiers), the Perch v2 ONNX published for BirdNET-Go.

### 3.1 TFLite -> ONNX (BirdNET-Analyzer custom classifiers and the stock BirdNET files)

BirdNET-Go's "custom classifier" is the BirdNET v2.4 backbone plus a new dense head, trained with BirdNET-Analyzer
and exported as a `.tflite` with a label file. BirdNET-Go takes it through `birdnet.modelpath` / `birdnet.labelpath`
(or `BIRDNET_MODELPATH` / `BIRDNET_LABELPATH`): the extension picks the backend (`.tflite` or `.onnx`), the input's
last dimension must be 144000, there must be one output (logits) or two (logits + 1024 embeddings), the label count
must equal the output width, and labels are `Scientific name_Common name` per line (it splits on `_`, so a
`Common_Name_Genus_species` file from elsewhere must be rewritten). The model outputs LOGITS: BirdNET-Analyzer
exports without the sigmoid and BirdNET-Go applies its own, and the engine does the same (`--post sigmoid`).

Check `Params.csv` first: `Hidden units N` > 0 means the head is Dense N -> ReLU -> Dense classes, which compiles
(section 2); `BirdNET Model version V2.4`, when the column exists (older BirdNET-Analyzer versions omit it), is the
backbone this engine knows; if it is absent, `check` tells you (it recognizes the v2.4 front end or it does not).
Convert with tf2onnx in a venv of its own (only this step needs TensorFlow; tested pair: tf2onnx 1.17.0, tensorflow
2.21.0, onnx 1.23.2, protobuf 7.36.2, numpy 2.5.4, Python 3.12 or 3.13, opset 17):

```
python -m venv tfenv && tfenv/bin/pip install tensorflow==2.21.0 tf2onnx==1.17.0 onnx==1.23.2 protobuf==7.36.2
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python tfenv/bin/python -I scripts/tflite_to_onnx.py model.tflite model.onnx 17
```

Two things the stock `python -m tf2onnx.convert --tflite` cannot do, which the script handles:

- Every BirdNET v2.4 graph lowers its spectrogram to `RFFT2D -> Squeeze -> Cast(complex64 -> float32)` (it keeps the
  real part of the STFT). tf2onnx only converts `RFFT2D` when `ComplexAbs` consumes it and stops with
  `Current implementation of RFFT2D only allows ComplexAbs as consumer not {'Squeeze'}`. The script registers one
  rewriter that replaces the `RFFT2D` by a MatMul with the cosine matrix `C[n,k] = cos(2 pi n k / N)`, exactly
  `Re(rfft(x))`. The result is NOT yet the form the recognizer matches: it still carries the Unsqueeze/Squeeze pair
  around that MatMul, a float -> float Cast, `ReverseSequence` for the mel reverse, standalone `Pad` nodes before
  the stride-2 depthwise convs and `GlobalAveragePool` for the means; the compiler's canonicalization pass
  (section 2) rewrites all of these, so `check` takes the converter's output as it is. The ONNX is ~10 MB bigger
  than the `.tflite` because of the cosine matrix; the engine replaces it with an FFT anyway.
- Success is the line `tf2onnx 1.17.0 opset 17 nodes N`; the TensorFlow deprecation and `W0000`/`I0000` log lines
  before it are harmless.
- With protobuf's default (upb) backend every tf2onnx optimizer pass leaks ~40 MB and the conversion is killed in a
  2 GB container; `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` keeps it under ~0.8 GB and ~6 s.

Then check the result with ONNX Runtime before going on (the compiler does this too, but a failure here is a
conversion problem, not an engine problem):

```
python -I - <<'EOF'
import numpy as np, onnxruntime as ort
s = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
i = s.get_inputs()[0]; print(i.name, i.shape, i.type, [(o.name, o.shape) for o in s.get_outputs()])
x = (np.random.default_rng(0).standard_normal((1, 144000)) * 0.05).astype(np.float32)
print(s.run(None, {i.name: x})[0].shape)
EOF
```

Expect an input `['unk__N', 144000] tensor(float)` (a dynamic batch dimension is fine; a dynamic window length is
not) and an output `[.., n_classes]`; then compare with the TFLite interpreter on a few windows (expect > 90 dB logit
SQNR between the two; `bnc check --tflite` repeats that comparison against the engine). Pitfalls:

- A `.tflite` saved with the sigmoid in the graph gives a double sigmoid in BirdNET-Go and, in the engine, a
  `Sigmoid` node the report refuses: export without it, or compile with `--logits <the Add before it>`.
- tf2onnx adds an unused `ai.onnx.ml` opset import and generated node names that differ between runs: harmless
  (the compiler matches structure, not names); the bytes of two conversions differ, their outputs do not.
- Labels: the compiler and the runtime check only the COUNT (one non-empty line per class, output order; a first
  line `inat2024_fsd50k`, Perch v2's catalog header, is not a class) and ship the file's bytes and sha256 in the
  package; nothing checks the names. Save it as plain text with `\n` or `\r\n` line ends: the compiler refuses a
  file that a host could number differently (other line breaks, a NUL byte, lines blank only by other white space;
  README, "Input and output per model"). The `Scientific name_Common name` form
  matters to BirdNET-Go, which splits each line on `_` (an optional third field is a code), so a file published as
  `Common_Name_Genus_species` (BirdNET-Analyzer's default) must be rewritten BEFORE compiling:

  ```
  python -I scripts/relabel_for_birdnetgo.py Labels.txt Labels_go.txt [--sci "Eurasian_Golden_Oriole=Oriolus oriolus"]
  ```

  turns `Barn_Swallow_Hirundo_rustica` into `Hirundo rustica_Barn Swallow`, keeps lines already in the target form,
  writes a non-species class (`Anthropogenic`, `Background_birds`, `Nothing`: one word, or not all capitalized) as
  `Name_Name` (`Anthropogenic_Anthropogenic`) and lists them, and REFUSES to write a line of two or more capitalized
  words that has no scientific name (`Eurasian_Golden_Oriole`: most likely a species the file lacks the name for)
  until you give it with `--sci` or accept the doubled form for all of them with `--allow-doubled`; `--check` only
  reports (flags are accepted before or after the file names). How BirdNET-Go then treats a
  `Name_Name` class (range filter, species lookups) is BirdNET-Go's business and not verified here: it reads the line
  as scientific name `Name`, common name `Name`. `check` and `compile` warn,
  for a BirdNET v2.4-backbone model, when a label file is not in that form (one or two `_`, a capitalized
  scientific part), naming the first offending line, by the same rule the rewrite tool uses; the stock v2.4 file
  passes, BattyBirdNET's noise classes (`audiomoth`, `NOISE`, 0-10 lines per region; `compile --family 224` does not
  warn) would be flagged, Perch's own scientific-name list is a different contract and is not checked. The rule is a
  heuristic: `--check` first and read the printed rewrites. BirdNET-Go treats `Name_Name` classes like
  any other class (they are reported under that name). CRLF line endings are kept as they are. A package's embedded
  file comes back out with `build/cli pkg.bnpk --labels > labels.txt`.
- Measured with the pinned versions, no hand edits: a published BirdNET-Analyzer custom classifier (41 classes,
  hidden units 1024): fp32 102.0 dB vs its TFLite, fp16 55.5 dB, PASS on both paths; a 3-class linear head trained
  with birdnet-analyzer 2.4.0: fp32 96.6 dB PASS, fp16 45.6 dB WARN (a tiny random-ish head amplifies the fp16
  loss: ship fp32 for such heads).
- Keep the `.tflite` next to the `.onnx`: it is the exact-DFT reference for `--tflite` (section 4), and the license
  of the compiled package is the `.tflite`'s (a BirdNET v2.4 derivative: CC BY-NC-SA 4.0, whatever the head's
  authors wrote on their record).

### 3.2 SavedModel / Keras

`python -m tf2onnx.convert --saved-model dir --output model.onnx --opset 17`. Fold BatchNorm (tf2onnx does by
default); the compiler folds a remaining per-channel Mul+Add after a conv but not a `BatchNormalization` op.

### 3.3 PyTorch

`torch.onnx.export(model, torch.zeros(1, n_samples), "model.onnx", opset_version=17, input_names=["pcm"])` with the
front end inside the graph (`torch.stft` exports to the ONNX `DFT` op from opset 17; the framed log-mel matcher
expects it).

## 4. One command: `bnc check`

```
bnc check model.onnx [--labels labels.txt] [--sample-rate HZ] [--post sigmoid|softmax]
    [--logits TENSOR] [--emb TENSOR] [--append head.onnx --at TENSOR] [--paths neon-fp16,neon-fp32]
    [--int8 CLASSES] [--wino MIN_C] [--audio a.wav b.wav ...] [--windows N] [--tflite ref.tflite] [--cli PATH | --docker CONTAINER]
    [--workdir DIR] [--keep] [--json [FILE]]
```

What it does, in order, stopping at the first failure:

1. Loads the ONNX, runs it once through ONNX Runtime (shapes), folds constants, canonicalizes exporter artifacts
   (verified against ONNX Runtime), runs the recognizer. Prints the IR (one line per fused op) and every unsupported
   node the requested outputs depend on, with its attributes, input/output shapes, the nearest supported pattern
   and the next step; nodes that fail only because an earlier one did are counted, not listed (`--verbose` lists
   them). Nodes after the logits tensor are ignored.
2. Checks the label file length against the logits width.
3. Compiles one package per path into the work directory (`eval_tmp/check-<model>/` by default).
4. Builds the runtime (`make OUT=build/check`) unless `--cli` names a built CLI, `--docker` names a container
   that mounts YOUR checkout at `/eng` (it refuses one that mounts another tree), or `--docker-image` names an
   image to start a throwaway container from (removed afterwards). The recipe for such a container is in the
   repository: `scripts/eval_container.sh build` builds `integration/Dockerfile.eval` (debian:trixie + gcc 14,
   make, libc; nothing of the repository baked in), `scripts/eval_container.sh start NAME` runs it with
   `--memory 2g` and the checkout at `/eng`, `stop NAME` removes it. The container build is the Raspberry Pi
   compiler (gcc 14, Linux), so its OUTPUTS are the ones a Pi would produce (its timings are not). Runs each
   package on the windows:
   `--windows-file w.npy|w.f32` (an array of n x n_samples float32, all of them unless `--windows N`), else cut
   from `--audio` files (at the model's rate, no resampling, near-silent windows skipped, at most `--windows N`,
   default 4), else N synthetic windows (chirps + noise).
5. Runs ONNX Runtime (and the TFLite file if `--tflite`) on the same windows.
6. Per path: logit SQNR, max |dconf| after `--post`, top-1 agreement; plus engine-fp16 vs engine-fp32 SQNR, the
   fp16 path's own loss without any reference.

Verdicts: PASS needs >= 60 dB (fp32 path) / >= 50 dB (fp16 path), max |dconf| <= 0.01 and full top-1 agreement;
WARN between 35 and the PASS line or |dconf| <= 0.05; FAIL below 35 dB, |dconf| > 0.05 or non-finite logits. Each
path's line names the SQNR it is judged by and any allowance in brackets, and there are exactly two allowances, both
only without `--tflite` on a graph whose DFT is an fp32 matrix (the BirdNET v2.4 family), where ONNX Runtime can be
the less exact side: the fp32 path is accepted at >= 40 dB (`[reference-limited: ... 40 dB accepted, pass --tflite
...]`), and the fp16 path is then judged by its own loss against the engine's fp32 path (`[judged by its own loss
...]`, only when the fp32 path itself sits below 60 dB from ONNX Runtime, i.e. the reference is the limit), never
by a lowered line. With `--tflite` there is no allowance: the exact reference alone decides (expect ~100 dB on the
fp32 path). How far the ONNX reference sits from the exact result depends on the export (the stock v2.4
ONNX: ~63 dB; a tf2onnx conversion with float64 cosines: ~109 dB), which is why it is measured, not assumed. With
`--post none` max |dconf| is in raw output units and its lines mean little. A WARN/FAIL on the fp16 path with a
PASS on fp32 means "ship fp32, or screen the fp16 precision" (section 7), except a WARN on an fp16 path whose GEMMs
run in int8 (BirdNET v3.0's; judged by the int8 floor, as `neon-int8`), which check sends to the gate (section 9); a FAIL on the fp32 path means a compiler
or kernel bug, not precision (section 8).

Reading the output on two models (synthetic windows, Apple M-series):

```
$ bnc check birdnet-v2.4.onnx --logits output --labels labels.txt --sample-rate 48000 --post sigmoid --tflite BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite
...
68 IR ops; 0 unsupported nodes (0 primary, the rest downstream); 0 uncovered nodes
logits: 'output' [6522]; window 144000 samples
ONNX Runtime vs TFLite: 63.4 dB
path neon-fp16: 59.9 dB vs ONNX Runtime, 62.4 dB vs TFLite, max |dconf| 0.0028, top-1 4/4: PASS
path neon-fp32: 63.4 dB vs ONNX Runtime, 100.4 dB vs TFLite, max |dconf| 0.0017, top-1 4/4: PASS
engine fp16 vs engine fp32: 62.4 dB (the fp16 path's own loss, reference-free)

PASS: every path agrees with ONNX Runtime on these windows; next: compile the shipping packages ...
```

```
$ bnc check perch_v2_fp32.onnx --labels perch_v2_labels.txt --sample-rate 32000 --post softmax
path neon-fp16: 59.1 dB vs ONNX Runtime, max |dconf| 0.0024, top-1 4/4: PASS
path neon-fp32: 109.0 dB vs ONNX Runtime, max |dconf| 0.0000, top-1 4/4: PASS
```

A BirdNET-Go custom classifier straight from `scripts/tflite_to_onnx.py` (41 classes, hidden units 1024, eight real
windows, Apple M-series, 1.4 s):

```
$ bnc check NFC.onnx --labels NFC_Labels.txt --sample-rate 48000 --post sigmoid --tflite NFC.tflite --windows-file eval_windows.f32
canonicalized: Unsqueeze/Squeeze around a MatMul x2, float->float Cast x2, ReverseSequence -> Slice x2, zero Pad folded into a Conv x4, GlobalAveragePool -> ReduceMean x14 (ONNX Runtime output unchanged: relative max |d| 5.5e-07)
69 IR ops; 0 unsupported nodes (0 primary, the rest downstream); 0 uncovered nodes
ONNX Runtime vs TFLite: 109.4 dB
path neon-fp16: 55.5 dB vs ONNX Runtime, 55.5 dB vs TFLite, max |dconf| 0.0039, top-1 8/8: PASS
path neon-fp32: 101.3 dB vs ONNX Runtime, 102.0 dB vs TFLite, max |dconf| 0.0000, top-1 8/8: PASS
PASS: every path agrees with ONNX Runtime on these windows; next: ...
```

On a CPU that cannot run a path (the `neon-fp16` one on a Raspberry Pi 4), check still compiles it, prints a `WARN`
naming its scratch package and the one to use, and leaves it out of the verdict: `PASS: every path this CPU can run
agrees with ONNX Runtime on these windows; skipped: neon-fp16 (cannot run here); next: ...`. Pass `--paths neon-fp32`
to check only the path that runs there. A path you name with `--paths` that this CPU cannot run is not left out
silently: the check ends with `WARN` (exit 1) and names it (`WARN: --paths names neon-fp16, which this CPU cannot run
(not checked); ...`); when no named path runs, it ends with `FAIL: no path ran on this CPU` (exit 6).

`--quiet` drops the one-line-per-op IR listing (the problems, path lines and verdict remain). `--json FILE` writes the same facts as JSON (`status`, `exit`, `next_step`, `coverage.unsupported[]` and
`coverage.primary[]`, `paths.<name>.{compiled,error,ran,sqnr_db,max_dconf,top1_agree,verdict,why}`,
`fp16_vs_fp32_sqnr_db`, `warnings[]`, `steps[]`); `--json -` prints it to stdout with the text on stderr. The file is
written on every exit path, including an internal error (`status` ERROR, exit 8). The exit code is the verdict
(appendix B). `check` does not take `--classes`: check the full head, then compile the slice.

`check` holds the model and ONNX Runtime's intermediate tensors in memory: expect a peak of several GB for a large
model (3.2 GB RSS for the 149 MB BirdNET v3.0 north-america-east file on a Mac; 3.3 GB for the 557 MB global file),
so run it on the host, not in a 2 GB container.

A few windows are a smoke test, not a gate: they catch wrong lowerings, dtype mistakes and gross fp16 loss, not a
0.3 pp recall shift. Section 9 is the gate.

## 5. Compile the shipping packages

```
make OUT=build                      # the CLI used below (check keeps its own copy in build/check/)
bnc compile model.onnx --labels labels_go.txt --model-id my_model --family 24 \
    --post sigmoid --sample-rate 48000 --paths neon-fp16 --license-file LICENSE.txt \
    --attribution "Model name, authors, license" [--logits TENSOR] [--emb TENSOR] [--hop SAMPLES] \
    [--notes TEXT] -o my_model-neon-fp16.bnpk
```

and the same with `--paths neon-fp32 -o my_model-neon-fp32.bnpk`. One package per path keeps each file small and
lets a host pick by CPU (`aarch64-a76` -> fp16, plain `aarch64` -> fp32); without `--paths` both go into one file,
which the compiler says. `compile` refuses to write a package without `--license-file` and `--attribution`, without
a `--family` it cannot infer, or with a sample rate the family cannot take (24 -> 48000 Hz, 224 -> 256000 Hz),
unless `--force`; for a BirdNET v2.4 backbone it infers `--family 24` and `--post sigmoid` and says so; a
BattyBirdNET head on that backbone is `--family 224 --sample-rate 256000`.

A license file for a custom classifier is two paragraphs: what the file is and who made it, and the license it
inherits. The downloaded files (`.tflite`, labels, `Params.csv`) do not carry the authors, the publication date or a
DOI: those come from the record you downloaded from (a Zenodo record page lists authors, version, publication date
and DOI; a Hugging Face model card or a GitHub release has the same in its README and metadata; a bare download
link has none), and the BirdNET-Analyzer version from `Params.csv` when it has the `BirdNET Model version` column
(older files record neither it nor the training tool's version). Mandatory, or the statement is not a license: the
backbone's license (CC BY-NC-SA 4.0 for BirdNET v2.4, with the BirdNET attribution), the head record's license, and
that the package is a derivative of both. For what you cannot establish, write it as such ("authors: the authors of
Zenodo record <DOI>", "training tool version not recorded in the published files", "date: the record's
publication date, <date>") rather than guessing; a model whose source and license you cannot find is not one to
compile. For a BirdNET-Analyzer classifier:

```
<Model name> (<N> classes), a custom classifier trained with BirdNET-Analyzer <version> by <authors>, published at
<URL or DOI>, <date>. Backbone: BirdNET v2.4 (BirdNET_GLOBAL_6K_V2.4), K. Lisa Yang Center for Conservation
Bioacoustics, Cornell Lab of Ornithology, and Chemnitz University of Technology (Kahl et al. 2021).
License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/), inherited from the BirdNET v2.4
backbone, whatever the head's own record says; non-commercial use only, derivatives keep this license. Changes:
compiled to a bnneon package (layout, packing, fp16 rounding on the fp16 path); no retraining.
```

(`weights/LICENSE-birdnet-v24.txt` is the stock example.) `--attribution` is the one-line credit a host shows.

- `--model-id`: a short identifier the host shows and keys on. Never reuse a stock id (`birdnet_v2.4`, `perch_v2`,
  ...) for a retrained or custom head: BirdNET-Go treats a 6522-class v2.4 package with the id `birdnet_v2.4` and a
  stock label file as the stock model and stores its detections under the stock model row. `--family`: an integer the host may use to recognize
  a model family (24 BirdNET v2.4, 120 Perch v2, 224 BattyBirdNET; 30 BirdNET v3.0, section 2.1; 0 for a new one). BirdNET-Go refuses families it
  does not know, so a custom classifier on the v2.4 backbone is compiled with `--family 24 --post sigmoid`.
- `--hop SAMPLES`: the model's native stride between windows (e.g. 144000 for 3 s windows with no overlap); it is
  informational metadata (`bnn_info.hop_samples`), 0 = the host decides. `--sample-rate HZ` is what the host must
  feed (48000 for BirdNET v2.4 models).
- `--post`: what the host applies to the logits (`sigmoid` for multi-label BirdNET heads, `softmax` for Perch);
  default: the Sigmoid/Softmax the graph itself applies to the logits tensor, else sigmoid (the choice is printed).
- `--logits`: the pre-activation tensor (default: the output of the dense or prototype head, or the input of a final
  Sigmoid/Softmax when only that is a graph output). `--emb`: the embedding tensor (default: the last spatial
  mean; a hidden layer's activated output can be named too).
- `--labels`: the label file goes into the package (`bnn_probe_meta "labels"`, `build/cli pkg --labels` prints it)
  with its sha256 (`bnn_info.labels_sha256`), so a host can verify it uses the right one (`bnn_verify_labels`,
  `build/cli pkg --verify-labels file`). Rewrite it into BirdNET-Go's form first (section 3.1).
- `--license-file`, `--attribution`: the upstream model's license text, verified at its source, and the credit line.
  A compiled package is a derivative of the model and carries the model's license (for BirdNET v2.4 and everything
  trained on it: CC BY-NC-SA 4.0).
- `--append head.onnx --at TENSOR`: stitch a separate head graph onto a backbone tensor (BattyBirdNET: the v2.4
  embedding model + a dense head), `--classes idx.txt`: keep a subset of classes (regional slices).
- fp16-path policies: `--dense f16|f32` (default: fp16 head weights above 64 classes, exact fp32 below),
  `--proto-s8` (int8 dotprod prototype head), `--proto32-h16` (fp16-stored prototypes on the fp32 path).
- `--paths neon-int8` (any AArch64; fp32 with int8 1x1 convs and heads) with `--int8 CLASSES` (default
  `pw,head,sed,lin,proto`; add `pwl` for the projections; `proto` is the prototype head, and `--proto32-h16` cannot be
  combined with it), and `--wino MIN_C` (fp32 Winograd F(4,3) for dense 3x3 stride-1
  convs with >= MIN_C input channels, `neon-fp32` and `neon-int8`): the BirdNET v3.0 options, section 2.1.

Packages are byte-reproducible from the same inputs and compiler: record the sha256 and the exact command, and
keep the `.onnx` the package was compiled from (and the `.tflite` it came from): a second tf2onnx conversion gives
a different `.onnx` (node names), whose sha256 is inside the package, so the package cannot be reproduced bit for bit
from the `.tflite` alone. The metadata key `compiler` names the compiler version that wrote the package; the runtime
quotes it in its error messages.

Inspect a package: `build/cli my_model-neon-fp16.bnpk --info` (metadata, the best path for this CPU, whether each
mode initializes and why not); `--labels` prints the embedded label file. Run it:
`build/cli my_model-neon-fp32.bnpk 2 windows.f32 out.f32 [reps] [warmup]` (mode 0 = the best path this CPU can run,
2 = neon-fp32, 3 = neon-fp16, 4 = neon-int8; `BNN_THREADS=N` in the environment runs it on N threads; `windows.f32` holds n x n_samples float32, made from a `.npy` with
`python -c "import numpy as np; np.load('w.npy').astype('<f4').tofile('windows.f32')"`; `out.f32` receives
n x n_classes logits; the timing line is ms per window) and `build/cli` alone prints the usage.

(BirdNET v3.0 packages use the `birdnetv3` keys instead: section 2.1.) Using the package in a BirdNET-Go build with the engine binding (not published yet; that build's documentation
describes the settings): set
`birdnet.modelpath` to the package and `birdnet.labelpath` to the label file, exactly like a custom `.tflite`:

```
birdnet:
  modelpath: /data/model/my_model-neon-fp32.bnpk     # or BIRDNET_MODELPATH
  labelpath: /data/model/my_model_labels_go.txt      # or BIRDNET_LABELPATH: the file compiled in (same bytes)
  # neonlibrarypath: /usr/lib/libbnneon.so           # optional; else /usr/lib, /usr/local/lib, /usr/lib/aarch64-linux-gnu, the linker path
```

The package choice is yours, by file: `-neon-fp16.bnpk` for a Cortex-A76 class CPU (Raspberry Pi 5), `-neon-fp32.bnpk`
for any other 64-bit ARM (Pi 4; it also runs on an A76, slower). BirdNET-Go checks the package against the model
contract (48 kHz, 144000 samples, one output per label, logits, post sigmoid) and the label file's sha256 against
the package's (`build/cli pkg --labels` prints the embedded file; a mismatch is refused naming both hashes), records
detections under the custom variant with the package as the classifier path, and has no fallback for a package: a
build without the bnneon tag, a `libbnneon.so` that cannot be loaded (`birdnet.neonlibrarypath`, else the standard
library locations; a library installed after start needs a restart), an API-generation-1 library, or a library too
old for the package (opcode 15 `gemv` of a hidden-layer head, the format version) each stop the classifier with a
message naming the package, the library and the install guide; the too-old case passes the engine's own text through
(`unknown opcode 0xf ... hint: update libbnneon`). A package file that does not exist is handled like a missing
custom `.tflite`: the installed or built-in model runs, with a notification. The `birdnet.backend: neon` +
`birdnet.neonweights` route remains for the stock weights only. Getting a BirdNET-Go with the engine comes down to
"build BirdNET-Go with the `bnneon` build tag and install `libbnneon.so`"; the build recipe (a backend-only
`go build -tags noembed,skipfrontend,bnneon` on an arm64 host, or a container image with the engine) and a tool to
try a package on a recording before deploying are in that build's documentation.

## 6. When the recognizer says no: extending the compiler

The report lists each unsupported node like this:

```
unsupported op Conv at node 23 Conv 'model/block_2/dwconv/Conv': dilated conv
    attributes: {'dilations': [2, 2], 'group': 72, 'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1], 'strides': [1, 1]}
    inputs: pw_out [1, 72, 48, 128], W [72, 1, 3, 3] (const), b [72] (const)
    outputs: dw_out [1, 72, 48, 128]
    nearest supported pattern: dense 1x1 conv (...), depthwise kxk (k 3|5|7, stride 1|2, SAME or VALID pads, swish), ...
    next step: add a recognizer rule in python/bnneon/compiler/recog.py r_conv and a lowering in python/bnneon/compiler/lower.py l_conv / l_dwconv ...
```

Fix the PRIMARY nodes first (`primary` in the JSON); the downstream ones usually disappear. Three layers can need
work, from cheapest to dearest:

1. **Export differently.** Many "unsupported" nodes are export artifacts: an unfolded BatchNormalization, a Gemm with
   `transB`, a Softmax in the graph, a dynamic window length, a matrix DFT. Re-export, or edit the ONNX graph
   (`onnx.helper`), rather than teaching the engine a new op.
2. **Recognizer rule** (`python/bnneon/compiler/recog.py`). `Recognizer.rule(n)` dispatches on `op_type`; a rule verifies the
   structure (shapes, constants), appends an IR op dict (`op`, `ins`, `out`, parameters, `nodes` = the ONNX node
   names it covers) and marks the nodes covered (`self.cov`). Raise `Unsupported("why")` with a precise reason when the
   shape is close but not right: that text is what the next person reads. Patterns that span several nodes (swish, SE,
   the front ends) are matched by walking consumers (`only_cons`). Add your pattern to `NEAREST` so the report can
   point at it.
3. **Lowering** (`python/bnneon/compiler/lower.py`). `Lowering.l_<irop>(i, o)` (the IR op's name in lower case) allocates buffers (`self.buf(dtype, h, w, c, ld, halo)`),
   packs tensors (`self.T(name, array)`; GEMM weights through `gemm_tensors`, which produces the fp16 or fp32 panel
   layout the kernels expect) and emits op records (`self.emit(opcode, buffers, tensors, ints)`). Buffers get memory
   slots by lifetime automatically. The op's int parameters are the kernel's contract: read the kernel's `init` to see
   what it checks. If no kernel fits:
4. **Kernel** (`src/rt_ops_f32.c` for fp32 math, baseline ARMv8-A objects; `src/rt_ops_f16.c` for fp16 / dotprod,
   Cortex-A76 objects). A kernel is `{init, run, fini, name, outmask, optmask}`. `init` must validate EVERY buffer,
   tensor and parameter it will touch, using `rt_need` (buffer dtype and size), `rt_need_t` (tensor bytes),
   `rt_need_args` (record shape), `RT_REQ(c, op, condition)` (geometry) and `rt_oom` for allocations: each records a
   message saying what was expected and found, and the loader adds the op index, kernel and source node. A bare
   `return -1` is reported as "init failed without a message (kernel bug)". Append the opcode to the enum in
   `src/rt.h` (never renumber), add the entry to the registry table, and the matching `OP[...]` in `lower.py`.
   Kernels only ever extend the shared GEMM files (`gemm_f16.c`, `gemm_f32.c`) additively. After touching shared
   code, run the bit-identity suite (appendix A) and the error tests (`python -I tests/test_errors.py ...`).

Then run `check` again. Keep the loop short: the report after each change tells you the next primary node.

### 6.1 Onboarding a new model family

The loop above fixes one node at a time. When `check` stops at node 0 or the report lists dozens of primary nodes,
the model belongs to a family the compiler does not know, and the first job is to size the gap, not to fix node 0.

1. **Size the gap.** When the front end fails, `check`/`report` print `behind the front end, op types no body rule
   handles: ...` (JSON `no_body_rule`); `report --verbose` lists every node. That census misses shapes of known op
   types without a kernel (a dense 3x3 Conv counts as `Conv`), so also survey the graph once with `onnx` (op type,
   kernel shape, group, strides per Conv; the constants of the front end) and sort what is missing into front end
   (graph input to the first conv), body (convs, activations, SE, residuals) and head (pooling to logits). Write the
   table and the MAC share of each missing layer class down before choosing a route.
2. **Choose the route.**
   - Re-export when the gap is export artifacts (section 6 step 1): cheapest, hours.
   - Extend the compiler when the missing pieces are a few reusable patterns: a front-end matcher in `recog.py`
     (`fe_stft2` / `fe_logmel` / `fe_dbimg` are the templates; it lowers to an FE op with float64-derived tables and checks itself
     against ONNX Runtime), new IR ops with their lowering in `lower.py`, and kernels. Days per layer class; the
     result is generic and every later model of that shape compiles.
   - A hand-written engine (as BirdNET v3.0's `.bnv3` engine was, now retired) only when a model is worth
     model-specific numerics the registry cannot express on a given CPU class.
   Worked example, BirdNET v3.0 (both routes exist). Missing at first were the front end (power, mel, dB, top-db,
   min/max, Resize to an image, three channels), dense multi-channel 3x3 convs (47 % of its 6.2 GMAC), GeM and two
   SED attention heads. The legacy engine came first, for the Cortex-A76 (int8 Winograd F(2x2, 3x3), int8 SDOT
   pointwise layers, fp16 parts: 130-150 ms per window on a Pi 5, gated). The compiler route came second, for CPUs
   without dotprod and fp16 (Pi 4): one front-end matcher (`fe_dbimg`), IR ops GEM and SED, a dense k x k lowering,
   a compile-time fold of the summed heads, and fp32 kernels checked against ONNX Runtime one op at a time; then a
   thread pool, fp32 Winograd F(4,3), and int8 1x1/head GEMMs without dotprod as a per-layer-class policy (section
   2.1). On a Pi 4 the int8 package is 2.4-2.8x faster than stock ONNX Runtime (1-4 threads), and on a Pi 5 still ~2.7x slower on one core than the
   hand-written engine. The third step closed that gap inside the registry: the hand-written engine's kernels became
   a kernel family of their own (the dotprod + fp16 family, `src/rt_ops_dot.c`: nine ops that wrap those kernels and
   their packers in the engine's call order) with its own lowering class (`python/bnneon/compiler/lower_dot.py`, chosen in
   `cmd_compile.lowering_class` for the `neon-fp16` path of a dB-image model). The package was proved bit-identical to the
   engine when both are fed the same image (a test harness that linked the retired engine with the package's front
   end), has the engine's speed, and runs on threads; the hand-written engine was then retired. That is the general recipe for model-specific numerics: put them in a
   kernel family and a lowering class that refuses, with a reason, whatever the family cannot run.
3. **Order of work.** fp32 kernels first, each new op checked against ONNX Runtime on real windows before the next
   (`check`, `--logits <intermediate tensor>` for vectors, section 8); fp16 or int8 only after the fp32 path passes,
   as a per-layer-class policy (section 7). Front-end patterns go into `recog.py` next to the three matchers, verified
   numerically against the graph's own subgraph. A new head (GeM: clip, pow p, mean, pow 1/p; an SED attention
   head: per-time-step fc, attention softmax over time, weighted sum) is a recognizer pattern on the pooled tensor plus
   one IR op and one kernel per path, like the prototype head. Registry rules: append opcodes to the enum in
   `src/rt.h`, never renumber or reuse one; extend the shared GEMM files additively; when a file format changes in a
   way an older reader would misread, bump the version the older reader checks so it refuses the file (the retired
   `.bnv3` format's version 2 for heads wider than 2048 classes was the example: older libraries rejected it with
   `bad version` and a host fell back instead of overflowing).
4. **Must not break.** After every change under `src/`: a bit-identity check of every gated model's outputs (appendix A),
   `tests/test_errors.py`, `check` on the stock models; before scoring a candidate, a declared gate (section 9); after
   a large change (a new family, a new path), a blind run of this guide by someone who did not write the change.
5. **Size assumptions.** A model bigger than the gated ones meets limits nobody hit before. Grep for fixed sizes and
   reused buffers (`2048`, `4096`, `MAX_`, `xalloc(sizeof(float) *` with a constant, `int` products of dimensions)
   and check each against the new class count, channel count and window length: the retired v3 engine reused its
   2048-float SE scratch as the logits buffer, which overflowed silently on the 11,560-class BirdNET v3.0 global head
   on Linux and happened to survive on a Mac. Run a new size in the arm64 container (gcc), not only on a Mac.

## 7. Precision: fp32 first, fp16 second

Always get the fp32 path to PASS first; it is the exact reference for everything after. Then look at the fp16 path:

- `engine fp16 vs engine fp32` in the check output is the fp16 path's loss on its own. >= 55 dB is normal for a
  bird classifier; < 50 dB means some layer class is fragile in fp16.
- The fp16 path is a per-layer-class policy, not a switch: the stems already run fp32 math on both paths, dense
  heads with <= 64 classes run in fp32, the depthwise and SE sums accumulate in fp32. New policies go into
  `Lowering` (`self.policy`, chosen by compile flags) so a package stays explicit about what it does.
- To find the fragile class, fake-quantize one class at a time in an fp32 reference (this repository's
  `bench/perch_oracle.c` with `PERCH_Q` is the worked example) and compare SQNR per class. The BattyBirdNET classifiers
  are the worked case
  where that readout said "every class contributes" (fp16 storage alone 59.7 dB, all fp16 weights 54.6, full fp16
  arithmetic 45.1 dB on the embedding, no single-class split recovering it), so they ship fp32 only.
- Do not ship an fp16 package that only PASSes on synthetic windows: the gate (section 9) decides.
- The int8 path (`neon-int8`) is a per-layer-class policy too: `--int8` names the classes (`pw`, `pwl`, `head`,
  `sed`, `lin`, `proto`). If its check or gate fails, drop a class, the projections (`pwl`) first: on BirdNET v3.0 NA-east the
  default `pw,head,sed,lin` (pwl in fp32) gave 57.0 dB, top-1 0.992 and 299 flips over the gate's 92,829 windows,
  against 53.0 dB, 0.986 and 468 for all five classes; both passed, and the all-class package is ~15 % faster.
  The opposite case is Perch v2: its prototype head is 77 % of the window on a Pi 4 and takes int8 well (`--int8
  proto`: 53.1 dB against fp32 on 64 real windows, the window 1.7x faster), while its 1x1 convolutions in int8 gain
  2 % and cost 6 dB (`--int8 proto,pw,head`: 46.8 dB). Screen each class for its time and its dB before keeping it.

## 8. Debugging parity (fp32 path FAILs)

A fp32 FAIL means the op list computes something else than the graph. Localize it:

1. `bnc report model.onnx` and read the IR: are the shapes, pads, activations and `bn_folded`
   what the graph says? A wrong `pads` or a missing `act` is the usual cause.
2. Compare intermediate tensors: `Graph(path).run_shapes(x, want_values=[tensor names])` returns ONNX Runtime's
   values; `BNN_PROF=2 build/cli pkg 2 windows.f32 out.f32` lists the ops in order with their kernels and source
   nodes, so you know which op computes which tensor. For a VECTOR tensor (an embedding, a head output), compile a
   copy with `--logits <that tensor>` and a matching label file: the package then returns it as its "logits". An
   activation [H][W][C] cannot be returned that way (the output buffer would be its padded layout); dump it from the
   kernel's `run` while debugging.
3. Non-finite logits on one path only: the fp16 path overflowed (activations > 65504); look for a layer without
   normalization, run it in fp32 (policy) or scale.
4. A window the runtime refuses at `bnn_infer` ("window not scored: op N ...") names the op; the host should fall
   back to the fp32 path for that window.

## 9. Gate, time and ship

A gate is a comparison of the candidate with the reference on real data, with a pass criterion written down BEFORE
scoring, every candidate reported, pass or fail. The repository ships the window-level half of the gates behind the
published results as `scripts/gate_generic.py`; the other half (replaying BirdNET-Go's detection post-processing over
annotated field recordings and measuring precision/recall shifts with bootstrap intervals over recording periods)
needs annotated data and a replay harness that are not part of this repository (appendix A; the declared criteria
and the evaluation data are in [GATES.md](GATES.md)).

1. Collect the audio: hours of YOUR recordings (the microphone, site and species the classifier will see), cut into
   windows (`--windows-file w.npy`, n x n_samples float32, or `--audio` files). The eight windows of a smoke test are
   not a gate; a few thousand windows are. Fix the window count before scoring, by detections rather than by time:
   at every declared threshold the hit set (reference and candidate detections together) needs a few hundred
   entries before a Jaccard of 0.97 means anything, since each flip costs 1 / union (at a union of 30, one flip is
   0.033). How to read a FAIL on Jaccard: if the union at that threshold is small (under ~300), every flip margin
   is small (<= ~0.03) and an fp32 control candidate passes, the gate had too little data at that threshold: the
   result is inconclusive, not a pass, and the answer is a gate on more audio (decided before it is scored). Flips
   with margins > 0.05, or a Jaccard failure on a large union, mean do not ship that candidate. (A 720-window
   v3.0 gate failed at 0.5 / 0.8 on unions of 96 / 30 with every flip within 0.03; the declared 92,829-window gate
   of the same package passed.) Labels are optional: with them the gate also reports precision and
   recall; without them it is an agreement gate, which is what a custom classifier with no annotated data gets.
2. Declare the criteria, once, before any candidate is scored:

   ```
   python -I scripts/gate_generic.py declare gate/criteria.json --thresholds 0.1,0.5 --jaccard 0.97 \
       --flip-margin 0.05 --max-dconf 0.01 --sqnr-db 50 --note "site X, 2 h, Nov 2026"
   ```

   The file carries a timestamp and is never overwritten (new criteria = a new file, and a reason). The statistics:
   window top-1 agreement; at each threshold the hit-set Jaccard over (window, class) detections, the flips
   (detections on one side only) with their margins |reference conf - threshold| (<= 0.005, <= 0.02, > 0.05, max);
   max |dconf|; logit SQNR; with `--truth` (CSV `window_index,label`, a header row allowed; a window with no row
   is all-negative; a label is matched by the whole line, its scientific or its common part, or a class index)
   precision and recall at each threshold for both sides and paired-bootstrap 95 % intervals of dP and dR over
   windows. Pass, as the published gates' rule: Jaccard > `jaccard` and no flip margin > `flip_margin` at every threshold
   (a detection is conf > threshold, as in BirdNET-Go; a threshold with no detections on either side counts as
   agreement), and with truth dP/dR means >= `mean_pp` with CI upper bounds >= 0. `--sqnr-db`, `--max-dconf` and
   `--top1` are optional, stricter lines (off unless declared; the published PASSing Perch gate had top-1 0.9965 and
   47.9 dB, so do not declare them casually). What it does NOT do: replay BirdNET-Go's detection post-processing
   (pending detections, confirmation counts, the 12 s window) or bootstrap over recording periods; that is the
   external replay harness's job (appendix A). The criteria file is created once and made read-only, and `score` prints its sha256: keep
   both with the results so a reader can see the criteria predate the scores.
3. Score every candidate against the exact reference (the `.tflite` for a BirdNET-family model, else the ONNX):

   ```
   python -I scripts/gate_generic.py score gate/criteria.json --ref model.tflite --cli build/cli --post sigmoid \
       --windows-file w.npy --cand my_model-neon-fp32.bnpk:2 --cand my_model-neon-fp16.bnpk:3 \
       [--truth truth.csv --labels labels_go.txt] --json gate/result.json
   ```

   prints each candidate's statistics and `gate: PASS|FAIL (why)`, then `OVERALL`; exit 0 only when every
   candidate passes. A failing fp16 candidate with a passing fp32 one means ship fp32 (section 7).

   `--ref-output TENSOR` picks the ONNX tensor the reference is read at (default: the last graph output); it may be
   an internal tensor (the gate adds it as an output on an in-memory copy of the model), and an unknown name stops
   with the graph's outputs and logits tensors listed. With `--post sigmoid` (or softmax) a tensor that is the
   output of that Sigmoid (Softmax) node is read before it, with a note, as `check` takes the logits, so the
   activation is never applied twice. BirdNET v3.0 has no `.tflite`: its reference is the fp32 ONNX, whose outputs
   are `embeddings` (the default, 1280 wide) and `predictions` (the sigmoid), so give `--ref-output predictions`:

   ```
   python -I scripts/gate_generic.py score gate/criteria.json \
       --ref birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx --ref-output predictions --post sigmoid \
       --cli build/cli --windows-file w.npy --cand birdnet_v3.0_north-america-east-neon-fp16.bnpk:3 \
       --cand birdnet_v3.0_north-america-east-neon-fp32.bnpk:2 --json gate/result.json
   ```

   (`note: 'predictions' is the Sigmoid of '/Add_8_output_0': scoring '/Add_8_output_0' ...`; naming the logits
   tensor `/Add_8_output_0`, the one `check` prints, gives the same result.)
4. Time on the target CPU, single thread, pinned (`taskset -c 3` on Linux; macOS cannot pin a process to a core, so
   a Mac's timings are only relative), with the background load recorded; `BNN_PROF=1` prints per-kernel
   times, `BNN_PROF=2` per op with the source node names. Numbers from another CPU (a Mac, a container) are not
   the target's numbers: a Mac's M-series runs a BirdNET v2.4 window in ~5 ms (fp16) / ~10 ms (fp32) where a Pi 5
   takes ~19 ms (fp16) and a Pi 4 ~33 ms (fp32 only); use them for relative comparisons (which kernel dominates),
   not for the deployment budget.
5. Ship: the package name, sha256, size, license and the selection key (which CPU class gets which path) go into the
   host's manifest; the host reads everything else (`bnn_probe`, `bnn_probe_meta`).

## 10. Troubleshooting: every message and its fix

Compiler (`bnc`), exit codes in appendix B:

| message (abridged) | meaning | fix |
|---|---|---|
| `bnc: no module named 'onnx'; hint: install the compiler extra: ...` (exit 3) | the package was installed without its compiler dependencies | `pip install -e '.[compiler]'` in the active environment |
| `model error: ... not a loadable ONNX file` | not ONNX (a .tflite, a truncated download) | section 3 (tf2onnx); check the sha256 |
| `expected exactly one graph input` | state/length inputs | re-export with the PCM window as the only input |
| `input ... expected [1, n_samples] ... fixed n_samples` | dynamic window length or wrong rank | re-export with a fixed window length (SavedModel: tf2onnx `--inputs name:0[1,144000]`; TFLite: resize the input before saving) |
| `ONNX Runtime cannot load the graph (...)` | a broken or too-new export (IR/opset) | `onnx.checker.check_model`; export with opset 17, IR <= your ONNX Runtime's |
| `no supported front end at the graph input: expected ... found [...]` | the graph starts with something the two front-end matchers do not know | the spectrogram must be in the graph in one of the two forms (section 2); for a preprocessed-input model add a front-end rule |
| `front end: expected the window Mul ...` / `... is not an identity kernel` / `pad layout ...` | a framed log-mel that deviates at that step | the message names the node; export closer to the Perch form or extend `fe_logmel` |
| `FE_STFT2 kernel supports the BirdNET v2.4 geometry only ... got ...` | a BirdNET-style STFT with other frame sizes | a generic STFT kernel is not written: section 6 step 4 (`RT_OP_FE_STFT2` is fixed-size) |
| `matrix-DFT magnitude chain (no_dft file): use the DFT-op file` | Perch "no_dft" export | use the export with the ONNX `DFT` op |
| `unsupported op X at node N ... : <reason>` + attributes/shapes/nearest/next step | the recognizer has no rule for that node in that shape | section 6; fix primary nodes first |
| `(downstream of the unsupported node producing 'T')` | not a problem of its own | fix the primary node |
| `dilated conv` / `grouped conv (group g) other than depthwise` / `conv with non-constant weights` | conv variants without kernels | re-export without dilation, or section 6 step 4 |
| `Gemm with attributes {...} / bias input` | Gemm with transB/alpha/beta/bias | export as MatMul + Add (or extend `rule`) |
| `depthwise conv without swish` | depthwise + ReLU / linear | the depthwise kernels fuse swish only; section 6 step 4 (an `act` parameter exists in DWK: 0/1; ReLU would need the kernel's epilogue) |
| `dense layer with activation X: dense layers fuse relu only` / `dense layer with K inputs on a C-wide vector` / `dense layer on a f16 buffer` | a head shape outside the dense chain | `--logits` at the pre-activation; check the export |
| `relu elsewhere needs a kernel` (Relu with `nearest: ... a hidden layer ...`) | a Relu that is neither after a conv nor between two dense layers | a hidden layer (Dense -> ReLU -> Dense) IS supported: the Relu here sits somewhere else; re-export |
| `graph output X not produced by a supported op` and `N uncovered nodes` before a PASS | a final Sigmoid/Softmax (or another output) the logits do not need, e.g. BirdNET v3.0's `predictions` Sigmoid | nothing: the host applies `--post`; only nodes the logits or embedding depend on must be covered |
| `canonicalization changed the graph output` | a canon rewrite was not exact on this graph | `--no-canon` and report it (a compiler bug) |
| `mean of a padded buffer or C % 4` / `prototype head needs an unpadded [cells % 8][C % 8] input` | geometry the kernels cannot take | pad channels in the export, or extend the kernel |
| `per-channel affine outside the v2.4 post BN + ReLU + 3x3 conv + mean pattern` | a standalone BN/affine | fold it into the preceding conv (export), or add an AFFINE kernel |
| `SE must directly follow a depthwise conv` / `SE input has other consumers` | SE placement the in-place kernel cannot do | section 6 step 2/3 (a non-in-place SE needs a copy) |
| `logits tensor 'T' is not produced by a supported op` | `--logits` default is a post-activation output | `--logits <pre-activation tensor> --post sigmoid|softmax` |
| `activation 'sigmoid' fused after a conv has no kernel` | a conv whose only consumer is a Sigmoid | the output activation: `--logits <the conv output>` |
| `conv k[..] s[..] pads [..] cin N act A has no <path> kernel` / `3x3 valid conv outside the v2.4 post/pc pattern` | a recognized conv shape without a kernel | section 6 step 4 (or re-export: a 1x1, a depthwise, or a 1-channel stem) |
| `depthwise conv k[..] s[..] pads [..]: the kernels take square odd k <= 7, equal strides <= 2 ...` | geometry outside the depthwise kernels | re-export or section 6 step 4 |
| `log-mel front end: nfft N ... the FFT kernel takes nfft a power of two in 16..2048 ...` | FFT limits | change the front end's n_fft (zero-pad to a power of two) |
| `log-mel front end: only the exact-DFT form is lowered` | matrix-DFT export | the DFT-op export |
| `embedding tensor T is not produced by a supported op` / `--classes needs the logits to come from a dense or prototype head` / `--classes: indices out of range or repeated` | argument points at the wrong tensor / head | as named |
| `ONNX Runtime fails to run the graph on a window` / `input ... is not float32` | the export itself fails or takes another dtype | fix the export first |
| `label file ... has N non-empty lines but the model's logits have M classes` (exit 7) | wrong label file or wrong `--logits`/`--classes` | use the head's label file, one per line, output order |
| `label file ...: non-blank line N (...) holds a line break other than \n or \r\n, a NUL byte, ...` (exit 7) | the file has `\r`-only, form-feed or Unicode line breaks, a NUL byte, or lines of other white space | save it as plain text, one label per line, `\n` or `\r\n` line ends |
| `usage error: -o is required` etc. (exit 3) | missing argument | as named |
| `--license-file and --attribution are required` / `--family is required` / `--post is required` (exit 3) | the package would carry no license/credit, an unknown family, or no post | give them (section 5 has the license template); `--force` writes without |
| `--sample-rate N with the BirdNET v2.4 front end ... takes 48000 Hz` (exit 3) | a wrong rate would be baked in | `--sample-rate 48000` (or `--force` if you know better) |
| `WARN: N of M label lines are not in BirdNET-Go's ... form (first: '...')` | labels in the published `Common_Name_Genus_species` form | `scripts/relabel_for_birdnetgo.py` before compiling |
| check: `container X does not mount this checkout at /eng` | `--docker` named a container of another tree | `scripts/eval_container.sh start`, or `--docker-image` |
| `cmp_suite: data/bitref/... is missing` (exit 2) | the gated reference outputs are not in the repository | `check` your own model instead (appendix A) |
| `internal error: <Exception> ... (at file:line)` (exit 8) | a compiler bug or a graph the recognizer's code did not foresee | report it with the model's `report --json` output; the JSON file is still written |
| `front end: a Conv-framed DFT (...) is recognized as a short-time Fourier transform but no front-end kernel lowers it ...` | a Conv-framed STFT whose tail is not the dB-image form (a modified BirdNET v3.0 front end, or another model) | compare it with the dB-image form (section 2; stock v3.0 compiles, section 2.1); otherwise section 6 |
| `front end: expected the window Mul ...` and every other `front end:` message | the v2.4, log-mel or dB-image matcher stopped at the named node (normalization, framing, window, `DFT MatMul is not the cos part`, Pow, mel reverse, Concat, BN; or the log-mel steps) | export the spectrogram in one of the three supported forms (section 2), or extend `fe_stft2` / `fe_logmel` / `fe_dbimg` |
| `no neon-fp16 lowering for IR op X` / `IR value ... has no buffer` | an IR op the recognizer kept standalone that this path cannot lower | the report shows the op; section 6 step 3 |
| check: `native build failed (...)` | no C compiler | install clang/gcc, or `--cli <built cli>` / `--docker <container>` |
| check: `the runtime refused or failed the <path> package: init: ...` | the compiled package fails validation (see the runtime messages below) | this is a compiler/kernel disagreement: report it with the full message |
| check: `the fp32 path disagrees with ONNX Runtime` | lowering/kernel bug | section 8 |
| check: `the fp32 path passes; the fp16 path does not` | fp16 precision | ship fp32 or section 7 |
| check: `no path compiles: ...` / `no path ran on this CPU` / `--cli X: not an executable` | lowering refusals on every path / no runnable path / wrong CLI | the message names the lowering error; `--paths`; the CLI path |

Runtime (`bnn_last_error()`, the CLI prints it after `init:` / `infer failed:` / `labels:`). Every message has the
form `<file> [path]: <what>: expected ... found ...; hint: ...`. Op messages are prefixed
`op N <kernel> (ONNX node '<name>' +k)` (the source node the compiler lowered it from, `+k` = the number of further
nodes it covers); packages written before compiler `bnc 2` have no source names and say so.

| message (abridged) | meaning | fix |
|---|---|---|
| `not a bnneon weights file (magic '...')` / `not a .bnpk package` | wrong file | compile it; check the download |
| `truncated (size S, header says T)` / `smaller than a package header` / `corrupt header` / `table offsets outside the file` / `tensor N lies outside the file` / `path N has corrupt tables` / `byte-order mark` | damaged file | re-download or recompile; compare the sha256 |
| `cannot open F (reason)` / `mmap of N bytes failed (reason)` | file system | the reason is the OS's |
| `the package needs reader version N, this library reads M` | package newer than the library | update libbnneon or recompile with the matching compiler |
| `this CPU (<features>) can run none of the package's paths: <path> (needs ...)` | e.g. an fp16 package on a Cortex-A72 | use the model's `-neon-fp32.bnpk` (named in the message) or compile one (`--paths neon-fp32`) |
| `path neon-fp32 is not in this package (it has: ...)` | mode 2/3/4 asked for a path the file lacks (the hint names the package by the path asked for, even one the model has no lowering for, e.g. neon-fp16 for a model the fp16 families cannot lower) | mode 0, or the other package |
| `mode N is not available for compiled packages` | mode 1 or out of range | 0, 2, 3 or 4 |
| `a legacy BirdNET v3.0 .bnv3 file, whose hand-written engine this library no longer has` | a `.bnv3` (the retired v3 engine's file) | the BirdNET v3.0 package of the same model (section 2.1), or a libbnneon from before the retirement |
| `op N <kernel> (ONNX node '...'): buffer argument i (buffer #b): expected dtype X with >= E elements, found dtype Y [...]` | the op list and the kernel disagree on a buffer | a compiler/kernel version mismatch or a corrupt package: recompile with the compiler matching this library; if it reproduces, it is a lowering bug (the source node tells you which layer) |
| `... tensor argument i (tensor #t 'name' dtype [dims]): expected N bytes, found M` | weight blob size | same |
| `... parameter check failed: <condition> (int params: [...])` | a geometry/int parameter violates the kernel's contract (the condition is the kernel's own source text) | same; the int params are the op record |
| `... op record shape: expected ...` / `buffer argument i is required ... but absent` / `... share memory slot ...` / `... writes the input window` / `buffer #N: bad descriptor` / `shares the input window's memory slot` | malformed op record or buffer table | same |
| any other `op N <kernel> (ONNX node ...): ...; hint: ...` | a kernel's init refused the op record | the message is the kernel's own check; same fix as above |
| `... unknown opcode 0x... (this library knows 1..N, the package was written by compiler '...')` | package needs a newer library | update libbnneon, or recompile |
| `... fp16 kernel in path X, which does not declare the fp16 requirement` (also `dotprod kernel ...`) | path table inconsistent | recompile |
| `out of memory: N bytes (M MB) for the activation buffers of path X` / `op N ...: out of memory allocating B bytes of kernel scratch` | really out of memory (distinct from any parameter error) | free memory; a smaller model or path |
| `path X needs more than 1 GB of activation buffers` | corrupt descriptor or a model far beyond the runtime's design | recompile; check the model |
| `metadata n_samples = N but the input buffer is ...` (also n_classes, emb_dim) | metadata and op list disagree | recompile |
| `init failed without a message (kernel bug)` | a kernel init returned -1 without `rt_bad`/`rt_oom` | fix the kernel's init (section 6 step 4) |
| `label file F (sha256 a..., N lines) is not the label file P was compiled with (sha256 b..., M classes)` | wrong labels in use | use the package's own label file (`bnn_probe_meta "labels"`) or recompile with `--labels F` |
| `window not scored: op N <kernel> (ONNX node ...) reported a failed window (non-finite values on the neon-fp16 path)` | fp16 overflow on this window | the host falls back to the fp32 path for the window; section 8 step 3 |

## 11. Scripted use: exit codes and JSON

For a script or CI job driving the compiler, run, in this order, and read the JSON rather than the prose:

1. `bnc check model.onnx --labels L --sample-rate HZ --post P --json check.json`; the exit code
   is the verdict (0 PASS, 1 WARN, 2 unsupported, 3 usage, 4 bad model, 5 parity FAIL, 6 runtime, 7 labels,
   8 internal error). The JSON is written on every exit.
2. On exit 2: `coverage.primary[]` lists the nodes to fix, each with `reason`, `attrs`, `inputs`/`outputs` shapes,
   `nearest` and `next_step`. Fix the first, re-run. Prefer re-exporting over new kernels. When `primary` is empty,
   the refusal came from the lowering or the logits choice: read `next_step` and `paths.<name>.error`.
3. On exit 6: `steps[-1].error` holds the runtime's message verbatim; it names the op, kernel and source node. That
   is a compiler/kernel disagreement to fix in the lowering or the kernel's init, never by editing the package.
4. On exit 1/5: `paths.<name>.{sqnr_db, max_dconf, top1_agree, why}` and `fp16_vs_fp32_sqnr_db` say which path and
   why; `next_step` says what to do.
5. A whole family that does not fit (exit 2 at node 0, or many primary nodes): section 6.1, in its order: size the gap
   from `no_body_rule` and a graph survey, choose re-export / compiler extension / hand-written engine, fp32 first.
6. Before shipping: `tests/test_errors.py` (with `--model` and `--labels`), a bit-identity check if you touched
   `src/` (appendix A; without the gated reference outputs, `check` the stock and your own models through
   `--docker-image`), and a declared
   gate (`scripts/gate_generic.py declare` BEFORE `score`).

Never:

- declare or change a gate criterion after seeing a candidate's score; write it down first, then score, then report
  every candidate including the failed ones;
- edit a package file to make it load, or relax a kernel's `init` check to make a package pass: the check is the
  contract;
- treat the few-window `check` as a gate, or PASS on synthetic windows as accuracy;
- compile a model whose license you have not read at its source, or drop the attribution;
- commit dead ends: log them (what, numbers, why, what could revive it) in `DEADENDS.md`;
- run a downloaded model file through anything but `python -I` in a clean directory, or execute downloaded code
  outside a container.

## Appendix A. What runs from a clean clone, and what needs external files

Runs from a clean clone (Python environments as in section 1):

- `make`, `build/cli`, the C API and the Python binding.
- Every `tests/test_*.py` and the lint gate `scripts/lint.sh`. `tests/test_python_binding.py`,
  `tests/test_readme_example.py` and `tests/test_scripts_units_s2.py` use the packages in `weights/` when present and
  skip the cases that need one when there is none; `tests/test_errors.py` runs parts B-D with `--model` and `--labels`
  (a BirdNET v2.4-family ONNX model and its label file) and part E with `--proto-package` (e.g. a Perch v2 package).
- `bnc check` and `bnc compile` on any ONNX model you have, and `scripts/gate_generic.py` (section 9) on your own
  windows.
- `scripts/compile_models.sh`: the shipped packages' exact commands. Its inputs are the original model and label
  files (it names them in its header; by default it looks under `data/`, which is git-ignored, and every input can be
  set by an environment variable). Each package records the sha256 of its ONNX (`source_sha256`) and label file
  (`labels_sha256`), shown by `build/cli MODEL --info`; `weights/SHA256SUMS` lists the packages these inputs gave.
  The Perch v2 north-america-east slice also needs the catalog's slice files (`perch_v2_north-america-east_indices.txt`
  and `_labels.txt` from tphakala/Perch-v2-Models) in `PNAE`; without them the script skips those packages and says
  so.
- The arm64 build container: `scripts/eval_container.sh build && scripts/eval_container.sh start <name>`
  (`check --docker-image` refuses a container that mounts another tree). Copy packages into a container with
  `docker cp` rather than reading them over the mount, and run containers with `--memory` <= 2 GB.

Needs files that are not in this repository:

- The bit-identity suite `scripts/cmp_suite.sh` (`CONTAINER=<name>`): it must print `v3 neon-fp16: bit-identical`,
  `v2.4 fp16: bit-identical`, `v2.4 fp32: bit-identical` after a change under `src/`, and it needs the gated
  reference outputs `data/bitref/{a17,c2,v3dev32}.f32` (+ `SHA256SUMS`, recorded in `results/bitref_SHA256SUMS.txt`),
  `data/devA64.f32` and the gated weights inside the container. Without them it stops with exit 2; use `check` on
  your own models instead.
- The detection-replay gates (`scripts/gate_v3_pkg.py`, `gate_sim_v3_pkg.py`, `overlap_replay_v3.py`,
  `gate_perch.sh`, `nae_calibration.py`, `perch_devset.py`, `decode_hash.py`, `pipe_score.py` as an external scorer):
  they replay BirdNET-Go's detection post-processing over the annotated evaluation set ([GATES.md](GATES.md)) with a
  harness that is not part of this repository (`simulate.py` / `simulate5s.py` / `infer.py` of the evaluation tree,
  `modelgate.py` with its model adapters, `perchgate.py`). The scripts load those modules by file path through
  `bnneon.extern`; when a module is missing they stop with Python's own `ModuleNotFoundError: No module named
  '<name>'`, and `nae_calibration.py` with the same error for `birdlab_tools` (the external harness's package name;
  `BIRDLAB_DATA` in `perch_devset.py` and `gate_perch.sh` names its data directory). Their result files in
  `results/` are the record of the published gates.
- Timing on a Raspberry Pi: single thread pinned with `taskset -c 3`, the background load recorded (`uptime`), one
  benchmark at a time; the sessions behind the published numbers are in `results/compare/` and `results/MENU.txt`.

BirdNET-Go integration: the manifest keys are `aarch64-a76/neon` (neon-fp16 package) and `aarch64/neon` (neon-fp32);
the Go side surfaces `bnn_last_error` unchanged with an `init <file> (mode N): ` prefix (its buffer is 512 bytes, the
engine's messages are at most 511).

A published BirdNET-Go custom classifier served as the test model for an independent run of this guide (41 classes,
Zenodo 10.5281/zenodo.20845361; CC BY 4.0 on the record, but a BirdNET v2.4 derivative and so CC BY-NC-SA 4.0).

## Appendix B. Reference

Exit codes of `bnc` (every command): 0 PASS; 1 WARN (also `check --paths` naming a path this CPU cannot run);
2 unsupported ops; 3 usage (also a missing or unknown command, and the compiler extra not installed); 4 the model
cannot be loaded or run by ONNX Runtime; 5 parity FAIL; 6 the runtime could not be built or refused the package;
7 label mismatch; 8 internal error. `bnc -h` / `--help` prints the usage and exits 0.

Runtime error contract (`src/bnneon.h`): every failing call sets a one-line, <= 511-byte message
`<what failed>: <expected> vs <found>; hint: <what to do>` (long node names are cut to keep the hint; the op's int
parameters are appended only when they fit); load and validation failures carry
`op <index> <kernel> (ONNX node '<name>' +k)`. `bnn_last_error_kind()` classifies it for hosts: `BNN_ERR_PACKAGE`
(bad, corrupt or incompatible file, or a kernel-contract violation), `BNN_ERR_OOM`, `BNN_ERR_CPU` (no runnable path
or mode), `BNN_ERR_WINDOW` (one window not scored), `BNN_ERR_LABELS`, `BNN_ERR_USAGE`. Return codes: `bnn_init`
NULL, `bnn_infer` nonzero (the window; the value is the kernel's, not a kind), `bnn_verify_labels` -1. Kernel inits
return `RT_E_PARAM` (-1) or `RT_E_OOM` (-2); the loader turns them into the two message families above.

Package format: `python/bnneon/compiler/pkgfmt.py` (writer) and `src/rt_pkg.c` (reader). Metadata keys a package carries:
`model_id, family, sample_rate, n_samples, hop_samples, n_classes, emb_dim, post, logit_scale, labels_sha256,
labels_file, labels, license, attribution, source_sha256, input, compiler, notes, class_subset` and, since `bnc 2`,
`src_nodes/<path>` (one source node per op; optional: older packages load unchanged). Packages from compilers before
`bnc 3` spell the license key `licence`; the library answers either spelling for old and new packages.

Tests: `python -I tests/test_errors.py [--cli build/cli] [--model M.onnx --labels L.txt [--logits T]]` feeds broken
graphs and mutated packages and checks every message names the op, kernel, source node, argument and values, then
splices a hidden layer and tf2onnx artifacts into the model and checks they compile and agree with ONNX Runtime, then
the round-3 tools (label rewrite and warning, compile refusals and defaults, `--labels` extraction, a declared gate)
(about 100 checks with a BirdNET v2.4-family model; `--logits` defaults to the head's output; the out-of-memory check
needs Linux `ulimit -v`; on a CPU without fp16 arithmetic the loader cases run on a `neon-fp32` package and the
`neon-fp16` run is skipped; the scratch directory, about 300 MB at its peak, goes into `--workdir DIR` or the system's
temporary directory and is removed at the end). Tools: `scripts/tflite_to_onnx.py` (conversion), `scripts/relabel_for_birdnetgo.py`
(labels), `scripts/gate_generic.py` (gate), `scripts/eval_container.sh` + `integration/Dockerfile.eval` (container).
