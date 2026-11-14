# Getting started

The full walk-through behind the README's [Quick start](../README.md#quick-start), with the outputs to expect at
each step, the Raspberry Pi 4 variants, the tests and troubleshooting.

These steps take you from nothing to bird detections on a recording, with BirdNET v2.4 (the simplest model to set up:
two files, a 67 MB download). They work on a Raspberry Pi 4 or 5 with 64-bit Raspberry Pi OS, on other 64-bit ARM
Linux machines, and on an Apple silicon Mac (for development; a Mac's timings are not a Pi's). There is no x86 build.

After step 2, run every command from the checkout's root. The outputs below are real, trimmed with `...`, and come
from an Apple silicon Mac unless they are marked as a Raspberry Pi 4's; on a Raspberry Pi 5 the numbers differ but the
shape is the same.

**On a Raspberry Pi 4** (or any other 64-bit ARM CPU without fp16 arithmetic), the `-neon-fp16.bnpk` package cannot
run: use the `-neon-fp32.bnpk` package wherever a command names a package. Step 5 puts the package's name in the shell
variable `PKG` and the later commands use `$PKG`; on a Pi 4, set it as the comment on that line says.

## 1. Install the prerequisites

You need a 64-bit ARM system, a C compiler (gcc >= 10 or clang), make, git, curl and Python 3.11 or newer.

On Raspberry Pi OS (64-bit) or Debian:

```sh
sudo apt update
sudo apt install -y git gcc make libc6-dev curl python3-venv
```

On macOS, the Xcode Command Line Tools provide clang, make, git and curl (`xcode-select --install`); install Python 3.11
or newer from python.org or Homebrew if `python3 --version` reports an older one.

Check:

```sh
uname -m
cc --version
python3 --version
```

```text
arm64                                              <- aarch64 on Linux
Apple clang version 21.0.0 (clang-2100.3.34.2)     <- gcc 14.2 on Raspberry Pi OS trixie, the reference compiler
Python 3.12.7                                      <- 3.11 or newer
```

On a Raspberry Pi, `dpkg --print-architecture` must print `arm64`: a 32-bit Raspberry Pi OS (`armhf`) cannot build
the engine, even when `uname -m` says `aarch64`.

## 2. Clone and build

```sh
git clone https://github.com/cdw9bf/birdnet-neon-engine.git
cd birdnet-neon-engine
make -j4
ls build
```

`make` prints one compiler line per file; it takes a few seconds on a Mac and about 33 s on a Pi 4. On macOS it also
prints one harmless warning, `'cblas_sgemm' is deprecated` (from the v2.4 fp32 reference). With gcc 14.2 (Raspberry
Pi OS trixie) it prints three warnings instead, each followed by a few `note:` lines: `-Wformat-truncation` at
`src/v24_model.c:105` and `src/bnneon.c:250`, and `-Wstringop-overflow` at `src/fe_dbimg.c:88`. They are harmless:
gcc does not see limits the code checks before those lines (the block count of a legacy `.bn24` file, the 511-byte
error message, an FFT size the loader has checked to be 64 to 8192). Then `ls build` shows:

```text
cli
libbnneon.so
obj
```

`build/libbnneon.so` is the library and `build/cli` the command-line tool. The library checks the CPU's features at run
time, so the same build runs on a Pi 4 and a Pi 5.

## 3. Install the Python package

The compiler is Python. Make a virtual environment in the checkout and install this package with the compiler's extra
dependencies (onnx and ONNX Runtime):

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[compiler]'
bnc --help | head -1
```

```text
...
Successfully installed bnneon-0.1.0 flatbuffers-25.12.19 ml_dtypes-0.6.0 numpy-2.5.4 onnx-1.23.2 onnxruntime-1.31.1 ...
bnc: the bnneon model compiler (ONNX -> .bnpk model package).
```

Every later step assumes the environment is active: in a new terminal, run `. .venv/bin/activate` again. The versions
above are the tested ones (`pip install onnx==1.23.2 onnxruntime==1.31.1 numpy==2.5.4` pins them).

## 4. Download BirdNET v2.4

Two files: the BirdNET v2.4 model as ONNX (from the Hugging Face repository BirdNET-Go downloads from) and BirdNET-Go's
English label file. The paths under `data/` (git-ignored) are where `scripts/compile_models.sh` looks for them.

```sh
mkdir -p data/battybirdnet data/birdnet
curl -fL -o data/battybirdnet/birdnet-v2.4-embeddings-fp32.onnx \
  https://huggingface.co/tphakala/BattyBirdNET-onnx/resolve/main/birdnet-v2.4-embeddings-fp32.onnx
curl -fL -o data/birdnet/birdnet_labels_v2.4.txt \
  https://raw.githubusercontent.com/tphakala/birdnet-go/5dca544a4939035d0d1de75f51a0e66c9a4c9aad/internal/classifier/data/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_en_us.txt
shasum -a 256 -c <<'EOF'
b6b8f24dc9c3d43f2deb14a6f2c7b5b233e7477b6baf1b52341291e714903fb0  data/battybirdnet/birdnet-v2.4-embeddings-fp32.onnx
b50b77b7c3dfe40cd637e8cccdca0173a0a4ddee8867b830ff3c1a566f477f16  data/birdnet/birdnet_labels_v2.4.txt
EOF
```

```text
data/battybirdnet/birdnet-v2.4-embeddings-fp32.onnx: OK
data/birdnet/birdnet_labels_v2.4.txt: OK
```

(`shasum` comes with perl; on Linux, `sha256sum -c` does the same.) The label file is pinned to the BirdNET-Go
commit the measurements used; the same file is on BirdNET-Go's `main` branch.

The model is licensed CC BY-NC-SA 4.0 (non-commercial; see [License](../README.md#license)), and so is every package you compile
from it.

## 5. Check and compile the model

`bnc check` first: it compiles a scratch package for each CPU path, runs it through the engine on test windows and
compares it with ONNX Runtime. On a Pi 4, add `--paths neon-fp32`: a Pi 4 cannot run the fp16 path (without the
option, check still compiles it, warns that it cannot run, and ends with `PASS: every path this CPU can run agrees
with ONNX Runtime on these windows; skipped: neon-fp16 (cannot run here); ...`).

```sh
bnc check data/battybirdnet/birdnet-v2.4-embeddings-fp32.onnx --labels data/birdnet/birdnet_labels_v2.4.txt \
  --sample-rate 48000 --post sigmoid --quiet
```

```text
data/battybirdnet/birdnet-v2.4-embeddings-fp32.onnx: 300 nodes after folding 0 constant nodes; input input [144000]; ...
68 IR ops; 0 unsupported nodes; 0 uncovered nodes
...
path neon-fp16: 59.9 dB vs ONNX Runtime, max |dconf| 0.0028, top-1 4/4: PASS
path neon-fp32: 63.4 dB vs ONNX Runtime, max |dconf| 0.0017, top-1 4/4: PASS
engine fp16 vs engine fp32: 62.4 dB (the fp16 path's own loss, reference-free)

PASS: every path agrees with ONNX Runtime on these windows; next: compile the shipping packages ...
```

Then compile the two packages, one per CPU path, into `weights/`:

```sh
scripts/compile_models.sh v24
```

```text
path neon-fp16: 66 ops, 72 buffers in 7 slots, 2.5 MB activations (11.0 MB unshared)
wrote weights/birdnet_v2.4-neon-fp16.bnpk: 28.4 MB, 201 tensors, 6522 classes, emb 1024
path neon-fp32: 66 ops, 85 buffers in 7 slots, 4.4 MB activations (21.3 MB unshared)
wrote weights/birdnet_v2.4-neon-fp32.bnpk: 52.1 MB, 166 tensors, 6522 classes, emb 1024
birdnet_v2.4-neon-fp16.bnpk 28360128 9f964bce9ff3c77e48ef60efbd2969b238d5dbda257d79fd4d8e95fb680604b6
birdnet_v2.4-neon-fp32.bnpk 52145536 53259796bd92976bbea2ee01a172f3995cb07eece2ebd7bc9b8956fa912b2533
```

The script runs `bnc compile` once per path; the fp16 one is:

```sh
bnc compile data/battybirdnet/birdnet-v2.4-embeddings-fp32.onnx --logits output \
  --labels data/birdnet/birdnet_labels_v2.4.txt --model-id birdnet_v2.4 --family 24 --post sigmoid \
  --sample-rate 48000 --paths neon-fp16 --license-file weights/LICENSE-birdnet-v24.txt \
  --attribution "BirdNET v2.4, K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology, and Chemnitz University of Technology" \
  -o weights/birdnet_v2.4-neon-fp16.bnpk
```

With the tested versions above, your packages match the hashes in `weights/SHA256SUMS` byte for byte. A package
records the base name of its label file (`labels_file`), but not the ONNX file's name, any directory, the output name
or a time: the hashes reproduce when the ONNX and label files have the contents above and the label file is named
`birdnet_labels_v2.4.txt` (in any directory, e.g. `V24=/path/model.onnx V24L=/path/birdnet_labels_v2.4.txt
scripts/compile_models.sh v24`). A label file of another name gives another hash and a package that works the same. A
package is an ordinary file: you can compile on one machine and copy the `.bnpk` files to another (the shipped
packages were compiled on a Mac and run on the Pis).

Check the hashes, then choose the package for this machine and inspect it:

```sh
(cd weights && shasum -a 256 --ignore-missing -c SHA256SUMS)
PKG=weights/birdnet_v2.4-neon-fp16.bnpk   # on a Raspberry Pi 4: PKG=weights/birdnet_v2.4-neon-fp32.bnpk
build/cli "$PKG" --info
```

```text
birdnet_v2.4-neon-fp16.bnpk: OK
birdnet_v2.4-neon-fp32.bnpk: OK
model_id birdnet_v2.4 family 24 format_version 1
sample_rate 48000 n_samples 144000 hop 0 n_classes 6522 emb_dim 1024 post 1 (sigmoid) logit_scale 1
best path for this CPU: neon-fp16
labels_sha256 b50b77b7c3dfe40cd637e8cccdca0173a0a4ddee8867b830ff3c1a566f477f16
compiler: bnc 3
license: BirdNET v2.4 (BirdNET_GLOBAL_6K_V2.4), ... Licensed CC BY-NC-SA 4.0 ...
...
input: float32 PCM in [-1, 1) at sample_rate, 144000 samples per window
mode 0: ok
...
```

On a Pi 4 the `-neon-fp32.bnpk` package says `best path for this CPU: neon-fp32`. The `-neon-fp16.bnpk` package
there says it cannot run and why, mode by mode (`--info` only reports, so it still exits 0):

```text
best path for this CPU: (none)
...
mode 0: weights/birdnet_v2.4-neon-fp16.bnpk: this CPU (any AArch64) can run none of the package's paths: neon-fp16 (needs fp16 arithmetic); hint: use the neon-fp32 package of this model (birdnet_v2.4-neon-fp32.bnpk, any AArch64) or compile one with --paths neon-fp32
mode 2: weights/birdnet_v2.4-neon-fp16.bnpk: path neon-fp32 is not in this package (it has: neon-fp16 (needs fp16 arithmetic)); hint: ...
...
```

`PKG` lives in this shell only: in a new terminal, set it again (and activate the environment, step 3).

## 6. Run it on a recording

The engine scores one fixed-length window of float32 samples at the model's sample rate per call; it does not read
audio files, resample or mix channels. For BirdNET v2.4 a window is 144000 samples: 3 s of mono audio at 48 kHz.

Get a test recording: the 2-minute example soundscape of BirdNET-Analyzer (48 kHz, mono, 16-bit WAV; see that
repository for its terms):

```sh
curl -fL -o data/soundscape.wav \
  https://raw.githubusercontent.com/birdnet-team/BirdNET-Analyzer/9db5740e47d353d1bdbe2747068ccc5ef4c9d8e1/birdnet_analyzer/example/soundscape.wav
shasum -a 256 -c <<'EOF'
df312b45bc82ce4c638c3e9e09d748702ea14a91ec29e4e8e0676d3e3e015fd7  data/soundscape.wav
EOF
```

Save this script as `classify.py`. It uses only the Python standard library, numpy and the `bnneon` binding:

```sh
cat > classify.py <<'EOF'
"""Scores a 16-bit mono WAV file in back-to-back windows; prints the best class per window."""
import sys
import wave

import numpy as np

import bnneon

model, wav = sys.argv[1], sys.argv[2]
with wave.open(wav, "rb") as w:
    rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
    pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
if width != 2 or channels != 1:
    sys.exit(f"{wav}: need 16-bit mono PCM (ffmpeg -i IN -ac 1 -ar RATE -sample_fmt s16 OUT.wav)")
audio = pcm.astype(np.float32) / 32768.0   # the engine takes float32 in [-1, 1)

with bnneon.Model(model, lib="build/libbnneon.so") as m:
    if rate != m.info.sample_rate:   # the engine does not resample
        sys.exit(f"{wav}: {rate} Hz, but the model needs {m.info.sample_rate} Hz")
    print(m.info.model_id.decode(), m.info.path.decode(), m.info.precision.decode())
    labels = m.labels()
    n = m.info.n_samples
    for start in range(0, len(audio) - n + 1, n):
        scores, _ = m.infer(audio[start:start + n])
        conf = m.confidences(scores)   # sigmoid for BirdNET, softmax for Perch
        best = int(np.argmax(conf))
        print(f"{start / rate:6.1f} s  {conf[best]:.2f}  {labels[best]}")
EOF
python classify.py "$PKG" data/soundscape.wav
```

```text
birdnet_v2.4 neon-fp16 fp16
   0.0 s  0.81  Poecile atricapillus_Black-capped Chickadee
   3.0 s  0.31  Poecile atricapillus_Black-capped Chickadee
   6.0 s  0.19  Baeolophus bicolor_Tufted Titmouse
   9.0 s  0.64  Haemorhous mexicanus_House Finch
...
  42.0 s  0.74  Junco hyemalis_Dark-eyed Junco
...
 117.0 s  0.39  Spinus tristis_American Goldfinch
```

Each line is a window's start, the confidence of its best class (0 to 1) and that class's label
(`Scientific name_Common name`). The script prints the best class of every window, so low values are normal; a
detector would keep only classes above a threshold. On a Pi 4 (`PKG` set to the `-neon-fp32.bnpk` package), the first
line is `birdnet_v2.4 neon-fp32 fp32` and the windows above print the same values.

Your own recordings: convert them to mono at the model's rate with ffmpeg (`sudo apt install ffmpeg`, or
`brew install ffmpeg`):

```sh
ffmpeg -i my-recording.mp3 -ac 1 -ar 48000 -sample_fmt s16 data/recording.wav
python classify.py "$PKG" data/recording.wav
```

| model | sample rate (`-ar`) | window |
|---|---|---|
| BirdNET v2.4 | 48000 | 144000 samples (3 s) |
| BirdNET v3.0, Perch v2 | 32000 | 160000 samples (5 s) |
| BattyBirdNET | 256000 | 144000 samples (0.5625 s) |

A file shorter than one window prints only the first line. The same for one window in C:
[Using it from C](REFERENCE.md#using-it-from-c).

## 7. Pick the package for your CPU

Each package file holds one CPU path, named at the end of the file name. Use:

| model | Raspberry Pi 5 (Cortex-A76) | Raspberry Pi 4 (Cortex-A72) |
|---|---|---|
| BirdNET v2.4 | `birdnet_v2.4-neon-fp16.bnpk` | `birdnet_v2.4-neon-fp32.bnpk` |
| BirdNET v3.0 | `birdnet_v3.0[_north-america-east]-neon-fp16.bnpk` | `birdnet_v3.0[_north-america-east]-neon-int8.bnpk` |
| Perch v2 | `perch_v2-neon-fp16-s8.bnpk` | `perch_v2-neon-int8.bnpk` |
| BattyBirdNET | `battybirdnet_<region>-neon-fp32.bnpk` | the same |

The bats ship as `neon-fp32` only: their `neon-fp16` package failed its gate (see [Measured speed](RESULTS.md)).
Ask the library which path a file would use on this machine:

```sh
build/cli weights/birdnet_v2.4-neon-fp32.bnpk --info | grep 'best path'
```

```text
best path for this CPU: neon-fp32
```

A package this CPU cannot run says `best path for this CPU: (none)`, and loading it fails with a message that names
the package to use instead, for example a `-neon-fp16.bnpk` on a Pi 4: `this CPU (any AArch64) can run none of the
package's paths: neon-fp16 (needs fp16 arithmetic); hint: use the neon-fp32 package of this model
(birdnet_v2.4-neon-fp32.bnpk, any AArch64) ...`. The full table is in [Supported CPUs](REFERENCE.md#supported-cpus).

## 8. Use more threads

A package's context can spread each window over several threads; the outputs were bit-identical at every thread count
checked (1 to 4).
`build/cli` scores a raw float32 file of whole windows and prints the time per window; `BNN_THREADS` sets the threads.
Make the raw file from the WAV, then time it at 1 and 4 threads:

```sh
python -c "import sys, wave, numpy as np; w = wave.open(sys.argv[1]); (np.frombuffer(w.readframes(w.getnframes()), '<i2') / 32768).astype(np.float32).tofile(sys.argv[2])" data/soundscape.wav data/soundscape.f32
build/cli "$PKG" 0 data/soundscape.f32 data/scores1.f32
BNN_THREADS=4 build/cli "$PKG" 0 data/soundscape.f32 data/scores4.f32
cmp data/scores1.f32 data/scores4.f32 && echo identical
```

```text
fp16: 40 windows x 1 reps: median 4.68 ms/window, min 4.55 ms/window
fp16: 40 windows x 1 reps: median 2.47 ms/window, min 2.46 ms/window
identical
```

Those are a Mac's timings. On Linux, `build/cli` also prints its peak and current memory (`VmHWM`, `VmRSS`) after
each timing; a Pi 4 with the `-neon-fp32.bnpk` package prints:

```text
fp32: 40 windows x 1 reps: median 129.61 ms/window, min 129.42 ms/window
VmHWM:	   82156 kB
VmRSS:	   82156 kB
fp32: 40 windows x 1 reps: median 65.26 ms/window, min 65.01 ms/window
VmHWM:	   83128 kB
VmRSS:	   83128 kB
identical
```

The Pi tables, pinned to cores, are in [Measured speed](RESULTS.md). In Python, pass `threads=4` to
`bnneon.Model`; in C, call `bnn_set_threads(ctx, 4)`. One thread uses the least CPU time per window; more threads
lower the latency of each window.

## 9. Run the tests

The tests are plain scripts. Each prints `N passed, 0 failed` (or a `skip` line when a file it needs is missing) and
exits nonzero on a failure:

```sh
for t in tests/test_*.py; do echo "$t: $(python -I "$t" 2>&1 | tail -1)"; done
```

```text
tests/test_agree.py: 17 passed, 0 failed
tests/test_compiler_units.py: 85 passed, 0 failed
tests/test_errors.py: 18 passed, 0 failed
...
tests/test_python_binding.py: 38 passed, 0 failed
tests/test_readme_example.py: 7 passed, 0 failed
...
tests/test_scripts_units_s2.py: 67 passed, 0 failed
```

The model-dependent tests use the packages in `weights/` that this CPU can run: the more models you compile, the
more they check. On a Pi 4, which cannot run the `-neon-fp16.bnpk` package, `tests/test_python_binding.py` reports
`24 passed, 0 failed` and `tests/test_readme_example.py` `5 passed, 0 failed` (each with a `skip` line for that
package). `tests/test_errors.py` runs its compiler and loader parts with the model you downloaded:

```sh
python -I tests/test_errors.py --model data/battybirdnet/birdnet-v2.4-embeddings-fp32.onnx \
  --labels data/birdnet/birdnet_labels_v2.4.txt | tail -1
```

```text
98 passed, 0 failed
```

That is the count measured on macOS, which skips the out-of-memory check (it needs Linux `ulimit -v`). A Pi 4 also
gives `98 passed, 0 failed` (measured): it runs the out-of-memory check and its loader cases on a `neon-fp32` package
(one more compile check), and skips the `neon-fp16` run. On Linux with fp16 arithmetic (a Pi 5) the out-of-memory
check runs and nothing is skipped, so the count is `99 passed, 0 failed` (from the test's code; not measured here).
It needs about 300 MB of scratch space in the system's temporary directory and removes it at the end. Where `/tmp` is
a small RAM disk (Raspberry Pi OS trixie mounts a tmpfs of half the memory there), add `--workdir DIR` to put the
scratch directory in `DIR` instead, for example `--workdir data`.

Before a pull request, also run the lint gate (in a git checkout): `pip install -e '.[lint]'`, then `scripts/lint.sh`
(no output and exit 0 when clean).

## 10. Where to go next

- **Other models:** download their inputs as listed in [Getting the models](REFERENCE.md#getting-the-models), then
  `scripts/compile_models.sh perch`, `bats` or `v3` (or `all`). BirdNET v3.0 compiles to three packages per model; the
  global one needs about 5 GB of memory to compile, so compile it on a larger machine and copy the packages.
- **Your own model** (an ONNX export, or a BirdNET-Analyzer custom classifier via TFLite):
  [docs/ADDING_A_MODEL.md](ADDING_A_MODEL.md), from `bnc check` to a gated package.
- **Accuracy:** how each package was checked against the fp32 reference: [docs/GATES.md](GATES.md).
- **From C:** [Using it from C](REFERENCE.md#using-it-from-c) and [Input and output per model](REFERENCE.md#input-and-output-per-model).
  **From Python:** [Python](REFERENCE.md#python). **From another language:** [Using it from other languages](REFERENCE.md#using-it-from-other-languages).
- **BirdNET-Go:** the engine runs behind an opt-in `neon` backend in a BirdNET-Go binding that is not published yet;
  until then, use the library from C or Python.

## Troubleshooting

**`bnc: no module named 'onnx'; hint: install the compiler extra: pip install -e '.[compiler]'`**: the package was
installed without the compiler's dependencies. Run `pip install -e '.[compiler]'` in the active environment.
**`bnc: command not found`**, or **`ModuleNotFoundError: No module named 'numpy'`** or **`'bnneon'`**: the environment
is not active (a new terminal starts without it); run `. .venv/bin/activate`.

**`OSError: build/libbnneon.so: cannot open shared object file: No such file or directory`** (Linux) or
**`OSError: dlopen(build/libbnneon.so ...)`** (macOS) from Python: the library is not built, or you are not in the
checkout's root (`classify.py` loads `build/libbnneon.so` relative to the current directory). Run `make` and run the
script from the checkout's root, or give `bnneon.Model` the library's full path (`lib=`; without `lib=`, the binding
reads `BNNEON_LIB`).

**`this CPU (...) can run none of the package's paths: neon-fp16 (needs fp16 arithmetic ...)`**, or
`best path for this CPU: (none)`: the package needs CPU features this board lacks (an fp16 package on a Pi 4; the
BirdNET v3.0 `neon-fp16` packages also need dotprod). Use the package the message names, usually `-neon-fp32.bnpk`
(v3.0 and Perch on a Pi 4: `-neon-int8.bnpk`), or compile one with `--paths neon-fp32` (step 7).

**`44100 Hz, but the model needs 48000 Hz`** (from `classify.py`), or confidences that make no sense: the audio is
not at the model's sample rate. The library does no resampling and cannot tell; it scores whatever samples it gets.
Convert with `ffmpeg -i IN -ac 1 -ar 48000 -sample_fmt s16 OUT.wav` (the rate per model is in step 6). A stereo file
must be mixed down the same way (`-ac 1`); `bnneon.Model.infer` raises `ValueError` for a window of the wrong length.

**`cannot open weights/....bnpk; hint: check the path and permissions`**, or
**`model error: data/....onnx: file not found`**: the file is not where the command looks. Packages are not in the
repository; compile them (step 5). `scripts/compile_models.sh` reads its inputs from the paths in step 4 and
[Getting the models](REFERENCE.md#getting-the-models), or from the variables named there (for example
`V24=/path/model.onnx V24L=/path/labels.txt scripts/compile_models.sh v24`).

**`warning: 'cblas_sgemm' is deprecated`** when building on macOS: harmless, from the v2.4 fp32 reference; the build
still succeeds. A build on x86 fails: the code is NEON only.

Every library and compiler message, with its fix: [docs/ADDING_A_MODEL.md](ADDING_A_MODEL.md), section 10.
