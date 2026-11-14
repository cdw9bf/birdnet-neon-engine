# birdnet-neon-engine: the libbnneon NEON engine

A NEON inference engine for bioacoustic classifiers on 64-bit ARM: BirdNET v2.4, BirdNET v3.0, Perch v2 and
BattyBirdNET-style heads. One C library (`libbnneon.so`) with a small C API (`src/bnneon.h`); the model file you
load decides what runs. A context runs on the calling thread, or on N threads for a compiled package
(`bnn_set_threads`; the outputs were bit-identical at every thread count checked, 1 to 4). It links only libc, libm and
pthread on Linux, and does no resampling. It is the engine behind an opt-in `neon` backend for BirdNET-Go (a binding not
published yet), but nothing in it depends on BirdNET-Go.

The repository holds the library, a command-line tool (`build/cli`), and the Python package `bnneon`: a ctypes binding
and `bnc`, the compiler that turns an ONNX model into a `.bnpk` model package. No model files are included: you
download the original models and compile them yourself ([Quick start](#quick-start)), and each model keeps its own
license.

It is for people who run BirdNET, Perch or bat classifiers on a Raspberry Pi 4 or 5 (or another 64-bit ARM board) and
want them to run faster, and for developers who want to call such models from C, Go or Python.

On a Raspberry Pi 5, BirdNET v2.4 takes 19.5 ms per 3-second window on one thread and 11.5 ms on four, where
BirdNET-Go's default model file, measured on one thread, takes 146.8 ms; on a Pi 4, 129.5 / 65.6 ms against 536.0 ms.
Every number, its conditions and its caveats: [docs/RESULTS.md](docs/RESULTS.md).

![ms per window, BirdNET-Go's stock default vs the NEON engine](results/figures/final_perf.png)

*The figure compares BirdNET-Go's default model file at one thread with the engine at one and four threads.
BirdNET-Go itself runs ONNX Runtime on as many threads as the CPU has cores (4 on a Pi). Like for like at 4 threads,
the default takes 68.5 ms (Pi 5) / 204.7 ms (Pi 4) for BirdNET v2.4 against the engine's 11.5 / 65.6 ms, and
337.7 / 885.8 ms for BirdNET v3.0 north-america-east against 52.7 / 325.6 ms.*

Median ms per window, pinned to cores: the engine at 1 / 4 threads, and the default model file BirdNET-Go `main`
selects for that model on that board, run at one thread by Python ONNX Runtime 1.31.1 in a timing harness (in an
earlier session on the same boards). All rows, the other stock options and peak memory:
[docs/RESULTS.md](docs/RESULTS.md).

| model | Pi 5 engine, 1 / 4 threads | Pi 5 default | Pi 4 engine, 1 / 4 threads | Pi 4 default |
|---|---|---|---|---|
| BirdNET v2.4 | 19.5 / 11.5 (`neon-fp16`) | 146.8 | 129.5 / 65.6 (`neon-fp32`) | 536.0 |
| BirdNET v3.0 north-america-east | 126.4 / 52.7 (`neon-fp16`) | 642.3 | 949.3 / 325.6 (`neon-int8`) | 2,321.8 |
| BirdNET v3.0 global | 146.9 / 66.9 (`neon-fp16`) | 754.3 | 1,125.8 / 375.4 (`neon-int8`) | 2,646.0 |
| Perch v2 | 142.0 / 49.2 (`neon-fp16-s8`) | 635.9 | 1,113.8 / 336.1 (`neon-int8`) | 2,210.4 |
| BattyBirdNET (EU) | 32.8 / 15.6 (`neon-fp32`) | 79.9 | 123.3 / 59.0 (`neon-fp32`) | 282.1 |

## What it runs

Each package file holds one CPU path, named at the end of the file name. Use:

| model | Raspberry Pi 5 (Cortex-A76) | Raspberry Pi 4 (Cortex-A72) |
|---|---|---|
| BirdNET v2.4 | `birdnet_v2.4-neon-fp16.bnpk` | `birdnet_v2.4-neon-fp32.bnpk` |
| BirdNET v3.0 | `birdnet_v3.0[_north-america-east]-neon-fp16.bnpk` | `birdnet_v3.0[_north-america-east]-neon-int8.bnpk` |
| Perch v2 | `perch_v2-neon-fp16-s8.bnpk` | `perch_v2-neon-int8.bnpk` |
| BattyBirdNET | `battybirdnet_<region>-neon-fp32.bnpk` | the same |

`neon-fp16` needs ARMv8.2 with fp16 arithmetic (+ dotprod for the int8 parts): a Raspberry Pi 5 or another
A55/A78-class CPU. `neon-fp32` and `neon-int8` run on any AArch64, including a Pi 4. The bats ship as `neon-fp32` only:
their `neon-fp16` package failed its gate. The library checks CPU features at run time, so one build loads on both a
Pi 4 and a Pi 5, and a package this CPU cannot run fails to load with a message that names the package to use instead.
It runs on the CPU only: **there is no x86 path** (the code is NEON only and the build fails on x86) and no GPU path.
Apple silicon builds and runs it for development. Details: [Supported CPUs](docs/REFERENCE.md#supported-cpus).

Accuracy: each engine row above passed its accuracy gate against the fp32 reference of its model
([docs/GATES.md](docs/GATES.md); result files in `results/`). The bird gates replay BirdNET-Go's detection
post-processing on recordings from the northeastern US, which exercises the 65 annotated species in them; every other
class is checked by window-level agreement only. The bat gate is window agreement on its own recording set.

## Quick start

You need 64-bit ARM Linux (on a Raspberry Pi: 64-bit Raspberry Pi OS) or an Apple silicon Mac, a C compiler
(gcc >= 10 or clang), make, git, curl and Python 3.11 or newer. On Raspberry Pi OS or Debian:
`sudo apt install -y git gcc make libc6-dev curl python3-venv`. Run every command from the checkout's root.

Clone, build the library and `build/cli`, and install the Python package with the compiler's dependencies (onnx and
ONNX Runtime):

```sh
git clone https://github.com/cdw9bf/birdnet-neon-engine.git
cd birdnet-neon-engine
make -j4
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[compiler]'
```

Download BirdNET v2.4 as ONNX and BirdNET-Go's English label file (the model is licensed CC BY-NC-SA 4.0,
non-commercial, and so is every package you compile from it; see [License](#license)):

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

Compile the two packages, one per CPU path, into `weights/`, and choose the one for this machine. **On a Raspberry
Pi 4** (or any other 64-bit ARM CPU without fp16 arithmetic) the `-neon-fp16.bnpk` package cannot run: set `PKG` as
the comment says.

```sh
scripts/compile_models.sh v24
PKG=weights/birdnet_v2.4-neon-fp16.bnpk   # on a Raspberry Pi 4: PKG=weights/birdnet_v2.4-neon-fp32.bnpk
build/cli "$PKG" --info | grep 'best path'
```

```text
...
wrote weights/birdnet_v2.4-neon-fp16.bnpk: 28.4 MB, 201 tensors, 6522 classes, emb 1024
...
wrote weights/birdnet_v2.4-neon-fp32.bnpk: 52.1 MB, 166 tensors, 6522 classes, emb 1024
...
best path for this CPU: neon-fp16                  <- neon-fp32 on a Pi 4, with the -neon-fp32.bnpk package
```

Run it on the 2-minute example soundscape of BirdNET-Analyzer (48 kHz, mono, 16-bit WAV; see that repository for its
terms) with a short script that uses only the Python standard library, numpy and the `bnneon` binding:

```sh
curl -fL -o data/soundscape.wav \
  https://raw.githubusercontent.com/birdnet-team/BirdNET-Analyzer/9db5740e47d353d1bdbe2747068ccc5ef4c9d8e1/birdnet_analyzer/example/soundscape.wav
shasum -a 256 -c <<'EOF'
df312b45bc82ce4c638c3e9e09d748702ea14a91ec29e4e8e0676d3e3e015fd7  data/soundscape.wav
EOF
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

The engine does not read audio files, resample or mix channels: convert your own recordings to mono at the model's
rate first (`ffmpeg -i my-recording.mp3 -ac 1 -ar 48000 -sample_fmt s16 data/recording.wav`). The environment and `PKG`
last for this shell only: in a new terminal, run `. .venv/bin/activate` and set `PKG` again.

Every step with its full output, `bnc check`, the package hashes, threads and the tests:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Using it from C

[`quickstart.c`](docs/REFERENCE.md#using-it-from-c) is a complete program that loads a model, scores one window and
prints the five best classes (`tests/test_readme_example.py` compiles and runs it). Save it as `quickstart.c`, build
it against the library, turn the recording above into a raw float32 file, cut its first 3-second window (144000
float32 samples = 576000 bytes) and run it on `PKG`:

```sh
cc -O2 -I src quickstart.c "$PWD/build/libbnneon.so" -lm -o quickstart
python -c "import sys, wave, numpy as np; w = wave.open(sys.argv[1]); (np.frombuffer(w.readframes(w.getnframes()), '<i2') / 32768).astype(np.float32).tofile(sys.argv[2])" data/soundscape.wav data/soundscape.f32
head -c 576000 data/soundscape.f32 > data/window.f32
./quickstart "$PKG" data/window.f32
```

```text
birdnet_v2.4, neon-fp16 path (fp16)
  4771  0.8125  Poecile atricapillus_Black-capped Chickadee
   636  0.0921  Baeolophus bicolor_Tufted Titmouse
  3294  0.0452  Loxia leucoptera_White-winged Crossbill
  5629  0.0292  Spinus tristis_American Goldfinch
  4774  0.0246  Poecile gambeli_Mountain Chickadee
```

On a Pi 4, `PKG` is the `-neon-fp32.bnpk` package; its output for the same window is in
[docs/REFERENCE.md](docs/REFERENCE.md#using-it-from-c), with the inputs, outputs and modes per model, threads, errors
and API stability. The API is plain C, so any language with a C FFI can use it (a BirdNET-Go binding does so with cgo
and dlopen; it is not published yet).

## Using it from Python

`classify.py` in the [Quick start](#quick-start) is a complete example. `bnneon.Model(path, mode=0, lib=None,
threads=1)` loads a model (`threads=4` spreads each window over four threads) and raises `bnneon.BnnError` with the
library's message; `infer` returns the scores exactly as `bnn_infer` wrote them (and the embedding), and
`confidences` applies `logit_scale` and the model's `post` (sigmoid or softmax). The binding needs only numpy
(`pip install -e .`, without the compiler extra), and it is one self-contained file: copy `python/bnneon/binding.py`
into your project as `bnneon.py`. The rest of the binding: [docs/REFERENCE.md](docs/REFERENCE.md#python).

## Documentation

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md): the quick start step by step, with every output (and the
  Raspberry Pi 4's), `bnc check`, reproducible package hashes, threads and timing, the tests and the lint gate, and
  troubleshooting.
- [docs/REFERENCE.md](docs/REFERENCE.md): supported CPUs, the build (and a Linux arm64 container), input and output per
  model, the model files, where to get every model, other languages and Python, API and format stability, the
  evaluation data.
- **Other models:** download their inputs as listed in [Getting the models](docs/REFERENCE.md#getting-the-models), then
  `scripts/compile_models.sh perch`, `bats` or `v3` (or `all`). BirdNET v3.0 compiles to three packages per model; the
  global one needs about 5 GB of memory to compile, so compile it on a larger machine and copy the packages.
- **Your own model** (an ONNX export, or a BirdNET-Analyzer custom classifier via TFLite):
  [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md), from `bnc check` to a gated package.
- [docs/GATES.md](docs/GATES.md): the accuracy gates, their declared criteria and the evaluation data.
- [docs/RESULTS.md](docs/RESULTS.md): every timing, its conditions, threads and peak memory.
- [MODEL_CARD.md](MODEL_CARD.md): the BirdNET v2.4 and v3.0 engine files.
- [results/COMPARISON.md](results/COMPARISON.md): the engine against BirdNET-Go's stock options on the same boards.
- [results/MENU.txt](results/MENU.txt): every measured option, session by session, with the gate verdicts.

## Troubleshooting

The common failures (the compiler extra not installed or the environment not active, the library not built, a package
this CPU cannot run, audio at the wrong sample rate, a file not where a command looks) and their fixes are in
[Getting started, Troubleshooting](docs/GETTING_STARTED.md#troubleshooting). Every library and compiler message, with
its fix: [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md#10-troubleshooting-every-message-and-its-fix), section 10.

## License

The engine code is licensed under the Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). The Apache
License does not cover the files under `weights/` or any compiled model package: the model files have their own
licenses, carried inside each file:

- BirdNET v2.4 (`.bn24`, `birdnet_v2.4-*.bnpk`): CC BY-NC-SA 4.0, with the BirdNET attribution (K. Lisa Yang Center
  for Conservation Bioacoustics, Cornell Lab of Ornithology, and Chemnitz University of Technology). The sources
  differ by one term: the [BirdNET-Analyzer README](https://github.com/birdnet-team/BirdNET-Analyzer) (section
  "License") and the `LICENSE` of [tphakala/BirdNET-v2.4](https://huggingface.co/tphakala/BirdNET-v2.4) state
  CC BY-NC-SA 4.0 for the models, while the Zenodo record
  [10.5281/zenodo.15050749](https://doi.org/10.5281/zenodo.15050749) lists CC BY-NC 4.0. The files here carry
  CC BY-NC-SA 4.0, which keeps the record's terms and adds ShareAlike.
- BirdNET v3.0 Developer Preview 3.1 (`birdnet_v3.0*.bnpk`): CC BY-SA 4.0 plus the BirdNET+ V3.0 Developer Preview
  Terms of Use, both from the Zenodo record [10.5281/zenodo.20703646](https://doi.org/10.5281/zenodo.20703646)
  (license field and `TERMS_OF_USE.txt`). The Terms provide the models for research and evaluation, require
  attribution (for example "Powered by BirdNET"), forbid any use for poaching or for a military purpose, and name the
  Museum für Naturkunde Berlin among the institutions involved, next to the two above.
- Perch v2: Apache-2.0 (Google Research), as tagged on the ONNX conversions
  [justinchuby/Perch-onnx](https://huggingface.co/justinchuby/Perch-onnx) and
  [tphakala/Perch-v2-Models](https://huggingface.co/tphakala/Perch-v2-Models).
- BattyBirdNET heads: CC BY-NC-SA 4.0, as the [BattyBirdNET-Analyzer README](https://github.com/rdz-oss/BattyBirdNET-Analyzer)
  states (section "License") and as derivatives of BirdNET v2.4.
- A custom classifier: whatever its source says, combined with its backbone's license
  ([docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md), section 5).

`weights/LICENSE-*.txt` hold the license notices compiled into the shipped packages (they name the license and link
its text) and `weights/LICENSE-WEIGHTS.txt` the notice for the `.bn24` files.
