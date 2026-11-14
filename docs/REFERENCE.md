# Reference

CPU paths, the build, the C example and the C API's inputs and outputs, the model files and where to get the models,
other languages and the Python binding, API and format stability, and the evaluation data.

## Supported CPUs

| CPU path | needs | for example | runs |
|---|---|---|---|
| `neon-fp16` (fp16 arithmetic, int8 dot products where the file says so) | ARMv8.2 with fp16 arithmetic (+ dotprod for the int8 parts) | Raspberry Pi 5 (Cortex-A76), A55/A78-class | everything |
| `neon-int8` (fp32 with int8 matrix products, no dotprod) | any AArch64 | Raspberry Pi 4 (Cortex-A72) | BirdNET v3.0, Perch v2 |
| `neon-fp32` | any AArch64 | Raspberry Pi 4 (Cortex-A72), Pi 3 (Cortex-A53, not measured) | BirdNET v2.4, BirdNET v3.0, Perch v2, BattyBirdNET |

BirdNET v3.0 compiles to three packages. `neon-fp16` (int8 dot products with fp16 parts: the dotprod + fp16 kernel
family) is the Raspberry Pi 5's path; `neon-int8` and `neon-fp32` run on any AArch64 and are the Pi 4's (they also run
on a Pi 5, about 2.7x slower than `neon-fp16`). The `neon-fp16` package replaced the legacy `.bnv3` file of the
hand-written v3 engine, which was retired: it reproduces that engine's numbers from the front end's image onwards at
the same speed and adds threads; this library refuses a `.bnv3` with `BNN_ERR_PACKAGE` and a message naming the
package to use. The `check` and `compile` commands
of each package are in [the guide, section 2.1](ADDING_A_MODEL.md) (the `neon-fp16` package takes
`--license-file weights/LICENSE-birdnet-v3-fp16.txt`), its gate command in section 9. The library checks CPU features at run time (HWCAP), so one
build loads on both a Pi 4 and a Pi 5; `bnn_init` refuses a file or mode the CPU cannot run instead of executing
an instruction the CPU lacks (the one unguarded case is mode 1, the slow fp32 reference for legacy `.bn24` files,
which is a development tool). **There is no x86 path**: the
code is NEON only and the build fails on x86. Apple silicon builds and runs it for development.

## Build

```
make                    # build/libbnneon.so and build/cli
make OUT=build-pi       # another output directory
```

Toolchain: an AArch64 Linux with gcc >= 10 or clang, and make. gcc 14 (Raspberry Pi OS / Debian trixie) is the
reference compiler; the gated outputs come from it. macOS on Apple silicon (clang) works for development and prints
one harmless warning (the deprecated `cblas_sgemm` of the v2.4 fp32 reference); the file is still named `libbnneon.so`. The Mac build compiles every file with fp16 and
dotprod (Apple silicon has both), so it shows what an A76-class CPU would pick; there is no switch to mask CPU features:
build the library you ship on AArch64 Linux (a Pi or the container below), and check a Pi 4's choice there. There is no `make install`: ship
`build/libbnneon.so` and `src/bnneon.h`.

Container (Linux arm64, gcc 14, from your checkout; its outputs matched a Pi's build bit for bit on every input
checked, 16 to 64 windows per model, but its timings are not a Pi's):

```
scripts/eval_container.sh build                 # image bnneon-eval:local (debian:trixie + gcc, make, libc)
scripts/eval_container.sh start bnneon-build    # --memory 2g, this checkout mounted at /eng
docker exec bnneon-build make -C /eng OUT=build-ctr
scripts/eval_container.sh stop bnneon-build
```

`build/cli` scores windows from a raw float32 file and prints the time per window, and inspects a file:
`build/cli MODEL --info` (metadata, license, best path for this CPU, which modes load), `build/cli MODEL --labels`
(the label file inside a package), `build/cli MODEL 0 windows.f32 out.f32` (n x n_samples float32 in, n x n_classes
float32 out).

## Using it from C

Save this as `quickstart.c`: it loads a model, scores one window and prints the five best classes.

<!-- tests/test_readme_example.py and tests/test_label_rule.py compile and run this block -->
```c
/* quickstart.c: load a model, score one window, print the five best classes. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bnneon.h"

/* the length of the label at l: its line without the '\n' and one '\r' before it (0 for NULL) */
static int label_len(const char *l) {
  size_t n = l ? strcspn(l, "\n") : 0;
  return (int)(n && l[n - 1] == '\r' ? n - 1 : n);
}

/* class i's label in a package's label file s, or NULL; the compiler's rule: a line of only spaces, tabs and '\r' is
 * blank and is not a label, and a first label line "inat2024_fsd50k" (Perch v2's full label file) is a header */
static const char *class_label(const char *s, int i) {
  for (int first = 1; s && *s; s = strchr(s, '\n') ? strchr(s, '\n') + 1 : NULL) {
    if (strspn(s, " \t\r") >= strcspn(s, "\n")) continue; /* blank */
    if (first && label_len(s) == 15 && !strncmp(s, "inat2024_fsd50k", 15)) i++;
    first = 0;
    if (i-- == 0) return s;
  }
  return NULL;
}

static void show_error(const char *what) { /* a failing call sets a one-line message: show it unchanged */
  fprintf(stderr, "%s: %s (kind %d)\n", what, bnn_last_error(), bnn_last_error_kind());
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s MODEL.bnpk|.bn24 [window.f32]\n", argv[0]); return 2; }
  if (bnn_api_version() != BNN_API_VERSION) { fprintf(stderr, "library API %d, header API %d\n", bnn_api_version(), BNN_API_VERSION); return 1; }

  bnn_info info = {.struct_size = sizeof info}; /* struct_size is required: the library fills that many bytes */
  if (bnn_probe(argv[1], &info)) { show_error("probe"); return 1; } /* header only: no weights loaded */
  if (!info.path[0]) { fprintf(stderr, "%s: this CPU can run none of the file's paths; hint: use the model's -neon-fp32.bnpk\n", argv[1]); return 1; }

  int rc = 1;
  bnn_ctx *ctx = NULL;
  char *labels = NULL;
  float *pcm = calloc((size_t)info.n_samples, sizeof *pcm), *out = malloc(sizeof *out * (size_t)info.n_classes);
  if (!pcm || !out) { fprintf(stderr, "out of memory\n"); goto done; }

  /* one window: info.n_samples float samples in [-1, 1) at info.sample_rate (from a file, or a 3 kHz tone) */
  if (argc > 2) {
    FILE *f = fopen(argv[2], "rb");
    size_t got = f ? fread(pcm, sizeof *pcm, (size_t)info.n_samples, f) : 0;
    if (f) fclose(f);
    if (got != (size_t)info.n_samples) { fprintf(stderr, "%s: need %d float32 samples\n", argv[2], info.n_samples); goto done; }
  } else {
    for (int i = 0; i < info.n_samples; i++) pcm[i] = 0.25f * sinf(6.2831853f * 3000.0f * i / info.sample_rate);
  }

  ctx = bnn_init(argv[1], BNN_MODE_DEFAULT); /* mode 0: the fastest path this CPU can run */
  if (!ctx) { show_error("init"); goto done; }
  bnn_get_info(ctx, &info); /* now with the chosen path and precision */
  if (bnn_infer(ctx, pcm, out, NULL)) { show_error("infer"); goto done; } /* this window was not scored: skip it */

  /* confidences: bnn_infer wrote logits (output_kind LOGITS) and info.post says what to apply (an older library also
   * ran legacy .bnv3 files, which wrote sigmoids: output_kind SIGMOID, nothing to apply) */
  float mx = out[0];
  for (int i = 0; i < info.n_classes; i++) if (out[i] > mx) mx = out[i];
  double sum = 0;
  for (int i = 0; info.post == BNN_POST_SOFTMAX && i < info.n_classes; i++) sum += exp((out[i] - mx) * info.logit_scale);
  for (int i = 0; info.output_kind == BNN_OUT_LOGITS && i < info.n_classes; i++) {
    float z = out[i] * info.logit_scale;
    out[i] = info.post == BNN_POST_SIGMOID ? 1.0f / (1.0f + expf(-z))
           : info.post == BNN_POST_SOFTMAX ? (float)(exp(z - mx * info.logit_scale) / sum) : z;
  }

  /* labels: a package carries its label file (class_label numbers it); legacy files do not: use the label file you
   * exported with */
  int n = bnn_meta(ctx, "labels", NULL, 0);
  if (n >= 0 && (labels = malloc((size_t)n + 1))) bnn_meta(ctx, "labels", labels, (size_t)n + 1);
  printf("%s, %s path (%s)\n", info.model_id, info.path, info.precision);
  for (int k = 0; k < 5 && k < info.n_classes; k++) {
    int best = 0;
    for (int i = 1; i < info.n_classes; i++) if (out[i] > out[best]) best = i;
    const char *l = class_label(labels, best);
    printf("%6d  %.4f  %.*s\n", best, out[best], label_len(l), l ? l : "");
    out[best] = -INFINITY;
  }
  rc = 0;
done:
  free(labels);
  free(pcm);
  free(out);
  bnn_free(ctx); /* NULL is fine */
  return rc;
}
```

Build it against the library, turn the recording of the [Quick start](../README.md#quick-start) into a raw float32
file, cut its first 3-second window (144000 float32 samples = 576000 bytes) and run it on `PKG` (on a Pi 4 the
`-neon-fp32.bnpk` package); without a window file it scores a 3 kHz tone:

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

With the `-neon-fp32.bnpk` package (a Pi 4, or any machine) the same window gives:

```text
birdnet_v2.4, neon-fp32 path (fp32)
  4771  0.8141  Poecile atricapillus_Black-capped Chickadee
   636  0.0921  Baeolophus bicolor_Tufted Titmouse
  3294  0.0452  Loxia leucoptera_White-winged Crossbill
  5629  0.0295  Spinus tristis_American Goldfinch
  4774  0.0246  Poecile gambeli_Mountain Chickadee
```

The library has no soname, so the executable records the path it was linked with: link the absolute path (as
above) or the path it will have on the target. On macOS the library's install name is the relative
`build/libbnneon.so`, so run the program from the checkout's root there.

## Input and output per model

The engine takes raw float32 PCM in [-1, 1) at the model's sample rate, one fixed-length window per call. Read the
numbers from `bnn_info` rather than hard-coding them; for the shipped files they are:

| family (`info.model`) | file | sample rate | window (`n_samples`) | `bnn_infer` writes | caller applies (`post`) | embedding (`emb_dim`) |
|---|---|---|---|---|---|---|
| BirdNET v2.4 (24) | `.bnpk`, legacy `.bn24` | 48000 | 144000 (3 s) | logits | sigmoid | 1024 |
| BirdNET v3.0 (30) | `.bnpk` | 32000 | 160000 (5 s) | logits | sigmoid | 1280 |
| Perch v2 (120) | `.bnpk` | 32000 | 160000 (5 s) | logits | softmax over the classes | 1536 |
| BattyBirdNET (224) | `.bnpk` | 256000 | 144000 (0.5625 s) | logits | sigmoid | 1024 |

- Multiply logits by `logit_scale` before `post` (1.0 for every shipped file). `hop_samples` is the model's
  suggested hop (0: the host decides).
- Pass `emb` (a buffer of `emb_dim` floats) to `bnn_infer` to get the embedding as well, or NULL.
- Class `i` is label line `i` of the label file, numbered as the compiler numbers them (`class_label` above, the
  Python binding's `labels()`): a line ends at `\n` (one `\r` before it is not part of the label), a line of only
  spaces and tabs is not a label, and a first line `inat2024_fsd50k` (Perch v2's full label file) is a header, not a
  class. The compiler refuses a label file that could be numbered another way (other line breaks, a NUL byte, lines
  blank only by other white space). A package carries its own label file (`bnn_meta(ctx, "labels", ...)`, or
  `build/cli MODEL --labels`) and its sha256 (`bnn_info.labels_sha256`; check yours with `bnn_verify_labels`).
  Legacy `.bn24` files carry only the hash: keep the label file you exported with.
- A regional Perch slice's plain softmax is over its own classes only; its confidences are not the full model's
  (about 1.65x higher at the median, `results/MENU.txt`).
- `bnn_infer` returning nonzero means only that window could not be scored (e.g. non-finite values on the fp16
  path): skip it, or score it with the `neon-fp32` path. Packages set `BNN_ERR_WINDOW` and a message; the legacy
  `.bn24` engine returns nonzero without setting one, so the message may be stale.
- Modes: 0 = the fastest path in the file this CPU can run; 2 = `neon-fp32`; 3 = `neon-fp16`; 4 = `neon-int8`;
  1 = fp32 reference (legacy `.bn24` files only, slow). A mode the file or the CPU cannot run is refused at
  `bnn_init`, with a message saying which file to use instead.
- One context per calling thread; contexts share nothing, so several may run on different threads. A package's
  context can also spread each window over N threads: `bnn_set_threads(ctx, N)` after `bnn_init` (1 by default, at
  most 64; `BNN_THREADS=N` for `build/cli`, `threads=N` in the Python binding). Every thread count checked (1
  to 4) gave bit-identical outputs. Legacy `.bn24` contexts take 1 only. `bnn_last_error` is per thread and is only
  meaningful right after a call failed.

## Model files

No model files are committed or hosted; you compile them from the original models (next section). `weights/` is
where `scripts/compile_models.sh` writes them (git-ignored), and where the tests look for them.

| file | what | how to make it |
|---|---|---|
| `.bnpk` | a compiled model package: metadata, license, label file, weights pre-packed per CPU path | `bnc compile model.onnx ...` from an ONNX model (after `pip install -e '.[compiler]'`; `python -I compiler/bnc.py compile ...` runs the same); `bnc check` first ([docs/ADDING_A_MODEL.md](ADDING_A_MODEL.md)); the shipped packages' exact commands are in `scripts/compile_models.sh` |
| `.bn24` | BirdNET v2.4 for the legacy v2.4 engine | `python -I scripts/export_weights.py BirdNET_v2.4_fp32.onnx labels.txt out.bn24`; the v2.4 packages give the same outputs bit for bit on the windows compared and are preferred |
| `.bnp2` | Perch v2 for the development tool `bench/perch_oracle.c` (`make build/perch_oracle`); the library does not load it | `python -I scripts/export_perch.py perch_v2_fp32.onnx labels.txt out.bnp2` |

Package names end in the path they hold (`-neon-fp16.bnpk`, `-neon-fp32.bnpk`, `-neon-int8.bnpk`); a host picks by CPU. Every file keeps
the license of the model it was made from, which is separate from the engine's: read it with
`bnn_probe_meta(path, "license", ...)` / `"attribution"` or `build/cli MODEL --info`. See [MODEL_CARD.md](../MODEL_CARD.md)
for the BirdNET v2.4 and v3.0 files. `weights/SHA256SUMS` lists the sha256 of every shipped package, and
`weights/SIZES.txt` their sizes.

### Getting the models

`scripts/compile_models.sh [v24|bats|perch|v3|all]` (in a Python with `pip install -e '.[compiler]'`, or with
`PY=<that python>`) reads the original models from `data/` (git-ignored; each path can be overridden by the variable
named in the script's header) and writes the packages into `weights/`. The shipped packages were compiled from the
ONNX conversions that BirdNET-Go downloads, from these Hugging Face repositories (each file downloads from
`https://huggingface.co/<repository>/resolve/main/<path>`; sizes and sha256 from the repositories' file listings,
checked 2026-11-11):

| model | repository and files | put them in (variable) |
|---|---|---|
| BirdNET v2.4 | [tphakala/BattyBirdNET-onnx](https://huggingface.co/tphakala/BattyBirdNET-onnx): `birdnet-v2.4-embeddings-fp32.onnx` (66.9 MB, sha256 b6b8f24d...; the official model with the 1,024-d embedding as a second output) | `data/battybirdnet/` (`V24`) |
| BirdNET v2.4 labels | BirdNET-Go's [`BirdNET_GLOBAL_6K_V2.4_Labels_en_us.txt`](https://github.com/tphakala/birdnet-go/blob/main/internal/classifier/data/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_en_us.txt) (sha256 b50b77b7..., the file compiled into the shipped package) | `data/birdnet/birdnet_labels_v2.4.txt` (`V24L`) |
| BattyBirdNET | the same repository: `birdnet-v2.4-embeddings-fp32-dfttrunc.onnx` (58.8 MB, sha256 b91139d3...), the 11 heads `fp32/BattyBirdNET-<region>_fp32.onnx` (46-157 KB) and their `labels/BattyBirdNET-<region>_Labels.txt` | `data/battybirdnet/` with `fp32/` and `labels/` (`B`) |
| Perch v2 | [tphakala/Perch-v2-Models](https://huggingface.co/tphakala/Perch-v2-Models): `full/perch_v2_fp32.onnx` (409 MB, sha256 bf0c8467...) and `full/perch_v2_labels.txt`; for the north-america-east slice `regional/north-america-east/perch_v2_north-america-east_indices.txt` and `_labels.txt` | `data/perch/` (`P`); the slice's two files in `data/perch/regional_nae/` (`PNAE`) |
| BirdNET v3.0 | [tphakala/BirdNET-v3.0-Models](https://huggingface.co/tphakala/BirdNET-v3.0-Models): `full/birdnet-v3.0-preview3.1-fp32-b1.onnx` (557 MB, sha256 05535c3e...), `full/birdnet-v3.0-preview3.1-labels-b1.txt`, `regional/north-america-east/birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx` (149 MB, sha256 d578c107...) and `-labels-b1.txt` | all four in `data/birdnet_v3/` (`V3`) |

The label files' sha256 are compiled into the packages (`build/cli PKG --info`). The original releases these files are
converted from (their licenses: see [License](../README.md#license)):

- BirdNET v2.4: Zenodo [10.5281/zenodo.15050749](https://doi.org/10.5281/zenodo.15050749) ("BirdNET Model V2.4":
  TFLite, Keras, protobuf and other formats, no ONNX); the fp32 ONNX conversion `BirdNET_v2.4_fp32.onnx` that
  `scripts/export_weights.py` takes is in [tphakala/BirdNET-v2.4](https://huggingface.co/tphakala/BirdNET-v2.4).
- BirdNET v3.0 Developer Preview 3.1: Zenodo [10.5281/zenodo.20703646](https://doi.org/10.5281/zenodo.20703646)
  (ONNX, TFLite, PyTorch; its `TERMS_OF_USE.txt` is the one in the Hugging Face repository above, byte for byte).
- Perch v2: Google's [Kaggle model page](https://www.kaggle.com/models/google/bird-vocalization-classifier); the
  ONNX conversion [justinchuby/Perch-onnx](https://huggingface.co/justinchuby/Perch-onnx) (its `perch_v2.onnx` has
  the same sha256 as `perch_v2_fp32.onnx` above).
- BattyBirdNET heads: [rdz-oss/BattyBirdNET-Analyzer](https://github.com/rdz-oss/BattyBirdNET-Analyzer).

## Using it from other languages

The API is plain C (ints, floats, C strings, one struct with a size handshake), so any language with a C FFI can
use it.

- **dlopen:** open `libbnneon.so` at run time, resolve the `bnn_*` symbols and call `bnn_api_version()` first; if
  the library is missing, keep your other runtime. Copy `bnn_info` exactly from `src/bnneon.h` and set
  `struct_size`.
- **Go:** a BirdNET-Go binding does exactly this with cgo and dlopen/dlsym (build tag `bnneon`; it reads both API
  generations, 1 and 2, runs the BirdNET v3.0 packages and sets `bnn_set_threads` from `birdnet.threads`). The
  binding is not published yet.
- **Python:** the `bnneon` package ([`python/bnneon/binding.py`](../python/bnneon/binding.py)), below.

### Python

ctypes, standard library + numpy. `pip install -e .` from this checkout installs the `bnneon` package with the binding
(numpy only; importing it needs no built library); `pip install -e '.[compiler]'` adds onnx and ONNX Runtime, which
the `bnc` compiler and its tools need, and `'.[plot]'` adds matplotlib for `scripts/plot_final_perf.py`. The binding is one self-contained file:
to use it without the package, copy `python/bnneon/binding.py` into your project as `bnneon.py`.

This example reads any format soundfile reads (`pip install soundfile`); `classify.py` in
the [Quick start](../README.md#quick-start) (and [Getting started, step 6](GETTING_STARTED.md#6-run-it-on-a-recording))
does the same for 16-bit WAV files with the standard library only.

```python
import bnneon   # python/bnneon/binding.py

import numpy as np
import soundfile as sf   # any reader that gives float32 samples in [-1, 1) will do

audio, rate = sf.read("recording.wav", dtype="float32")   # mono, at the model's rate (48 kHz for BirdNET v2.4)
with bnneon.Model("weights/birdnet_v2.4-neon-fp16.bnpk", lib="build/libbnneon.so") as m:  # or BNNEON_LIB=...; Pi 4: -neon-fp32
    print(m.info.model_id, m.info.sample_rate, m.info.n_samples, m.meta("license"))
    assert rate == m.info.sample_rate and audio.ndim == 1   # the engine does not resample or mix down
    labels = m.labels()               # None for legacy files
    for start in range(0, len(audio) - m.info.n_samples + 1, m.info.n_samples):   # back-to-back windows
        pcm = audio[start:start + m.info.n_samples]   # m.info.n_samples float32 samples
        scores, emb = m.infer(pcm)    # emb is None if emb_dim == 0
        conf = m.confidences(scores)  # logit_scale + post (sigmoid / softmax), as in the C example
        best = int(np.argmax(conf))
        print(f"{start / rate:7.1f} s  {conf[best]:.2f}  {labels[best] if labels else best}")
```

`bnneon.Model(path, mode=0, lib=None, threads=1)` (also `m.set_threads(n)`) raises `bnneon.BnnError` with the library's message and `.kind`
(`"PACKAGE"`, `"OOM"`, `"CPU"`, `"WINDOW"`, `"LABELS"`, `"USAGE"`); `infer` raises `ValueError` for a window of the
wrong length. `bnneon.probe(path)` and `bnneon.probe_meta(path, key)` read a file's metadata without loading the
weights; `bnneon.verify_labels(path, labels_txt)` raises (kind `"LABELS"`) unless the label file is the one compiled
in. `info` string fields (`model_id`, `path`, `precision`) are bytes; `info.labels_sha256_hex` is the label hash as hex. `scores` is exactly what `bnn_infer` wrote:
the test checks it equals the CLI's output bit for bit:

```
python -I tests/test_python_binding.py [--lib build/libbnneon.so] [--cli build/cli] [--model FILE ...]
```

(it uses the packages in `weights/` when present, and skips cleanly when there is no library or no model).
`python -I tests/test_readme_example.py` compiles the C example above with `-Werror` and runs it the same way.

## API and format stability

- **API version:** `BNN_API_VERSION` in the header and `bnn_api_version()` in the library, currently 2. Check they
  match when you dlopen. `bnn_info` grows only by appending fields, and the `struct_size` handshake lets an older
  host use a newer library (and the reverse): the library fills `min(yours, its)` bytes.
- **Package format:** `.bnpk` format version 1. Each package records the minimum reader version it needs; a library
  that is too old refuses it at load with `BNN_ERR_PACKAGE` and a message naming both versions, and an unknown
  opcode is refused the same way. Packages from older compilers (without per-op source names) load unchanged.
- **Legacy files:** `.bn24` version 1. `.bnv3` (BirdNET v3.0 for the retired hand-written engine) is refused with
  `BNN_ERR_PACKAGE`; libraries before the retirement read versions 1 and 2 (2 = heads wider than 2048 classes).
- **Errors:** every failure (except a legacy engine's `bnn_infer`, above, and an absent metadata key, which only
  returns -1) sets a one-line message, at most 511 bytes, of the form
  `<what failed>: <expected> vs <found>; hint: <what to do>`, plus a kind from `bnn_last_error_kind()`. Load errors
  name the op, kernel, source ONNX node and values (docs/ADDING_A_MODEL.md, section 10 and appendix B). A
  successful call does not clear the message.

## Evaluation data

The accuracy gates use public recordings; none of the audio is in this repository (sources, versions and the gate
criteria: [docs/GATES.md](GATES.md)):

- 28 hours of Sapsucker Woods soundscapes: Kahl, Charif, Klinck (2022), Zenodo
  [10.5281/zenodo.7079380](https://doi.org/10.5281/zenodo.7079380), CC BY 4.0.
- 4 dawn-chorus recordings from Powdermill Nature Reserve: Chronister, Rhinehart, Place, Kitzes (2021), Dryad
  [10.5061/dryad.d2547d81z](https://doi.org/10.5061/dryad.d2547d81z), CC0 1.0.
- 26,505 bat windows from four Dryad packages, all CC0 1.0 (Brinkløv et al. 2023; Jameson 2024; Fu, Kinniry,
  Kloepper 2019; Bergmann et al. 2022; DOIs in docs/GATES.md).

The BirdNET v3.0 detection replays apply the geomodel range filter at each recording's own location and at one fixed
evaluation site in the eastern US, whose coordinates are not published (docs/GATES.md).

