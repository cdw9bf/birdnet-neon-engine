# BirdNET v2.4 weights for the NEON engine (`BirdNET_v2.4_neon.bn24`)

**Status:** not published. You make the file from the original model with `scripts/export_weights.py`; no copy is
hosted.

## What it is
A re-layout of the published BirdNET v2.4 model (`BirdNET_v2.4_fp32.onnx`, 6522 classes, 3 s at 48 kHz) into the
flat, mmap-able `.bn24` format read by the bnneon engine. Tensors are copied from the ONNX file (BatchNorm is already
folded there); only the memory layout changes (OHWI convolutions, `[out][in]` linear layers, transposed mel
matrices). No retraining, quantization or re-fitting is stored in the file; the engine converts weights to fp16 or
keeps them in fp32 at load time.

- Producer: `scripts/export_weights.py BirdNET_v2.4_fp32.onnx <labels> out.bn24` (engine repository).
- Labels: the sha256 of the label file used for the export is stored in the header; a host can compare it with its
  label file (`bnn_verify_labels`).
- Size: ~62 MB.

## License
CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/), inherited from BirdNET v2.4: the
BirdNET-Analyzer README (https://github.com/birdnet-team/BirdNET-Analyzer, section "License") states this license for
the models. The Zenodo record of the v2.4 model files (https://doi.org/10.5281/zenodo.15050749) lists CC BY-NC 4.0;
CC BY-NC-SA 4.0 keeps its terms and adds ShareAlike. Attribution:
K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology, and Chemnitz University of Technology
(Kahl et al. 2021, BirdNET). Non-commercial use only; derivatives keep this license. The engine code is licensed
separately, under the Apache License 2.0, which does not cover this file; the weights carry the BirdNET attribution
in addition.

## Accuracy (34.4 h evaluation set: 28 SSW hours + 4 Chronister dawn streams, 227,022 windows; docs/GATES.md)
Replay of BirdNET-Go's post-processing at the upstream default, the Raspberry Pi template and the overlap-1.5 configs,
against the official TFLite FP32 model:
- fp16 engine path (ARMv8.2 with fp16 arithmetic, e.g. Raspberry Pi 5): every period's detections identical;
  163 window-level threshold flips, all within 0.0044 of the threshold; logit SQNR 54 dB.
- fp32 engine path (any AArch64, e.g. Raspberry Pi 4): 0 flips, identical saved detections, 81 dB.

## Speed (1 thread)
This file, in the sessions of 2026-11-02/04 (`results/MENU.txt`): Raspberry Pi 5 19.3 ms per window (ONNX Runtime
with the stock INT8_ARM model: 147 ms), Raspberry Pi 4 131 ms (ONNX Runtime with this project's fused fp32 file, not
hosted: 213 ms). The BirdNET v2.4 packages give the same outputs bit for bit on the windows compared and are
preferred; in the later thread tables of `results/timing/` they take 19.5 ms (`neon-fp16`, Pi 5) and 129.5 ms
(`neon-fp32`, Pi 4).

## Limitations
Same as BirdNET v2.4. The file is only useful with the bnneon engine; it is not an ONNX/TFLite model.

---

# BirdNET v3.0 model packages (`birdnet_v3.0[_north-america-east]-neon-fp16|int8|fp32.bnpk`)

**Status:** not published. You build the packages from the original models with `scripts/compile_models.sh v3`; no
copy is hosted. Where to get the inputs: [docs/REFERENCE.md, "Getting the models"](docs/REFERENCE.md#getting-the-models).

## What it is
The published BirdNET v3.0 Developer Preview 3.1 fp32 ONNX models (global `birdnet-v3.0-preview3.1-fp32-b1.onnx`,
sha256 05535c3e..., 11,560 classes; the north-america-east slice
`birdnet-v3.0-preview3.1-north-america-east-fp32-b1.onnx`, sha256 d578c107..., 800 classes; 5 s at 32 kHz) compiled
by the engine's general compiler into `.bnpk` packages, one per CPU path:
- `neon-fp16`: for CPUs with fp16 arithmetic and dotprod (Cortex-A76 class, Raspberry Pi 5). The precision plan of the
  retired hand-written v3 engine (whose `.bnv3` files this library now refuses) as registry kernels: fp32 stem, fp16 Winograd for the 24-channel 3x3 convolutions, int8
  Winograd / im2col dot products for the dense 3x3 expansions, int8 dot products with per-row dynamic activations for
  every 1x1 convolution and the heads, fp16 depthwise, fp32 SE / GeM / attention pooling. The file holds fp32 weights
  (the attention heads' output scales folded into their class layers); the engine rounds them when it loads them.
- `neon-int8`: fp32 activations; the 1x1 convolutions and the heads (`--int8 pw,pwl,head,sed,lin`: every 1x1 convolution and head GEMM) as int8 matrix
  products with per-output-channel int8 weights (round half away from zero) and per-row dynamic int8 activations; the
  dense 3x3 convolutions as fp32 (Winograd F(4,3) above 24 input channels), depthwise, SE, GeM, front end and
  softmax in fp32. Any AArch64 CPU, no dotprod needed.
- `neon-fp32`: everything fp32 (Winograd F(4,3) as above).
Changes to the model: layout and packing, exact float64 folding of the summed linear heads and their scales, the
Winograd weight transform (float64), int8 rounding of the int8 layers' weights; no retraining. The package carries the
label file (and its sha256), the license and the attribution, and returns logits (post sigmoid).
- Producer: `scripts/compile_models.sh v3` (engine repository); the exact `bnc.py compile` commands are in it.
- Sizes and sha256 (bnc 3, this repository's `scripts/compile_models.sh v3`; byte-reproducible):
  `birdnet_v3.0_north-america-east-neon-fp16.bnpk` 131,695,232 bytes, 87b80320e7ff29d6...;
  `birdnet_v3.0-neon-fp16.bnpk` 540,017,664 bytes, e7440d476be1d8a0...;
  `birdnet_v3.0_north-america-east-neon-int8.bnpk` 57,168,960 bytes, ec661f1e47244396...;
  `birdnet_v3.0_north-america-east-neon-fp32.bnpk` 140,836,416 bytes, fdd66795dd196a7d...;
  `birdnet_v3.0-neon-int8.bnpk` 160,164,928 bytes, c795c7a3bb09bae9...;
  `birdnet_v3.0-neon-fp32.bnpk` 549,158,528 bytes, a4f068acb99f4d00... (full hashes in `weights/SHA256SUMS`).
  The previous files (bnc 2; their neon-fp16 pair 70189524... / be0c418a... are the K2B-G1 gated candidates) differ
  from these only in metadata (the license key's spelling and the compiler version): same tensors and ops,
  bit-identical outputs.
  Labels: the model's label file, compiled in (NA-east sha256 85b6357d..., global 4f4ef82f...).

## License
CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/) plus the BirdNET+ V3.0 Developer Preview Terms of Use,
inherited from BirdNET v3.0 (never NonCommercial; ShareAlike). Source: the Zenodo record of Developer Preview 3.1
(https://doi.org/10.5281/zenodo.20703646): its license field is CC BY-SA 4.0, and its `TERMS_OF_USE.txt` provides the
models for research and evaluation, requires attribution (for example "Powered by BirdNET") and forbids any use for
poaching or for a military purpose. Attribution: K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of
Ornithology, and Chemnitz University of Technology; the Terms of Use also name the Museum für Naturkunde Berlin. The notice compiled into the packages is
`weights/LICENSE-birdnet-v3.txt` (`neon-int8`, `neon-fp32`) or `weights/LICENSE-birdnet-v3-fp16.txt` (`neon-fp16`).

## Accuracy (34.4 h: 28 SSW hours + 4 Chronister streams, 92,829 windows on the overlap-2.2 grid; gates V002 and K2B-G1)
Against fp32 ONNX Runtime of the same model, BirdNET-Go post-processing (threshold 0.84, overlap 2.2, false-positive
filter level 3, geomodel range filter 0.05) at a fixed evaluation site in the eastern US and at each recording's own
location (docs/GATES.md), paired bootstrap over 37 periods; window agreement over all the model's classes. Every candidate scored is listed.
- NA-east fp16 (gate K2B-G1, the V002 criterion): 172 TP / 5 FP / 158 FN against 171 / 5 / 159 for fp32 (+1 TP), dP
  +0.02 pp [-0.03, +0.10], dR +0.32 pp [-0.66, +1.37]; Jaccard 0.976 at 0.84, every flip within 0.022 of the threshold,
  top-1 0.983, 50.8 dB. PASS.
- Global fp16 (K2B-G1): 175 / 5 / 155 against 174 / 5 / 156 (+1 TP), dP +0.02 pp [-0.03, +0.09], dR +0.32 pp [-0.66,
  +1.37]; over all 11,560 classes Jaccard 0.976, every flip within 0.022, top-1 0.975, 58.9 dB (60.8 dB on the 10,760
  classes outside the NA-east slice). PASS.
- NA-east int8: detections identical in total (171 TP / 5 FP / 159 FN), dP -0.00 pp [-0.05, +0.04], dR +0.01 pp
  [-0.87, +0.83]; Jaccard 0.980 at 0.84, every flip within 0.021 of the threshold, top-1 0.986, 53.0 dB. PASS.
- NA-east fp32: identical detections in every period, Jaccard 1.000, 1 flip (margin < 0.0001), 99.0 dB. PASS.
- NA-east int8 with fp32 projections (int8b, not shipped): 172 / 5 / 158, dR +0.30 pp [0.00, +0.88], 57.0 dB. PASS.
- Global int8: detections identical in total (174 / 5 / 156), dP -0.00 pp [-0.05, +0.04], dR +0.01 pp [-0.87, +0.83];
  over all 11,560 classes Jaccard 0.980 at 0.84, every flip within 0.021 of the threshold, top-1 0.981, 61.2 dB; its
  800 NA-east columns equal the NA-east package's. PASS.
- Global fp32: identical detections in every period, Jaccard 1.000, 1 flip (margin < 0.0001), 106.3 dB. PASS.
- Global int8 with fp32 projections (int8b, not shipped): 175 / 5 / 155, dR +0.30 pp [0.00, +0.88], 65.2 dB. PASS.
Only North American species with annotations (65) are exercised by the detection replay; the other classes by window
agreement only. Results: `results/gate_K2B_G1_*_result.txt` (fp16) and `results/gate_V002_*_result.txt`.

## Speed and memory of the `neon-fp16` packages (Raspberry Pi 5, Cortex-A76 2.4 GHz; ms per window at 1 / 2 / 3 / 4 threads; peak RSS)
- NA-east 126.4 / 74.5 / 58.7 / 52.7 ms, 81-91 MB. Global 146.9 / 87.6 / 72.2 / 66.9 ms, 336 MB.
- Timed with BirdNET-Go's detector stopped (`results/timing/pi5_detector_stopped.txt`). The retired `.bnv3` engine in
  the same session: 126.1 ms and 137 MB (NA-east) on its one thread. Outputs were bit-identical at every thread count (hash over the timing inputs).

## Speed and memory of the `neon-int8` / `neon-fp32` packages (Raspberry Pi 4, Cortex-A72 1.5 GHz; ms per window at 1 / 2 / 4 threads; peak RSS)
- NA-east int8 949 / 521 / 326 ms, 78-81 MB; global int8 1126 / 610 / 375 ms, 179-182 MB (`results/timing/pi4.txt`).
- NA-east fp32 1234 / 682 / 440 ms, 163 MB; global fp32 1532 / 861 / 616 ms, 564 MB (an earlier session, K068).
- Stock ONNX Runtime fp32 NA-east in an earlier session (H001): 2311 / 1406 / 927 ms (2,321.8 ms at 1 thread in the
  comparison session C001). Outputs were bit-identical at every thread count (hash over the timing inputs).

## Limitations
Same as BirdNET v3.0 Developer Preview 3.1 (not the GA build). Only useful with the bnneon engine. On a Cortex-A76
class CPU (Raspberry Pi 5) use the `neon-fp16` package (126-147 ms on one core, against ~360-430 ms for `neon-int8`);
the `neon-fp16` package needs fp16 arithmetic and dotprod and is refused on a Raspberry Pi 4.
