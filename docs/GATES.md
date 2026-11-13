# Accuracy gates and evaluation data

Every engine row in the README and in `results/MENU.txt` passed an accuracy gate: the candidate package was compared
with an fp32 reference of the same model on real recordings, against a pass criterion that was written down before
any candidate was scored. Every candidate scored is reported, pass or fail (the failures are in `DEADENDS.md`). This
page gives the evaluation data, the harness and the declared criteria; the result files are in `results/`.

The result files and session logs keep the labels of the development repository the runs were made in: commit ids
(for example the engine commit of a timing session), experiment ids (A017, K068, ...), names of session scripts, and
paths beginning `lab-data/`, `kernel-lab/` or `<lab-root>/`. They are provenance labels: that repository's history,
scripts and directories are not published, so they do not resolve here.

The detection-replay half of these gates needs the evaluation audio and a replay harness that are not part of this
repository (docs/ADDING_A_MODEL.md, appendix A). `scripts/gate_generic.py` is the window-level half, usable on any
model and any audio (guide, section 9).

## Evaluation data

None of the audio is in this repository; result files name recordings by file name and report aggregate counts only.

- **Bird set (34.4 h, 32 streams).** 28 hours of soundscapes from Sapsucker Woods, New York: S. Kahl, R. Charif,
  H. Klinck (2022), "A collection of fully-annotated soundscape recordings from the Northeastern United States",
  Zenodo, [10.5281/zenodo.7079380](https://doi.org/10.5281/zenodo.7079380), CC BY 4.0. Four dawn-chorus recordings
  (77 five-minute segments) from Powdermill Nature Reserve, Pennsylvania: L. M. Chronister, T. A. Rhinehart,
  A. Place, J. Kitzes (2021), "An annotated set of audio recordings of Eastern North American birds containing
  frequency, time, and species information", Dryad,
  [10.5061/dryad.d2547d81z](https://doi.org/10.5061/dryad.d2547d81z), CC0 1.0. The version matters: the gates used
  the 44.1 kHz WAV files of the record's earlier versions (Dryad versions 2 to 4, the last of 2021-04-01;
  `wav_Files.zip` of 1,428,295,296 bytes, sha256 8216cd49..., as Dryad's file listing for those versions gives it).
  The current version 5 of 2021-04-06 (`wav_Files.zip` sha256 ba60a906...) holds 32 kHz re-exports, which do not
  reproduce the recorded scores.
- **Development windows.** The precision screens (`devA`, `devP` in the result files) use windows from 8 further
  Sapsucker Woods hours of the same Zenodo record, disjoint from the 28 gate hours.
- **Bat set (26,505 windows, 1,052 streams).** Ultrasonic recordings resampled to 256 kHz as BirdNET-Go does,
  from four Dryad packages, all CC0 1.0 (descriptions from each record): Brinkløv, Macaulay, Bergler et al. (2023),
  "Open-source workflow approaches to passive acoustic monitoring of bats", AudioMoth and SM4BAT recordings at
  woodland edges in Denmark, annotated in Raven Lite,
  [10.5061/dryad.4xgxd25fh](https://doi.org/10.5061/dryad.4xgxd25fh); Jameson (2024), "buzzfindr: Automating the
  detection of feeding buzzes in bat echolocation recordings", recordings from five sites in Ontario,
  [10.5061/dryad.v9s4mw72b](https://doi.org/10.5061/dryad.v9s4mw72b); Fu, Kinniry, Kloepper (2019, the data of their
  2018 article), "Data from: The Chirocopter: a UAV for recording sound and video of bats at altitude", recordings
  from New Mexico, [10.5061/dryad.cf472n3](https://doi.org/10.5061/dryad.cf472n3); Bergmann, Burchardt, Wimmer,
  Kugelschafter et al. (2022), "The soundscape of swarming", Myotis recordings at autumn swarming sites in northern
  Germany with identified reference sequences,
  [10.5061/dryad.wdbrv15s8](https://doi.org/10.5061/dryad.wdbrv15s8). 520 recordings carry species labels (the
  Danish and German sets); the rest are agreement-only. The BattyBirdNET-Analyzer README names its training data
  as the NABat machine-learning data set, the Animal Sound Archive Berlin, ChiroVox, xeno-canto and recordings by
  individuals; none of the four Dryad packages above is among the named sources.

## Harness

- Decoding: ffmpeg PCM16 / 32768, then soxr (VHQ) to the model's rate (bird set), or BirdNET-Go's resampler (bat set
  and the Perch path); references and candidates are decoded identically.
- Detection replay: BirdNET-Go's post-processing (release 20261014: pending detections, the false-positive filter
  levels, the privacy filter) replayed over the annotated streams; precision and recall per configuration, with
  paired-bootstrap 95 % intervals of dP and dR (2000 resamples over the 37 recording periods).
- Window statistics: top-1 agreement, hit-set Jaccard at each threshold over (window, class) detections, flips
  (detections on one side only) with their margin |reference confidence - threshold|, max |dconf|, logit SQNR.
- Candidates are scored by the arm64 container build (gcc 14, the Raspberry Pi OS compiler); before each scored run its
  outputs were checked bit-identical with the Pi's on a sample (16 to 64 windows per model).

### The evaluation site (BirdNET v3.0 replays)

The BirdNET v3.0 replays apply the v3 geomodel range filter (threshold 0.05) at two locations: at each stream's own
recording location, and at one fixed evaluation site in the eastern US with each stream's own week. The site's
coordinates are not published: the result files show them as `<site-lat>, <site-lon>`. The scripts take the site as
a required argument (`scripts/gate_sim_v3_pkg.py ... --site LAT,LON`). A rerun at another site is a valid replay but
not the recorded one: the range filter's species list depends on the location, so even a nearby site can change the
counts. The replay at each stream's own location uses only public metadata and reproduces the recorded numbers; each
v3.0 verdict below required a pass at both locations.

## Declared criteria

### BirdNET v2.4 (A013, A017-A019; stock-runtime comparisons C009, C010)

227,022 windows of the bird set (the union of the 0.8 s and 1.5 s stride grids). References: the official TFLite
FP32 model (primary) and the fp32 ONNX model on ONNX Runtime. Replayed configurations: `upstream_default` (overlap 0,
threshold 0.8, FP filter level 0, range filter 0.01), `pi_template` (2.2, 0.8, level 2, 0.05) and `c4_deploy` (1.5,
0.8, level 2, 0.05).

PASS when, for each configuration, dP and dR are not significantly negative (95 % CI upper bound >= 0) and both mean
shifts are >= -0.5 pp, and at the window level the hit-set Jaccard at 0.8 is > 0.97 with no flip whose margin
exceeds 0.05. The fp32 ONNX model itself, scored against TFLite, marks the noise floor
(`results/gate_onnx_fp32_result.txt`). Results: `gate_A013_result.txt`, `gate_A017_result.txt` (fp16 path, PASS),
`gate_A018_result.txt` (int8 head, FAIL: DEADENDS D6), `gate_A019_result.txt` (fp32 path, PASS).

### Perch v2 (N012-N014, K110; stock TFLite C013)

142,293 windows of the bird set through BirdNET-Go's Perch replay (softmax, threshold, overlap scaled by 5/3).
Reference: `perch_v2_fp32.onnx` on ONNX Runtime fp32. The same rule as v2.4 for each configuration
(`upstream_default`, `pi_template`, `c4_deploy`; `perch_override` 2.2 / 0.5 / level 2 for information), the window
rule at each configuration's threshold, top-1 agreement over windows where the reference confidence reaches the
threshold. The stock `perch_v2_int8_arm.onnx` is numerically broken (it quantizes the mel MatMul: 10.9 dB against
fp32) and is not a comparator. Results: `gate_N012_result.txt` (`neon-fp16-s8`), `gate_N013_result.txt`
(`neon-fp16`), `gate_N014_result.txt` (`neon-fp32`), `gate_K110_result.txt` (`neon-int8`, the Pi 4 package): all
PASS.

### Perch v2 north-america-east slice (N016-N018)

The bird set is North American, so the north-america-east slice (999 classes, the catalog's index file) is the slice
that can be gated. Reference: the fp32 ONNX model's logits restricted to the 999 kept classes and softmaxed over that
set (what an exact fp32 slice computes). Rule: the Perch rule above, with `perch_override` among the gated
configurations. Results: `gate_N016_result.txt` (`neon-fp32`), `gate_N017_result.txt` (`neon-fp16-s8`),
`gate_N018_result.txt` (`neon-fp16`): all PASS. A calibration report without verdict compares the slice's
confidences with the full model's (`results/nae_calibration.txt`).

### BattyBirdNET (N008; stock comparisons C014)

The bat set at BirdNET-Go's bat threshold 0.5, all 11 regional heads. A: against the TFLite-numerics reference (the
official BirdNET v2.4 TFLite FP32 embeddings plus the fp32 heads, what BattyBirdNET-Analyzer runs), hit-set Jaccard
> 0.97 per region (regions with at least 20 reference hit windows) and pooled, no flip with margin > 0.05, top-1
agreement >= 0.99 on windows where either side has a hit. B: against the stock ONNX pipeline, the candidate must stay
within the TFLite reference's own distance from B (Jaccard within 0.005, at most 2 more flips above 0.05, top-1
within 0.005). Result: `neon-fp32` PASS (Jaccard 0.9997, max flip margin 0.021), `neon-fp16` FAIL (DEADENDS D8);
summarized in `results/MENU.txt`.

### BirdNET v3.0 (G003, V002, K2B-G1; stock comparisons C011, C012)

All 32 streams of the bird set on the overlap-2.2 grid (stride 8/6 s; 92,829 windows), decoded to 32 kHz. Reference:
fp32 ONNX Runtime of the same model, on the same windows, in the same run (`scripts/gate_v3_pkg.py`; only the top-10
per window of each side is stored). Replay (`scripts/gate_sim_v3_pkg.py`): threshold 0.84, FP filter level 3, top-10
over the model's labels, the v3 geomodel range filter at 0.05 at both locations above.

PASS when all of:

- (a) dP: 95 % CI lower bound > -0.5 pp, at both locations;
- (b) dR: 95 % CI lower bound > -1.0 pp, at both locations;
- (c) window hit-set Jaccard at 0.84 over all the model's classes > 0.97;
- (d) no flip at 0.84 with margin > 0.05;
- (e) top-1 agreement over all the model's classes >= 0.97;
- (f) global model only: logit SQNR on the 10,760 classes outside the north-america-east slice >= that on the 800
  inside it minus 3 dB.

G003 (the retired hand-written engine's global file) was gated by (a)-(d) at the evaluation site, with the stream
locations reported. V002 gated the `neon-fp32` and `neon-int8` packages of both models; K2B-G1 the `neon-fp16`
packages, with the V002 criterion unchanged. Before the K2B-G1 runs the scorer gained a per-stream completeness check
(a stream whose engine output arrives short is scored again from its first window); it cannot change a score, since
a stream's statistics are added only from a complete output. Results: `gate_G003_v3global_result.txt`,
`gate_V002_{nae,glob}_{fp32,int8,int8b}_result.txt`, `gate_K2B_G1_{nae,glob}_fp16_result.txt`: all PASS. The
`neon-fp16` packages reproduce the retired engine's numbers from the front end's image onwards (the package and the
engine were run on the same images and gave bit-identical outputs); the two front ends differ in the last bits of the
image (at most 2.4e-6), which is why the packages have their own gate.

## Bit-identity

The evidence for bit-identity is hash equality of the outputs: at 1 to 4 threads over the timing inputs (16 to 32
windows per model, `results/timing/`), and between the container build and a Pi's build on 16 to 64 windows per model
(the timing inputs, and the samples above). It is not a proof for every input. After a change
under `src/`, `scripts/cmp_suite.sh` compares the gated models' outputs with the gated reference outputs
(`results/bitref_SHA256SUMS.txt`); those reference files are not in the repository.
