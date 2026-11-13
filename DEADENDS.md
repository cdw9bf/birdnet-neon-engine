# Dead ends

Approaches that were tried and dropped. Their code was never committed; each entry records the numbers and what might
revive it. The ids in parentheses (A0xx, N0xx, K1xx, ...) are the experiment ids that the result files under `results/` also use.

## D1 int8 everywhere (v3-style dynamic per-row activations + per-channel weights) on v2.4
- What: oracle fake quant, all GEMM layers int8 (A004). 34.7 dB logit SQNR vs TFLite on devA, max|dconf| 0.070,
  top-1 95.3 %. Weight-only int8: 35.2 dB, i.e. the weights (not the activations) are the problem.
  Per-tensor activations are worse (pwl 31.9 dB alone).
- Why: v2.4 per-channel weight ranges are too wide for 8 bits in many layers; the stem, the first block of each stage
  and p0 are the worst (41.8-47 dB alone).
- Revive: per-group weight scales (e.g. 32 along K) or int8 only for the >= 58 dB layers; QAT.

## D2 depthwise 3x3 in fp32 arithmetic (fp16 storage) on the A76 fp16 path (A009)
- What: padded fp16 input rows converted once into a 3-row fp32 ring, fp32 weights/accumulation,
  LUT SiLU, fp16 output; two layouts (8 ch x 1 px; 4 ch x 2 px).
- Numbers (Pi 5, core 3): dw 4.0-4.2 ms vs 1.67 ms for the fp16 kernel (kern_f16.c dw3x3_f16); accuracy vs TFLite on
  devA 55.3 / 55.1 dB vs 54.0 dB (the oracle predicted +3.5 dB; fp16 GEMM accumulation dominates the remaining error).
- Why dropped: +2.4 ms (+10 %) for +1.3 dB.
- Revive: if the gate is short of margin, try fp16 hi/lo weight pairs in the fp16 dw kernel (2 FMLA per tap, the
  oracle shows the loss is the fp16 rounding of the large cancelling taps, not the 9-term accumulation).

## D3 front end: DCT-I (even-part) trick to halve the FFT (A011)
- What: Re X_k of a real frame = DCT-I of y_j = x_j + x_{N-j}; DCT-I via a real FFT of half the
  length (cosft1 pre-processing, complex FFT of N/4 points) with odd bins from a running sum.
- Numbers: image vs the exact double-FFT front end 63.5 dB (max |d| 0.14) vs 97.2 dB for the radix-2 half-complex
  real FFT; not timed further.
- Why: the odd bins are a running sum of ~k/2 terms plus a 512-term cosine sum with cancellation, so fp32 error grows
  with k; |mel|^0.46 amplifies absolute errors on weak bins.
- Revive: only with double (or compensated) accumulation of the running sum and F_1; probably not worth it.

## D4 head GEMV with 8 rows in flight (A014)
- 16 independent fp16 FMLA chains instead of 8: head 1.48 -> 1.76 ms on the Pi 5 (DRAM-bound 13.4 MB stream; more
  concurrent row streams hurt the prefetcher). Reverted to 4 rows.

## D5 fp32 (Pi 4) path experiments that did not pay (A019a)
- fp16 head weight storage + FCVTL: head 6.8 -> 5.9 ms but 89.1 dB vs TFLite (below the 90 dB target). Removed.
- degree-4 exp2 polynomial in the fp32 SiLU: 90.7 dB vs 92.0, saves ~1 of ~18 ops. Not taken.
- explicit weight PRFM in the fp32 GEMM: pwl -3.5 ms but pc +0.7-1.4 ms, net within noise (kept only for panels
  that do not fit L1, fixed setting, no env switch).
- dw formulations: 2 px x 8 ch with weight reloads 21-29 ms (stack traffic), 8-ch blocks 20.5, channel-inner 2 px x 4
  ch 20.2, sliding window + split chains + column prefetch 20.0 (kept). The stride-2 first blocks are DRAM-latency
  bound (padded pw output 1.9-2.0 MB > 1 MB L2). Revive: fuse pw + dw per row band (est. -6 to -8 ms), asm SiLU pass
  without gcc's constant moves (est. -3 ms).

## D6 int8 SDOT head on the fp16 path (A015/A018; former mode 4)
- What: per-class symmetric int8 head weights x per-window asymmetric int8 features (SDOT GEMV).
  Pi 5: 18.3-18.5 ms vs 19.25-19.3 ms (-0.85 ms, head 13.4 -> 6.7 MB of DRAM).
- Full gate (results/gate_A018_result.txt): FAIL at c4_deploy vs TFLite (135/4/195 vs 137/4/193, dR -0.60 pp
  [-1.55, 0]); 364 flips (max margin 0.012), Jaccard 0.9927, 53.0 dB. Same failure pattern as the stock-runtime study's dynamic-int8
  head files and today's stock int8 default (2 TPs at one config): FAIL by the pre-declared -0.5 pp mean rule
  (1 TP = 0.30 pp passes, 2 TPs fail); the CI includes 0, so not statistically significant.
- Removed from the engine and the API (mode 4 no longer exists).
- Revive: per-group head weight scales or an fp16 head with int8 storage (FCVTL + scale), gated again.

# Compiler, runtime, Perch v2, BattyBirdNET

## D7 copying package weights into anonymous memory (N007b)
- What: rt_pkg.c copies the mmap'd package into one aligned malloc block at open (experiment switch, removed).
- Numbers (Pi 5 idle, core 3, devA64 x5): v2.4 fp16 package 19.32-19.34 ms (mmap) vs 19.24-19.27 ms (copy);
  legacy .bn24 engine 18.93-18.96 ms. Perch fp16 234.9 (mmap) vs 233.6 ms (copy).
- Why dropped: -0.07 ms (0.4 %) for a second copy of the weights in RAM (page cache + anonymous); the remaining
  0.35 ms gap to the legacy engine is not the weight memory.
- Revive: if huge pages become available on the Pi kernel (no THP today), a copy into a MAP_HUGETLB block.

## D8 BattyBirdNET on the fp16 path (N008, the bat gate)
- What: the bat packages' neon-fp16 path (the gated BirdNET v2.4 fp16 body; its embeddings are bit-identical to the
  v2.4 fp16 engine's) on ultrasonic audio.
- Numbers: the 26,505-window CC0 bat set (docs/GATES.md), 11 heads, threshold 0.5: FAIL vs the TFLite-numerics reference (USA and
  USA-EAST-high flips with margin 0.062 / 0.068 > 0.05; pooled Jaccard 0.986, 1,075 flips) and vs the stock ONNX
  (pooled Jaccard 0.985 vs floor 0.9975, 1,153 flips vs 194); embedding 44.8 dB vs TFLite. The fp32 path PASSES
  (vs TFLite pooled Jaccard 0.9997, 21 flips, max margin 0.021).
- Diagnosis (oracle fake quant on 256 windows of the bat set, embedding SQNR vs the exact oracle): fp16 storage alone
  59.7 dB, all fp16 weights 54.6 (dw 57.2, pw 63.9, pwl 63.9, pc 65.0, c0 65.3), full fp16 arithmetic 45.1; fp32
  c0 46.0; fp16 arithmetic per class on top of fp16 storage + weights (53.5): pw 49.3, pwl 49.6, pc 50.3, c0 51.2,
  p0 53.0, dw 53.5. The loss is spread over every layer class: bat embeddings are ~9 dB more fp16-fragile than bird
  embeddings (birds: 54.3 dB vs TFLite for the same code). No single-class split recovers it.
- Decision: bats ship the neon-fp32 package on every AArch64 (Pi 5 33.1 ms, still 2.4x the stock 78.2 ms).
- Revive: fp16 storage + fp32 arithmetic GEMMs (FCVTL in the K loop; ~fp32 speed, half the weight bytes), or fp32
  accumulation of fp16 products (FMLAL, not on the A76).

# BirdNET v3.0 global

## D9 BirdNET v3.0 through the generic compiler (G001, out-of-the-box test) - REVIVED
- What: `bnc.py check` / `compile` on the stock birdnet-v3.0-preview3.1-fp32-b1.onnx, as the guide said.
- Result then: exit 2 at node 0 (reflect Pad + Conv-framed DFT front end); no package. Gap behind it: the front end (power,
  mel, dB with top-db, min/max, bilinear Resize to 224 x 468, 3-channel affine), dense multi-channel 3x3 convs
  (47.5 % of 6.2 GMAC), GeM pooling, two SED attention heads, the 5-head sum.
- Why dropped then: several days for fp32 + fp16 compiled paths, and still ~2-4x slower on a Pi 5 than the legacy C3
  engine (int8 Winograd + SDOT, gated); the global model ships as a .bnv3 (G003 PASS, 150 ms on the Pi 5).
- Revived (V001-V002): the trigger named here happened (the Pi 4, no fp16 + dotprod). v3.0 now
  compiles through `bnc.py` (front-end family FE_DBIMG, dense k x k CONVK, GeM, SED, head fold) to neon-fp32 and
  neon-int8 packages with threads and Winograd F(4,3); gate V002. The Pi 5 part of the reasoning held: the compiled
  path is 2.6-2.8x slower than C3 on one A76 core (K066), so C3 stays the Pi 5 path. Still not done: a neon-fp16
  lowering of the dense k x k convs (no use while C3 exists). Later (K2-B, K203): the neon-fp16 path came as the
  dotprod + fp16 kernel family with C3's own kernels and gated (K2B-G1), and C3 was retired.

# BirdNET v3.0 on the Pi 4

## D10 gemm_f32: all rows in one M block for few-row GEMMs with large weights (K050)
- What: a design-review suggestion: when M x Kp x 4 bytes fit 768 KB (or 1 MB) and the weights exceed 128 KB, one M block
  (each weight panel read once from DRAM instead of once per 8-row block). Loop order only (outputs bit-identical).
- Numbers (Pi 4, core 3, NA-east fp32 direct package, 8 windows): 1641.9 ms (off) vs 1661.5 (768 KB) vs 1687.4 ms
  (1 MB). Slower: the A-row block then streams from L2 for every panel instead of staying in L1, and the A72's
  prefetcher already hides most of the weight stream of these layers.
- Revive: only with a K-blocked variant (A tile of a K slice in L1, partial sums in registers across panels).

## D11 depthwise conv in channel blocks on one thread (K054)
- What: run the channel-block form (the thread split, bit-identical) also on one thread when the 3-row band of the
  padded input exceeds 48 KB / 16 KB, so a block's rows stay in L1.
- Numbers (Pi 4, core 3): v3 NA-east int8 package 1118.6 -> 1113.3 / 1112.6 ms (-0.5 %); v2.4 fp32 package 132.0 ->
  135.5 ms (+2.6 %). Not taken.
- Revive: no; the depthwise time is in the SiLU and the stride-2 DRAM reads (D5), not the band.

## D12 int8 (no dotprod) GEMM: weight prefetch and other row-block sizes (K057)
- What: PRFM of the weight stream 512 / 1024 B ahead in the 4x4 SMULL tile; int8 row blocks of 8 KB / 24 KB of
  activation rows instead of 16 KB.
- Numbers (Pi 4, core 3, NA-east neon-int8 package): 1072.5 ms (default) vs 1086.6 (PRFM 512), 1087.7 (8 KB),
  1119.5 ms (24 KB); identical outputs. Not taken (the 1024 B variant's run did not complete: not pursued).
- Revive: an asm micro-kernel with a software-pipelined K loop; the intrinsics loop has no spills (checked).

# TFLite + XNNPACK comparison (C006-C014)

## D13 Perch v2 TFLite from the DFT-op ONNX (C013)
- Tried: `perch_v2_fp32.onnx` (ONNX DFT op, sha256 bf0c8467...) through onnxsim + onnx2tf 2.6.9, as the v3 files were built.
- Result: onnxsim with a static input shape fails (`[ShapeInferenceError] Inferred shape and existing shape differ in
  dimension 0: (4) vs (5)`); `onnx2tf -ois inputs:1,160000` lowers DFT to a custom op `ONNX_DFT`, which TFLite 2.17.1
  refuses ("Encountered unresolved custom op: ONNX_DFT ... Node number 27 failed to prepare").
- Instead: `perch_v2_no_dft_fp32.onnx` (DFT as a matmul) with the three rank > 2 constant-weight MatMuls rewritten
  as Reshape / Gemm / Reshape (ORT output bit-identical), so they become FULLY_CONNECTED rather than BATCH_MATMUL
  (450 -> 251 ms on the Mac).
- Revive: only if TFLite gains a DFT builtin.

## D14 The per-row int8 v3 TFLite files on a Pi 4 at TFLite 2.17.1's default delegate flags (C007)
- Tried: `v3na_fft_i8dyn_s0se.tflite` and `v3g_fft_i8dyn_s0se.tflite` through the C API 2.17.1 (BirdNET-Go's library
  and set-up) on the Cortex-A72, 1-4 threads.
- Result: SIGSEGV at the first invoke, a call through a null function pointer inside libtensorflowlite_c (gdb: pc 0x0
  under TfLiteInterpreterInvoke). The same with ENABLE_LATEST_OPERATORS. Runs with the delegate's QS8 flag cleared
  (1,401 / 1,584 ms) or without the delegate (1,445 ms); LiteRT 2.3 from Python runs it at its defaults (1,333 ms).
  The Pi 5 (A76, dotprod) and x86 are unaffected.
- Revive: with a newer TFLite library, or with delegate flags exposed in go-tflite (QS8 off on CPUs without dotprod).

## D15 int8 (no dotprod) GEMM on the A72: epilogue and accumulate variants (K061)
- What: (a) a joint 8-row dequant + SiLU epilogue (two 4x4 K loops per panel visit, 8 SiLU chains in flight);
  (b) SADDW/SADDW2 instead of SADALP for the int16 -> int32 accumulate; (c) an asm 4x4 K loop with software-pipelined
  loads (D12's revive note); (d) a saturating-narrow quantization store; (e) the fused 64-row quantize + GEMM loop on
  one thread.
- Numbers (Pi 4, 1.5 GHz, nae-int8, 1 thread total / pw_s8n ms): (a) 1060.4 / 541.9 vs 1048.5 / 530.6 (+2 %; 0 at 4
  threads: the FP pipe is throughput-bound and 8 chains spill, 35 vs 8 vector stores); (b) micro-benchmark 26.7 vs 22.5
  cycles per 16 multiplies (SADDW only partly dual-issues), not built; (c) not written: measured SMULL, SMLAL2 and
  SADALP all issue on the same pipe at 1/cycle, so the scheme's ceiling is 8.0 GMAC/s and gcc's loop (48 pipe ops + 6
  loads + 5 scalar per 256 MACs) leaves <= 2 %; (d) -1 ms, within noise (and -Inf would map to -128, not -127);
  (e) +3 ms. Every variant kept the output hashes.
- Revive: (c) only on an in-order core (Cortex-A53); (b) on another non-dotprod core with a second integer pipe.

## D16 fp32 Winograd F(4x4, 3x3) scheduling variants on the A72 (K062)
- What and numbers (Pi 4, nae-int8, convw_f32 x8 ms/window, 1 thread unless noted; baseline 261, final 234.5):
  panel-outer order (8 channels per pass, M block in L1) 302.5 at S8 / 250.6 at S32, worse with an A prefetch (337.7);
  larger strips with the L1 aliasing removed (S16 255.2, S32 261.8 at 1 thread; 4 threads 106-141 vs 93); prefetch of
  output lines / residual / input patches 264-266 (neutral or worse); a two-phase layer form (weights read once per
  layer, 3 MB V buffer per C48 layer) 267-285 (4 threads 103-109 vs 93); a bounded rendezvous of the 4 workers per
  channel block (shared L2 weights) 88-93 at 4 threads for 0-1000 us (no effect); Nb 24 / 96-192 at 1 thread and 64 at
  4 threads lose to 64 / 40.
- Why: the A block's 8 short row streams (3-4 lines) fall below what the A72 prefetcher tracks, so re-reading it
  per panel exposes L2 latency; the output transform is latency/ILP-bound (~1000 cycles per tile and 4 channels for
  ~500 dependent NEON ops), not memory-bound; at 4 threads per-worker V + M blocks beyond S8 overflow the shared L2.
- Revive: a K-major (prefetchable) A layout for the point-GEMM kernel; a cheaper SiLU only with a re-gate (D5).

## D17 depthwise 3x3 + SiLU loop forms on the A72 (K064)
- What and numbers (micro-benchmark cycles per 4-wide output vector; base 117 / 90 / 124 for C 512 / 960 / 1536,
  adopted staged ring 50-61): pixel-major loops 93-112 (64-channel chunks 68-85; taps re-streamed from L2); SiLU fused
  into pass 1 in registers 97-149 (spills, 4 latency-bound chains); channel-group-outer without staging (D11's form):
  no gain; staging per row (3 copies of each line) 66-90; L1 instead of L2 prefetch (evicted before use); prefetch of
  the adjacent group: noise; DC ZVA on output lines: 1 thread 63.3 -> 68.4 ms (int8) / 62.5 -> 79.4 (fp32), no gain
  at 4 threads; transparent huge pages: not available on the Pi 4 kernel (no sysfs node, madvise EINVAL).
- Not raced: a pixel-row split over threads (the channel sums' pixel order is the bit identity; a serial sum pass
  would cost more than the gain); dwk_f32 (0.5 % of the window).
- Pass 1 alone (86-88 / 56-58 / 92-96 cycles) was timed in the micro-benchmark by calling dw_row_s1 per row
  (K064). gemm_f32.c also carried an older bench switch, DW_NOSILU, that compiled the SiLU
  pass out of dw3x3_f32's pixel-major loop; no recorded number used it, and the simplify series removed it.
- Revive: huge pages on a kernel that has them (every pixel line is a different 4 KB page).

# Pi 5 kernel audit and the dotprod family (K200-)

## D22 Perch v2 and BirdNET v2.4: the fp16 1x1 convs in int8 SDOT on the A76 (K201)
- What: the 1x1 convs are the largest fp16 cost on the A76 (Perch 43.8 of 144 ms, v2.4 10.6 of 19.6 ms) and int8 SDOT
  has twice the fp16 ceiling. Screened with the oracles (per-row asymmetric int8 activations, per-channel weights).
- Numbers: Perch (devP 256 windows, vs the fp32 oracle): shipped fp16 + int8 head 51.5 dB; expansions in int8 46.9;
  projections in int8 42.0; both 41.3 (max |dconf| 0.029, 1 flip at 0.8). v2.4 (A004-A006): expansions
  40.8 dB, projections 40.1 dB, against 55 dB for the fp16 path, whose int8 head at 59.5 dB already failed the gate (D6).
- Why not: expansions-only would save about 10 ms of 144 on Perch for 4.6 dB; the projections (linear bottlenecks,
  wide dynamic range per row) are the sensitive class in both models. The fp16 projections already run at 87 % of
  the fp16 ceiling; the expansions' gap is their SiLU epilogue, which is the table-lookup SiLU already.
- Revive: a per-layer mix chosen by a per-layer screen, or per-channel activation scales (needs calibration data).

## D23 GEMMs with few rows split over threads by rows (K205)
- What: the dotprod family's first thread split cut every int8 GEMM into row chunks, also the heads (7 and 15 rows x
  11,568 classes for the global model) and the late 1x1 convs (105 rows).
- Numbers (Pi 5, detector running): heads_s8 3.1 ms on 1 thread -> 4.1 ms on 4; global model 150.8 / 111.6 / 88.6 /
  91.7 ms at 1 / 2 / 3 / 4 threads. Each chunk streams the whole weight matrix (23.7 MB for a global class layer), so
  one-row chunks multiply the weight traffic.
- Instead: GEMMs with fewer than 64 rows per thread are split over output channels (a view of the packed weights per
  16-channel block range), the rows quantized once first. Same bytes.
# Pi 4 kernels, round 2 (K100-)

## D18 Perch v2 prototype head through a low-rank factorization of the prototypes (K100)
- What: P [59180][1536] (unit rows) as A [59180][k] x B [k][1536] by a truncated SVD, so the head costs
  64 x k x (1536 + 59180) MACs instead of 64 x 59180 x 1536 (5.8 GMAC).
- Numbers (eigenvalues of P^T P, float64): the top 256 / 512 / 768 / 1024 / 1280 directions hold 53.5 / 72.9 / 83.7 /
  90.9 / 96.2 % of the energy, i.e. a rank-k P has 3.3 / 5.7 / 7.9 / 10.4 / 14.2 dB against a white cell. Singular values
  29.4 (first), 7.9 (256th), 5.7 (512th), 3.7 (1024th), 2.5 (last): near full rank.
- K104, the data-dependent version: real cells (8 windows x 64 cells) have only 73.6 / 80.8 / 85.9 / 95.5 % of their
  energy inside the top 256 / 384 / 512 / 1024 directions; the column maxima through a rank-k P reach 17.6 / 21.0 /
  23.9 / 32.6 dB (the head needs > 50).
- Revive: no.

## D19 int8 SMULL stream with an fp32 FMLA stream beside it on the A72 (K103)
- What: part of K in fp32 on the "free" second FP/SIMD pipe while the integer multiplies hold the first.
- Numbers (Pi 4, per call, 12,288 int8 MACs + 1,536 fp32 MACs): 2449 ns together, 1658 ns + 912 ns apart; the same
  at 2-4x the fp32 share. FMLA's micro-ops take the multiply pipe too: nothing is free.
- Revive: no.

## D20 Perch v2 body 1x1 convs in int8 on the A72 (K106)
- What: `--int8 proto,pw,head` (25 of 51 1x1 convs + the head conv in int8) against `--int8 proto`.
- Numbers: Pi 4 1 thread 1156 vs 1182 ms (the int8 convs take 165 ms where fp32 took 178: small K, the per-row
  quantization eats the kernel's gain); devP64 vs fp32 46.8 dB, top-1 0.984 against 53.1 dB, 1.000. With pwl 41 dB.
- Revive: no.

## D21 Prototype head without dotprod: prefetch of the next panels, inlined tiles (K108)
- What: PRFM of the next panel group's bytes spread over the 16 cell tiles; the two 4x4 tile loops inlined into the
  panel loop instead of two out-of-line functions.
- Numbers (Pi 4, proto_s8n ms/window at 1 / 4 threads): prefetch 723 / 188 vs 708 / 184 (2 panels per pass) and
  712 / 182 vs 689 / 175 (4 per pass); inlined 714 / 183 vs 689 / 175. Outputs identical.
- Why: the prototypes are one sequential stream the hardware prefetcher already follows; inlining both K loops into
  one frame costs registers across the float epilogue.
- Revive: no.

## D24 int8 (no dotprod): 8 or 16 products per int16 lane as independent 4-product chains joined by int16 adds (K113)
- What: the serial SMLAL chain is what makes n = 8 / 16 lose (K102), so run 2 or 4 chains of 4 products, add them in
  int16 (an add can use the second SIMD pipe) and widen once: 4c multiplies + 1 SADALP per 32c MACs. It would need
  7-bit activations (|q| <= 63) to be provably exact for Perch's prototypes, i.e. a re-gate.
- Numbers (Pi 4, L1-resident 4x4 tile, K 1536 / 256, GMAC/s): n=4 9.48 / 8.49; 2 chains 8.71 / 7.55; 4 chains
  7.10 / 6.70. Slower than n=4: the K loop is not bound by the multiply pipe alone (2.53 cycles per 16 MACs at n=4
  against 2.25 from the pipe count), and the extra loads and adds cost more than the saved SADALPs.
- So about 9.5 GMAC/s is the practical ceiling of the widening-multiply scheme on the A72; the head runs at 8.5.
- Revive: no.

## D25 fp32 GEMM: prefetch-for-write of the output rows 8-32 rows ahead (K119)
- What: while an M block's first panel runs, PRFM (store intent) the output lines of the rows 8 / 16 / 32 ahead, for
  layers whose block outputs exceed L1 (the early expansions write 3-9 MB per layer).
- Numbers (Pi 4, bench on Perch shapes, ms): M 15687 N 144 K 24 SiLU 27.3 -> 32.1-32.3; M 4000 N 192 K 32 10.1 ->
  12.0-12.1; M 1008 N 288 K 48 4.63 -> 5.2-5.3; the projections unchanged. Worse at every distance.
- Why: the output rows of a block are already in L2 after its first panel; the prefetches only add 9 line requests
  per row to a loop that is SiLU-bound.
- Revive: no.

# Experiment switches removed from the code (simplify series, 2026-11-10)
The switches below were left in the kernels after their experiments had finished. Each kept its default; the other
settings are recorded here and the code no longer has them.

## D26 Winograd F(2,3) int8: tiles per GEMM call other than 8 (K024)
- What: the tile chunk of wino_f23_tiles was the environment variable BNV3_WCHUNK (default 8, scratch laid out
  for up to 32); a sweep over 4 / 8 / 16 / 32 tiles (a design-review suggestion).
- Numbers (Pi 5, BirdNET v3.0 hand-written engine, ms per window median): 136.7 / 138.1 / 137.1 / 137.4; the
  Winograd stage alone 35.5 / 34.9 / 35.0 / 35.3 ms. Outputs bit-identical for every chunk.
- Why: no memory effect to find; the transforms are bound by compute and latency, not by the cache.
- Now: the constant WINO_CHUNK 8 (kern.h). Revive: no.

## D27 fp16 SiLU by exp2 polynomial and reciprocal estimate (replaced by the table lookup: I002, K022)
- What: BNV3_SILU chose the fp16 SiLU of the dotprod / fp16 kernels at build time: 0 = degree-3 exp2 polynomial +
  FRECPE + one Newton step (silu_f16_a, the original), 1 = degree-2 polynomial + FRECPE + Newton, 2 = degree-2 +
  FRECPE only (about 8-bit reciprocal), 3 = round trip through the fp32 SiLU, 4 = table-lookup piecewise quadratic
  (silu_lut.h); also tried and never selected: a bit-trick reciprocal for fp16 and for fp32 (silu_f16_b, silu_f32_b).
- Numbers (I002, Pi 5): table 1.10 cycles per element in 4 chains (1.20 in 8) against 1.61 for the polynomial in 8
  chains; in the engine 139.0 / 139.6 against 142.6 / 142.1 ms per window in two A/B pairs, and -3.0 to -3.4 ms summed
  over the stages (pw -1.0, dw -1.1, exp -1.2). Exhaustive over fp16
  |x| <= 64: max abs error 2.7e-3 against 4.8e-3, rms 5.5e-4 against 9.9e-4, so the table is also the more accurate
  form. Not bit-identical to the polynomial: it was integrated in K022 and passed the gate as part of C3 (K025).
- Now: the table form is plain code in act.h; variants 1, 2 and 3 and the bit-trick reciprocals are deleted.
  silu_f16_a stays as an uncalled reference (wino_s0.c's 4-chain silu4w is that polynomial and is live).
- Revive: no.

## D28 int8 SDOT GEMM: per-tensor epilogue with the scales folded (a design-review suggestion)
- What: for a per-tensor quantized input, fold the activation scale into the weight scales and the zero-point term
  into the bias once per 16 channels, leaving one FMA per 4 outputs (the switch PT_FOLD in gemm_s8.c, default 0).
- Numbers: the only record is the note left at the switch, "+0.2 ms only" (per window, the hand-written v3
  engine); no experiment row was written.
- Why not: the folded form rounds differently, so the outputs are not bit-identical to the gated build; 0.2 ms did
  not justify a re-gate.
- Revive: only together with another change that needs a re-gate of the dotprod path anyway.

## D29 depthwise k x k staged form: prefetch distance other than 8 pixels (K117)
- What: DWK_ST_PF, the distance of the software prefetch ahead of the input-row copy and of the output lines.
- Numbers (Pi 4, cycles per 4-wide output vector, 25 taps with SiLU, shapes 63x16x288 / 32x8x816 / 125x32x192
  stride 2 / 16x4x1392): 8 pixels 78 / 84 / 109 / 96; 4 and 16 pixels 81 / 87 / 118 / 104; no prefetch 85 / 99 /
  123 / 124.
- Now: the constant 8 (pc_kern_f32.c). Revive: no.

## D30 int8 Winograd F(4x4, 3x3) for BirdNET v3.0's dense 3x3 convs (src/wino43.c, removed with the v3 engine)
- What: V = B^T d B (6x6) in fp32 with one asymmetric scale per tile (or per layer, static), U = G g G^T with one
  symmetric scale per output channel over all 36 positions, 36 raw-int32 SDOT GEMMs, integer output transform
  (coefficients 1, 2, 4, 8). 36 MACs per 16 outputs (144 direct, 64 for the shipped F(2x2, 3x3)).
- Numbers: post-training quantization with the shared scales 30.8 dB (oracle mode 11; with per-position scales the
  loss was 1.8 dB); a min/max PTQ student with static transform-domain ranges 14.8 dB (the original v3 engine tree's
  notes); quantization-aware training of the static F(4,3) scales reached about 20 dB at most (fast.c's note).
- Why dropped: only usable for a student trained Winograd-aware with exactly this fake quantization, and that
  student's static transform-domain scales did not train. The fp32 F(4x4, 3x3) of the neon-fp32 / neon-int8 paths
  (wino_f32.c) is a different, live kernel. Revive: only with a QAT student that trains these scales.

## D31 alternative int8 SDOT GEMM with an asm epilogue (src/gemm_s8_alt.c + silu_asm.h, removed with the v3 engine)
- What: gemm_s8's signature and packing with the K loop and the fp16-SiLU tile epilogue in inline asm, accumulators
  started at the 0x4B400000 magic (one FSUB instead of SCVTF for Kp <= 256), SiLU through a 1/(1+e) polynomial
  without FRECPE, N-chunk / M-outer loop order (experiments K101-K104, K012 of the development log, not published).
- Numbers: per layer 4-16 % faster in isolation (b1.exp 10.09 -> 9.47 ms, b4.pwl 1.05 -> 0.90); in the engine about
  -3 % at the minimum only, median within noise (149.3 / 160.6 vs 152.1 / 160.8 ms, two A/B rounds), and -0.85 dB
  systematic on two dev sets from its SiLU / epilogue. The asm K loop itself was adopted into gemm_s8.c (it is the
  K loop the dotprod family runs). M-chunk / N-outer order (K104): no gain.
- Revive: no; the remaining time of the family is not in this epilogue (K200).

## D32 Winograd F(2x2, 3x3) int8 with the positions as the inner loop (WINO_FUSED, K023)
- What: one fused micro-kernel per tile x 16 output channels doing all 16 positions' SDOT sums and the output
  transform in registers, instead of 16 GEMMs over a chunk of tiles and a separate integer output transform.
- Numbers (Pi 5): 39.9 ms vs 34.8 ms for the six layers: no weight reuse across tiles, load-bound.
- Now: the 16-GEMM form (wino.c wino_f23_tiles). Revive: no.
