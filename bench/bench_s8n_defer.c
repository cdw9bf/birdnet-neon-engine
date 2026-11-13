// Micro-benchmark (Cortex-A72 and other AArch64 cores without dotprod): int8 x int8 -> int32 GEMM K loops that widen
// the int16 partial sums every n products per lane instead of every 2 (SMULL + SMLAL2 + SADALP per 16 MACs today).
// A lane of an int16 accumulator takes n products before the SADALP; with |a| <= 127 that is exact when the sum of
// |w| over the n weights of the lane is <= 258 (the compiler checks it per output row). Per 8 n MACs: n multiplies
// (SMULL / SMLAL / SMLAL2) + 1 SADALP on the integer multiply pipe, against 3 n / 2 today.
//   bench_s8n_defer [K] [reps]     tile 4 rows x 4 channels, everything in L1; prints ns and cycles per 16 MACs
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define AI static inline __attribute__((always_inline))
#define NI __attribute__((noinline))
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* weights packed [K/16][4 channels][16], activations 4 rows [K] */
#define MAC2(acc, a, w)                                    \
  do {                                                     \
    int16x8_t p_ = vmull_s8(vget_low_s8(a), vget_low_s8(w)); \
    p_ = vmlal_high_s8(p_, a, w);                          \
    acc = vpadalq_s16(acc, p_);                            \
  } while (0)

/* n = 2: today's loop (gemm_s8n.c acc4x4) */
NI void k_base(const int8_t *A, int K, const int8_t *wp, int32_t *out) {
  const int8_t *A0 = A, *A1 = A + K, *A2 = A + 2 * K, *A3 = A + 3 * K;
  int32x4_t c[16];
  for (int i = 0; i < 16; i++) c[i] = vdupq_n_s32(0);
  for (int k = 0; k < K; k += 16, wp += 64) {
    int8x16_t w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16), w2 = vld1q_s8(wp + 32), w3 = vld1q_s8(wp + 48);
    int8x16_t a = vld1q_s8(A0 + k);
    MAC2(c[0], a, w0); MAC2(c[1], a, w1); MAC2(c[2], a, w2); MAC2(c[3], a, w3);
    a = vld1q_s8(A1 + k);
    MAC2(c[4], a, w0); MAC2(c[5], a, w1); MAC2(c[6], a, w2); MAC2(c[7], a, w3);
    a = vld1q_s8(A2 + k);
    MAC2(c[8], a, w0); MAC2(c[9], a, w1); MAC2(c[10], a, w2); MAC2(c[11], a, w3);
    a = vld1q_s8(A3 + k);
    MAC2(c[12], a, w0); MAC2(c[13], a, w1); MAC2(c[14], a, w2); MAC2(c[15], a, w3);
  }
  for (int i = 0; i < 16; i++) out[i] = vaddvq_s32(c[i]);
}

/* n = 4, weights of both chunks in registers (8), activations 2 registers per row */
#define MAC4(acc, a0, a1, w0, w1)                              \
  do {                                                         \
    int16x8_t p_ = vmull_s8(vget_low_s8(a0), vget_low_s8(w0)); \
    p_ = vmlal_high_s8(p_, a0, w0);                            \
    p_ = vmlal_s8(p_, vget_low_s8(a1), vget_low_s8(w1));       \
    p_ = vmlal_high_s8(p_, a1, w1);                            \
    acc = vpadalq_s16(acc, p_);                                \
  } while (0)
NI void k_n4(const int8_t *A, int K, const int8_t *wp, int32_t *out) {
  const int8_t *A0 = A, *A1 = A + K, *A2 = A + 2 * K, *A3 = A + 3 * K;
  int32x4_t c[16];
  for (int i = 0; i < 16; i++) c[i] = vdupq_n_s32(0);
  for (int k = 0; k < K; k += 32, wp += 128) {
    int8x16_t w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16), w2 = vld1q_s8(wp + 32), w3 = vld1q_s8(wp + 48);
    int8x16_t v0 = vld1q_s8(wp + 64), v1 = vld1q_s8(wp + 80), v2 = vld1q_s8(wp + 96), v3 = vld1q_s8(wp + 112);
    int8x16_t a = vld1q_s8(A0 + k), b = vld1q_s8(A0 + k + 16);
    MAC4(c[0], a, b, w0, v0); MAC4(c[1], a, b, w1, v1); MAC4(c[2], a, b, w2, v2); MAC4(c[3], a, b, w3, v3);
    a = vld1q_s8(A1 + k); b = vld1q_s8(A1 + k + 16);
    MAC4(c[4], a, b, w0, v0); MAC4(c[5], a, b, w1, v1); MAC4(c[6], a, b, w2, v2); MAC4(c[7], a, b, w3, v3);
    a = vld1q_s8(A2 + k); b = vld1q_s8(A2 + k + 16);
    MAC4(c[8], a, b, w0, v0); MAC4(c[9], a, b, w1, v1); MAC4(c[10], a, b, w2, v2); MAC4(c[11], a, b, w3, v3);
    a = vld1q_s8(A3 + k); b = vld1q_s8(A3 + k + 16);
    MAC4(c[12], a, b, w0, v0); MAC4(c[13], a, b, w1, v1); MAC4(c[14], a, b, w2, v2); MAC4(c[15], a, b, w3, v3);
  }
  for (int i = 0; i < 16; i++) out[i] = vaddvq_s32(c[i]);
}

/* n = 2 m with the weights loaded per use (m chunks of one channel are contiguous: packed [K/(16 m)][4][m][16]),
 * activations m registers per row */
#define DEFER_KERNEL(NAME, M)                                                                       \
  AI int32x4_t NAME##_dot(int32x4_t acc, const int8x16_t *a, const int8_t *w) {                     \
    int8x16_t w0 = vld1q_s8(w);                                                                     \
    int16x8_t p = vmull_s8(vget_low_s8(a[0]), vget_low_s8(w0));                                     \
    p = vmlal_high_s8(p, a[0], w0);                                                                 \
    for (int j = 1; j < M; j++) {                                                                   \
      int8x16_t wj = vld1q_s8(w + 16 * j);                                                          \
      p = vmlal_s8(p, vget_low_s8(a[j]), vget_low_s8(wj));                                          \
      p = vmlal_high_s8(p, a[j], wj);                                                               \
    }                                                                                               \
    return vpadalq_s16(acc, p);                                                                     \
  }                                                                                                 \
  NI void NAME(const int8_t *A, int K, const int8_t *wp, int32_t *out) {                            \
    int32x4_t c[16];                                                                                \
    for (int i = 0; i < 16; i++) c[i] = vdupq_n_s32(0);                                             \
    for (int k = 0; k < K; k += 16 * M, wp += 64 * M) {                                             \
      for (int r = 0; r < 4; r++) {                                                                 \
        int8x16_t a[M];                                                                             \
        for (int j = 0; j < M; j++) a[j] = vld1q_s8(A + (size_t)r * K + k + 16 * j);                \
        c[4 * r + 0] = NAME##_dot(c[4 * r + 0], a, wp);                                             \
        c[4 * r + 1] = NAME##_dot(c[4 * r + 1], a, wp + 16 * M);                                    \
        c[4 * r + 2] = NAME##_dot(c[4 * r + 2], a, wp + 32 * M);                                    \
        c[4 * r + 3] = NAME##_dot(c[4 * r + 3], a, wp + 48 * M);                                    \
      }                                                                                             \
    }                                                                                               \
    for (int i = 0; i < 16; i++) out[i] = vaddvq_s32(c[i]);                                         \
  }
DEFER_KERNEL(k_n4m, 2)
DEFER_KERNEL(k_n8, 4)
DEFER_KERNEL(k_n16, 8)
DEFER_KERNEL(k_n32, 16)

/* n = 8, 8 rows x 2 channels per pass with the 8 weight registers held (the activations load per use) */
NI void k_n8w(const int8_t *A, int K, const int8_t *wp, int32_t *out) {
  /* same packing as k_n8; channels 0-1 then 2-3 (two passes over K) */
  for (int half = 0; half < 2; half++) {
    int32x4_t c[8];
    for (int i = 0; i < 8; i++) c[i] = vdupq_n_s32(0);
    const int8_t *w = wp + half * 128;
    for (int k = 0; k < K; k += 64, w += 256) {
      int8x16_t u0 = vld1q_s8(w), u1 = vld1q_s8(w + 16), u2 = vld1q_s8(w + 32), u3 = vld1q_s8(w + 48);
      int8x16_t v0 = vld1q_s8(w + 64), v1 = vld1q_s8(w + 80), v2 = vld1q_s8(w + 96), v3 = vld1q_s8(w + 112);
      for (int r = 0; r < 4; r++) {
        const int8_t *ar = A + (size_t)r * K + k;
        int8x16_t a0 = vld1q_s8(ar), a1 = vld1q_s8(ar + 16), a2 = vld1q_s8(ar + 32), a3 = vld1q_s8(ar + 48);
        int16x8_t p = vmull_s8(vget_low_s8(a0), vget_low_s8(u0)), q = vmull_s8(vget_low_s8(a0), vget_low_s8(v0));
        p = vmlal_high_s8(p, a0, u0); q = vmlal_high_s8(q, a0, v0);
        p = vmlal_s8(p, vget_low_s8(a1), vget_low_s8(u1)); q = vmlal_s8(q, vget_low_s8(a1), vget_low_s8(v1));
        p = vmlal_high_s8(p, a1, u1); q = vmlal_high_s8(q, a1, v1);
        p = vmlal_s8(p, vget_low_s8(a2), vget_low_s8(u2)); q = vmlal_s8(q, vget_low_s8(a2), vget_low_s8(v2));
        p = vmlal_high_s8(p, a2, u2); q = vmlal_high_s8(q, a2, v2);
        p = vmlal_s8(p, vget_low_s8(a3), vget_low_s8(u3)); q = vmlal_s8(q, vget_low_s8(a3), vget_low_s8(v3));
        p = vmlal_high_s8(p, a3, u3); q = vmlal_high_s8(q, a3, v3);
        c[2 * r] = vpadalq_s16(c[2 * r], p); c[2 * r + 1] = vpadalq_s16(c[2 * r + 1], q);
      }
    }
    for (int r = 0; r < 4; r++) { out[4 * r + 2 * half] = vaddvq_s32(c[2 * r]); out[4 * r + 2 * half + 1] = vaddvq_s32(c[2 * r + 1]); }
  }
}

typedef void (*kfn)(const int8_t *, int, const int8_t *, int32_t *);

/* repack [K/16][4][16] (base) into [K/(16 m)][4][m][16] */
static void repack(const int8_t *wb, int K, int m, int8_t *wo) {
  for (int g = 0; g < K / (16 * m); g++)
    for (int ch = 0; ch < 4; ch++)
      for (int j = 0; j < m; j++) memcpy(wo + ((size_t)(g * 4 + ch) * m + j) * 16, wb + ((size_t)(g * m + j) * 4 + ch) * 16, 16);
}

int main(int argc, char **argv) {
  int K = argc > 1 ? atoi(argv[1]) : 1536, reps = argc > 2 ? atoi(argv[2]) : 20000;
  double ghz = argc > 3 ? atof(argv[3]) : 1.5;
  int wmax = argc > 4 ? atoi(argv[4]) : 7; /* |w| <= wmax: 32 products per lane stay exact for wmax <= 8 */
  if (K % 256) { fprintf(stderr, "K must be a multiple of 256\n"); return 1; }
  int8_t *A = aligned_alloc(64, 4 * (size_t)K), *wb = aligned_alloc(64, 4 * (size_t)K), *wr = aligned_alloc(64, 4 * (size_t)K);
  srand(1);
  for (int i = 0; i < 4 * K; i++) { A[i] = (int8_t)(rand() % 255 - 127); wb[i] = (int8_t)(rand() % (2 * wmax + 1) - wmax); }
  int32_t ref[16], out[16];
  k_base(A, K, wb, ref);
  struct { const char *name; kfn f; int m; int n; } v[] = {
    {"base n=2 (SMULL+SMLAL2+SADALP)", k_base, 1, 2}, {"n=4, weights in registers", k_n4, 2, 4}, {"n=4, weights loaded per use", k_n4m, 2, 4},
    {"n=8, weights loaded per use", k_n8, 4, 8}, {"n=8, 4x2 weights in registers", k_n8w, 4, 8}, {"n=16", k_n16, 8, 16}, {"n=32", k_n32, 16, 32}};
  printf("K %d, tile 4x4, %d reps, clock %.2f GHz, |w| <= %d\n", K, reps, ghz, wmax);
  for (unsigned i = 0; i < sizeof v / sizeof v[0]; i++) {
    const int8_t *w = wb;
    if (v[i].f == k_n4) { /* [K/32][2 chunks][4][16]: the base packing already */ }
    else if (v[i].m > 1) { repack(wb, K, v[i].m, wr); w = wr; }
    v[i].f(A, K, w, out);
    int ok = !memcmp(out, ref, sizeof ref);
    double best = 1e30;
    for (int t = 0; t < 5; t++) {
      double t0 = now_s();
      for (int r = 0; r < reps; r++) v[i].f(A, K, w, out);
      double dt = now_s() - t0;
      if (dt < best) best = dt;
    }
    double macs = (double)reps * 16 * K;
    printf("  %-34s %s  %6.3f ns/16 MACs  %5.2f cycles/16 MACs  %5.2f GMAC/s\n", v[i].name, ok ? "exact" : "WRONG",
           1e9 * best / (macs / 16), 1e9 * best / (macs / 16) * ghz, macs / best * 1e-9);
  }
  return 0;
}
