// int8 GEMM for CPUs without dotprod (baseline ARMv8-A, e.g. Cortex-A72 / A53): per-output-channel symmetric int8
// weights, per-row dynamic activations (asymmetric, mapped into [-127, 127] with a zero point). The multiply is the
// widening pair SMULL (8 x 8b -> 16b) + SMLAL2 (the upper 8) into int16 lanes, then SADALP into int32: 16 MACs per
// three instructions. With |q| <= 127 on both sides a pair sum is <= 32258 < 32767 (no int16 overflow), and the int32
// lanes take K up to ~33000. Tile 4 rows x 4 channels (16 int32x4 accumulators); weights packed [Np/4][Kp/16][4][16].
// Epilogue: y = sa[m] * sw[n] * (acc - z[m] * colsum[n]) + b[n], activation, residual, (padded) store, as gemm_f32.
#include <arm_neon.h>
#include <math.h>
#include <string.h>

#include "gemm_s8n.h"

#define AI static inline __attribute__((always_inline))

void s8n_quant_rows(const float *x, int ld, int r0, int r1, int K, int Kp, int8_t *q, float *sa, int32_t *za) {
  for (int m = r0; m < r1; m++) {
    const float *r = x + (size_t)m * ld;
    float32x4_t mn = vdupq_n_f32(INFINITY), mx = vdupq_n_f32(-INFINITY);
    int k = 0;
    for (; k + 4 <= K; k += 4) { float32x4_t v = vld1q_f32(r + k); mn = vminq_f32(mn, v); mx = vmaxq_f32(mx, v); }
    float lo = vminvq_f32(mn), hi = vmaxvq_f32(mx);
    for (; k < K; k++) { lo = fminf(lo, r[k]); hi = fmaxf(hi, r[k]); }
    int8_t *o = q + (size_t)m * Kp;
    float inv;
    int32_t z;
    if (!(hi - lo > 1e-30f * fmaxf(fabsf(hi), fabsf(lo))) || !(hi - lo > 0.f)) { /* constant row: symmetric */
      float a = fmaxf(fabsf(hi), 1e-30f);
      inv = 127.f / a; z = 0; sa[m] = a / 127.f;
    } else {
      inv = 254.f / (hi - lo);
      z = -127 - (int32_t)lrintf(lo * inv);
      sa[m] = (hi - lo) / 254.f;
    }
    za[m] = z;
    const float32x4_t vi = vdupq_n_f32(inv);
    const int32x4_t vz = vdupq_n_s32(z), lo127 = vdupq_n_s32(-127), hi127 = vdupq_n_s32(127);
    k = 0;
    for (; k + 8 <= K; k += 8) {
      int32x4_t a = vaddq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(r + k), vi)), vz);
      int32x4_t b = vaddq_s32(vcvtnq_s32_f32(vmulq_f32(vld1q_f32(r + k + 4), vi)), vz);
      a = vminq_s32(vmaxq_s32(a, lo127), hi127); b = vminq_s32(vmaxq_s32(b, lo127), hi127);
      int16x8_t h = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
      vst1_s8(o + k, vmovn_s16(h));
    }
    for (; k < K; k++) {
      int32_t v = (int32_t)lrintf(r[k] * inv) + z;
      o[k] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v);
    }
    for (; k < Kp; k++) o[k] = (int8_t)z; /* padded k: (q - z) = 0 against zero weights */
  }
}

/* s8n_quant_rows of x[m][k] * gate[k] (K % 4 == 0, K <= 4096): each row is gated into an L1 buffer and quantized from
 * there, so a squeeze-excitation gate needs no in-place pass over the activation (same products: bit-identical) */
void s8n_quant_rows_gate(const float *x, int ld, int r0, int r1, int K, int Kp, const float *gate, int8_t *q, float *sa,
                         int32_t *za) {
  float t[4096] __attribute__((aligned(16)));
  for (int m = r0; m < r1; m++) {
    const float *r = x + (size_t)m * ld;
    for (int k = 0; k < K; k += 4) vst1q_f32(t + k, vmulq_f32(vld1q_f32(r + k), vld1q_f32(gate + k)));
    s8n_quant_rows(t, 0, m, m + 1, K, Kp, q, sa, za);
  }
}

#define MAC(acc, a, w)                                                                                      \
  do {                                                                                                      \
    int16x8_t p_ = vmull_s8(vget_low_s8(a), vget_low_s8(w));                                                \
    p_ = vmlal_high_s8(p_, a, w);                                                                           \
    acc = vpadalq_s16(acc, p_);                                                                             \
  } while (0)

AI size_t orow(const epi32 *e, int m) {
  if (!e->opad) return (size_t)m;
  return (size_t)((m / e->ow) + e->opad) * (e->ow + 2 * e->opad) + (m % e->ow) + e->opad;
}
AI size_t rrow(const epi32 *e, int m) {
  if (!e->rpad) return (size_t)m;
  return (size_t)((m / e->rw) + e->rpad) * (e->rw + 2 * e->rpad) + (m % e->rw) + e->rpad;
}

/* the int32 sums of rows m0..m0+3 against the 4-channel panel wp, reduced to one int32x4 (4 channels) per row */
AI void acc4x4(const int8_t *q, int Kp, int M, int m0, const int8_t *wp, int32x4_t r[4]) {
  const int8_t *A0 = q + (size_t)m0 * Kp;
  const int8_t *A1 = m0 + 1 < M ? A0 + Kp : A0, *A2 = m0 + 2 < M ? A0 + 2 * (size_t)Kp : A0, *A3 = m0 + 3 < M ? A0 + 3 * (size_t)Kp : A0;
  int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  int32x4_t c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
  for (int k = 0; k < Kp; k += 16, wp += 64) {
    int8x16_t w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16), w2 = vld1q_s8(wp + 32), w3 = vld1q_s8(wp + 48);
    int8x16_t a = vld1q_s8(A0 + k);
    MAC(c00, a, w0); MAC(c01, a, w1); MAC(c02, a, w2); MAC(c03, a, w3);
    a = vld1q_s8(A1 + k);
    MAC(c10, a, w0); MAC(c11, a, w1); MAC(c12, a, w2); MAC(c13, a, w3);
    a = vld1q_s8(A2 + k);
    MAC(c20, a, w0); MAC(c21, a, w1); MAC(c22, a, w2); MAC(c23, a, w3);
    a = vld1q_s8(A3 + k);
    MAC(c30, a, w0); MAC(c31, a, w1); MAC(c32, a, w2); MAC(c33, a, w3);
  }
  r[0] = vpaddq_s32(vpaddq_s32(c00, c01), vpaddq_s32(c02, c03));
  r[1] = vpaddq_s32(vpaddq_s32(c10, c11), vpaddq_s32(c12, c13));
  r[2] = vpaddq_s32(vpaddq_s32(c20, c21), vpaddq_s32(c22, c23));
  r[3] = vpaddq_s32(vpaddq_s32(c30, c31), vpaddq_s32(c32, c33));
}

/* dequantize 4 rows (rows past M take row m0's scale: discarded later) */
AI void deq4(const int32x4_t r[4], int M, int m0, int n0, const s8pack *W, const float *sa, const int32_t *za, float32x4_t y[4]) {
  const int32x4_t cs = vld1q_s32(W->colsum + n0);
  const float32x4_t sw = vld1q_f32(W->sw + n0), bb = vld1q_f32(W->b + n0);
  int mr = M - m0 < 4 ? M - m0 : 4;
  for (int i = 0; i < 4; i++) {
    int mi = i < mr ? m0 + i : m0;
    float32x4_t f = vcvtq_f32_s32(vsubq_s32(r[i], vmulq_n_s32(cs, za[mi])));
    y[i] = vaddq_f32(vmulq_f32(f, vmulq_n_f32(sw, sa[mi])), bb);
  }
}

AI void store_rows(const float32x4_t *y, int m0, int mr, int n0, const epi32 *e) {
  for (int i = 0; i < mr; i++) {
    float32x4_t v = y[i];
    if (e->res) v = vaddq_f32(v, vld1q_f32(e->res + rrow(e, m0 + i) * e->ldr + n0));
    vst1q_f32(e->out + orow(e, m0 + i) * e->ldo + n0, v);
  }
}

/* one 4 x 4 tile: rows m0.., channels n0.. */
AI void tile4x4(const int8_t *q, int Kp, int M, int m0, const int8_t *wp, int n0, const s8pack *W, const float *sa,
                const int32_t *za, const epi32 *e) {
  int32x4_t r[4];
  float32x4_t y[4];
  acc4x4(q, Kp, M, m0, wp, r);
  deq4(r, M, m0, n0, W, sa, za, y);
  if (e->act == ACT32_SILU) silu4_f32x4(y);
  else if (e->act == ACT32_RELU)
    for (int i = 0; i < 4; i++) y[i] = vmaxq_f32(y[i], vdupq_n_f32(0));
  store_rows(y, m0, M - m0 < 4 ? M - m0 : 4, n0, e);
}

/* R < 4 rows against one panel (the last tile of an M that is not a multiple of 4, e.g. M = 105 in every b5 layer):
 * a 4-row tile would run the whole K loop for 4 - R rows of discarded work; per k-step this does 12 R pipe ops instead
 * of 48, same loads of the panel */
#define TAIL_TILE(R)                                                                                              \
  AI void tile##R##x4(const int8_t *q, int Kp, int M, int m0, const int8_t *wp, int n0, const s8pack *W, const float *sa, \
                      const int32_t *za, const epi32 *e) {                                                        \
    const int8_t *A0 = q + (size_t)m0 * Kp, *A1 = R > 1 ? A0 + Kp : A0, *A2 = R > 2 ? A0 + 2 * (size_t)Kp : A0;    \
    int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00;   \
    int32x4_t c20 = c00, c21 = c00, c22 = c00, c23 = c00;                                                          \
    for (int k = 0; k < Kp; k += 16, wp += 64) {                                                                   \
      int8x16_t w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16), w2 = vld1q_s8(wp + 32), w3 = vld1q_s8(wp + 48);        \
      int8x16_t a = vld1q_s8(A0 + k);                                                                              \
      MAC(c00, a, w0); MAC(c01, a, w1); MAC(c02, a, w2); MAC(c03, a, w3);                                          \
      if (R > 1) { a = vld1q_s8(A1 + k); MAC(c10, a, w0); MAC(c11, a, w1); MAC(c12, a, w2); MAC(c13, a, w3); }     \
      if (R > 2) { a = vld1q_s8(A2 + k); MAC(c20, a, w0); MAC(c21, a, w1); MAC(c22, a, w2); MAC(c23, a, w3); }     \
    }                                                                                                              \
    int32x4_t r[4] = {vpaddq_s32(vpaddq_s32(c00, c01), vpaddq_s32(c02, c03)),                                      \
                      vpaddq_s32(vpaddq_s32(c10, c11), vpaddq_s32(c12, c13)),                                      \
                      vpaddq_s32(vpaddq_s32(c20, c21), vpaddq_s32(c22, c23)), vdupq_n_s32(0)};                     \
    float32x4_t y[4];                                                                                              \
    deq4(r, M, m0, n0, W, sa, za, y);                                                                              \
    if (e->act == ACT32_SILU) silu4_f32x4(y);                                                                      \
    else if (e->act == ACT32_RELU)                                                                                 \
      for (int i = 0; i < 4; i++) y[i] = vmaxq_f32(y[i], vdupq_n_f32(0));                                          \
    store_rows(y, m0, R, n0, e);                                                                                   \
  }
TAIL_TILE(1)
TAIL_TILE(2)
TAIL_TILE(3)
AI void tile_tail(const int8_t *q, int Kp, int M, int m0, const int8_t *wp, int n0, const s8pack *W, const float *sa,
                  const int32_t *za, const epi32 *e) {
  switch (M - m0) {
  case 1: tile1x4(q, Kp, M, m0, wp, n0, W, sa, za, e); break;
  case 2: tile2x4(q, Kp, M, m0, wp, n0, W, sa, za, e); break;
  case 3: tile3x4(q, Kp, M, m0, wp, n0, W, sa, za, e); break;
  default: tile4x4(q, Kp, M, m0, wp, n0, W, sa, za, e);
  }
}

void s8n_gemm_range(const int8_t *q, int Kp, const float *sa, const int32_t *za, int M, const s8pack *W, const epi32 *e,
                    int m0, int m1, int p0, int p1) {
  /* p0, p1 count 4-channel panels; rows in blocks that keep the int8 rows in L1 while the panels stream */
  int mb = (int)(16384 / (size_t)Kp) & ~3;
  if (mb < 4) mb = 4;
  for (int b0 = m0; b0 < m1; b0 += mb) {
    int b1 = m1 - b0 < mb ? m1 : b0 + mb;
    for (int pn = p0; pn < p1; pn++) {
      const int8_t *wp = W->w + (size_t)pn * Kp * 4;
      int m = b0;
      for (; b1 - m >= 4; m += 4) tile4x4(q, Kp, M, m, wp, pn * 4, W, sa, za, e);
      if (m < b1) tile_tail(q, Kp, M, m, wp, pn * 4, W, sa, za, e);
    }
  }
}
