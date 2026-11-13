// int8 prototype head for CPUs without dotprod (baseline ARMv8-A, e.g. Cortex-A72): column maxima of the
// cell-by-prototype similarity matrix, the arithmetic of pc_proto_s8.c (the SDOT head) with the widening multiply of
// gemm_s8n.c:
//   s[p][j] = sx[p] * sw[j] * sum_k qx[p][k] qw[j][k]; colmax[j] = sw[j] * max_p sx[p] * acc[p][j]   (sw > 0)
// per-cell symmetric int8 activations, per-prototype symmetric int8 weights, exact int32 sums: for the same int8
// inputs the column maxima are bit-identical to the SDOT kernel's.
// Layout: prototypes [Np/4][Kp/16][4][16] (a 4-prototype panel is 4 Kp bytes: 6 KB at K = 1536, in L1 while the cells
// stream from L2), cells [P][Kp] row-major. Panel-outer: the prototypes (N x K bytes, far beyond the caches) are read
// from memory once per window. Tile 4 cells x 4 prototypes, 16 int32x4 accumulators.
// Two K loops: SMULL + SMLAL2 + SADALP per 16 MACs (any weights: a pair of products fits an int16 lane), and for
// panels the compiler proved safe ("n4": for every int16 lane, the 4 weights it accumulates over two 16-byte steps
// have sum |w| <= 258 = 32767 / 127, so 4 products with |q| <= 127 stay below 2^15) SMULL + 3 SMLAL + SADALP per
// 32 MACs. The sums are the same integers either way. The bound needs activations in [-127, 127]:
// pc_quant_cells_f32 guarantees it; a caller with bytes of its own must not pass -128 with n4 flags.
#include <arm_neon.h>
#include <math.h>
#include <string.h>

#include "pc_kern.h"

#define AI static inline __attribute__((always_inline))

/* x [P][ld] fp32 -> qa [P][Kp] int8 (q = round(x * 127 / max|x_p|), ties away from zero, clamped to [-127, 127];
 * zero beyond C) and the scale of the NORMALISED cell: sx[p] = max|x_p| / (127 (||x_p|| + eps)). The sums run as
 * pc_quant_cells_f16's do (two 4-lane fused accumulators over 8-channel steps), so cells whose values are
 * fp16-representable get the bytes and scales of the fp16 quantizer (which never emits -128 either: no finite fp16
 * magnitude is small enough to make its scale infinite). C % 8 == 0, 0 < C <= Kp. */
void pc_quant_cells_f32(const float *x, int P, int C, int ld, float eps, int8_t *qa, int Kp, float *sx) {
  for (int p = 0; p < P; p++) {
    const float *r = x + (size_t)p * ld;
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, m0 = s0, m1 = s0;
    for (int c = 0; c < C; c += 8) {
      float32x4_t u0 = vld1q_f32(r + c), u1 = vld1q_f32(r + c + 4);
      s0 = vfmaq_f32(s0, u0, u0); s1 = vfmaq_f32(s1, u1, u1);
      m0 = vmaxq_f32(m0, vabsq_f32(u0)); m1 = vmaxq_f32(m1, vabsq_f32(u1));
    }
    float den = sqrtf(vaddvq_f32(vaddq_f32(s0, s1))) + eps, mx = vmaxvq_f32(vmaxq_f32(m0, m1));
    if (!(mx > 0)) mx = 1e-30f;
    sx[p] = mx / (127.f * den);
    const float32x4_t inv = vdupq_n_f32(127.f / mx);
    int8_t *q = qa + (size_t)p * Kp;
    for (int c = 0; c < C; c += 8) {
      int32x4_t i0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(r + c), inv));
      int32x4_t i1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(r + c + 4), inv));
      /* never -128: the n4 K loop's bound and the symmetric range both need |q| <= 127 (a cell whose largest value
       * is below ~4e-37 has an infinite scale, and its negative values would saturate to -128) */
      vst1_s8(q + c, vmax_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))), vdup_n_s8(-127)));
    }
    memset(q + C, 0, (size_t)(Kp - C));
  }
}

#define MAC2(acc, a, w)                                      \
  do {                                                       \
    int16x8_t p_ = vmull_s8(vget_low_s8(a), vget_low_s8(w)); \
    p_ = vmlal_high_s8(p_, a, w);                            \
    acc = vpadalq_s16(acc, p_);                              \
  } while (0)
#define MAC4(acc, a0, a1, w0, w1)                              \
  do {                                                         \
    int16x8_t p_ = vmull_s8(vget_low_s8(a0), vget_low_s8(w0)); \
    p_ = vmlal_high_s8(p_, a0, w0);                            \
    p_ = vmlal_s8(p_, vget_low_s8(a1), vget_low_s8(w1));       \
    p_ = vmlal_high_s8(p_, a1, w1);                            \
    acc = vpadalq_s16(acc, p_);                                \
  } while (0)

#define ACC16 \
  int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00, \
            c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00
#define REDUCE(r)                                                  \
  do {                                                             \
    r[0] = vpaddq_s32(vpaddq_s32(c00, c01), vpaddq_s32(c02, c03)); \
    r[1] = vpaddq_s32(vpaddq_s32(c10, c11), vpaddq_s32(c12, c13)); \
    r[2] = vpaddq_s32(vpaddq_s32(c20, c21), vpaddq_s32(c22, c23)); \
    r[3] = vpaddq_s32(vpaddq_s32(c30, c31), vpaddq_s32(c32, c33)); \
  } while (0)

/* 4 cells (rows A0.., stride Kp) x one 4-prototype panel -> r[i] = the 4 int32 sums of cell i */
static __attribute__((noinline)) void tile_n2(const int8_t *A0, int Kp, const int8_t *wp, int32x4_t r[4]) {
  const int8_t *A1 = A0 + Kp, *A2 = A1 + Kp, *A3 = A2 + Kp;
  ACC16;
  for (int k = 0; k < Kp; k += 16, wp += 64) {
    int8x16_t w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16), w2 = vld1q_s8(wp + 32), w3 = vld1q_s8(wp + 48);
    int8x16_t a = vld1q_s8(A0 + k);
    MAC2(c00, a, w0); MAC2(c01, a, w1); MAC2(c02, a, w2); MAC2(c03, a, w3);
    a = vld1q_s8(A1 + k);
    MAC2(c10, a, w0); MAC2(c11, a, w1); MAC2(c12, a, w2); MAC2(c13, a, w3);
    a = vld1q_s8(A2 + k);
    MAC2(c20, a, w0); MAC2(c21, a, w1); MAC2(c22, a, w2); MAC2(c23, a, w3);
    a = vld1q_s8(A3 + k);
    MAC2(c30, a, w0); MAC2(c31, a, w1); MAC2(c32, a, w2); MAC2(c33, a, w3);
  }
  REDUCE(r);
}
/* the same sums with 4 products per int16 lane (Kp % 32 == 0; the panel is n4-safe) */
static __attribute__((noinline)) void tile_n4(const int8_t *A0, int Kp, const int8_t *wp, int32x4_t r[4]) {
  const int8_t *A1 = A0 + Kp, *A2 = A1 + Kp, *A3 = A2 + Kp;
  ACC16;
  for (int k = 0; k < Kp; k += 32, wp += 128) {
    int8x16_t w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16), w2 = vld1q_s8(wp + 32), w3 = vld1q_s8(wp + 48);
    int8x16_t v0 = vld1q_s8(wp + 64), v1 = vld1q_s8(wp + 80), v2 = vld1q_s8(wp + 96), v3 = vld1q_s8(wp + 112);
    int8x16_t a = vld1q_s8(A0 + k), b = vld1q_s8(A0 + k + 16);
    MAC4(c00, a, b, w0, v0); MAC4(c01, a, b, w1, v1); MAC4(c02, a, b, w2, v2); MAC4(c03, a, b, w3, v3);
    a = vld1q_s8(A1 + k); b = vld1q_s8(A1 + k + 16);
    MAC4(c10, a, b, w0, v0); MAC4(c11, a, b, w1, v1); MAC4(c12, a, b, w2, v2); MAC4(c13, a, b, w3, v3);
    a = vld1q_s8(A2 + k); b = vld1q_s8(A2 + k + 16);
    MAC4(c20, a, b, w0, v0); MAC4(c21, a, b, w1, v1); MAC4(c22, a, b, w2, v2); MAC4(c23, a, b, w3, v3);
    a = vld1q_s8(A3 + k); b = vld1q_s8(A3 + k + 16);
    MAC4(c30, a, b, w0, v0); MAC4(c31, a, b, w1, v1); MAC4(c32, a, b, w2, v2); MAC4(c33, a, b, w3, v3);
  }
  REDUCE(r);
}

int pc_proto_n4_check(const int8_t *qw, int Kp, const uint8_t *n4, int npanels) {
  if (Kp % 32) return -1; /* the n4 loop is never used */
  const uint16x8_t lim = vdupq_n_u16(258);
  for (int pn = 0; pn < npanels; pn++) {
    if (!n4[pn]) continue;
    const int8_t *w = qw + (size_t)pn * Kp * 4;
    uint16x8_t over = vdupq_n_u16(0);
    for (int k = 0; k < Kp; k += 32, w += 128)
      for (int ch = 0; ch < 4; ch++) { /* channel ch of the panel: bytes [16 ch, 16 ch + 16) of each 64-byte step */
        uint8x16_t a = vreinterpretq_u8_s8(vabsq_s8(vld1q_s8(w + 16 * ch)));
        uint8x16_t b = vreinterpretq_u8_s8(vabsq_s8(vld1q_s8(w + 64 + 16 * ch)));
        uint16x8_t s = vaddq_u16(vaddl_u8(vget_low_u8(a), vget_high_u8(a)), vaddl_u8(vget_low_u8(b), vget_high_u8(b)));
        over = vorrq_u16(over, vcgtq_u16(s, lim));
      }
    if (vmaxvq_u16(over)) return pn;
  }
  return -1;
}

/* panels [p0, p1) (4 prototypes each) -> colmax[4 p0 .. 4 p1); P % 4 == 0, Kp % 16 == 0. n4 (may be NULL): one byte
 * per panel, nonzero = the panel is n4-safe (used only when Kp % 32 == 0). PROTO_PG panels share each pass over the
 * cells: a 4-cell tile (4 Kp bytes) stays in L1 for the group's later panels, and 4 panels + the tile are 30 KB at
 * K = 1536 (Pi 4, K108: 767 ms for 1 panel per pass, 708 for 2, 693 for 3, 689 for 4). */
#define PROTO_PG 4
void pc_proto_colmax_s8n(const int8_t *qa, int Kp, const float *sx, int P, const int8_t *qw, const float *sw,
                         const uint8_t *n4, int p0, int p1, float *colmax) {
  const int k32 = Kp % 32 == 0;
  const size_t pb = (size_t)Kp * 4; /* bytes per panel */
  for (int pn = p0; pn < p1; pn += PROTO_PG) {
    const int g1 = p1 - pn < PROTO_PG ? p1 - pn : PROTO_PG;
    const int8_t *wp = qw + (size_t)pn * pb;
    float32x4_t mx[PROTO_PG];
    int fast[PROTO_PG];
    for (int g = 0; g < g1; g++) { mx[g] = vdupq_n_f32(-INFINITY); fast[g] = n4 && k32 && n4[pn + g]; }
    for (int m0 = 0; m0 < P; m0 += 4) {
      const int8_t *A0 = qa + (size_t)m0 * Kp;
      for (int g = 0; g < g1; g++) {
        int32x4_t r[4];
        if (fast[g]) tile_n4(A0, Kp, wp + g * pb, r);
        else tile_n2(A0, Kp, wp + g * pb, r);
        mx[g] = vmaxq_f32(mx[g], vmulq_n_f32(vcvtq_f32_s32(r[0]), sx[m0]));
        mx[g] = vmaxq_f32(mx[g], vmulq_n_f32(vcvtq_f32_s32(r[1]), sx[m0 + 1]));
        mx[g] = vmaxq_f32(mx[g], vmulq_n_f32(vcvtq_f32_s32(r[2]), sx[m0 + 2]));
        mx[g] = vmaxq_f32(mx[g], vmulq_n_f32(vcvtq_f32_s32(r[3]), sx[m0 + 3]));
      }
    }
    for (int g = 0; g < g1; g++)
      vst1q_f32(colmax + 4 * (size_t)(pn + g), vmulq_f32(mx[g], vld1q_f32(sw + 4 * (size_t)(pn + g))));
  }
}
