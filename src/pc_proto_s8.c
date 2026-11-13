// int8 SDOT prototype head (Cortex-A76: dotprod): column maxima of the cell-by-prototype similarity matrix.
//   s[p][j] = sx[p] * sw[j] * sum_k qx[p][k] qw[j][k]   (per-cell symmetric int8 activations, per-prototype
//   symmetric int8 weights, exact int32 accumulation); colmax[j] = sw[j] * max_p sx[p] * acc[p][j] (sw > 0).
// Micro-kernel 8 cells x 12 prototypes: 24 int32x4 accumulators, A = 2 registers (8 rows x 4 k), B = 3 registers
// (12 columns x 4 k), 24 by-element SDOT per 4 k. Weights packed [N/12][K/4][12][4], activations [M/8][K/4][8][4];
// a 12-column panel (K = 1536: 18 KB) stays in L1 across the 8 row tiles.
#include <arm_neon.h>
#include <math.h>
#include <string.h>

#include "pc_kern.h"

/* x [P][ld] fp16 -> qa packed [P/8][K/4][8][4] int8 (q = round(x * 127 / max|x_p|), ties away from zero) and the
 * per-cell scale of the NORMALISED cell: sx[p] = max|x_p| / (127 (||x_p|| + eps)) */
void pc_quant_cells_f16(const f16 *x, int P, int C, int ld, float eps, int8_t *qa, float *sx) {
  for (int p = 0; p < P; p++) {
    const f16 *r = x + (size_t)p * ld;
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, m0 = s0, m1 = s0;
    for (int c = 0; c < C; c += 8) {
      float16x8_t v = vld1q_f16(r + c);
      float32x4_t u0 = vcvt_f32_f16(vget_low_f16(v)), u1 = vcvt_high_f32_f16(v);
      s0 = vfmaq_f32(s0, u0, u0); s1 = vfmaq_f32(s1, u1, u1);
      m0 = vmaxq_f32(m0, vabsq_f32(u0)); m1 = vmaxq_f32(m1, vabsq_f32(u1));
    }
    float den = sqrtf(vaddvq_f32(vaddq_f32(s0, s1))) + eps, mx = vmaxvq_f32(vmaxq_f32(m0, m1));
    if (!(mx > 0)) mx = 1e-30f;
    sx[p] = mx / (127.f * den);
    const float32x4_t inv = vdupq_n_f32(127.f / mx);
    int8_t *q = qa + (size_t)(p / 8) * C * 8 + (p % 8) * 4;
    for (int c = 0; c < C; c += 8) {
      float16x8_t v = vld1q_f16(r + c);
      int32x4_t i0 = vcvtaq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v)), inv));
      int32x4_t i1 = vcvtaq_s32_f32(vmulq_f32(vcvt_high_f32_f16(v), inv));
      int16x8_t h = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
      int8x8_t b = vqmovn_s16(h);
      /* k groups of 4: [c/4][8 rows][4] */
      memcpy(q + (size_t)(c / 4) * 32, &b, 4);
      int8_t t[8];
      vst1_s8(t, b);
      memcpy(q + (size_t)(c / 4 + 1) * 32, t + 4, 4);
    }
  }
}

void pc_proto_colmax_s8(const int8_t *qa, const float *sx, int P, int K, const int8_t *qw, const float *sw, int Np,
                        float *colmax) {
  pc_proto_colmax_s8_range(qa, sx, P, K, qw, sw, colmax, 0, Np);
}
/* prototypes [n_begin, n_end) (multiples of 12): each 12-column panel's maxima are independent of the others */
void pc_proto_colmax_s8_range(const int8_t *qa, const float *sx, int P, int K, const int8_t *qw, const float *sw,
                              float *colmax, int n_begin, int n_end) {
  const int k4 = K / 4;
  for (int n0 = n_begin; n0 < n_end; n0 += 12) {
    const int8_t *wp = qw + (size_t)(n0 / 12) * K * 12;
    float32x4_t mx0 = vdupq_n_f32(-INFINITY), mx1 = mx0, mx2 = mx0;
    for (int m0 = 0; m0 < P; m0 += 8) {
      const int8_t *ap = qa + (size_t)(m0 / 8) * K * 8, *bp = wp;
      int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c10 = c00, c11 = c00, c12 = c00, c20 = c00, c21 = c00,
                c22 = c00, c30 = c00, c31 = c00, c32 = c00, c40 = c00, c41 = c00, c42 = c00, c50 = c00, c51 = c00,
                c52 = c00, c60 = c00, c61 = c00, c62 = c00, c70 = c00, c71 = c00, c72 = c00;
      for (int k = 0; k < k4; k++) {
        int8x16_t a0 = vld1q_s8(ap), a1 = vld1q_s8(ap + 16);
        int8x16_t b0 = vld1q_s8(bp), b1 = vld1q_s8(bp + 16), b2 = vld1q_s8(bp + 32);
        ap += 32; bp += 48;
        c00 = vdotq_laneq_s32(c00, b0, a0, 0); c01 = vdotq_laneq_s32(c01, b1, a0, 0); c02 = vdotq_laneq_s32(c02, b2, a0, 0);
        c10 = vdotq_laneq_s32(c10, b0, a0, 1); c11 = vdotq_laneq_s32(c11, b1, a0, 1); c12 = vdotq_laneq_s32(c12, b2, a0, 1);
        c20 = vdotq_laneq_s32(c20, b0, a0, 2); c21 = vdotq_laneq_s32(c21, b1, a0, 2); c22 = vdotq_laneq_s32(c22, b2, a0, 2);
        c30 = vdotq_laneq_s32(c30, b0, a0, 3); c31 = vdotq_laneq_s32(c31, b1, a0, 3); c32 = vdotq_laneq_s32(c32, b2, a0, 3);
        c40 = vdotq_laneq_s32(c40, b0, a1, 0); c41 = vdotq_laneq_s32(c41, b1, a1, 0); c42 = vdotq_laneq_s32(c42, b2, a1, 0);
        c50 = vdotq_laneq_s32(c50, b0, a1, 1); c51 = vdotq_laneq_s32(c51, b1, a1, 1); c52 = vdotq_laneq_s32(c52, b2, a1, 1);
        c60 = vdotq_laneq_s32(c60, b0, a1, 2); c61 = vdotq_laneq_s32(c61, b1, a1, 2); c62 = vdotq_laneq_s32(c62, b2, a1, 2);
        c70 = vdotq_laneq_s32(c70, b0, a1, 3); c71 = vdotq_laneq_s32(c71, b1, a1, 3); c72 = vdotq_laneq_s32(c72, b2, a1, 3);
      }
#define ROWMAX(r, A, B, C)                                                                \
  do {                                                                                    \
    float s = sx[m0 + (r)];                                                               \
    mx0 = vmaxq_f32(mx0, vmulq_n_f32(vcvtq_f32_s32(A), s));                               \
    mx1 = vmaxq_f32(mx1, vmulq_n_f32(vcvtq_f32_s32(B), s));                               \
    mx2 = vmaxq_f32(mx2, vmulq_n_f32(vcvtq_f32_s32(C), s));                               \
  } while (0)
      ROWMAX(0, c00, c01, c02); ROWMAX(1, c10, c11, c12); ROWMAX(2, c20, c21, c22); ROWMAX(3, c30, c31, c32);
      ROWMAX(4, c40, c41, c42); ROWMAX(5, c50, c51, c52); ROWMAX(6, c60, c61, c62); ROWMAX(7, c70, c71, c72);
#undef ROWMAX
    }
    vst1q_f32(colmax + n0, vmulq_f32(mx0, vld1q_f32(sw + n0)));
    vst1q_f32(colmax + n0 + 4, vmulq_f32(mx1, vld1q_f32(sw + n0 + 4)));
    vst1q_f32(colmax + n0 + 8, vmulq_f32(mx2, vld1q_f32(sw + n0 + 8)));
  }
}
