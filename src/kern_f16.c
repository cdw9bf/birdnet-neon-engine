// fp32 stem conv and fp16 depthwise 3x3 (all NHWC, padded inputs with a 1-pixel zero halo).
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "act.h"
#include "kern.h"

/* 4 SiLU chains in lock-step */
static inline __attribute__((always_inline)) void silu4(float16x8_t *v) { silu_lut4(v, silu_lut_tab()); }

/* stem: 3x3 stride 2, 3 -> 24, SiLU, for an EVEN image height H and width W: TF-same padding of a 3x3 stride-2 conv
 * is then bottom / right only, which is what this kernel implements (an odd size would need a top / left pad). out
 * is padded fp16 [(H/2+2)][(W/2+2)][24] (halo must already be zero). */
void stem_conv_f32_h(const float *img, int H, int W, const float *w, const float *b, f16 *out) {
  const int Ho = (H + 1) / 2, Wo = (W + 1) / 2, C = 24;
  float wt[27][24];
  for (int co = 0; co < C; co++)
    for (int i = 0; i < 27; i++) wt[i][co] = w[co * 27 + i]; /* OHWI: i = (ky*3+kx)*3+ci */
  for (int oy = 0; oy < Ho; oy++) {
    int ox = 0;
    if (oy * 2 + 2 < H) {
      /* interior fast path: two output pixels per pass share the weight loads; taps contiguous per row */
      for (; ox + 2 <= Wo && ox * 2 + 4 < W; ox += 2) {
        float32x4_t a0 = vld1q_f32(b), a1 = vld1q_f32(b + 4), a2 = vld1q_f32(b + 8);
        float32x4_t a3 = vld1q_f32(b + 12), a4 = vld1q_f32(b + 16), a5 = vld1q_f32(b + 20);
        float32x4_t c0 = a0, c1 = a1, c2 = a2, c3 = a3, c4 = a4, c5 = a5;
        for (int ky = 0; ky < 3; ky++) {
          const float *p = img + ((size_t)(oy * 2 + ky) * W + ox * 2) * 3; /* 9 floats for px0, px1 starts at +6 */
          for (int i = 0; i < 9; i++) {
            const float *wr = wt[ky * 9 + i];
            float32x4_t w0 = vld1q_f32(wr), w1 = vld1q_f32(wr + 4), w2 = vld1q_f32(wr + 8);
            float32x4_t w3 = vld1q_f32(wr + 12), w4 = vld1q_f32(wr + 16), w5 = vld1q_f32(wr + 20);
            float u = p[i], v = p[i + 6];
            a0 = vfmaq_n_f32(a0, w0, u); a1 = vfmaq_n_f32(a1, w1, u); a2 = vfmaq_n_f32(a2, w2, u);
            a3 = vfmaq_n_f32(a3, w3, u); a4 = vfmaq_n_f32(a4, w4, u); a5 = vfmaq_n_f32(a5, w5, u);
            c0 = vfmaq_n_f32(c0, w0, v); c1 = vfmaq_n_f32(c1, w1, v); c2 = vfmaq_n_f32(c2, w2, v);
            c3 = vfmaq_n_f32(c3, w3, v); c4 = vfmaq_n_f32(c4, w4, v); c5 = vfmaq_n_f32(c5, w5, v);
          }
        }
        f16 *o = out + ((size_t)(oy + 1) * (Wo + 2) + ox + 1) * C;
        vst1q_f16(o, silu_h(vcombine_f16(vcvt_f16_f32(a0), vcvt_f16_f32(a1))));
        vst1q_f16(o + 8, silu_h(vcombine_f16(vcvt_f16_f32(a2), vcvt_f16_f32(a3))));
        vst1q_f16(o + 16, silu_h(vcombine_f16(vcvt_f16_f32(a4), vcvt_f16_f32(a5))));
        vst1q_f16(o + 24, silu_h(vcombine_f16(vcvt_f16_f32(c0), vcvt_f16_f32(c1))));
        vst1q_f16(o + 32, silu_h(vcombine_f16(vcvt_f16_f32(c2), vcvt_f16_f32(c3))));
        vst1q_f16(o + 40, silu_h(vcombine_f16(vcvt_f16_f32(c4), vcvt_f16_f32(c5))));
      }
    }
    for (; ox < Wo; ox++) {
      float32x4_t a[6];
      for (int j = 0; j < 6; j++) a[j] = vld1q_f32(b + 4 * j);
      for (int ky = 0; ky < 3; ky++) {
        int iy = oy * 2 + ky;
        if (iy >= H) continue;
        for (int kx = 0; kx < 3; kx++) {
          int ix = ox * 2 + kx;
          if (ix >= W) continue;
          const float *p = img + ((size_t)iy * W + ix) * 3;
          for (int ci = 0; ci < 3; ci++) {
            const float *wr = wt[(ky * 3 + kx) * 3 + ci];
            float v = p[ci];
            for (int j = 0; j < 6; j++) a[j] = vfmaq_n_f32(a[j], vld1q_f32(wr + 4 * j), v);
          }
        }
      }
      f16 *o = out + ((size_t)(oy + 1) * (Wo + 2) + ox + 1) * C;
      for (int j = 0; j < 3; j++) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(a[2 * j]), vcvt_f16_f32(a[2 * j + 1]));
        vst1q_f16(o + 8 * j, silu_h(h));
      }
    }
  }
}

/* depthwise 3x3 v2: pixel-major (contiguous channel chunks), weights [9][C] L1-resident, fp16 accumulate, SiLU,
 * per-row fp16 channel partial sums flushed to fp32 (for the SE mean) */
void dw3x3_f16(const f16 *xp, int H, int W, int C, int s, int pt, int pl, const f16 *w, const f16 *b, f16 *out,
               int Ho, int Wo, float *chsum) {
  dw3x3_f16_cr(xp, H, W, C, s, pt, pl, w, b, out, Ho, Wo, chsum, 0, C);
}
/* channels [c0, c1) only (multiples of 8): every channel lane is computed on its own, in the same pixel order */
void dw3x3_f16_cr(const f16 *xp, int H, int W, int C, int s, int pt, int pl, const f16 *w, const f16 *b, f16 *out,
                  int Ho, int Wo, float *chsum, int c0, int c1) {
  const int Wp = W + 2;
  (void)H;
  f16 rs[2048] __attribute__((aligned(16)));
  for (int c = c0; c < c1; c++) chsum[c] = 0.f;
  for (int oy = 0; oy < Ho; oy++) {
    int iy0 = oy * s - pt + 1;
    memset(rs + c0, 0, sizeof(f16) * (size_t)(c1 - c0));
    int ox = 0;
    /* four output pixels per pass: weights loaded once, 4 independent FMLA chains, SiLU in lock-step
     * (same per-pixel op order as the 1-pixel path -> bit-identical) */
    for (; ox + 4 <= Wo; ox += 4) {
      int ix0 = ox * s - pl + 1;
      const f16 *r0 = xp + ((size_t)iy0 * Wp + ix0) * C;
      const f16 *r1 = r0 + (size_t)Wp * C, *r2 = r1 + (size_t)Wp * C;
      f16 *o = out + ((size_t)oy * Wo + ox) * C;
      const size_t d = (size_t)s * C;
      for (int c = c0; c < c1; c += 8) {
        float16x8_t w0 = vld1q_f16(w + c), w1 = vld1q_f16(w + C + c), w2 = vld1q_f16(w + 2 * C + c);
        float16x8_t w3 = vld1q_f16(w + 3 * C + c), w4 = vld1q_f16(w + 4 * C + c), w5 = vld1q_f16(w + 5 * C + c);
        float16x8_t w6 = vld1q_f16(w + 6 * C + c), w7 = vld1q_f16(w + 7 * C + c), w8 = vld1q_f16(w + 8 * C + c);
        float16x8_t bb = vld1q_f16(b + c);
        float16x8_t a[4], a2[4], y[4];
        for (int q = 0; q < 4; q++) {
          const f16 *q0 = r0 + q * d + c, *q1 = r1 + q * d + c, *q2 = r2 + q * d + c;
          a[q] = vfmaq_f16(bb, vld1q_f16(q0), w0);
          a2[q] = vmulq_f16(vld1q_f16(q0 + C), w1);
          a[q] = vfmaq_f16(a[q], vld1q_f16(q0 + 2 * C), w2);
          a2[q] = vfmaq_f16(a2[q], vld1q_f16(q1), w3);
          a[q] = vfmaq_f16(a[q], vld1q_f16(q1 + C), w4);
          a2[q] = vfmaq_f16(a2[q], vld1q_f16(q1 + 2 * C), w5);
          a[q] = vfmaq_f16(a[q], vld1q_f16(q2), w6);
          a2[q] = vfmaq_f16(a2[q], vld1q_f16(q2 + C), w7);
          a[q] = vfmaq_f16(a[q], vld1q_f16(q2 + 2 * C), w8);
          y[q] = vaddq_f16(a[q], a2[q]);
        }
        silu4(y);
        float16x8_t acc = vld1q_f16(rs + c);
        for (int q = 0; q < 4; q++) {
          vst1q_f16(o + q * C + c, y[q]);
          acc = vaddq_f16(acc, y[q]);
        }
        vst1q_f16(rs + c, acc);
      }
    }
    for (; ox < Wo; ox++) {
      int ix0 = ox * s - pl + 1;
      const f16 *r0 = xp + ((size_t)iy0 * Wp + ix0) * C;
      const f16 *r1 = r0 + (size_t)Wp * C, *r2 = r1 + (size_t)Wp * C;
      f16 *o = out + ((size_t)oy * Wo + ox) * C;
      for (int c = c0; c < c1; c += 8) {
        float16x8_t a = vfmaq_f16(vld1q_f16(b + c), vld1q_f16(r0 + c), vld1q_f16(w + c));
        float16x8_t a2 = vmulq_f16(vld1q_f16(r0 + C + c), vld1q_f16(w + C + c));
        a = vfmaq_f16(a, vld1q_f16(r0 + 2 * C + c), vld1q_f16(w + 2 * C + c));
        a2 = vfmaq_f16(a2, vld1q_f16(r1 + c), vld1q_f16(w + 3 * C + c));
        a = vfmaq_f16(a, vld1q_f16(r1 + C + c), vld1q_f16(w + 4 * C + c));
        a2 = vfmaq_f16(a2, vld1q_f16(r1 + 2 * C + c), vld1q_f16(w + 5 * C + c));
        a = vfmaq_f16(a, vld1q_f16(r2 + c), vld1q_f16(w + 6 * C + c));
        a2 = vfmaq_f16(a2, vld1q_f16(r2 + C + c), vld1q_f16(w + 7 * C + c));
        a = vfmaq_f16(a, vld1q_f16(r2 + 2 * C + c), vld1q_f16(w + 8 * C + c));
        float16x8_t y = silu_h(vaddq_f16(a, a2));
        vst1q_f16(o + c, y);
        vst1q_f16(rs + c, vaddq_f16(vld1q_f16(rs + c), y));
      }
    }
    for (int c = c0; c < c1; c += 8) {
      float16x8_t h = vld1q_f16(rs + c);
      vst1q_f32(chsum + c, vaddq_f32(vld1q_f32(chsum + c), vcvt_f32_f16(vget_low_f16(h))));
      vst1q_f32(chsum + c + 4, vaddq_f32(vld1q_f32(chsum + c + 4), vcvt_high_f32_f16(h)));
    }
  }
}
