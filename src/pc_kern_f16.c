// fp16 kernels for compiled models (Cortex-A76 objects): depthwise kxk with asymmetric pads, padding copies, mean,
// L2 normalisation. The depthwise kernel accumulates in fp16 from the fp16 bias over the taps in (ky, kx) order
// (the oracle's h16 mode models exactly this) and applies the LUT SiLU of the GEMM epilogues.
#include <arm_neon.h>
#include <math.h>
#include <string.h>

#include "pc_kern.h"
#include "silu_lut.h"

void pc_dwk_f16(const f16 *xp, int H, int W, int C, int k, int s, int pt, int pl, const f16 *w, const f16 *b, int act,
                f16 *out, int Ho, int Wo, float *chsum) {
  pc_dwk_f16_cr(xp, H, W, C, k, s, pt, pl, w, b, act, out, Ho, Wo, chsum, 0, C);
}
/* channels [c0, c1) only (multiples of 8): every channel lane is computed on its own, in the same pixel order */
void pc_dwk_f16_cr(const f16 *xp, int H, int W, int C, int k, int s, int pt, int pl, const f16 *w, const f16 *b, int act,
                   f16 *out, int Ho, int Wo, float *chsum, int c0, int c1) {
  (void)H;
  const silu_lut_t lt = silu_lut_load();
  const int h = (k - 1) / 2, Wp = W + 2 * h;
  for (int c = c0; c < c1; c++) chsum[c] = 0.f;
  for (int oy = 0; oy < Ho; oy++) {
    int ox = 0;
    /* four output pixels per pass, two partial sums each (even / odd taps): 8 independent FMA chains, so the loop
     * is load-bound rather than FMA-latency-bound; weights loaded once per tap for the four pixels */
    for (; ox + 4 <= Wo; ox += 4) {
      const f16 *x0 = xp + ((size_t)(oy * s - pt + h) * Wp + (ox * s - pl + h)) * C;
      const size_t d = (size_t)s * C;
      f16 *o = out + ((size_t)oy * Wo + ox) * C;
      for (int c = c0; c < c1; c += 8) {
        float16x8_t bb = vld1q_f16(b + c), z = vdupq_n_f16(0);
        float16x8_t a0 = bb, a1 = bb, a2 = bb, a3 = bb, e0 = z, e1 = z, e2 = z, e3 = z;
        int t = 0;
        for (int ky = 0; ky < k; ky++) {
          const f16 *r = x0 + (size_t)ky * Wp * C + c;
          const f16 *wr = w + (size_t)ky * k * C + c;
          for (int kx = 0; kx < k; kx++, t++) {
            float16x8_t wv = vld1q_f16(wr + (size_t)kx * C);
            const f16 *q = r + (size_t)kx * C;
            if (t & 1) {
              e0 = vfmaq_f16(e0, vld1q_f16(q), wv); e1 = vfmaq_f16(e1, vld1q_f16(q + d), wv);
              e2 = vfmaq_f16(e2, vld1q_f16(q + 2 * d), wv); e3 = vfmaq_f16(e3, vld1q_f16(q + 3 * d), wv);
            } else {
              a0 = vfmaq_f16(a0, vld1q_f16(q), wv); a1 = vfmaq_f16(a1, vld1q_f16(q + d), wv);
              a2 = vfmaq_f16(a2, vld1q_f16(q + 2 * d), wv); a3 = vfmaq_f16(a3, vld1q_f16(q + 3 * d), wv);
            }
          }
        }
        float16x8_t v[4] = {vaddq_f16(a0, e0), vaddq_f16(a1, e1), vaddq_f16(a2, e2), vaddq_f16(a3, e3)};
        if (act == 1) silu_lut4(v, &lt);
        for (int q = 0; q < 4; q++) vst1q_f16(o + (size_t)q * C + c, v[q]);
        float16x8_t sm = vaddq_f16(vaddq_f16(v[0], v[1]), vaddq_f16(v[2], v[3]));
        vst1q_f32(chsum + c, vaddq_f32(vld1q_f32(chsum + c), vcvt_f32_f16(vget_low_f16(sm))));
        vst1q_f32(chsum + c + 4, vaddq_f32(vld1q_f32(chsum + c + 4), vcvt_high_f32_f16(sm)));
      }
    }
    for (; ox + 2 <= Wo; ox += 2) {
      const f16 *x0 = xp + ((size_t)(oy * s - pt + h) * Wp + (ox * s - pl + h)) * C, *x1 = x0 + (size_t)s * C;
      f16 *o = out + ((size_t)oy * Wo + ox) * C;
      for (int c = c0; c < c1; c += 8) {
        float16x8_t bb = vld1q_f16(b + c), a0 = bb, a1 = bb;
        for (int ky = 0; ky < k; ky++) {
          const f16 *r0 = x0 + (size_t)ky * Wp * C + c, *r1 = x1 + (size_t)ky * Wp * C + c;
          const f16 *wr = w + (size_t)ky * k * C + c;
          for (int kx = 0; kx < k; kx++) {
            float16x8_t wv = vld1q_f16(wr + (size_t)kx * C);
            a0 = vfmaq_f16(a0, vld1q_f16(r0 + (size_t)kx * C), wv);
            a1 = vfmaq_f16(a1, vld1q_f16(r1 + (size_t)kx * C), wv);
          }
        }
        if (act == 1) { a0 = silu_lut_h(a0, &lt); a1 = silu_lut_h(a1, &lt); }
        vst1q_f16(o + c, a0);
        vst1q_f16(o + C + c, a1);
        float16x8_t sm = vaddq_f16(a0, a1);
        vst1q_f32(chsum + c, vaddq_f32(vld1q_f32(chsum + c), vcvt_f32_f16(vget_low_f16(sm))));
        vst1q_f32(chsum + c + 4, vaddq_f32(vld1q_f32(chsum + c + 4), vcvt_high_f32_f16(sm)));
      }
    }
    for (; ox < Wo; ox++) {
      const f16 *x0 = xp + ((size_t)(oy * s - pt + h) * Wp + (ox * s - pl + h)) * C;
      f16 *o = out + ((size_t)oy * Wo + ox) * C;
      for (int c = c0; c < c1; c += 8) {
        float16x8_t a0 = vld1q_f16(b + c);
        for (int ky = 0; ky < k; ky++) {
          const f16 *r0 = x0 + (size_t)ky * Wp * C + c;
          const f16 *wr = w + (size_t)ky * k * C + c;
          for (int kx = 0; kx < k; kx++) a0 = vfmaq_f16(a0, vld1q_f16(r0 + (size_t)kx * C), vld1q_f16(wr + (size_t)kx * C));
        }
        if (act == 1) a0 = silu_lut_h(a0, &lt);
        vst1q_f16(o + c, a0);
        vst1q_f32(chsum + c, vaddq_f32(vld1q_f32(chsum + c), vcvt_f32_f16(vget_low_f16(a0))));
        vst1q_f32(chsum + c + 4, vaddq_f32(vld1q_f32(chsum + c + 4), vcvt_high_f32_f16(a0)));
      }
    }
  }
}

void pc_pad_f16(const f16 *x, int H, int W, int C, int ld, int h, f16 *xp) {
  for (int y = 0; y < H; y++)
    for (int xx = 0; xx < W; xx++)
      memcpy(xp + ((size_t)(y + h) * (W + 2 * h) + xx + h) * C, x + ((size_t)y * W + xx) * ld, sizeof(f16) * C);
}

void pc_cvt_pad_f16(const float *x, int M, int W, int C, int ld, int h, int ldo, f16 *xp) {
  for (int m = 0; m < M; m++) {
    const float *r = x + (size_t)m * ld;
    f16 *o = xp + ((size_t)(m / W + h) * (W + 2 * h) + m % W + h) * ldo;
    int c = 0;
    for (; c + 8 <= C; c += 8)
      vst1q_f16(o + c, vcombine_f16(vcvt_f16_f32(vld1q_f32(r + c)), vcvt_f16_f32(vld1q_f32(r + c + 4))));
    for (; c < C; c++) o[c] = (f16)r[c];
  }
}

void pc_mean_f16(const f16 *x, int P, int C, int ld, float *out) {
  for (int c = 0; c < C; c += 4) {
    float32x4_t s = vdupq_n_f32(0);
    for (int p = 0; p < P; p++) s = vaddq_f32(s, vcvt_f32_f16(vld1_f16(x + (size_t)p * ld + c)));
    vst1q_f32(out + c, vdivq_f32(s, vdupq_n_f32((float)P)));
  }
}

void pc_l2norm_f16(const f16 *x, int P, int C, int ld, float eps, f16 *a) {
  for (int p = 0; p < P; p++) {
    const f16 *r = x + (size_t)p * ld;
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    for (int c = 0; c < C; c += 8) {
      float16x8_t v = vld1q_f16(r + c);
      float32x4_t u0 = vcvt_f32_f16(vget_low_f16(v)), u1 = vcvt_high_f32_f16(v);
      s0 = vfmaq_f32(s0, u0, u0); s1 = vfmaq_f32(s1, u1, u1);
    }
    float32x4_t d = vdupq_n_f32(sqrtf(vaddvq_f32(vaddq_f32(s0, s1))) + eps);
    for (int c = 0; c < C; c += 8) {
      float16x8_t v = vld1q_f16(r + c);
      float32x4_t u0 = vdivq_f32(vcvt_f32_f16(vget_low_f16(v)), d), u1 = vdivq_f32(vcvt_high_f32_f16(v), d);
      vst1q_f16(a + (size_t)p * C + c, vcombine_f16(vcvt_f16_f32(u0), vcvt_f16_f32(u1)));
    }
  }
}
