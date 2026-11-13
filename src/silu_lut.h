/* fp16 SiLU via a 16-segment piecewise-quadratic table (TBL coefficient lookup), ideas-kernels I002.
 * silu(x) = max(x,0) - |x| * q(|x|),  q(u) = sigmoid(-u) = 1/(1+e^u), u clamped to [0, 15.4].
 * Segment index n = round(u) via the fp16 magic add (1536 = 1.5*2^10: ulp 1, low byte of the bits = n);
 * TBL index bytes [n, 16+n] = TBL(tm bytes, even-byte dup) + 0x1000;
 * f = u - n in [-0.5, 0.5]; q ~ c0[n] + c1[n] f + c2[n] f^2 with fp16 coefficients fetched by a 2-register TBL.
 * All ops are 0.5-cycle NEON ops on the A76 (no FRECPE (V0, 1/4 rate), no shifts (V1), no fcvt).
 * Accuracy (numpy model): rms err 7.0e-4 vs 9.6e-4 for the engine's silu_f16_a on a realistic distribution;
 * max abs err on [-8,8] 4.2e-3 vs 7.6e-3. Not bit-identical to silu_f16_a. */
#ifndef SILU_LUT_H
#define SILU_LUT_H
#include <arm_neon.h>
#include <stdint.h>
#include "silu_lut_tables_deg2.h"

#define SL_AI static inline __attribute__((always_inline))

typedef struct { uint8x16x2_t t0, t1, t2; uint8x16_t dup; } silu_lut_t;
static const uint8_t SILU_LUT_DUP[16] __attribute__((aligned(16))) = {0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14};
SL_AI silu_lut_t silu_lut_load(void) {
  silu_lut_t t;
  t.t0.val[0] = vld1q_u8(SILU_LUT_C0); t.t0.val[1] = vld1q_u8(SILU_LUT_C0 + 16);
  t.t1.val[0] = vld1q_u8(SILU_LUT_C1); t.t1.val[1] = vld1q_u8(SILU_LUT_C1 + 16);
  t.t2.val[0] = vld1q_u8(SILU_LUT_C2); t.t2.val[1] = vld1q_u8(SILU_LUT_C2 + 16);
  t.dup = vld1q_u8(SILU_LUT_DUP);
  return t;
}

SL_AI float16x8_t silu_lut_h(float16x8_t x, const silu_lut_t *t) {
  const float16x8_t magic = vdupq_n_f16((__fp16)1536.0f), umax = vdupq_n_f16((__fp16)15.4f), zero = vdupq_n_f16((__fp16)0.f);
  const int16x8_t jofs = vdupq_n_s16(0x1000); /* [n, n] + [0, 16] = bytes [n, 16+n] */
  float16x8_t u = vabsq_f16(x);
  float16x8_t uc = vminq_f16(u, umax);
  float16x8_t tm = vaddq_f16(uc, magic);
  float16x8_t f = vsubq_f16(uc, vsubq_f16(tm, magic));
  uint8x16_t j = vreinterpretq_u8_s16(vaddq_s16(vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_f16(tm), t->dup)), jofs));
  float16x8_t c0 = vreinterpretq_f16_u8(vqtbl2q_u8(t->t0, j));
  float16x8_t c1 = vreinterpretq_f16_u8(vqtbl2q_u8(t->t1, j));
  float16x8_t c2 = vreinterpretq_f16_u8(vqtbl2q_u8(t->t2, j));
  float16x8_t p = vfmaq_f16(c1, c2, f);
  p = vfmaq_f16(c0, p, f);
  return vfmsq_f16(vmaxq_f16(x, zero), u, p);
}

/* 8 vectors in lock-step (drop-in for the engine's silu8) */
SL_AI void silu_lut8(float16x8_t *v, const silu_lut_t *t) {
  const float16x8_t magic = vdupq_n_f16((__fp16)1536.0f), umax = vdupq_n_f16((__fp16)15.4f), zero = vdupq_n_f16((__fp16)0.f);
  const int16x8_t jofs = vdupq_n_s16(0x1000);
  float16x8_t u[8], uc[8], tm[8], f[8], p[8];
  uint8x16_t j[8];
  for (int i = 0; i < 8; i++) u[i] = vabsq_f16(v[i]);
  for (int i = 0; i < 8; i++) uc[i] = vminq_f16(u[i], umax);
  for (int i = 0; i < 8; i++) tm[i] = vaddq_f16(uc[i], magic);
  for (int i = 0; i < 8; i++) f[i] = vsubq_f16(uc[i], vsubq_f16(tm[i], magic));
  for (int i = 0; i < 8; i++) j[i] = vreinterpretq_u8_s16(vaddq_s16(vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_f16(tm[i]), t->dup)), jofs));
  for (int i = 0; i < 8; i++) p[i] = vfmaq_f16(vreinterpretq_f16_u8(vqtbl2q_u8(t->t1, j[i])), vreinterpretq_f16_u8(vqtbl2q_u8(t->t2, j[i])), f[i]);
  for (int i = 0; i < 8; i++) p[i] = vfmaq_f16(vreinterpretq_f16_u8(vqtbl2q_u8(t->t0, j[i])), p[i], f[i]);
  for (int i = 0; i < 8; i++) v[i] = vfmsq_f16(vmaxq_f16(v[i], zero), u[i], p[i]);
}
SL_AI void silu_lut4(float16x8_t *v, const silu_lut_t *t) {
  const float16x8_t magic = vdupq_n_f16((__fp16)1536.0f), umax = vdupq_n_f16((__fp16)15.4f), zero = vdupq_n_f16((__fp16)0.f);
  const int16x8_t jofs = vdupq_n_s16(0x1000);
  float16x8_t u[4], uc[4], tm[4], f[4], p[4];
  uint8x16_t j[4];
  for (int i = 0; i < 4; i++) u[i] = vabsq_f16(v[i]);
  for (int i = 0; i < 4; i++) uc[i] = vminq_f16(u[i], umax);
  for (int i = 0; i < 4; i++) tm[i] = vaddq_f16(uc[i], magic);
  for (int i = 0; i < 4; i++) f[i] = vsubq_f16(uc[i], vsubq_f16(tm[i], magic));
  for (int i = 0; i < 4; i++) j[i] = vreinterpretq_u8_s16(vaddq_s16(vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_f16(tm[i]), t->dup)), jofs));
  for (int i = 0; i < 4; i++) p[i] = vfmaq_f16(vreinterpretq_f16_u8(vqtbl2q_u8(t->t1, j[i])), vreinterpretq_f16_u8(vqtbl2q_u8(t->t2, j[i])), f[i]);
  for (int i = 0; i < 4; i++) p[i] = vfmaq_f16(vreinterpretq_f16_u8(vqtbl2q_u8(t->t0, j[i])), p[i], f[i]);
  for (int i = 0; i < 4; i++) v[i] = vfmsq_f16(vmaxq_f16(v[i], zero), u[i], p[i]);
}
#endif
