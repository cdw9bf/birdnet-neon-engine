// Inline SiLU / sigmoid approximations (NEON fp32 and fp16). All variants avoid FDIV.
#ifndef BNV3_ACT_H
#define BNV3_ACT_H
#include <arm_neon.h>
#include <pthread.h>
#include <stdatomic.h>

/* ---- fp32: exp2 via magic-number split, degree-5 poly; reciprocal via FRECPE + Newton ---- */
static inline float32x4_t exp2_p5_f32(float32x4_t t) {
  const float32x4_t magic = vdupq_n_f32(12582912.0f);
  float32x4_t tm = vaddq_f32(t, magic);
  float32x4_t n = vsubq_f32(tm, magic);
  float32x4_t f = vsubq_f32(t, n);
  int32x4_t ni = vshlq_n_s32(vreinterpretq_s32_f32(tm), 23);
  float32x4_t p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), vdupq_n_f32(1.3333558e-3f), f);
  p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, f);
  p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
  return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), ni));
}
/* log2 for x > 0 (normal): exponent + degree-7 polynomial of the mantissa in [sqrt(.5), sqrt(2)), ~1e-7 abs */
static inline float32x4_t log2_f32(float32x4_t x) {
  int32x4_t xi = vreinterpretq_s32_f32(x);
  /* shift the mantissa range to [0.7071, 1.4142) */
  int32x4_t t = vsubq_s32(xi, vdupq_n_s32(0x3F3504F3));
  int32x4_t e = vshrq_n_s32(t, 23);
  float32x4_t mnt = vreinterpretq_f32_s32(vsubq_s32(xi, vshlq_n_s32(e, 23)));
  float32x4_t z = vsubq_f32(mnt, vdupq_n_f32(1.0f));
  float32x4_t u = vdivq_f32(z, vaddq_f32(mnt, vdupq_n_f32(1.0f))); /* atanh series: ln m = 2(u + u^3/3 + ...) */
  float32x4_t u2 = vmulq_f32(u, u);
  float32x4_t pp = vfmaq_f32(vdupq_n_f32(1.f / 7), u2, vdupq_n_f32(1.f / 9));
  pp = vfmaq_f32(vdupq_n_f32(1.f / 5), pp, u2);
  pp = vfmaq_f32(vdupq_n_f32(1.f / 3), pp, u2);
  pp = vfmaq_f32(vdupq_n_f32(1.f), pp, u2);
  float32x4_t ln = vmulq_f32(vmulq_n_f32(u, 2.f), pp);
  return vfmaq_n_f32(vcvtq_f32_s32(e), ln, 1.44269504f);
}

/* reference-quality: poly5 + FRECPE + 2 Newton */
static inline float32x4_t silu_f32_a(float32x4_t x) {
  float32x4_t t = vmulq_n_f32(x, -1.44269504f);
  t = vminq_f32(vmaxq_f32(t, vdupq_n_f32(-126.f)), vdupq_n_f32(126.f));
  float32x4_t d = vaddq_f32(exp2_p5_f32(t), vdupq_n_f32(1.0f));
  float32x4_t r = vrecpeq_f32(d);
  r = vmulq_f32(r, vrecpsq_f32(d, r));
  r = vmulq_f32(r, vrecpsq_f32(d, r));
  return vmulq_f32(x, r);
}
/* ---- fp16 (8 lanes): exp2 via magic split (fp16 has 10 mantissa bits), poly3; FRECPE + 1 Newton. silu_f16_a is
 * the polynomial SiLU the kernels used before the table (DEADENDS D27); nothing calls it. It stays as the
 * single-chain statement of what wino_s0.c's silu4w computes and of what silu_lut.h's accuracy is compared with. ---- */
static inline float16x8_t exp2_p3_f16(float16x8_t t) {
  const float16x8_t magic = vdupq_n_f16((__fp16)1536.0f); /* 1.5 * 2^10 */
  float16x8_t tm = vaddq_f16(t, magic);
  float16x8_t n = vsubq_f16(tm, magic);
  float16x8_t f = vsubq_f16(t, n);
  int16x8_t ni = vshlq_n_s16(vreinterpretq_s16_f16(tm), 10);
  float16x8_t p = vfmaq_f16(vdupq_n_f16((__fp16)0.2402265f), vdupq_n_f16((__fp16)0.0555041f), f);
  p = vfmaq_f16(vdupq_n_f16((__fp16)0.6931472f), p, f);
  p = vfmaq_f16(vdupq_n_f16((__fp16)1.0f), p, f);
  return vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(p), ni));
}
static inline float16x8_t silu_f16_a(float16x8_t x) {
  float16x8_t t = vmulq_n_f16(x, (__fp16)-1.44269504f);
  t = vminq_f16(vmaxq_f16(t, vdupq_n_f16((__fp16)-14.f)), vdupq_n_f16((__fp16)14.f));
  float16x8_t d = vaddq_f16(exp2_p3_f16(t), vdupq_n_f16((__fp16)1.0f));
  float16x8_t r = vrecpeq_f16(d);
  r = vmulq_f16(r, vrecpsq_f16(d, r));
  return vmulq_f16(x, r);
}
#include "silu_lut.h"
/* The table registers, loaded once per translation unit. The kernels run on several threads (and hosts run several
 * contexts), so the first load is serialised by pthread_once and published through an atomic flag; afterwards a call
 * costs one load of the flag. */
static silu_lut_t g_silu_lut;
static pthread_once_t g_silu_lut_once = PTHREAD_ONCE_INIT;
static atomic_int g_silu_lut_ready;
__attribute__((unused)) static void silu_lut_tab_init(void) {
  g_silu_lut = silu_lut_load();
  atomic_store_explicit(&g_silu_lut_ready, 1, memory_order_release);
}
__attribute__((unused)) static inline const silu_lut_t *silu_lut_tab(void) {
  if (__builtin_expect(!atomic_load_explicit(&g_silu_lut_ready, memory_order_acquire), 0)) pthread_once(&g_silu_lut_once, silu_lut_tab_init);
  return &g_silu_lut;
}
/* the fp16 SiLU of the kernels: the table-lookup piecewise-quadratic form (silu_lut.h) */
static inline float16x8_t silu_h(float16x8_t x) { return silu_lut_h(x, silu_lut_tab()); }
/* 4 SiLU chains in lock-step */
static inline __attribute__((always_inline)) void silu4_h(float16x8_t *v) { silu_lut4(v, silu_lut_tab()); }
#endif
