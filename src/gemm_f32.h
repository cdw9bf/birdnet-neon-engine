// fp32 GEMM kernels (gemm_f32.c) for the portable NEON path of the v2.4 engine (plain ARMv8-A: Cortex-A72 / Pi 4).
#ifndef GEMM_F32_H
#define GEMM_F32_H
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  int N, K, Kp, Np; /* Kp: multiple of 4 (>= K; the A row stride covers it; zero weights beyond K); Np = N rounded to 8 */
  float *w;         /* [Np/8][Kp][8] (+64 B slack) */
  float *b;         /* [Np + 8] */
} fpack;

enum { ACT32_NONE = 0, ACT32_SILU = 1, ACT32_RELU = 2, ACT32_RELU_POOL2 = 3 /* ReLU + [max|avg] of row pairs */ };
typedef struct {
  int act;
  float *out;
  int ldo;          /* output row stride (floats); >= N rounded to 8 (pad channels get 0) */
  int ow, opad;     /* if opad: pixel m -> padded position ((m/ow)+opad)*(ow+2opad) + m%ow + opad */
  const float *res; /* optional residual (added after the activation), row stride ldr */
  int ldr;
  int rpad, rw;     /* residual in a zero-haloed buffer: pixel m at ((m/rw)+rpad)*(rw+2rpad) + m%rw + rpad (0: row m) */
  int pool_off;     /* ACT32_RELU_POOL2: channel offset of the avg half */
} epi32;

void fp_pack(fpack *h, const float *w /*[N][K]*/, const float *bias, int N, int K, int Kp);
void fp_free(fpack *h);
/* C[m][0..N) = act(A[m][0..Kp) . W + b) for m < M; A rows at stride lda (floats), >= Kp. The kernel works on
 * 8-row tiles: rows up to M rounded up to 8 (plus 16 B) must be readable (their contents are ignored). */
void gemm_f32(const float *A, int lda, int M, const fpack *W, const epi32 *e);
/* the M block gemm_f32 uses for this shape, and one part of the GEMM: rows [m0, m1) (m0 a multiple of the block),
 * weight panels [p0, p1) (8 channels each). Any partition into parts computes every output exactly as gemm_f32 */
int gemm_f32_mblock(int M, const fpack *W);
void gemm_f32_range(const float *A, int lda, int M, const fpack *W, const epi32 *e, int m0, int m1, int p0, int p1);
/* gemm_f32_range on A[m][k] * gate[k] (squeeze-excitation gate fused into the GEMM's A blocks; needs K == Kp); scr:
 * gemm_f32_gate_scratch(M, W) floats per caller. Bit-identical to gating A in place, then gemm_f32_range. */
size_t gemm_f32_gate_scratch(int M, const fpack *W);
void gemm_f32_range_gate(const float *A, int lda, int M, const fpack *W, const epi32 *e, int m0, int m1, int p0, int p1,
                         const float *gate, float *scr);
/* K = nseg segments of segk (multiple of 4) floats: row m, segment s starts at A + m * lda + s * segstride
 * (floats). Used by the 4x8 stem conv on a padded image (segments = the 4 patch rows; no im2col). */
/* colmax[n] = max over the M rows of (A . W^T)[m][n] (no bias), n < Np; M % 8 == 0 */
void gemm_f32_colmax(const float *A, int lda, int M, const fpack *W, float *colmax);
void gemm_f32_seg(const float *A, int lda, int M, int nseg, int segk, long segstride, const fpack *W,
                  const epi32 *e);
/* depthwise 3x3 (pad 1, stride s in {1, 2}) + SiLU on the padded input xp [(H+2)][(W+2)][C] (C % 8 == 0) ->
 * out [Ho][Wo][C]; chsum[c] = sum of the outputs over the pixels */
void dw3x3_f32(const float *xp, int H, int W, int C, int s, const float *w /*[9][C]*/, const float *b, float *out,
               int Ho, int Wo, float *chsum);
void dw3x3_f32_cr(const float *xp, int H, int W, int C, int s, const float *w, const float *b, float *out, int Ho,
                  int Wo, float *chsum, int c0, int c1);
/* in place SiLU over n floats (n % 8 == 0); optional chsum[c % C] += x (C % 8 == 0) */
void silu_rows_f32(float *x, size_t n, int C, float *chsum);

/* log2 x for x > 0 (normal): the exponent plus log2 of the mantissa in [sqrt(.5), sqrt(2)) by a degree-9 polynomial
 * in (m - 1) (max abs error 7e-8, fit in fp32 Horner). The front ends' log (v24_fe.c, pc_kern_f32.c). */
static inline __attribute__((always_inline)) float32x4_t log2_acc_f32x4(float32x4_t x) {
  int32x4_t xi = vreinterpretq_s32_f32(x);
  int32x4_t t = vsubq_s32(xi, vdupq_n_s32(0x3f3504f3));
  int32x4_t e = vshrq_n_s32(t, 23);
  float32x4_t z = vsubq_f32(vreinterpretq_f32_s32(vsubq_s32(xi, vshlq_n_s32(e, 23))), vdupq_n_f32(1.f));
  float32x4_t p = vfmaq_f32(vdupq_n_f32(-0.20643105f), z, vdupq_n_f32(0.12125539f));
  p = vfmaq_f32(vdupq_n_f32(0.21717128f), z, p);
  p = vfmaq_f32(vdupq_n_f32(-0.23910326f), z, p);
  p = vfmaq_f32(vdupq_n_f32(0.28774428f), z, p);
  p = vfmaq_f32(vdupq_n_f32(-0.3606941f), z, p);
  p = vfmaq_f32(vdupq_n_f32(0.4809185f), z, p);
  p = vfmaq_f32(vdupq_n_f32(-0.7213475f), z, p);
  p = vfmaq_f32(vdupq_n_f32(1.4426949f), z, p);
  return vfmaq_f32(vcvtq_f32_s32(e), z, p);
}

/* 2^t for t in [-126, 126] (clamped): degree-6 polynomial on the fraction, rel. err ~1e-7 */
static inline float32x4_t exp2_f32x4(float32x4_t t) {
  t = vminq_f32(vmaxq_f32(t, vdupq_n_f32(-126.f)), vdupq_n_f32(126.f));
  float32x4_t n = vrndnq_f32(t), f = vsubq_f32(t, n);
  float32x4_t p = vdupq_n_f32(1.5403530e-4f);
  p = vfmaq_f32(vdupq_n_f32(1.3333558e-3f), p, f);
  p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), p, f);
  p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, f);
  p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(1.f), p, f);
  return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(vcvtq_s32_f32(n), 23)));
}
/* 1 / d for d >= 1: FRECPE + 2 Newton steps (rel. err ~1e-7) */
static inline float32x4_t rcp_f32x4(float32x4_t d) {
  float32x4_t r = vrecpeq_f32(d);
  r = vmulq_f32(r, vrecpsq_f32(d, r));
  r = vmulq_f32(r, vrecpsq_f32(d, r));
  return r;
}
static inline float32x4_t sigmoid_f32x4(float32x4_t x) {
  float32x4_t e = exp2_f32x4(vmulq_n_f32(x, -1.44269504f));
  return rcp_f32x4(vaddq_f32(vdupq_n_f32(1.f), e));
}

/* SiLU for the GEMM / depthwise epilogues: x * 1 / (1 + 2^(-x log2 e)); exp2 by the magic-number split with a
 * degree-5 polynomial (rel. err ~2e-7), t clamped to +-60 so 2^t stays finite (the gate underflows to 0 /
 * saturates to 1 as in the exact function); reciprocal FRECPE + one Newton step (rel. err ~1e-5). Cheaper variants
 * measured on devA vs TFLite: this one 91.9 dB; degree 4: 90.7 dB; two Newton steps or FDIV: 92.2 dB. */
static inline float32x4_t silu_f32x4(float32x4_t x) {
  const float32x4_t magic = vdupq_n_f32(12582912.0f); /* 1.5 * 2^23 */
  float32x4_t t = vmulq_n_f32(x, -1.44269504f);
  t = vminq_f32(vmaxq_f32(t, vdupq_n_f32(-60.f)), vdupq_n_f32(60.f));
  float32x4_t tm = vaddq_f32(t, magic);
  float32x4_t f = vsubq_f32(t, vsubq_f32(tm, magic));
  int32x4_t ni = vshlq_n_s32(vreinterpretq_s32_f32(tm), 23);
  float32x4_t p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), vdupq_n_f32(1.3333558e-3f), f);
  p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, f);
  p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
  float32x4_t d = vaddq_f32(vdupq_n_f32(1.f), vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), ni)));
  float32x4_t r = vrecpeq_f32(d);
  r = vmulq_f32(r, vrecpsq_f32(d, r));
  return vmulq_f32(x, r);
}

/* N SiLU chains in lock-step (same ops as silu_f32x4, bit-identical): the chain is ~18 dependent ops, so one
 * vector at a time is latency-bound on the A72 */
#define SILU_N_F32X4(NAME, N)                                                                                  \
  static inline void NAME(float32x4_t *v) {                                                                    \
    const float32x4_t magic = vdupq_n_f32(12582912.0f), lo = vdupq_n_f32(-60.f), hi = vdupq_n_f32(60.f);       \
    const float32x4_t c5 = vdupq_n_f32(1.3333558e-3f), c4 = vdupq_n_f32(9.6181291e-3f);                        \
    const float32x4_t c3 = vdupq_n_f32(5.5504109e-2f), c2 = vdupq_n_f32(2.4022651e-1f);                        \
    const float32x4_t c1 = vdupq_n_f32(6.9314718e-1f), one = vdupq_n_f32(1.f);                                 \
    float32x4_t t[N], tm[N], f[N], p[N], d[N], r[N];                                                           \
    for (int j = 0; j < N; j++) t[j] = vminq_f32(vmaxq_f32(vmulq_n_f32(v[j], -1.44269504f), lo), hi);          \
    for (int j = 0; j < N; j++) tm[j] = vaddq_f32(t[j], magic);                                                \
    for (int j = 0; j < N; j++) f[j] = vsubq_f32(t[j], vsubq_f32(tm[j], magic));                               \
    for (int j = 0; j < N; j++) p[j] = vfmaq_f32(c4, c5, f[j]);                                                \
    for (int j = 0; j < N; j++) p[j] = vfmaq_f32(c3, p[j], f[j]);                                              \
    for (int j = 0; j < N; j++) p[j] = vfmaq_f32(c2, p[j], f[j]);                                              \
    for (int j = 0; j < N; j++) p[j] = vfmaq_f32(c1, p[j], f[j]);                                              \
    for (int j = 0; j < N; j++) p[j] = vfmaq_f32(one, p[j], f[j]);                                             \
    for (int j = 0; j < N; j++)                                                                                \
      d[j] = vaddq_f32(one, vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p[j]),                      \
                                                            vshlq_n_s32(vreinterpretq_s32_f32(tm[j]), 23))));  \
    for (int j = 0; j < N; j++) r[j] = vrecpeq_f32(d[j]);                                                      \
    for (int j = 0; j < N; j++) r[j] = vmulq_f32(r[j], vrecpsq_f32(d[j], r[j]));                               \
    for (int j = 0; j < N; j++) v[j] = vmulq_f32(v[j], r[j]);                                                  \
  }
SILU_N_F32X4(silu4_f32x4, 4)
SILU_N_F32X4(silu8_f32x4, 8)
#endif
