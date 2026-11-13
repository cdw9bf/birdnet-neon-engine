// int8 SDOT GEMM: 4 pixels x 16 output channels micro-kernel (by-element SDOT, c4-packed weights), fused epilogue;
// int8 quantisation passes (per-row and per-tensor asymmetric) and 3x3 im2col.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "act.h"
#include "kern.h"

void qw_pack(qweights *q, const float *w, const float *bias, int N, int K) {
  int Kp = (K + 15) & ~15;
  q->N = N; q->K = K; q->Kp = Kp;
  q->w = al64((size_t)N * Kp + 64);
  q->scale = malloc(sizeof(float) * N);
  q->swsum = malloc(sizeof(float) * N);
  q->bias = malloc(sizeof(float) * N);
  memset(q->w, 0, (size_t)N * Kp);
  for (int n = 0; n < N; n++) {
    const float *r = w + (size_t)n * K;
    float mx = 0;
    for (int k = 0; k < K; k++) mx = fmaxf(mx, fabsf(r[k]));
    float s = mx > 0 ? mx / 127.f : 1.f;
    q->scale[n] = s;
    q->bias[n] = bias ? bias[n] : 0.f;
    int32_t sum = 0;
    int nb = n / 16, nl = n % 16;
    for (int k = 0; k < K; k++) {
      int v = (int)nearbyintf(r[k] / s);
      v = v > 127 ? 127 : (v < -127 ? -127 : v);
      sum += v;
      q->w[(size_t)nb * 16 * Kp + (size_t)(k / 4) * 64 + nl * 4 + (k % 4)] = (int8_t)v;
    }
    q->swsum[n] = s * (float)sum;
  }
}

void qw_free(qweights *q) {
  free(q->w); free(q->scale); free(q->swsum); free(q->bias);
  memset(q, 0, sizeof *q);
}

/* ---------------- epilogue for one 4x16 tile ---------------- */
static inline size_t opix(const epi *e, int m) {
  if (!e->opad) return (size_t)m;
  int y = m / e->ow, x = m - y * e->ow;
  return (size_t)(y + e->opad) * (e->ow + 2 * e->opad) + x + e->opad;
}

/* int32 -> fp32: SCVTF is V0-only at half rate; the magic-number form (integer add + fsub, any pipe) is exact for
 * |acc| < 2^22, guaranteed when Kp <= 256 (|acc| <= Kp*128*127). */
static inline float32x4_t i2f(int32x4_t c, int mg) {
  if (mg) return vsubq_f32(vreinterpretq_f32_s32(vaddq_s32(c, vdupq_n_s32(0x4B400000))), vdupq_n_f32(12582912.0f));
  return vcvtq_f32_s32(c);
}
static inline void tile_row(int32x4_t c0, int32x4_t c1, int32x4_t c2, int32x4_t c3, float sa, float zs,
                            const float *sw, const float *sws, const float *bb, const epi *e, int m, int n, int mg) {
  /* v = sa*(sw*acc) + (sa*zp)*swsum + b */
  float32x4_t v0 = vmulq_f32(i2f(c0, mg), vld1q_f32(sw));
  float32x4_t v1 = vmulq_f32(i2f(c1, mg), vld1q_f32(sw + 4));
  float32x4_t v2 = vmulq_f32(i2f(c2, mg), vld1q_f32(sw + 8));
  float32x4_t v3 = vmulq_f32(i2f(c3, mg), vld1q_f32(sw + 12));
  v0 = vfmaq_n_f32(vfmaq_n_f32(vld1q_f32(bb), v0, sa), vld1q_f32(sws), zs);
  v1 = vfmaq_n_f32(vfmaq_n_f32(vld1q_f32(bb + 4), v1, sa), vld1q_f32(sws + 4), zs);
  v2 = vfmaq_n_f32(vfmaq_n_f32(vld1q_f32(bb + 8), v2, sa), vld1q_f32(sws + 8), zs);
  v3 = vfmaq_n_f32(vfmaq_n_f32(vld1q_f32(bb + 12), v3, sa), vld1q_f32(sws + 12), zs);
  size_t po = opix(e, m) * e->ldo + n;
  if (e->out16) {
    float16x8_t h0 = vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1));
    float16x8_t h1 = vcombine_f16(vcvt_f16_f32(v2), vcvt_f16_f32(v3));
    if (e->act == ACT_SILU) { h0 = silu_h(h0); h1 = silu_h(h1); }
    vst1q_f16(e->out16 + po, h0);
    vst1q_f16(e->out16 + po + 8, h1);
  } else {
    if (e->act == ACT_SILU) { v0 = silu_f32_a(v0); v1 = silu_f32_a(v1); v2 = silu_f32_a(v2); v3 = silu_f32_a(v3); }
    else if (e->act == ACT_RELU) {
      float32x4_t z = vdupq_n_f32(0);
      v0 = vmaxq_f32(v0, z); v1 = vmaxq_f32(v1, z); v2 = vmaxq_f32(v2, z); v3 = vmaxq_f32(v3, z);
    }
    if (e->res32) {
      const float *r = e->res32 + (size_t)m * e->ldo + n;
      v0 = vaddq_f32(v0, vld1q_f32(r)); v1 = vaddq_f32(v1, vld1q_f32(r + 4));
      v2 = vaddq_f32(v2, vld1q_f32(r + 8)); v3 = vaddq_f32(v3, vld1q_f32(r + 12));
    }
    float *o = e->out32 + po;
    vst1q_f32(o, v0); vst1q_f32(o + 4, v1); vst1q_f32(o + 8, v2); vst1q_f32(o + 12, v3);
  }
}

#define AI static inline __attribute__((always_inline))

/* asm K loop taken from the alternative GEMM (K101-K104; the rest of it is DEADENDS D31): fixed accumulators v16-v31,
 * A in v0-v3, weights double-buffered v4-v11 - removes the register-rotation movs gcc emits (review #2 item 2). */
/* ---------------- K loop: 4 rows x 16 channels, Kp/16 iterations ---------------- */
#define SDOT_GROUP(W0, W1, W2, W3, L)                                                                  \
  "sdot v16.4s, " W0 ".16b, v0.4b[" L "]\n sdot v20.4s, " W0 ".16b, v1.4b[" L "]\n"                   \
  "sdot v24.4s, " W0 ".16b, v2.4b[" L "]\n sdot v28.4s, " W0 ".16b, v3.4b[" L "]\n"                   \
  "sdot v17.4s, " W1 ".16b, v0.4b[" L "]\n sdot v21.4s, " W1 ".16b, v1.4b[" L "]\n"                   \
  "sdot v25.4s, " W1 ".16b, v2.4b[" L "]\n sdot v29.4s, " W1 ".16b, v3.4b[" L "]\n"                   \
  "sdot v18.4s, " W2 ".16b, v0.4b[" L "]\n sdot v22.4s, " W2 ".16b, v1.4b[" L "]\n"                   \
  "sdot v26.4s, " W2 ".16b, v2.4b[" L "]\n sdot v30.4s, " W2 ".16b, v3.4b[" L "]\n"                   \
  "sdot v19.4s, " W3 ".16b, v0.4b[" L "]\n sdot v23.4s, " W3 ".16b, v1.4b[" L "]\n"                   \
  "sdot v27.4s, " W3 ".16b, v2.4b[" L "]\n sdot v31.4s, " W3 ".16b, v3.4b[" L "]\n"

#define KLOOP_BODY                                                                                     \
  "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[ini]]\n"                                                   \
  "ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[ini]]\n"                                                   \
  "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[ini]]\n"                                                   \
  "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[ini]]\n"                                                   \
  "ldp q4, q5, [%[w]]\n ldp q6, q7, [%[w], #32]\n"                                                     \
  "1:\n"                                                                                               \
  "ldr q0, [%[p0]], #16\n ldr q1, [%[p1]], #16\n ldr q2, [%[p2]], #16\n ldr q3, [%[p3]], #16\n"        \
  "ldp q8, q9, [%[w], #64]\n ldp q10, q11, [%[w], #96]\n"                                              \
  SDOT_GROUP("v4", "v5", "v6", "v7", "0")                                                              \
  "ldp q4, q5, [%[w], #128]\n ldp q6, q7, [%[w], #160]\n"                                              \
  SDOT_GROUP("v8", "v9", "v10", "v11", "1")                                                            \
  "ldp q8, q9, [%[w], #192]\n ldp q10, q11, [%[w], #224]\n"                                            \
  "add %[w], %[w], #256\n"                                                                             \
  SDOT_GROUP("v4", "v5", "v6", "v7", "2")                                                              \
  "ldp q4, q5, [%[w]]\n ldp q6, q7, [%[w], #32]\n"                                                     \
  "subs %w[c], %w[c], #1\n"                                                                            \
  SDOT_GROUP("v8", "v9", "v10", "v11", "3")                                                            \
  "b.ne 1b\n"

/* loop only; accumulators handed to the C epilogue through register variables v16..v31 */
#define KLOOP(a0, a1, a2, a3, w, cnt, init)                                                            \
  asm volatile(KLOOP_BODY                                                                              \
      : [p0] "+r"(a0), [p1] "+r"(a1), [p2] "+r"(a2), [p3] "+r"(a3), [w] "+r"(w), [c] "+r"(cnt),        \
        "=w"(c00), "=w"(c01), "=w"(c02), "=w"(c03), "=w"(c10), "=w"(c11), "=w"(c12), "=w"(c13),       \
        "=w"(c20), "=w"(c21), "=w"(c22), "=w"(c23), "=w"(c30), "=w"(c31), "=w"(c32), "=w"(c33)        \
      : [ini] "r"(init)                                                                                \
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "cc", "memory")

static const int32_t g_zero16[16] __attribute__((aligned(16)));


/* 8 fp16 SiLU values: two table-lookup groups of 4 chains (a single chain is a long dependent run that the A76 cannot
 * overlap with the next one; review #2, alt-gemm K102/K103) */
AI void silu8(float16x8_t *v) {
  silu_lut4(v, silu_lut_tab());
  silu_lut4(v + 4, silu_lut_tab());
}

/* per-row affine, identical op order to tile_row: v = (b + (sw*acc)*sa) + swsum*(sa*zp) */
AI void affine4(int32x4_t c0, int32x4_t c1, int32x4_t c2, int32x4_t c3, float sa, float zs, const float32x4_t *sw,
                const float32x4_t *sws, const float32x4_t *bb, int mg, float32x4_t *v) {
  v[0] = vfmaq_n_f32(vfmaq_n_f32(bb[0], vmulq_f32(i2f(c0, mg), sw[0]), sa), sws[0], zs);
  v[1] = vfmaq_n_f32(vfmaq_n_f32(bb[1], vmulq_f32(i2f(c1, mg), sw[1]), sa), sws[1], zs);
  v[2] = vfmaq_n_f32(vfmaq_n_f32(bb[2], vmulq_f32(i2f(c2, mg), sw[2]), sa), sws[2], zs);
  v[3] = vfmaq_n_f32(vfmaq_n_f32(bb[3], vmulq_f32(i2f(c3, mg), sw[3]), sa), sws[3], zs);
}

/* kind 1: fp16 out + SiLU (optionally padded output); kind 2: fp32 out, act none/relu, +res32; 3: raw int32;
 * 0: generic */
AI void gemm_core(const int8_t *A, int lda, const rowq *rq, int per_tensor, int M, const qweights *W, const epi *e,
                  const int kind) {
  const int Kp = W->Kp, N = W->N, mg = Kp <= 256;
  const int ow = e->ow, opad = e->opad, Wp = ow + 2 * opad, ldo = e->ldo;
  for (int n = 0; n < N; n += 16) {
    const int8_t *wb = W->w + (size_t)(n / 16) * 16 * Kp;
    const float *swp = W->scale + n, *swsp = W->swsum + n, *bbp = W->bias + n;
    int y = 0, x = 0; /* carried output coordinates of row m (padded layouts) */
    for (int m = 0; m < M; m += 4) {
      int rows = M - m < 4 ? M - m : 4;
      const int8_t *a0 = A + (size_t)m * lda;
      const int8_t *a1 = rows > 1 ? a0 + lda : a0;
      const int8_t *a2 = rows > 2 ? a0 + 2 * (size_t)lda : a0;
      const int8_t *a3 = rows > 3 ? a0 + 3 * (size_t)lda : a0;
      int32x4_t c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33;
      {
        /* GCC only guarantees local register variables as asm operands: copy them into ordinary variables at
         * once, otherwise the out-of-line tile_row calls (caller-saved v16-v31) clobber them (found by the
         * container-vs-gated bit check: heads/conv_head differed under gcc, not clang). */
        register int32x4_t r00 asm("v16"), r01 asm("v17"), r02 asm("v18"), r03 asm("v19");
        register int32x4_t r10 asm("v20"), r11 asm("v21"), r12 asm("v22"), r13 asm("v23");
        register int32x4_t r20 asm("v24"), r21 asm("v25"), r22 asm("v26"), r23 asm("v27");
        register int32x4_t r30 asm("v28"), r31 asm("v29"), r32 asm("v30"), r33 asm("v31");
        const int8_t *p0 = a0, *p1 = a1, *p2 = a2, *p3 = a3, *w = wb;
        int cnt = Kp / 16;
        const int32_t *init = g_zero16;
#define c00 r00
#define c01 r01
#define c02 r02
#define c03 r03
#define c10 r10
#define c11 r11
#define c12 r12
#define c13 r13
#define c20 r20
#define c21 r21
#define c22 r22
#define c23 r23
#define c30 r30
#define c31 r31
#define c32 r32
#define c33 r33
        KLOOP(p0, p1, p2, p3, w, cnt, init);
#undef c00
#undef c01
#undef c02
#undef c03
#undef c10
#undef c11
#undef c12
#undef c13
#undef c20
#undef c21
#undef c22
#undef c23
#undef c30
#undef c31
#undef c32
#undef c33
        c00 = r00; c01 = r01; c02 = r02; c03 = r03; c10 = r10; c11 = r11; c12 = r12; c13 = r13;
        c20 = r20; c21 = r21; c22 = r22; c23 = r23; c30 = r30; c31 = r31; c32 = r32; c33 = r33;
      }
      const rowq *r = per_tensor ? rq : rq + m;
      const int st = per_tensor ? 0 : 1;
      if (kind == 3) { /* raw int32 tile store (Winograd: the output transform runs on integers) */
        int32_t *o = (int32_t *)e->out32 + (size_t)m * ldo + n;
        int32x4_t cr[16] = {c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33};
        for (int p = 0; p < rows; p++)
          for (int j = 0; j < 4; j++) vst1q_s32(o + (size_t)p * ldo + 4 * j, cr[4 * p + j]);
        continue;
      }
      if (kind == 0 || rows < 4) {
        tile_row(c00, c01, c02, c03, r[0].scale, r[0].scale * r[0].zp, swp, swsp, bbp, e, m, n, mg);
        if (rows > 1) tile_row(c10, c11, c12, c13, r[st].scale, r[st].scale * r[st].zp, swp, swsp, bbp, e, m + 1, n, mg);
        if (rows > 2) tile_row(c20, c21, c22, c23, r[2 * st].scale, r[2 * st].scale * r[2 * st].zp, swp, swsp, bbp, e, m + 2, n, mg);
        if (rows > 3) tile_row(c30, c31, c32, c33, r[3 * st].scale, r[3 * st].scale * r[3 * st].zp, swp, swsp, bbp, e, m + 3, n, mg);
        continue;
      }
      float32x4_t sw[4] = {vld1q_f32(swp), vld1q_f32(swp + 4), vld1q_f32(swp + 8), vld1q_f32(swp + 12)};
      float32x4_t sws[4] = {vld1q_f32(swsp), vld1q_f32(swsp + 4), vld1q_f32(swsp + 8), vld1q_f32(swsp + 12)};
      float32x4_t bb[4] = {vld1q_f32(bbp), vld1q_f32(bbp + 4), vld1q_f32(bbp + 8), vld1q_f32(bbp + 12)};
      float32x4_t v[16];
      affine4(c00, c01, c02, c03, r[0].scale, r[0].scale * r[0].zp, sw, sws, bb, mg, v);
      affine4(c10, c11, c12, c13, r[st].scale, r[st].scale * r[st].zp, sw, sws, bb, mg, v + 4);
      affine4(c20, c21, c22, c23, r[2 * st].scale, r[2 * st].scale * r[2 * st].zp, sw, sws, bb, mg, v + 8);
      affine4(c30, c31, c32, c33, r[3 * st].scale, r[3 * st].scale * r[3 * st].zp, sw, sws, bb, mg, v + 12);
      size_t po[4];
      if (opad) {
        for (int p = 0; p < 4; p++) {
          po[p] = ((size_t)(y + opad) * Wp + x + opad) * ldo + n;
          if (++x == ow) { x = 0; y++; }
        }
      } else {
        for (int p = 0; p < 4; p++) po[p] = (size_t)(m + p) * ldo + n;
      }
      if (kind == 1) {
        float16x8_t h[8];
        for (int j = 0; j < 8; j++) h[j] = vcombine_f16(vcvt_f16_f32(v[2 * j]), vcvt_f16_f32(v[2 * j + 1]));
        silu8(h);
        for (int p = 0; p < 4; p++) { vst1q_f16(e->out16 + po[p], h[2 * p]); vst1q_f16(e->out16 + po[p] + 8, h[2 * p + 1]); }
      } else {
        if (e->act == ACT_RELU)
          for (int j = 0; j < 16; j++) v[j] = vmaxq_f32(v[j], vdupq_n_f32(0));
        if (e->res32)
          for (int p = 0; p < 4; p++) {
            const float *rr = e->res32 + (size_t)(m + p) * ldo + n;
            for (int j = 0; j < 4; j++) v[4 * p + j] = vaddq_f32(v[4 * p + j], vld1q_f32(rr + 4 * j));
          }
        for (int p = 0; p < 4; p++)
          for (int j = 0; j < 4; j++) vst1q_f32(e->out32 + po[p] + 4 * j, v[4 * p + j]);
      }
    }
  }
}

void gemm_s8(const int8_t *A, int lda, const rowq *rq, int per_tensor, int M, const qweights *W, const epi *e) {
  if (e->act == ACT_RAW) {
    gemm_core(A, lda, rq, per_tensor, M, W, e, 3);
  } else if (e->out16 && e->act == ACT_SILU) {
    if (per_tensor) gemm_core(A, lda, rq, 1, M, W, e, 1);
    else gemm_core(A, lda, rq, 0, M, W, e, 1);
  } else if (e->out32 && e->act != ACT_SILU) {
    gemm_core(A, lda, rq, per_tensor, M, W, e, 2);
  } else {
    gemm_core(A, lda, rq, per_tensor, M, W, e, 0);
  }
}

/* ---------------- quantisation ---------------- */
/* q = round_even(x/s) - zp via magic add: (x*inv + 2^23 + 1.5*2^23...) -> we use 12582912 (1.5*2^23) whose low
 * mantissa bits hold round(x*inv) + 2^22; subtracting (2^22 + zp) as integer gives q; then saturate to int8. */
static inline int8x8_t q8_f32x8(float32x4_t a, float32x4_t b, float32x4_t inv, int32x4_t zofs) {
  const float32x4_t magic = vdupq_n_f32(12582912.0f);
  int32x4_t qa = vsubq_s32(vreinterpretq_s32_f32(vfmaq_f32(magic, a, inv)), zofs);
  int32x4_t qb = vsubq_s32(vreinterpretq_s32_f32(vfmaq_f32(magic, b, inv)), zofs);
  qa = vminq_s32(vmaxq_s32(qa, vdupq_n_s32(-128)), vdupq_n_s32(127));
  qb = vminq_s32(vmaxq_s32(qb, vdupq_n_s32(-128)), vdupq_n_s32(127));
  int16x8_t h = vcombine_s16(vmovn_s32(qa), vmovn_s32(qb));
  return vmovn_s16(h);
}

static inline void params(float mn, float mx, rowq *rq, float *inv, int32x4_t *zofs) {
  mn = fminf(mn, 0.f); mx = fmaxf(mx, 0.f);
  float s = (mx - mn) > 0 ? (mx - mn) / 255.f : 1.f;
  float iv = 1.f / s;
  float zp = nearbyintf(mn * iv) + 128.f;
  rq->scale = s; rq->zp = zp;
  *inv = iv;
  /* magic bits: 0x4B400000 = 12582912.0f; integer of (magic + v) bits = 0x4B400000 + round(v) */
  *zofs = vdupq_n_s32(0x4B400000 + (int32_t)zp);
}

void quant_rows_f16(const f16 *x, int ldx, int M, int C, const f16 *gate, int8_t *a, int lda, int Cp, rowq *rq) {
  /* all-fp16 arithmetic (0.5-cycle ops only): optional per-channel gate multiply (SE), row min/max, then
   * q = round(x*inv) - zp via the fp16 magic 1536 (ulp 1 in [1024,2048)); x*inv in [-255,255]. */
  const float16x8_t hm = vdupq_n_f16((f16)1536.0f);
  const int16x8_t lo = vdupq_n_s16(-128), hi = vdupq_n_s16(127);
  for (int m = 0; m < M; m++) {
    const f16 *r = x + (size_t)m * ldx;
    int8_t *o = a + (size_t)m * lda;
    float16x8_t mnh = vdupq_n_f16(0), mxh = vdupq_n_f16(0);
    int c = 0;
    if (gate)
      for (; c + 8 <= C; c += 8) {
        float16x8_t h = vmulq_f16(vld1q_f16(r + c), vld1q_f16(gate + c));
        mnh = vminq_f16(mnh, h); mxh = vmaxq_f16(mxh, h);
      }
    else
      for (; c + 8 <= C; c += 8) {
        float16x8_t h = vld1q_f16(r + c);
        mnh = vminq_f16(mnh, h); mxh = vmaxq_f16(mxh, h);
      }
    float fmn = vminvq_f16(mnh), fmx = vmaxvq_f16(mxh);
    for (; c < C; c++) {
      float v = (float)r[c] * (gate ? (float)gate[c] : 1.f);
      fmn = fminf(fmn, v); fmx = fmaxf(fmx, v);
    }
    float inv; int32x4_t zofs;
    params(fmn, fmx, &rq[m], &inv, &zofs);
    float16x8_t hinv = vdupq_n_f16((f16)inv);
    int16x8_t zo = vdupq_n_s16((int16_t)(0x6600 + (int)rq[m].zp));
    c = 0;
    for (; c + 16 <= C; c += 16) {
      float16x8_t h0 = vld1q_f16(r + c), h1 = vld1q_f16(r + c + 8);
      if (gate) { h0 = vmulq_f16(h0, vld1q_f16(gate + c)); h1 = vmulq_f16(h1, vld1q_f16(gate + c + 8)); }
      int16x8_t q0 = vsubq_s16(vreinterpretq_s16_f16(vfmaq_f16(hm, h0, hinv)), zo);
      int16x8_t q1 = vsubq_s16(vreinterpretq_s16_f16(vfmaq_f16(hm, h1, hinv)), zo);
      q0 = vminq_s16(vmaxq_s16(q0, lo), hi); q1 = vminq_s16(vmaxq_s16(q1, lo), hi);
      vst1q_s8(o + c, vuzp1q_s8(vreinterpretq_s8_s16(q0), vreinterpretq_s8_s16(q1)));
    }
    for (; c < C; c++) {
      float q = nearbyintf((float)r[c] * (gate ? (float)gate[c] : 1.f) * inv) - rq[m].zp;
      o[c] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
    }
    for (; c < Cp; c++) o[c] = 0;
  }
}

void quant_rows_f32(const float *x, int ldx, int M, int C, int8_t *a, int lda, int Cp, rowq *rq) {
  for (int m = 0; m < M; m++) {
    const float *r = x + (size_t)m * ldx;
    int8_t *o = a + (size_t)m * lda;
    float32x4_t mn = vdupq_n_f32(0), mx = vdupq_n_f32(0);
    int c = 0;
    for (; c + 4 <= C; c += 4) {
      float32x4_t v = vld1q_f32(r + c);
      mn = vminq_f32(mn, v); mx = vmaxq_f32(mx, v);
    }
    float fmn = vminvq_f32(mn), fmx = vmaxvq_f32(mx);
    for (; c < C; c++) { fmn = fminf(fmn, r[c]); fmx = fmaxf(fmx, r[c]); }
    float inv; int32x4_t zofs;
    params(fmn, fmx, &rq[m], &inv, &zofs);
    float32x4_t vinv = vdupq_n_f32(inv);
    c = 0;
    for (; c + 8 <= C; c += 8) vst1_s8(o + c, q8_f32x8(vld1q_f32(r + c), vld1q_f32(r + c + 4), vinv, zofs));
    for (; c < C; c++) {
      float q = nearbyintf(r[c] * inv) - rq[m].zp;
      o[c] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
    }
    for (; c < Cp; c++) o[c] = 0;
  }
}

void quant_tensor_pad_f32(const float *x, int H, int W, int C, int8_t *out, rowq *q) {
  size_t n = (size_t)H * W * C;
  float32x4_t mn = vdupq_n_f32(0), mx = vdupq_n_f32(0);
  for (size_t i = 0; i < n; i += 4) {
    float32x4_t v = vld1q_f32(x + i);
    mn = vminq_f32(mn, v); mx = vmaxq_f32(mx, v);
  }
  float inv; int32x4_t zofs;
  params(vminvq_f32(mn), vmaxvq_f32(mx), q, &inv, &zofs);
  float32x4_t vinv = vdupq_n_f32(inv);
  int8_t halo = (int8_t)(-(int)q->zp);
  int Wp = W + 2;
  memset(out, halo, (size_t)Wp * C);
  memset(out + (size_t)(H + 1) * Wp * C, halo, (size_t)Wp * C);
  for (int y = 0; y < H; y++) {
    int8_t *o = out + (size_t)(y + 1) * Wp * C;
    memset(o, halo, C);
    memset(o + (size_t)(W + 1) * C, halo, C);
    const float *r = x + (size_t)y * W * C;
    o += C;
    for (int i = 0; i < W * C; i += 8) vst1_s8(o + i, q8_f32x8(vld1q_f32(r + i), vld1q_f32(r + i + 4), vinv, zofs));
  }
}

void im2col_s8(const int8_t *xp, int H, int W, int C, int s, int pt, int pl, int Wo, int p0, int np, int8_t *col,
               int Kp) {
  const int Wp = W + 2, K = 9 * C;
  (void)H;
  for (int p = 0; p < np; p++) {
    int m = p0 + p, oy = m / Wo, ox = m - oy * Wo;
    int8_t *d = col + (size_t)p * Kp;
    for (int ky = 0; ky < 3; ky++) {
      const int8_t *row = xp + ((size_t)(oy * s + ky - pt + 1) * Wp + (ox * s - pl + 1)) * C;
      if (C % 16 == 0) {
        for (int i = 0; i < 3 * C; i += 16) vst1q_s8(d + i, vld1q_s8(row + i));
      } else {
        memcpy(d, row, 3 * C);
      }
      d += 3 * C;
    }
    for (int k = K; k < Kp; k++) col[(size_t)p * Kp + k] = 0;
  }
}
