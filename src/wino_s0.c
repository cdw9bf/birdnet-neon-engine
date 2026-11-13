// I005: stage-0 3x3 conv (C = K = 24, stride 1, pad 1, fp16, SiLU + residual) as Winograd F(2x2,3x3) in fp16.
// Transform coefficients are 0, +-1, +-1/2 (exact in fp16), 2.25x fewer MACs than direct. The transformed-domain
// GEMM reuses the engine's by-element FMLA asm block (4 tiles x 24 co, 8 ci per block). Not bit-identical to the
// direct fp16 conv it replaced in the retired hand-written engine (different fp16 summation order).
// Layouts (same as the engine): x / out / res padded [(H+2)][(W+2)][24] fp16, zero halo; H, W even.
#include <stdlib.h>
#include <string.h>
#include "act.h"
#include "kern.h"


/* 4 SiLU chains in lock-step, the polynomial form: the same ops as act.h's silu_f16_a (bit-identical to it; the
 * other kernels' SiLU is the table form) */
static inline __attribute__((always_inline)) void silu4w(float16x8_t *v) {
  const float16x8_t k = vdupq_n_f16((__fp16)-1.44269504f), lo = vdupq_n_f16((__fp16)-14.f), hi = vdupq_n_f16((__fp16)14.f);
  const float16x8_t magic = vdupq_n_f16((__fp16)1536.0f), c3 = vdupq_n_f16((__fp16)0.0555041f);
  const float16x8_t c2 = vdupq_n_f16((__fp16)0.2402265f), c1 = vdupq_n_f16((__fp16)0.6931472f), one = vdupq_n_f16((__fp16)1.0f);
  float16x8_t t[4], tm[4], f[4], pp[4], d[4], r[4];
  for (int j = 0; j < 4; j++) t[j] = vminq_f16(vmaxq_f16(vmulq_f16(v[j], k), lo), hi);
  for (int j = 0; j < 4; j++) tm[j] = vaddq_f16(t[j], magic);
  for (int j = 0; j < 4; j++) f[j] = vsubq_f16(t[j], vsubq_f16(tm[j], magic));
  for (int j = 0; j < 4; j++) pp[j] = vfmaq_f16(c2, c3, f[j]);
  for (int j = 0; j < 4; j++) pp[j] = vfmaq_f16(c1, pp[j], f[j]);
  for (int j = 0; j < 4; j++) pp[j] = vfmaq_f16(one, pp[j], f[j]);
  for (int j = 0; j < 4; j++)
    d[j] = vaddq_f16(vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(pp[j]), vshlq_n_s16(vreinterpretq_s16_f16(tm[j]), 10))), one);
  for (int j = 0; j < 4; j++) r[j] = vrecpeq_f16(d[j]);
  for (int j = 0; j < 4; j++) r[j] = vmulq_f16(r[j], vrecpsq_f16(d[j], r[j]));
  for (int j = 0; j < 4; j++) v[j] = vmulq_f16(v[j], r[j]);
}

typedef struct wino_s0_t {
  f16 *U;    /* [16 pos][24 ci][24 co] */
  f16 bias[24];
  f16 *V, *M; /* scratch: [16][tiles_pad][24] each */
  int tiles_pad;
} wino_s0_t;

/* U = G g G^T per (co, ci), G = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]; w is OHWI [24co][3][3][24ci] fp32 */
wino_s0_t *wino_s0_pack(const float *w, const float *bias, int W) {
  wino_s0_t *p = calloc(1, sizeof *p);
  p->U = aligned_alloc(64, sizeof(f16) * 16 * 24 * 24);
  p->tiles_pad = ((W / 2 + 3) / 4) * 4;
  p->V = aligned_alloc(64, sizeof(f16) * 16 * p->tiles_pad * 24);
  p->M = aligned_alloc(64, sizeof(f16) * 16 * p->tiles_pad * 24);
  memset(p->V, 0, sizeof(f16) * 16 * p->tiles_pad * 24);
  static const float G[4][3] = {{1, 0, 0}, {.5f, .5f, .5f}, {.5f, -.5f, .5f}, {0, 0, 1}};
  for (int co = 0; co < 24; co++) {
    p->bias[co] = (f16)bias[co];
    for (int ci = 0; ci < 24; ci++) {
      float g[3][3], t[4][3], u[4][4];
      for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++) g[ky][kx] = w[((co * 3 + ky) * 3 + kx) * 24 + ci];
      for (int i = 0; i < 4; i++)
        for (int kx = 0; kx < 3; kx++) t[i][kx] = G[i][0] * g[0][kx] + G[i][1] * g[1][kx] + G[i][2] * g[2][kx];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) u[i][j] = t[i][0] * G[j][0] + t[i][1] * G[j][1] + t[i][2] * G[j][2];
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) p->U[((size_t)(i * 4 + j) * 24 + ci) * 24 + co] = (f16)u[i][j];
    }
  }
  return p;
}
void wino_s0_free(wino_s0_t *p) { if (p) { free(p->U); free(p->V); free(p->M); free(p); } }

#define LD(r, c) vld1q_f16(d + ((size_t)(r) * Wp + (c)) * 24 + 8 * g)
/* input transform of one tile (4x4x24 at padded rows r0.., cols c0..) into V[pos][tile][24] */
static inline __attribute__((always_inline)) void in_tf(const f16 *d, int Wp, f16 *V, size_t vstride) {
  for (int g = 0; g < 3; g++) {
    float16x8_t t[4][4];
    for (int j = 0; j < 4; j++) {
      float16x8_t d0 = LD(0, j), d1 = LD(1, j), d2 = LD(2, j), d3 = LD(3, j);
      t[0][j] = vsubq_f16(d0, d2); t[1][j] = vaddq_f16(d1, d2); t[2][j] = vsubq_f16(d2, d1); t[3][j] = vsubq_f16(d1, d3);
    }
    for (int i = 0; i < 4; i++) {
      vst1q_f16(V + (size_t)(i * 4 + 0) * vstride + 8 * g, vsubq_f16(t[i][0], t[i][2]));
      vst1q_f16(V + (size_t)(i * 4 + 1) * vstride + 8 * g, vaddq_f16(t[i][1], t[i][2]));
      vst1q_f16(V + (size_t)(i * 4 + 2) * vstride + 8 * g, vsubq_f16(t[i][2], t[i][1]));
      vst1q_f16(V + (size_t)(i * 4 + 3) * vstride + 8 * g, vsubq_f16(t[i][1], t[i][3]));
    }
  }
}
#undef LD

/* 4 tiles x 24 co, 8 ci: the engine's s0 asm block (element regs v0-v3, weights v4-v6, accs v16-v27) */
#define FMLA8(L)                                                                                     \
  "fmla v16.8h, v4.8h, v0.h[" L "]\n fmla v17.8h, v5.8h, v0.h[" L "]\n fmla v18.8h, v6.8h, v0.h[" L "]\n" \
  "fmla v19.8h, v4.8h, v1.h[" L "]\n fmla v20.8h, v5.8h, v1.h[" L "]\n fmla v21.8h, v6.8h, v1.h[" L "]\n" \
  "fmla v22.8h, v4.8h, v2.h[" L "]\n fmla v23.8h, v5.8h, v2.h[" L "]\n fmla v24.8h, v6.8h, v2.h[" L "]\n" \
  "fmla v25.8h, v4.8h, v3.h[" L "]\n fmla v26.8h, v5.8h, v3.h[" L "]\n fmla v27.8h, v6.8h, v3.h[" L "]\n"
#define WBLK "ldp q4, q5, [%[w]], #32\n ldr q6, [%[w]], #16\n"

static inline __attribute__((always_inline)) void gemm4(const f16 *v0, const f16 *v1, const f16 *v2, const f16 *v3,
                                                        const f16 *U, f16 *m0, f16 *m1, f16 *m2, f16 *m3) {
  register float16x8_t a00 __asm__("v16") = vdupq_n_f16(0), a01 __asm__("v17") = a00, a02 __asm__("v18") = a00;
  register float16x8_t a10 __asm__("v19") = a00, a11 __asm__("v20") = a00, a12 __asm__("v21") = a00;
  register float16x8_t a20 __asm__("v22") = a00, a21 __asm__("v23") = a00, a22 __asm__("v24") = a00;
  register float16x8_t a30 __asm__("v25") = a00, a31 __asm__("v26") = a00, a32 __asm__("v27") = a00;
  for (int g = 0; g < 3; g++) {
    const f16 *wg = U + (size_t)(8 * g) * 24;
    __asm__ volatile(
        "ldr q0, [%[p0]]\n ldr q1, [%[p1]]\n ldr q2, [%[p2]]\n ldr q3, [%[p3]]\n"
        WBLK FMLA8("0") WBLK FMLA8("1") WBLK FMLA8("2") WBLK FMLA8("3")
        WBLK FMLA8("4") WBLK FMLA8("5") WBLK FMLA8("6") WBLK FMLA8("7")
        : [w] "+r"(wg), "+w"(a00), "+w"(a01), "+w"(a02), "+w"(a10), "+w"(a11), "+w"(a12), "+w"(a20), "+w"(a21),
          "+w"(a22), "+w"(a30), "+w"(a31), "+w"(a32)
        : [p0] "r"(v0 + 8 * g), [p1] "r"(v1 + 8 * g), [p2] "r"(v2 + 8 * g), [p3] "r"(v3 + 8 * g)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "memory");
  }
  vst1q_f16(m0, a00); vst1q_f16(m0 + 8, a01); vst1q_f16(m0 + 16, a02);
  vst1q_f16(m1, a10); vst1q_f16(m1 + 8, a11); vst1q_f16(m1 + 16, a12);
  vst1q_f16(m2, a20); vst1q_f16(m2 + 8, a21); vst1q_f16(m2 + 16, a22);
  vst1q_f16(m3, a30); vst1q_f16(m3 + 8, a31); vst1q_f16(m3 + 16, a32);
}

/* output transform A^T M A (+ bias, SiLU, + residual) of one tile; M[pos*mstride + 8g] */
static inline __attribute__((always_inline)) void out_tf(const f16 *M, size_t mstride, const f16 *bias, f16 *o, const f16 *r,
                                                         int Wp) {
  for (int g = 0; g < 3; g++) {
    float16x8_t m[4][4], s[2][4], y[4];
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) m[i][j] = vld1q_f16(M + (size_t)(i * 4 + j) * mstride + 8 * g);
    for (int j = 0; j < 4; j++) {
      s[0][j] = vaddq_f16(vaddq_f16(m[0][j], m[1][j]), m[2][j]);
      s[1][j] = vsubq_f16(vsubq_f16(m[1][j], m[2][j]), m[3][j]);
    }
    float16x8_t b = vld1q_f16(bias + 8 * g);
    for (int i = 0; i < 2; i++) {
      y[2 * i] = vaddq_f16(vaddq_f16(vaddq_f16(s[i][0], s[i][1]), s[i][2]), b);
      y[2 * i + 1] = vaddq_f16(vsubq_f16(vsubq_f16(s[i][1], s[i][2]), s[i][3]), b);
    }
    silu4w(y);
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) {
        size_t off = ((size_t)i * Wp + j) * 24 + 8 * g;
        vst1q_f16(o + off, vaddq_f16(y[2 * i + j], vld1q_f16(r + off)));
      }
  }
}

/* elements of each of the two scratch arrays V and M of the row-range form (V zero-initialised by the caller) */
size_t wino_s0_scratch_elems(const wino_s0_t *p) { return (size_t)16 * p->tiles_pad * 24; }
/* tile rows [ty0, ty1) (output rows 2 ty0 .. 2 ty1 - 1) with scratch V, M of the caller: tile rows are independent,
 * so any split over threads (one V, M pair per thread) writes the same bytes */
void conv3x3_s1_wino_f16_c24_rows(const f16 *x, int H, int W, const wino_s0_t *p, f16 *out, const f16 *res, f16 *Vs,
                                  f16 *Ms, int ty0, int ty1) {
  const int Wp = W + 2, T = W / 2, TP = p->tiles_pad;
  const size_t vs = (size_t)TP * 24; /* position stride in V / M */
  (void)H;
  if (!Vs) { Vs = p->V; Ms = p->M; } /* the scratch packed with the weights (one thread at a time may use it) */
  for (int ty = ty0; ty < ty1; ty++) {
    const f16 *rows = x + (size_t)(2 * ty) * Wp * 24;
    for (int tx = 0; tx < T; tx++) in_tf(rows + (size_t)(2 * tx) * 24, Wp, Vs + (size_t)tx * 24, vs);
    for (int pos = 0; pos < 16; pos++) {
      const f16 *V = Vs + pos * vs, *U = p->U + (size_t)pos * 24 * 24;
      f16 *M = Ms + pos * vs;
      for (int tx = 0; tx < TP; tx += 4)
        gemm4(V + (size_t)tx * 24, V + (size_t)(tx + 1) * 24, V + (size_t)(tx + 2) * 24, V + (size_t)(tx + 3) * 24, U,
              M + (size_t)tx * 24, M + (size_t)(tx + 1) * 24, M + (size_t)(tx + 2) * 24, M + (size_t)(tx + 3) * 24);
    }
    f16 *orow = out + ((size_t)(2 * ty + 1) * Wp + 1) * 24;
    const f16 *rrow = res + ((size_t)(2 * ty + 1) * Wp + 1) * 24;
    for (int tx = 0; tx < T; tx++) out_tf(Ms + (size_t)tx * 24, vs, p->bias, orow + (size_t)(2 * tx) * 24, rrow + (size_t)(2 * tx) * 24, Wp);
  }
}
