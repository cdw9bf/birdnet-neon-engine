// NEON kernels of the dotprod + fp16 package ops (Cortex-A76: dotprod + fp16 arith; rt_ops_dot.c), first written for
// the retired hand-written BirdNET v3.0 engine
#ifndef BNV3_KERN_H
#define BNV3_KERN_H
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

typedef __fp16 f16;
#include "rt.h"
static inline void *al64(size_t n) { return rt_al64(n); } /* the kernels' name for it */

/* ---- int8 weights, per-output-channel symmetric, packed "c4": [N/16][Kp/4][16 ch][4 k] ---- */
typedef struct {
  int N, K, Kp;      /* Kp = K rounded up to 16 */
  int8_t *w;         /* packed */
  float *scale;      /* [N] weight scale s_w */
  float *swsum;      /* [N] s_w * sum_k q_w  (zero-point fold, fp32) */
  float *bias;       /* [N] */
} qweights;

void qw_pack(qweights *q, const float *w /*[N][K] (K contiguous)*/, const float *bias, int N, int K);
void qw_free(qweights *q);

/* ---- per-row asymmetric int8 activation rows: x ~ (q + zp) * scale ---- */
typedef struct {
  float scale;
  float zp; /* integer-valued */
} rowq;

enum { ACT_NONE = 0, ACT_SILU = 1, ACT_RELU = 2, ACT_RAW = 3 /* raw int32 accumulators to out32 (as int32_t*) */ };
typedef struct {
  int act;
  /* output: exactly one of out16 / out32 */
  f16 *out16;
  float *out32;
  int ldo;            /* channel stride of an output pixel */
  int ow, opad;       /* if opad: pixel m -> padded position ((m/ow)+opad)*(ow+2opad) + m%ow + opad */
  const float *res32; /* optional fp32 residual, same pixel indexing as out (no pad), stride ldo */
} epi;

/* C[m][n] = epi( sa[m] * (sw[n]*acc + zp[m]*swsum[n]) + b[n] ), acc = sum_k A[m][k]*Wq[n][k]
 * A: M rows of Kp int8 (stride lda). rq: per-row params (if per_tensor, rq[0] for all rows). N % 16 == 0. */
void gemm_s8(const int8_t *A, int lda, const rowq *rq, int per_tensor, int M, const qweights *W, const epi *e);

/* quantisation passes */
void quant_rows_f16(const f16 *x, int ldx, int M, int C, const f16 *gate /*[C] or NULL*/, int8_t *a, int lda,
                    int Cp, rowq *rq);
void quant_rows_f32(const float *x, int ldx, int M, int C, int8_t *a, int lda, int Cp, rowq *rq);
/* per-tensor asym quant of an NHWC fp32 tensor into a padded int8 buffer [(H+2)][(W+2)][C] with halo = -zp */
void quant_tensor_pad_f32(const float *x, int H, int W, int C, int8_t *out, rowq *q);
/* Winograd F(2,3) int8 (wino.c), in two steps (threads): the padded copy once, then tile ranges with a work area per
 * thread */
void *wino_pack(const float *w, const float *bias, int K, int C);
void wino_free(void *p);
#define WINO_CHUNK 8 /* tiles per GEMM call (4, 16 and 32 were measured: DEADENDS D26) */
size_t wino_xpad_bytes(int H, int W, int C);
size_t wino_work_bytes(int C, int K);
void wino_f23_pad(const float *x, int H, int W, int C, float *xp);
void wino_f23_tiles(const void *wp, const float *xp, int H, int W, f16 *out, void *work,
                    void (*gemm)(const int8_t *, int, const rowq *, int, int, const qweights *, const epi *),
                    int t_begin, int t_end);
/* im2col 3x3 from a padded int8 tensor (pad 1 all round; TF-same offsets pt/pl in {0,1}) for output rows
 * [p0, p0+np) of an Ho x Wo map; rows of Kp bytes (K = 9*C, tail zero) */
void im2col_s8(const int8_t *xp, int H, int W, int C, int s, int pt, int pl, int Wo, int p0, int np, int8_t *col,
               int Kp);

/* fp16 / fp32 kernels */
void stem_conv_f32_h(const float *img3 /*[H][W][3]*/, int H, int W, const float *w /*OHWI [24][3][3][3]*/,
                     const float *b, f16 *out /*[H/2 + 2][W/2 + 2][24], interior written; H and W even*/);
/* depthwise 3x3 fp16 on a padded input [(H+2)][(W+2)][C] (zero halo), stride s, TF-same offsets; SiLU; writes
 * [Ho][Wo][C] and per-channel sums (fp32, for SE) */
void dw3x3_f16(const f16 *xp, int H, int W, int C, int s, int pt, int pl, const f16 *w /*[9][C]*/,
               const f16 *b, f16 *out, int Ho, int Wo, float *chsum);
void dw3x3_f16_cr(const f16 *xp, int H, int W, int C, int s, int pt, int pl, const f16 *w, const f16 *b, f16 *out,
                  int Ho, int Wo, float *chsum, int c0, int c1); /* channels [c0, c1), 8-aligned (threads) */

/* fp16 Winograd F(2,3) of the 24 -> 24 channel 3x3 conv (wino_s0.c): packed weights with one caller's scratch; the
 * _rows form takes tile rows [ty0, ty1) and a scratch pair per thread of wino_s0_scratch_elems fp16 values each
 * (NULL: the scratch packed with the weights) */
struct wino_s0_t;
struct wino_s0_t *wino_s0_pack(const float *w, const float *bias, int W);
void wino_s0_free(struct wino_s0_t *p);
size_t wino_s0_scratch_elems(const struct wino_s0_t *p);
void conv3x3_s1_wino_f16_c24_rows(const f16 *x, int H, int W, const struct wino_s0_t *p, f16 *out, const f16 *res, f16 *Vs,
                                  f16 *Ms, int ty0, int ty1);
#endif
