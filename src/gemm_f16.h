// fp16 GEMM kernels (gemm_f16.c) shared by the v2.4 engine
#ifndef GEMM_F16_H
#define GEMM_F16_H
#include <stddef.h>
#include <stdint.h>

typedef __fp16 f16;

#define FLUSH_K 128 /* max fp16 accumulation block (k steps); longer K is split into balanced blocks summed in fp32 */

typedef struct {
  int N, K, Kp, Np; /* Kp: multiple of 8 (>= K, the A row stride covers it, zero weights beyond K); Np = N rounded to 8 */
  f16 *w;           /* [Np/16][Kp][16] then, if Np % 16, a tail panel [Kp][8] */
  float *b32;       /* [Np] */
  f16 *b16;         /* [Np] */
} hpack;

enum { ACT16_NONE = 0, ACT16_SILU = 1, ACT16_RELU = 2, ACT16_RELU_POOL2 = 3 /* ReLU + [max|avg] of row pairs */ };
typedef struct {
  int act;
  f16 *out;
  int ldo;          /* output row stride (channels); >= N rounded to 8 (pad channels get 0) */
  int ow, opad;     /* if opad: pixel m -> padded position ((m/ow)+opad)*(ow+2opad) + m%ow + opad */
  const f16 *res;   /* optional residual (added after the activation), row stride ldr */
  int ldr;
  int pool_off;     /* ACT16_RELU_POOL2: channel offset of the avg half */
} epi16;

void hp_pack(hpack *h, const float *w /*[N][K]*/, const float *bias, int N, int K, int Kp);
void hp_free(hpack *h);
void gemm_f16(const f16 *A, int lda, int M, const hpack *W, const epi16 *e);
/* rows [m_begin, m_end) only (m_begin % 8 == 0; m_end % 8 == 0 or m_end == M): the union over a partition of
 * [0, M) writes the same bytes as gemm_f16 (threads) */
void gemm_f16_range(const f16 *A, int lda, int M, const hpack *W, const epi16 *e, int m_begin, int m_end);
/* colmax[n] = max over the M rows of (A . W^T)[m][n] (no bias), fp32, n < Np */
void gemm_f16_colmax(const f16 *A, int lda, int M, const hpack *W, float *colmax);
void gemm_f16_colmax_range(const f16 *A, int lda, int M, const hpack *W, float *colmax, int nb0, int nb1); /* 16-column panels [nb0, nb1) */
void gemm_f16_rows(const f16 *const *rows, int nseg, int segk, int M, const hpack *W, const epi16 *e);

#endif
