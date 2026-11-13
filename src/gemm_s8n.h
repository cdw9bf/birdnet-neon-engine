/* int8 GEMM without dotprod (gemm_s8n.c): SMULL + SMLAL2 + SADALP, per-channel weights, per-row dynamic activations */
#ifndef GEMM_S8N_H
#define GEMM_S8N_H
#include <stdint.h>

#include "gemm_f32.h"

typedef struct {
  int N, K, Kp, Np;       /* Kp: K rounded up to 16; Np: N rounded up to 8 (zero weights beyond N) */
  const int8_t *w;        /* [Np/4][Kp/16][4][16] */
  const float *sw, *b;    /* [Np] per-channel weight scale, bias */
  const int32_t *colsum;  /* [Np] sum over k of the int8 weights (zero-point correction) */
} s8pack;

/* rows [r0, r1) of x (row stride ld, K used) -> q [M][Kp] int8 (padded k = the zero point), scale sa, zero point za */
void s8n_quant_rows(const float *x, int ld, int r0, int r1, int K, int Kp, int8_t *q, float *sa, int32_t *za);
/* the same on x[m][k] * gate[k] (an SE gate fused into the quantization; K % 4 == 0, K <= 4096) */
void s8n_quant_rows_gate(const float *x, int ld, int r0, int r1, int K, int Kp, const float *gate, int8_t *q, float *sa,
                         int32_t *za);
/* rows [m0, m1) x 4-channel panels [p0, p1) of out = act(dequant(q . W) + b) (+ residual), epi32 store rules; a 4-row
 * tile past M repeats row m0 (the last 1-3 rows of M run a tile of their own) */
void s8n_gemm_range(const int8_t *q, int Kp, const float *sa, const int32_t *za, int M, const s8pack *W, const epi32 *e,
                    int m0, int m1, int p0, int p1);
#endif
