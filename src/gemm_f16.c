// fp16 GEMM for 1x1 convolutions / im2col convs (ARMv8.2 fp16 arithmetic, Cortex-A76):
//   C[m][n] = act( sum_k A[m][k] * W[n][k] + b[n] ) (+ residual), A and C fp16 NHWC rows.
// Micro-kernels 8 rows x 16 channels and 8 rows x 8 channels (N tail): by-element FMLA .8h in one inline-asm K loop
// (gemm_f16_kblock.h; gcc turns the intrinsics into DUP + FMLA and spills). Weights packed in 16-channel panels
// [Kp][16] plus an optional 8-channel tail panel [Kp][8]. Accumulation is fp16 within K blocks of <= FLUSH_K and fp32
// across blocks (the precision study behind it: DEADENDS.md and docs/ADDING_A_MODEL.md, section 7).
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_f16.h"
#include "gemm_f16_kblock.h"
#include "silu_lut.h"

#define AI static inline __attribute__((always_inline))

void hp_pack(hpack *h, const float *w, const float *bias, int N, int K, int Kp) {
  memset(h, 0, sizeof *h);
  h->N = N; h->K = K; h->Kp = Kp; h->Np = (N + 7) & ~7;
  size_t n = (size_t)h->Np * Kp, bytes = ((n * 2) + 127) & ~(size_t)63; /* +64 B slack: the K loop reads ahead */
  h->w = aligned_alloc(64, bytes);
  memset(h->w, 0, bytes);
  int n16 = h->Np / 16;
  for (int o = 0; o < N; o++) {
    int nb = o / 16, j = o % 16;
    for (int k = 0; k < K; k++) {
      size_t idx = nb < n16 ? ((size_t)nb * Kp + k) * 16 + j : (size_t)n16 * Kp * 16 + (size_t)k * 8 + j;
      h->w[idx] = (__fp16)w[(size_t)o * K + k];
    }
  }
  h->b32 = calloc(h->Np + 16, sizeof(float));
  h->b16 = aligned_alloc(64, ((size_t)(h->Np + 16) * 2 + 63) & ~(size_t)63);
  for (int o = 0; o < h->Np + 16; o++) {
    h->b32[o] = o < N && bias ? bias[o] : 0.f;
    h->b16[o] = (__fp16)h->b32[o];
  }
}

void hp_free(hpack *h) {
  free(h->w); free(h->b32); free(h->b16);
  memset(h, 0, sizeof *h);
}

AI size_t out_row(const epi16 *e, int m) {
  if (!e->opad) return (size_t)m;
  return (size_t)((m / e->ow) + e->opad) * (e->ow + 2 * e->opad) + (m % e->ow) + e->opad;
}

/* K blocks: as few as possible with <= FLUSH_K k-steps each, balanced, multiples of 8 */
AI void kblocks(int Kp, int *nblk, int *bs) {
  *nblk = (Kp + FLUSH_K - 1) / FLUSH_K;
  *bs = ((Kp + *nblk - 1) / *nblk + 7) & ~7;
}

AI void flush_add(float *fb, const f16 *cb, int nv /* 8-lane vectors */) {
  for (int i = 0; i < nv; i++) {
    float16x8_t c = vld1q_f16(cb + i * 8);
    vst1q_f32(fb + i * 8, vaddq_f32(vld1q_f32(fb + i * 8), vcvt_f32_f16(vget_low_f16(c))));
    vst1q_f32(fb + i * 8 + 4, vaddq_f32(vld1q_f32(fb + i * 8 + 4), vcvt_high_f32_f16(c)));
  }
}

/* epilogue for one tile: v[r][0..nv) = pre-activation fp16 rows (bias added); act, residual / pool, store */
AI void store_tile(float16x8_t v[8][2], int nv, int mr, int m0, int n0, const epi16 *e, const silu_lut_t *lt) {
  if (e->act == ACT16_RELU_POOL2) {
    /* rows 2q, 2q+1 are adjacent pixels along W: ReLU, then [max | avg] into pooled pixel m0/2 + q */
    for (int q = 0; 2 * q < mr; q++)
      for (int h = 0; h < nv; h++) {
        float16x8_t a = vmaxq_f16(v[2 * q][h], vdupq_n_f16(0)), b = vmaxq_f16(v[2 * q + 1][h], vdupq_n_f16(0));
        f16 *op = e->out + (size_t)(m0 / 2 + q) * e->ldo + n0 + 8 * h;
        vst1q_f16(op, vmaxq_f16(a, b));
        vst1q_f16(op + e->pool_off, vmulq_f16(vaddq_f16(a, b), vdupq_n_f16(0.5f)));
      }
    return;
  }
  size_t orow = out_row(e, m0);
  int ocol = e->opad ? m0 % e->ow : 0;
  for (int r = 0; r < mr; r += 2) {
    float16x8_t x[4];
    if (nv == 2) {
      x[0] = v[r][0]; x[1] = v[r][1]; x[2] = v[r + 1][0]; x[3] = v[r + 1][1];
      if (e->act == ACT16_SILU) silu_lut4(x, lt);
    } else {
      x[0] = v[r][0]; x[1] = v[r + 1][0];
      if (e->act == ACT16_SILU) { x[0] = silu_lut_h(x[0], lt); x[1] = silu_lut_h(x[1], lt); }
    }
    if (e->act == ACT16_RELU)
      for (int q = 0; q < 2 * nv; q++) x[q] = vmaxq_f16(x[q], vdupq_n_f16(0));
    for (int q = 0; q < 2 && r + q < mr; q++) {
      float16x8_t x0 = x[nv * q], x1 = nv == 2 ? x[2 * q + 1] : x0;
      if (e->res) {
        const f16 *rp = e->res + (size_t)(m0 + r + q) * e->ldr + n0;
        x0 = vaddq_f16(x0, vld1q_f16(rp));
        if (nv == 2) x1 = vaddq_f16(x1, vld1q_f16(rp + 8));
      }
      f16 *op = e->out + orow * e->ldo + n0;
      vst1q_f16(op, x0);
      if (nv == 2) vst1q_f16(op + 8, x1);
      orow++;
      if (e->opad && ++ocol == e->ow) { ocol = 0; orow += 2 * e->opad; }
    }
  }
}

/* one tile of 8 rows x (16 | 8) channels from row pointers ar[8] (contiguous K) */
AI void tile(const f16 *const *ar, int m0, int mr, const f16 *wp, int wide, const hpack *W, int n0, const epi16 *e,
             const silu_lut_t *lt, float *fb, f16 *cb) {
  int nblk, bs;
  kblocks(W->Kp, &nblk, &bs);
  int nv = wide ? 2 : 1, ws = wide ? 16 : 8;
  if (nblk > 1) memset(fb, 0, sizeof(float) * 8 * ws);
  for (int kb = 0; kb < W->Kp; kb += bs) {
    int kn = W->Kp - kb < bs ? W->Kp - kb : bs;
    if (wide)
      kblock0_8x16(ar[0] + kb, ar[1] + kb, ar[2] + kb, ar[3] + kb, ar[4] + kb, ar[5] + kb, ar[6] + kb, ar[7] + kb,
                   wp + (size_t)kb * 16, kn / 8, cb);
    else
      kblock0_8x8(ar[0] + kb, ar[1] + kb, ar[2] + kb, ar[3] + kb, ar[4] + kb, ar[5] + kb, ar[6] + kb, ar[7] + kb,
                  wp + (size_t)kb * 8, kn / 8, cb);
    if (nblk > 1) flush_add(fb, cb, nv * 8);
  }
  float16x8_t v[8][2];
  if (nblk > 1) {
    for (int r = 0; r < 8; r++)
      for (int h = 0; h < nv; h++) {
        const float *q = fb + r * ws + h * 8;
        v[r][h] = vcombine_f16(vcvt_f16_f32(vaddq_f32(vld1q_f32(q), vld1q_f32(W->b32 + n0 + 8 * h))),
                               vcvt_f16_f32(vaddq_f32(vld1q_f32(q + 4), vld1q_f32(W->b32 + n0 + 8 * h + 4))));
      }
  } else {
    for (int r = 0; r < 8; r++)
      for (int h = 0; h < nv; h++) v[r][h] = vaddq_f16(vld1q_f16(cb + r * ws + h * 8), vld1q_f16(W->b16 + n0 + 8 * h));
  }
  store_tile(v, nv, mr, m0, n0, e, lt);
}

AI void tile_rows(const f16 *A, int lda, int M, int m0, const f16 **ar) {
  int mr = M - m0 < 8 ? M - m0 : 8;
  for (int r = 0; r < 8; r++) ar[r] = A + (size_t)(m0 + (r < mr ? r : mr - 1)) * lda;
}

/* rows [m_begin, m_end) of the GEMM (m_begin a multiple of 8, m_end a multiple of 8 or M): every 8-row tile is
 * computed and stored on its own, so any split of the rows over threads gives the same bytes as one call */
void gemm_f16_range(const f16 *A, int lda, int M, const hpack *W, const epi16 *e, int m_begin, int m_end) {
  silu_lut_t lt = silu_lut_load();
  float fb[128] __attribute__((aligned(64)));
  f16 cb[128] __attribute__((aligned(64)));
  const f16 *ar[8];
  int n16 = W->Np / 16, tail = W->Np % 16 != 0;
  const f16 *wt = W->w + (size_t)n16 * W->Kp * 16;
  if ((size_t)W->Kp * W->Np * 2 <= 48 * 1024) {
    /* small weights (fit L1): rows outer, so A rows are read once and output rows are written whole */
    for (int m0 = m_begin; m0 < m_end; m0 += 8) {
      int mr = M - m0 < 8 ? M - m0 : 8;
      tile_rows(A, lda, M, m0, ar);
      for (int nb = 0; nb < n16; nb++) tile(ar, m0, mr, W->w + (size_t)nb * W->Kp * 16, 1, W, nb * 16, e, &lt, fb, cb);
      if (tail) tile(ar, m0, mr, wt, 0, W, n16 * 16, e, &lt, fb, cb);
    }
  } else {
    for (int nb = 0; nb < n16 + tail; nb++) {
      for (int m0 = m_begin; m0 < m_end; m0 += 8) {
        int mr = M - m0 < 8 ? M - m0 : 8;
        tile_rows(A, lda, M, m0, ar);
        if (nb < n16) tile(ar, m0, mr, W->w + (size_t)nb * W->Kp * 16, 1, W, nb * 16, e, &lt, fb, cb);
        else tile(ar, m0, mr, wt, 0, W, n16 * 16, e, &lt, fb, cb);
      }
    }
  }
}
void gemm_f16(const f16 *A, int lda, int M, const hpack *W, const epi16 *e) { gemm_f16_range(A, lda, M, W, e, 0, M); }

/* GEMM whose K is nseg segments of segk (multiple of 8) taken from separate row pointers: row m, segment s starts
 * at rows[s * M + m]. Used by the 4x8 stem conv on a padded fp16 image (no im2col). nseg * segk <= FLUSH_K. */
void gemm_f16_rows(const f16 *const *rows, int nseg, int segk, int M, const hpack *W, const epi16 *e) {
  silu_lut_t lt = silu_lut_load();
  f16 cb[128] __attribute__((aligned(64)));
  int n16 = W->Np / 16, tail = W->Np % 16 != 0;
  for (int m0 = 0; m0 < M; m0 += 8) {
    int mr = M - m0 < 8 ? M - m0 : 8;
    for (int nb = 0; nb < n16 + tail; nb++) {
      int wide = nb < n16, ws = wide ? 16 : 8, nv = wide ? 2 : 1;
      const f16 *wp = W->w + (size_t)nb * W->Kp * 16; /* the tail panel starts right after the 16-panels */
      for (int s = 0; s < nseg; s++) {
        const f16 *const *rp = rows + (size_t)s * M;
        const f16 *a[8];
        for (int r = 0; r < 8; r++) a[r] = rp[m0 + (r < mr ? r : mr - 1)];
        const f16 *w = wp + (size_t)s * segk * ws;
        if (wide) {
          if (s == 0) kblock0_8x16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], w, segk / 8, cb);
          else kblock_8x16(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], w, segk / 8, cb);
        } else {
          if (s == 0) kblock0_8x8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], w, segk / 8, cb);
          else kblock_8x8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], w, segk / 8, cb);
        }
      }
      float16x8_t v[8][2];
      for (int r = 0; r < 8; r++)
        for (int h = 0; h < nv; h++)
          v[r][h] = vaddq_f16(vld1q_f16(cb + r * ws + h * 8), vld1q_f16(W->b16 + nb * 16 + 8 * h));
      store_tile(v, nv, mr, m0, nb * 16, e, &lt);
    }
  }
}

/* column maxima of A . W^T (no bias, no activation): colmax[n] = max_m sum_k A[m][k] W[n][k] for n < Np, fp32.
 * Same K blocking and fp16 accumulation / fp32 flush as gemm_f16; the per-tile results are reduced over the rows
 * in fp32 instead of being stored (prototype heads: M cells x N prototypes, only the max over cells is needed).
 * Weight panels outer (each panel is streamed once and stays in L1 across the row tiles). */
void gemm_f16_colmax(const f16 *A, int lda, int M, const hpack *W, float *colmax) {
  gemm_f16_colmax_range(A, lda, M, W, colmax, 0, W->Np / 16 + (W->Np % 16 != 0));
}
/* weight panels [nb0, nb1) (16 columns each, the 8-column tail last): each panel's maxima are independent */
void gemm_f16_colmax_range(const f16 *A, int lda, int M, const hpack *W, float *colmax, int nb0, int nb1) {
  float fb[128] __attribute__((aligned(64)));
  f16 cb[128] __attribute__((aligned(64)));
  const f16 *ar[8];
  int n16 = W->Np / 16;
  int nblk, bs;
  kblocks(W->Kp, &nblk, &bs);
  for (int nb = nb0; nb < nb1; nb++) {
    int wide = nb < n16, ws = wide ? 16 : 8, nv = wide ? 2 : 1;
    const f16 *wp = W->w + (size_t)nb * W->Kp * 16;
    float32x4_t mx[4];
    for (int q = 0; q < 4; q++) mx[q] = vdupq_n_f32(-INFINITY);
    for (int m0 = 0; m0 < M; m0 += 8) {
      int mr = M - m0 < 8 ? M - m0 : 8;
      tile_rows(A, lda, M, m0, ar);
      if (nblk > 1) memset(fb, 0, sizeof(float) * 8 * ws);
      for (int kb = 0; kb < W->Kp; kb += bs) {
        int kn = W->Kp - kb < bs ? W->Kp - kb : bs;
        if (wide)
          kblock0_8x16(ar[0] + kb, ar[1] + kb, ar[2] + kb, ar[3] + kb, ar[4] + kb, ar[5] + kb, ar[6] + kb, ar[7] + kb,
                       wp + (size_t)kb * 16, kn / 8, cb);
        else
          kblock0_8x8(ar[0] + kb, ar[1] + kb, ar[2] + kb, ar[3] + kb, ar[4] + kb, ar[5] + kb, ar[6] + kb, ar[7] + kb,
                      wp + (size_t)kb * 8, kn / 8, cb);
        if (nblk > 1) flush_add(fb, cb, nv * 8);
      }
      for (int r = 0; r < mr; r++)
        for (int h = 0; h < nv; h++) {
          float32x4_t lo, hi;
          if (nblk > 1) { lo = vld1q_f32(fb + r * ws + h * 8); hi = vld1q_f32(fb + r * ws + h * 8 + 4); }
          else {
            float16x8_t c = vld1q_f16(cb + r * ws + h * 8);
            lo = vcvt_f32_f16(vget_low_f16(c)); hi = vcvt_high_f32_f16(c);
          }
          mx[2 * h] = vmaxq_f32(mx[2 * h], lo);
          mx[2 * h + 1] = vmaxq_f32(mx[2 * h + 1], hi);
        }
    }
    for (int q = 0; q < 2 * nv; q++) vst1q_f32(colmax + (size_t)nb * 16 + 4 * q, mx[q]);
  }
}
