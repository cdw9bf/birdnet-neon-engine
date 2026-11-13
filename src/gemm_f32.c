// fp32 GEMM for 1x1 convolutions / im2col convs on plain ARMv8-A NEON (Cortex-A72 / Pi 4 path):
//   C[m][n] = act( sum_k A[m][k] * W[n][k] + b[n] ) (+ residual), A and C fp32 NHWC rows.
// Micro-kernel 8 rows x 8 channels: by-element FMLA .4s in one inline-asm K loop (gemm_f32_kblock.h; gcc spills and
// rewrites the intrinsics version) whose 16 accumulators stay in registers for the fused epilogue (bias, SiLU / ReLU,
// residual, padded / pooled store). Weights packed in 8-channel panels [Kp][8]. fp32 accumulation over the whole K.
// Loop order: M blocks whose A rows fit half of L1 (16 KB); inside, weight panels outer, 8-row tiles inner, so each
// panel is read from L2/DRAM once per M block and the A rows stay in L1.
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_f32.h"
#include "gemm_f32_kblock.h"

#define AI static inline __attribute__((always_inline))

/* weight-stream prefetch: 1 = for panels not resident in L1 (measured best on the A72), 2048 B ahead */
static const int g_pf = 1;
static const long g_pfd = 2048;

void fp_pack(fpack *h, const float *w, const float *bias, int N, int K, int Kp) {
  memset(h, 0, sizeof *h);
  h->N = N; h->K = K; h->Kp = Kp; h->Np = (N + 7) & ~7;
  size_t n = (size_t)h->Np * Kp, bytes = ((n * 4) + 127) & ~(size_t)63; /* +64 B slack: the K loop reads ahead */
  h->w = aligned_alloc(64, bytes);
  memset(h->w, 0, bytes);
  for (int o = 0; o < N; o++) {
    int nb = o / 8, j = o % 8;
    for (int k = 0; k < K; k++) h->w[((size_t)nb * Kp + k) * 8 + j] = w[(size_t)o * K + k];
  }
  h->b = calloc(h->Np + 8, sizeof(float));
  for (int o = 0; o < N; o++) h->b[o] = bias ? bias[o] : 0.f;
}

void fp_free(fpack *h) {
  free(h->w); free(h->b);
  memset(h, 0, sizeof *h);
}

AI size_t out_row(const epi32 *e, int m) {
  if (!e->opad) return (size_t)m;
  return (size_t)((m / e->ow) + e->opad) * (e->ow + 2 * e->opad) + (m % e->ow) + e->opad;
}

/* one output row of the tile (bias and activation already applied): residual, store */
AI void store_row(float32x4_t x0, float32x4_t x1, int m, size_t orow, int n0, const epi32 *e) {
  if (e->res) {
    size_t rm = e->rpad ? (size_t)((m / e->rw) + e->rpad) * (e->rw + 2 * e->rpad) + m % e->rw + e->rpad : (size_t)m;
    const float *rp = e->res + rm * e->ldr + n0;
    x0 = vaddq_f32(x0, vld1q_f32(rp));
    x1 = vaddq_f32(x1, vld1q_f32(rp + 4));
  }
  float *op = e->out + orow * e->ldo + n0;
  vst1q_f32(op, x0);
  vst1q_f32(op + 4, x1);
}

/* ReLU + [max | avg] of the adjacent pixel pair (rows 2q, 2q+1) into pooled pixel m0/2 + q */
AI void store_pool(float32x4_t u0, float32x4_t u1, float32x4_t v0, float32x4_t v1, float32x4_t b0, float32x4_t b1,
                   int q, int n0, const epi32 *e) {
  const float32x4_t z = vdupq_n_f32(0), hf = vdupq_n_f32(0.5f);
  float32x4_t a0 = vmaxq_f32(vaddq_f32(u0, b0), z), a1 = vmaxq_f32(vaddq_f32(u1, b1), z);
  float32x4_t c0 = vmaxq_f32(vaddq_f32(v0, b0), z), c1 = vmaxq_f32(vaddq_f32(v1, b1), z);
  float *op = e->out + (size_t)q * e->ldo + n0;
  vst1q_f32(op, vmaxq_f32(a0, c0));
  vst1q_f32(op + 4, vmaxq_f32(a1, c1));
  vst1q_f32(op + e->pool_off, vmulq_f32(vaddq_f32(a0, c0), hf));
  vst1q_f32(op + e->pool_off + 4, vmulq_f32(vaddq_f32(a1, c1), hf));
}

#define KB(A, LDA, W, N4, NSEG, SKIP) \
  KB8X8F(A, LDA, W, N4, NSEG, SKIP, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)
#define KB_PF(A, LDA, W, N4, NSEG, SKIP, PFD) \
  KB8X8F_PF(A, LDA, W, N4, NSEG, SKIP, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, PFD)
#define DECL_ACCS float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15

/* bias + activation of a row pair (4 vectors: SiLU chains interleaved) */
#define ACT_PAIR(x0, x1, x2, x3)                                                                               \
  do {                                                                                                         \
    x0 = vaddq_f32(x0, b0); x1 = vaddq_f32(x1, b1); x2 = vaddq_f32(x2, b0); x3 = vaddq_f32(x3, b1);            \
    if (e_->act == ACT32_SILU) {                                                                               \
      float32x4_t v4[4] = {x0, x1, x2, x3};                                                                    \
      silu4_f32x4(v4);                                                                                         \
      x0 = v4[0]; x1 = v4[1]; x2 = v4[2]; x3 = v4[3];                                                          \
    } else if (e_->act == ACT32_RELU) {                                                                        \
      const float32x4_t z = vdupq_n_f32(0);                                                                    \
      x0 = vmaxq_f32(x0, z); x1 = vmaxq_f32(x1, z); x2 = vmaxq_f32(x2, z); x3 = vmaxq_f32(x3, z);              \
    }                                                                                                          \
  } while (0)
/* row r of a tile written to its padded position; orow / ocol track the position of the next row */
#define PADROW(r, ar, br)                                                                                      \
  if (mr_ > (r)) {                                                                                             \
    store_row(ar, br, m0_ + (r), orow, n0_, e_);                                                               \
    orow++;                                                                                                    \
    if (++ocol == e_->ow) { ocol = 0; orow += 2 * e_->opad; }                                                  \
  }

/* epilogue of one tile (accumulators in registers) */
#define EPILOGUE(mr, m0, n0, W, e)                                                                             \
  do {                                                                                                         \
    const int mr_ = (mr), m0_ = (m0), n0_ = (n0);                                                              \
    const epi32 *e_ = (e);                                                                                     \
    float32x4_t b0 = vld1q_f32((W)->b + n0_), b1 = vld1q_f32((W)->b + n0_ + 4);                                \
    if ((e)->act == ACT32_RELU_POOL2) {                                                                        \
      int q = (m0) / 2;                                                                                        \
      store_pool(a0, a1, a2, a3, b0, b1, q, n0, e);                                                            \
      if ((mr) > 2) store_pool(a4, a5, a6, a7, b0, b1, q + 1, n0, e);                                          \
      if ((mr) > 4) store_pool(a8, a9, a10, a11, b0, b1, q + 2, n0, e);                                        \
      if ((mr) > 6) store_pool(a12, a13, a14, a15, b0, b1, q + 3, n0, e);                                      \
    } else {                                                                                                   \
      ACT_PAIR(a0, a1, a2, a3);                                                                                \
      if (mr_ > 2) ACT_PAIR(a4, a5, a6, a7);                                                                   \
      if (mr_ > 4) ACT_PAIR(a8, a9, a10, a11);                                                                 \
      if (mr_ > 6) ACT_PAIR(a12, a13, a14, a15);                                                               \
      if (e_->opad) {                                                                                          \
        size_t orow = out_row(e_, m0_);                                                                        \
        int ocol = m0_ % e_->ow;                                                                               \
        PADROW(0, a0, a1); PADROW(1, a2, a3); PADROW(2, a4, a5); PADROW(3, a6, a7);                            \
        PADROW(4, a8, a9); PADROW(5, a10, a11); PADROW(6, a12, a13); PADROW(7, a14, a15);                      \
      } else {                                                                                                 \
        store_row(a0, a1, m0_, m0_, n0_, e_);                                                                  \
        if (mr_ > 1) store_row(a2, a3, m0_ + 1, m0_ + 1, n0_, e_);                                             \
        if (mr_ > 2) store_row(a4, a5, m0_ + 2, m0_ + 2, n0_, e_);                                             \
        if (mr_ > 3) store_row(a6, a7, m0_ + 3, m0_ + 3, n0_, e_);                                             \
        if (mr_ > 4) store_row(a8, a9, m0_ + 4, m0_ + 4, n0_, e_);                                             \
        if (mr_ > 5) store_row(a10, a11, m0_ + 5, m0_ + 5, n0_, e_);                                           \
        if (mr_ > 6) store_row(a12, a13, m0_ + 6, m0_ + 6, n0_, e_);                                           \
        if (mr_ > 7) store_row(a14, a15, m0_ + 7, m0_ + 7, n0_, e_);                                           \
      }                                                                                                        \
    }                                                                                                          \
  } while (0)

int gemm_f32_mblock(int M, const fpack *W) {
  int mb = (int)(16384 / ((size_t)W->Kp * 4)) & ~7;
  if (mb < 8) mb = 8;
  return mb;
}

/* rows [m0, m1) (m0 a multiple of the M block, or 0) x weight panels [p0, p1) of gemm_f32 */
void gemm_f32_range(const float *A, int lda, int M, const fpack *W, const epi32 *e, int m0, int m1, int p0, int p1) {
  size_t wbytes = (size_t)W->Kp * 8 * 4;
  int npan = W->Np / 8;
  int pf = g_pf == 2 || (g_pf == 1 && wbytes * npan > 32 * 1024);
  long pfd = g_pfd, ldb = (long)lda * 4, n4 = W->Kp / 4;
  int mb = gemm_f32_mblock(M, W);
  for (int mb0 = m0; mb0 < m1; mb0 += mb) {
    int mb1 = m1 - mb0 < mb ? m1 : mb0 + mb;
    for (int nb = p0; nb < p1; nb++) {
      const float *wp = W->w + (size_t)nb * W->Kp * 8;
      for (int m0 = mb0; m0 < mb1; m0 += 8) { /* (EPILOGUE declares m0_ itself: never name a variable that) */
        int mr = M - m0 < 8 ? M - m0 : 8;
        const float *p = A + (size_t)m0 * lda, *w = wp;
        DECL_ACCS;
        if (pf) KB_PF(p, ldb, w, n4, 1, 0, pfd);
        else KB(p, ldb, w, n4, 1, 0);
        EPILOGUE(mr, m0, nb * 8, W, e);
      }
    }
  }
}

/* gemm_f32_range on A scaled per input channel (A[m][k] * gate[k], K == Kp): each M block's rows are scaled into scr
 * (gemm_f32_gate_scratch floats) just before its panels run, so a squeeze-excitation gate costs no pass of its own over
 * the activation. The products are the ones an in-place gating multiply would store: outputs are bit-identical. */
size_t gemm_f32_gate_scratch(int M, const fpack *W) { return (size_t)((gemm_f32_mblock(M, W) + 7) & ~7) * W->Kp + 16; }
void gemm_f32_range_gate(const float *A, int lda, int M, const fpack *W, const epi32 *e, int m0, int m1, int p0, int p1,
                         const float *gate, float *scr) {
  size_t wbytes = (size_t)W->Kp * 8 * 4;
  int npan = W->Np / 8, Kp = W->Kp;
  int pf = g_pf == 2 || (g_pf == 1 && wbytes * npan > 32 * 1024);
  long pfd = g_pfd, ldb = (long)Kp * 4, n4 = Kp / 4;
  int mb = gemm_f32_mblock(M, W);
  for (int mb0 = m0; mb0 < m1; mb0 += mb) {
    int mb1 = m1 - mb0 < mb ? m1 : mb0 + mb, rows = (mb1 - mb0 + 7) & ~7; /* whole 8-row tiles (rows past M: ignored) */
    for (int r = 0; r < rows; r++) {
      const float *x = A + (size_t)(mb0 + r) * lda;
      float *y = scr + (size_t)r * Kp;
      for (int k = 0; k < Kp; k += 4) vst1q_f32(y + k, vmulq_f32(vld1q_f32(x + k), vld1q_f32(gate + k)));
    }
    for (int nb = p0; nb < p1; nb++) {
      const float *wp = W->w + (size_t)nb * W->Kp * 8;
      for (int m0 = mb0; m0 < mb1; m0 += 8) { /* (EPILOGUE declares m0_ itself: never name a variable that) */
        int mr = M - m0 < 8 ? M - m0 : 8;
        const float *p = scr + (size_t)(m0 - mb0) * Kp, *w = wp;
        DECL_ACCS;
        if (pf) KB_PF(p, ldb, w, n4, 1, 0, pfd);
        else KB(p, ldb, w, n4, 1, 0);
        EPILOGUE(mr, m0, nb * 8, W, e);
      }
    }
  }
}

void gemm_f32(const float *A, int lda, int M, const fpack *W, const epi32 *e) {
  gemm_f32_range(A, lda, M, W, e, 0, M, 0, W->Np / 8);
}

void gemm_f32_seg(const float *A, int lda, int M, int nseg, int segk, long segstride, const fpack *W,
                  const epi32 *e) {
  int npan = W->Np / 8;
  long ldb = (long)lda * 4, n4 = segk / 4, skip = segstride * 4 - (n4 + 1) * 16;
  for (int m0 = 0; m0 < M; m0 += 8) {
    int mr = M - m0 < 8 ? M - m0 : 8;
    for (int nb = 0; nb < npan; nb++) {
      const float *p = A + (size_t)m0 * lda, *w = W->w + (size_t)nb * W->Kp * 8;
      DECL_ACCS;
      KB(p, ldb, w, n4, nseg, skip);
      EPILOGUE(mr, m0, nb * 8, W, e);
    }
  }
}

/* ---- depthwise 3x3 + SiLU ---- */
/* pass 1 per output row: 4-channel blocks outer with the 9 taps' weights, the bias and the 3x3 input window in
 * registers, pixels inner: 3 (stride 1) or 6 (stride 2) loads + 7 FMLA + 2 FMUL + 2 FADD + 1 store per 4 outputs, the window
 * sliding by register renaming (period-3 unrolling, no moves). The pass footprint (3 input rows, 64 B
 * lines) stays in L1 across the C/4 channel blocks. pass 2: SiLU (chains interleaved) + channel sums over the row. */
/* three partial sums (one per tap column) so the dependent chain is 3 FMLA + 2 FADD deep instead of 9 FMLA: the
 * A72's FMLA latency (~7 cycles) and small NEON issue queues make a 9-deep chain cost ~30 cycles per step */
#define DW_ACC(A0, A1, A2, B0, B1, B2, C0, C1, C2)                                                             \
  acc = vfmaq_f32(bias, w0, A0); acb = vmulq_f32(w1, B0); acd = vmulq_f32(w2, C0);                             \
  acc = vfmaq_f32(acc, w3, A1); acb = vfmaq_f32(acb, w4, B1); acd = vfmaq_f32(acd, w5, C1);                    \
  acc = vfmaq_f32(acc, w6, A2); acb = vfmaq_f32(acb, w7, B2); acd = vfmaq_f32(acd, w8, C2);                    \
  acc = vaddq_f32(acc, vaddq_f32(acb, acd))
#define DW_LD3(X0, X1, X2, Q) X0 = vld1q_f32(Q); X1 = vld1q_f32((Q) + rs1); X2 = vld1q_f32((Q) + rs2)
/* the column loads are strided by C floats (one cache line per column and row), which the A72's hardware
 * prefetcher does not follow; the padded input of the stride-2 blocks (~2 MB) comes from DRAM, so prefetch the
 * three rows of the column DW_PFD pixels ahead */
#define DW_PFD 6
#define DW_PF3(Q) __builtin_prefetch(Q); __builtin_prefetch((Q) + rs1); __builtin_prefetch((Q) + rs2)
/* stride 1: window (A, B, C) = columns ox, ox+1, ox+2; loads column ox+2 */
#define DW_STEP1(A0, A1, A2, B0, B1, B2, C0, C1, C2)                                                           \
  do {                                                                                                         \
    DW_PF3(q + (2 + DW_PFD) * C);                                                                              \
    DW_LD3(C0, C1, C2, q + 2 * C);                                                                             \
    DW_ACC(A0, A1, A2, B0, B1, B2, C0, C1, C2);                                                                \
    vst1q_f32(oc, acc);                                                                                        \
    q += C; oc += C;                                                                                           \
  } while (0)
/* stride 2: window = columns 2ox, 2ox+1, 2ox+2; loads the last two */
#define DW_STEP2(A0, A1, A2, B0, B1, B2, C0, C1, C2)                                                           \
  do {                                                                                                         \
    DW_PF3(q + (1 + 2 * DW_PFD) * C);                                                                          \
    DW_PF3(q + (2 + 2 * DW_PFD) * C);                                                                          \
    DW_LD3(B0, B1, B2, q + C);                                                                                 \
    DW_LD3(C0, C1, C2, q + 2 * C);                                                                             \
    DW_ACC(A0, A1, A2, B0, B1, B2, C0, C1, C2);                                                                \
    vst1q_f32(oc, acc);                                                                                        \
    q += 2 * C; oc += C;                                                                                       \
  } while (0)
/* the three input rows at r0, r0 + rs1, r0 + rs2 (signed float offsets: the staged ring below keeps them out of order) */
AI void dw_row_s1(const float *r0, long rs1, long rs2, int C, const float *w, const float *b, float *o, int Wo, int c0,
                  int c1) {
  for (int c = c0; c < c1; c += 4) {
    const float *wc = w + c;
    const float32x4_t w0 = vld1q_f32(wc), w1 = vld1q_f32(wc + C), w2 = vld1q_f32(wc + 2 * C);
    const float32x4_t w3 = vld1q_f32(wc + 3 * C), w4 = vld1q_f32(wc + 4 * C), w5 = vld1q_f32(wc + 5 * C);
    const float32x4_t w6 = vld1q_f32(wc + 6 * C), w7 = vld1q_f32(wc + 7 * C), w8 = vld1q_f32(wc + 8 * C);
    const float32x4_t bias = vld1q_f32(b + c);
    float32x4_t x0, x1, x2, y0, y1, y2, z0, z1, z2, acc, acb, acd;
    const float *q = r0 + c;
    float *oc = o + c;
    DW_LD3(x0, x1, x2, q);
    DW_LD3(y0, y1, y2, q + C);
    int ox = 0;
    for (; ox + 3 <= Wo; ox += 3) {
      DW_STEP1(x0, x1, x2, y0, y1, y2, z0, z1, z2);
      DW_STEP1(y0, y1, y2, z0, z1, z2, x0, x1, x2);
      DW_STEP1(z0, z1, z2, x0, x1, x2, y0, y1, y2);
    }
    if (ox < Wo) {
      DW_STEP1(x0, x1, x2, y0, y1, y2, z0, z1, z2);
      if (ox + 1 < Wo) DW_STEP1(y0, y1, y2, z0, z1, z2, x0, x1, x2);
    }
  }
}
AI void dw_row_s2(const float *r0, int Wp, int C, const float *w, const float *b, float *o, int Wo, int c0, int c1) {
  const long rs1 = (long)Wp * C, rs2 = 2 * rs1;
  for (int c = c0; c < c1; c += 4) {
    const float *wc = w + c;
    const float32x4_t w0 = vld1q_f32(wc), w1 = vld1q_f32(wc + C), w2 = vld1q_f32(wc + 2 * C);
    const float32x4_t w3 = vld1q_f32(wc + 3 * C), w4 = vld1q_f32(wc + 4 * C), w5 = vld1q_f32(wc + 5 * C);
    const float32x4_t w6 = vld1q_f32(wc + 6 * C), w7 = vld1q_f32(wc + 7 * C), w8 = vld1q_f32(wc + 8 * C);
    const float32x4_t bias = vld1q_f32(b + c);
    float32x4_t x0, x1, x2, y0, y1, y2, z0, z1, z2, acc, acb, acd;
    const float *q = r0 + c;
    float *oc = o + c;
    DW_LD3(x0, x1, x2, q);
    int ox = 0;
    for (; ox + 3 <= Wo; ox += 3) { /* the newest column rotates z -> y -> x: period 3 */
      DW_STEP2(x0, x1, x2, y0, y1, y2, z0, z1, z2);
      DW_STEP2(z0, z1, z2, x0, x1, x2, y0, y1, y2);
      DW_STEP2(y0, y1, y2, z0, z1, z2, x0, x1, x2);
    }
    if (ox < Wo) {
      DW_STEP2(x0, x1, x2, y0, y1, y2, z0, z1, z2);
      if (ox + 1 < Wo) DW_STEP2(z0, z1, z2, x0, x1, x2, y0, y1, y2);
    }
  }
}

void silu_rows_f32(float *x, size_t n, int C, float *chsum) {
  size_t i = 0;
  int c = 0;
  for (; i + 32 <= n; i += 32) {
    float32x4_t v[8];
    for (int j = 0; j < 8; j++) v[j] = vld1q_f32(x + i + 4 * j);
    silu8_f32x4(v);
    for (int j = 0; j < 8; j++) vst1q_f32(x + i + 4 * j, v[j]);
    if (chsum)
      for (int j = 0; j < 8; j += 2) {
        vst1q_f32(chsum + c, vaddq_f32(vld1q_f32(chsum + c), v[j]));
        vst1q_f32(chsum + c + 4, vaddq_f32(vld1q_f32(chsum + c + 4), v[j + 1]));
        if ((c += 8) == C) c = 0;
      }
  }
  for (; i < n; i += 8) {
    float32x4_t v[4] = {vld1q_f32(x + i), vld1q_f32(x + i + 4), vdupq_n_f32(0), vdupq_n_f32(0)};
    silu4_f32x4(v);
    vst1q_f32(x + i, v[0]); vst1q_f32(x + i + 4, v[1]);
    if (chsum) {
      vst1q_f32(chsum + c, vaddq_f32(vld1q_f32(chsum + c), v[0]));
      vst1q_f32(chsum + c + 4, vaddq_f32(vld1q_f32(chsum + c + 4), v[1]));
      if ((c += 8) == C) c = 0;
    }
  }
}

/* ---- stride 1 with C >= 16: 16-channel groups outer, output rows inner, the band staged in L1 ----
 * The real buffers' pixel stride (C * 4 B) is a multiple of 64 B with a short period in the A72's 2-way L1 (8 sets
 * for C = 512 / 1536, 64 for C = 960), so a 16-channel group's three input rows, its taps and its output row all
 * fall into the same few sets and evict each other: pass 1 took 60-95 cycles per 4 outputs for ~12 instructions
 * (measured on the Pi 4; every line is also a different 4 KB page). Here each group's input lines are gathered once
 * per output row into a contiguous 3-row ring (one 64 B line per pixel, the line we must fetch anyway), the taps
 * into [9][16], and the sliding-window kernel and the SiLU read only contiguous L1 data; the group's line one
 * input row ahead is prefetched into L2 (not L1: it would be evicted before use for the same reason). Same
 * instructions per element and the channel sums still add the pixels in order: bit-identical to the row path. */
#define DW_ST_G 16
#define DW_ST_WMAX 64 /* ring 3 * Wp * 16 floats + staged output row on the stack, within L1 */
/* SiLU + channel sums of one staged output row into the real output: two pixels = 8 chains, an odd last pixel 4 */
AI void dw_silu_row16(const float *sout, float *o, int C, int Wo, float *cs) {
  int px = 0;
  for (; px + 2 <= Wo; px += 2) {
    float32x4_t v[8];
    for (int j = 0; j < 8; j++) v[j] = vld1q_f32(sout + px * 16 + 4 * j);
    silu8_f32x4(v);
    float *o0 = o + (size_t)px * C, *o1 = o0 + C;
    for (int j = 0; j < 4; j++) {
      vst1q_f32(o0 + 4 * j, v[j]);
      vst1q_f32(o1 + 4 * j, v[4 + j]);
      vst1q_f32(cs + 4 * j, vaddq_f32(vaddq_f32(vld1q_f32(cs + 4 * j), v[j]), v[4 + j]));
    }
  }
  if (px < Wo) {
    float32x4_t v[4];
    for (int k = 0; k < 4; k++) v[k] = vld1q_f32(sout + px * 16 + 4 * k);
    silu4_f32x4(v);
    float *op = o + (size_t)px * C;
    for (int k = 0; k < 4; k++) {
      vst1q_f32(op + 4 * k, v[k]);
      vst1q_f32(cs + 4 * k, vaddq_f32(vld1q_f32(cs + 4 * k), v[k]));
    }
  }
}
static void dw3_staged_cr(const float *xp, int W, int C, const float *w, const float *b, float *out, int Ho, int Wo,
                          float *chsum, int c0, int c1) {
  const int Wp = W + 2, G = DW_ST_G;
  const long rowf = (long)Wp * C;
  float ring[3 * DW_ST_WMAX * DW_ST_G] __attribute__((aligned(64)));
  float sout[DW_ST_WMAX * DW_ST_G] __attribute__((aligned(64)));
  float wst[9 * DW_ST_G] __attribute__((aligned(64)));
  for (int c = c0; c < c1; c++) chsum[c] = 0.f;
  for (int g = c0; g < c1; g += G) {
    for (int t = 0; t < 9; t++)
      for (int j = 0; j < G; j += 4) vst1q_f32(wst + t * G + j, vld1q_f32(w + (size_t)t * C + g + j));
    for (int r = 0; r < 2; r++) { /* the band's first two rows; row oy + 2 joins per output row */
      const float *src = xp + r * rowf + g;
      float *dst = ring + r * Wp * G;
      for (int px = 0; px < Wp; px++)
        for (int j = 0; j < G; j += 4) vst1q_f32(dst + px * G + j, vld1q_f32(src + (size_t)px * C + j));
    }
    for (int oy = 0; oy < Ho; oy++) {
      const float *src = xp + (oy + 2) * rowf + g;
      float *dst = ring + ((oy + 2) % 3) * Wp * G;
      for (int px = 0; px < Wp; px++) {
        __builtin_prefetch(src + (size_t)px * C + rowf, 0, 2);
        for (int j = 0; j < G; j += 4) vst1q_f32(dst + px * G + j, vld1q_f32(src + (size_t)px * C + j));
      }
      int top = oy % 3; /* ring slot of input row oy */
      long rs1 = ((top + 1) % 3 - top) * (long)Wp * G, rs2 = ((top + 2) % 3 - top) * (long)Wp * G;
      dw_row_s1(ring + top * Wp * G, rs1, rs2, G, wst, b + g, sout, Wo, 0, G);
      dw_silu_row16(sout, out + (size_t)oy * Wo * C + g, C, Wo, chsum + g);
    }
  }
}

void dw3x3_f32(const float *xp, int H, int W, int C, int s, const float *w, const float *b, float *out, int Ho,
               int Wo, float *chsum) {
  (void)H;
  const int Wp = W + 2;
  if (s == 1 && Wp <= DW_ST_WMAX && C % DW_ST_G == 0) { dw3_staged_cr(xp, W, C, w, b, out, Ho, Wo, chsum, 0, C); return; }
  for (int c = 0; c < C; c++) chsum[c] = 0.f;
  for (int oy = 0; oy < Ho; oy++) {
    const float *r0 = xp + (size_t)(oy * s) * Wp * C;
    float *o = out + (size_t)oy * Wo * C;
    if (s == 1) dw_row_s1(r0, (long)Wp * C, 2L * Wp * C, C, w, b, o, Wo, 0, C);
    else dw_row_s2(r0, Wp, C, w, b, o, Wo, 0, C);
    silu_rows_f32(o, (size_t)Wo * C, C, chsum);
  }
}

/* the same for channels [c0, c1) only (multiples of 8): per element the same instructions as dw3x3_f32 (SiLU is
 * element-wise; each channel's sum still adds the pixels in order), so a channel split over threads is bit-identical */
void dw3x3_f32_cr(const float *xp, int H, int W, int C, int s, const float *w, const float *b, float *out, int Ho,
                  int Wo, float *chsum, int c0, int c1) {
  (void)H;
  const int Wp = W + 2;
  if (s == 1 && Wp <= DW_ST_WMAX && c0 % DW_ST_G == 0 && (c1 - c0) % DW_ST_G == 0) {
    dw3_staged_cr(xp, W, C, w, b, out, Ho, Wo, chsum, c0, c1);
    return;
  }
  for (int c = c0; c < c1; c++) chsum[c] = 0.f;
  for (int oy = 0; oy < Ho; oy++) {
    const float *r0 = xp + (size_t)(oy * s) * Wp * C;
    float *o = out + (size_t)oy * Wo * C;
    if (s == 1) dw_row_s1(r0, (long)Wp * C, 2L * Wp * C, C, w, b, o, Wo, c0, c1);
    else dw_row_s2(r0, Wp, C, w, b, o, Wo, c0, c1);
    for (int px = 0; px < Wo; px++) {
      float *q = o + (size_t)px * C;
      for (int c = c0; c < c1; c += 8) {
        float32x4_t v[4] = {vld1q_f32(q + c), vld1q_f32(q + c + 4), vdupq_n_f32(0), vdupq_n_f32(0)};
        silu4_f32x4(v);
        vst1q_f32(q + c, v[0]); vst1q_f32(q + c + 4, v[1]);
        vst1q_f32(chsum + c, vaddq_f32(vld1q_f32(chsum + c), v[0]));
        vst1q_f32(chsum + c + 4, vaddq_f32(vld1q_f32(chsum + c + 4), v[1]));
      }
    }
  }
}

/* column maxima of A . W^T (no bias): colmax[n] = max_m (A . W^T)[m][n], n < Np; M % 8 == 0. Panels outer. */
void gemm_f32_colmax(const float *A, int lda, int M, const fpack *W, float *colmax) {
  int npan = W->Np / 8;
  long ldb = (long)lda * 4, n4 = W->Kp / 4;
  for (int nb = 0; nb < npan; nb++) {
    const float *wp = W->w + (size_t)nb * W->Kp * 8;
    float32x4_t m0v = vdupq_n_f32(-INFINITY), m1v = m0v;
    for (int m0 = 0; m0 < M; m0 += 8) {
      const float *p = A + (size_t)m0 * lda, *w = wp;
      DECL_ACCS;
      KB(p, ldb, w, n4, 1, 0);
      m0v = vmaxq_f32(m0v, vmaxq_f32(vmaxq_f32(vmaxq_f32(a0, a2), vmaxq_f32(a4, a6)),
                                     vmaxq_f32(vmaxq_f32(a8, a10), vmaxq_f32(a12, a14))));
      m1v = vmaxq_f32(m1v, vmaxq_f32(vmaxq_f32(vmaxq_f32(a1, a3), vmaxq_f32(a5, a7)),
                                     vmaxq_f32(vmaxq_f32(a9, a11), vmaxq_f32(a13, a15))));
    }
    vst1q_f32(colmax + (size_t)nb * 8, m0v);
    vst1q_f32(colmax + (size_t)nb * 8 + 4, m1v);
  }
}
