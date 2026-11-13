// fp32 Winograd F(4x4, 3x3) for dense 3x3 / stride 1 / pad 1 convolutions (baseline ARMv8-A): the image is cut into
// strips of S output tiles; per strip the 6x6 input patches are transformed (B^T d B, 4 channels per vector), then for
// each block of Nb output channels the 36 transform-domain GEMMs run as 8-tile x 8-channel tiles of the engine's fp32
// K loop (gemm_f32_kblock.h) and the block is transformed back (A^T m A) with bias + activation into the (zero-haloed)
// output. Strip and channel block keep the working set (V strip + M block + one weight block per point) in L2.
// Weights are transformed at compile time in float64 (U = G g G^T, 36 fpack panels).
#include <arm_neon.h>
#include <string.h>

#include "gemm_f32.h"
#include "gemm_f32_kblock.h"
#include "wino_f32.h"

/* B^T rows (points 0, +-1, +-2): applied to a column of 6 vectors */
#define BT_COL(d0, d1, d2, d3, d4, d5, o0, o1, o2, o3, o4, o5)                                                  \
  do {                                                                                                          \
    float32x4_t t4 = vsubq_f32(d4, vmulq_n_f32(d2, 4.f)), t3 = vsubq_f32(d3, vmulq_n_f32(d1, 4.f));           \
    float32x4_t u4 = vsubq_f32(d4, d2), u3 = vmulq_n_f32(vsubq_f32(d3, d1), 2.f);                              \
    o0 = vaddq_f32(vsubq_f32(vmulq_n_f32(d0, 4.f), vmulq_n_f32(d2, 5.f)), d4);                                 \
    o1 = vaddq_f32(t4, t3);                                                                                     \
    o2 = vsubq_f32(t4, t3);                                                                                     \
    o3 = vaddq_f32(u4, u3);                                                                                     \
    o4 = vsubq_f32(u4, u3);                                                                                     \
    o5 = vaddq_f32(vsubq_f32(vmulq_n_f32(d1, 4.f), vmulq_n_f32(d3, 5.f)), d5);                                 \
  } while (0)

/* V[xi][s][c..c+4) for one tile from its 6x6 patch p (row stride rs floats, pixel stride ld) */
static inline void in_tile(const float *p, size_t rs, int ld, int c, float *V, size_t vxs) {
  float32x4_t d[6][6], t[6][6];
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++) d[i][j] = vld1q_f32(p + i * rs + (size_t)j * ld + c);
  for (int j = 0; j < 6; j++) /* columns: t = B^T d */
    BT_COL(d[0][j], d[1][j], d[2][j], d[3][j], d[4][j], d[5][j], t[0][j], t[1][j], t[2][j], t[3][j], t[4][j], t[5][j]);
  for (int i = 0; i < 6; i++) { /* rows: V = t B */
    float32x4_t o[6];
    BT_COL(t[i][0], t[i][1], t[i][2], t[i][3], t[i][4], t[i][5], o[0], o[1], o[2], o[3], o[4], o[5]);
    for (int j = 0; j < 6; j++) vst1q_f32(V + (size_t)(i * 6 + j) * vxs + c, o[j]);
  }
}

/* A^T rows: 6 vectors -> 4 */
#define AT_COL(m0, m1, m2, m3, m4, m5, y0, y1, y2, y3)                                                          \
  do {                                                                                                          \
    float32x4_t a12 = vaddq_f32(m1, m2), s12 = vsubq_f32(m1, m2), a34 = vaddq_f32(m3, m4), s34 = vsubq_f32(m3, m4); \
    y0 = vaddq_f32(vaddq_f32(m0, a12), a34);                                                                    \
    y1 = vaddq_f32(s12, vmulq_n_f32(s34, 2.f));                                                                 \
    y2 = vaddq_f32(a12, vmulq_n_f32(a34, 4.f));                                                                 \
    y3 = vaddq_f32(vaddq_f32(s12, vmulq_n_f32(s34, 8.f)), m5);                                                  \
  } while (0)

/* plane strides (floats) of the V strip [36][S8][C] and the M block [36][S8][Nb]: one cache line more than the plane,
 * so the 36 planes of a tile (read or written together by the transforms) land in distinct sets of the 2-way L1
 * whatever S x C is; a plane of exactly 2 KB (S 8 x C 64) or 8 KB (S 32 x C 64) folds the 36 accesses onto a handful
 * of sets and misses on nearly every one */
static inline size_t plane_v(const wino43_cfg *g) { return (size_t)((g->S + 7) & ~7) * g->C + 16; }
static inline size_t plane_m(const wino43_cfg *g) { return (size_t)((g->S + 7) & ~7) * g->Nb + 16; }

/* 8 tiles x 8 channels x K = C transform-domain products on the engine's 8x8 K loop, raw store of the 16 accumulators
 * plus +0.0 (the zero bias the generic GEMM epilogue added: it turns a -0.0 from an underflowing negative product
 * into +0.0, so the bits stay those of gemm_f32_range in every case) and a weight prefetch pfd bytes ahead.
 * Why not gemm_f32_range: its prefetch runs 2 KB ahead within the panel block, but a point's block is only Nb x C x 4
 * bytes and the next point's block lies Np x C x 4 bytes further, so the first lines of every point were demand
 * misses from DRAM; pfd = Np x C x 4 prefetches the same panel of the next point instead */
static inline void pt_tile(const float *A, long lda_b, const float *w, long n4, long pfd, float *M, int ldm) {
  float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
  KB8X8F_PF(A, lda_b, w, n4, 1, 0, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, pfd);
  const float32x4_t z = vdupq_n_f32(0.f);
  a0 = vaddq_f32(a0, z); a1 = vaddq_f32(a1, z); a2 = vaddq_f32(a2, z); a3 = vaddq_f32(a3, z);
  a4 = vaddq_f32(a4, z); a5 = vaddq_f32(a5, z); a6 = vaddq_f32(a6, z); a7 = vaddq_f32(a7, z);
  a8 = vaddq_f32(a8, z); a9 = vaddq_f32(a9, z); a10 = vaddq_f32(a10, z); a11 = vaddq_f32(a11, z);
  a12 = vaddq_f32(a12, z); a13 = vaddq_f32(a13, z); a14 = vaddq_f32(a14, z); a15 = vaddq_f32(a15, z);
  vst1q_f32(M, a0); vst1q_f32(M + 4, a1); M += ldm;
  vst1q_f32(M, a2); vst1q_f32(M + 4, a3); M += ldm;
  vst1q_f32(M, a4); vst1q_f32(M + 4, a5); M += ldm;
  vst1q_f32(M, a6); vst1q_f32(M + 4, a7); M += ldm;
  vst1q_f32(M, a8); vst1q_f32(M + 4, a9); M += ldm;
  vst1q_f32(M, a10); vst1q_f32(M + 4, a11); M += ldm;
  vst1q_f32(M, a12); vst1q_f32(M + 4, a13); M += ldm;
  vst1q_f32(M, a14); vst1q_f32(M + 4, a15);
}

size_t wino43_scratch_floats(const wino43_cfg *g) {
  return 36 * plane_v(g) + 36 * plane_m(g) + (size_t)36 * g->C /* edge patch */ + 64;
}

/* output transform of one tile for channels [n, n + 4) of the M block (row s): rows in pairs so 8 SiLU chains run in
 * lock-step (the chain is ~18 dependent ops; 4 at a time left the A72's pipes half idle) */
static inline void out_tile(const wino43_cfg *g, const float *M, size_t mxs, float32x4_t bv, int ty, int tx, int ny, int nx,
                            const float *res, float *out) {
  float32x4_t t[4][6];
  for (int j = 0; j < 6; j++) {
    float32x4_t m0 = vld1q_f32(M + (size_t)(0 * 6 + j) * mxs), m1 = vld1q_f32(M + (size_t)(1 * 6 + j) * mxs);
    float32x4_t m2 = vld1q_f32(M + (size_t)(2 * 6 + j) * mxs), m3 = vld1q_f32(M + (size_t)(3 * 6 + j) * mxs);
    float32x4_t m4 = vld1q_f32(M + (size_t)(4 * 6 + j) * mxs), m5 = vld1q_f32(M + (size_t)(5 * 6 + j) * mxs);
    AT_COL(m0, m1, m2, m3, m4, m5, t[0][j], t[1][j], t[2][j], t[3][j]);
  }
  const int Wop = g->Wo + 2 * g->opad;
  for (int i = 0; i < ny; i += 2) {
    float32x4_t y[8];
    AT_COL(t[i][0], t[i][1], t[i][2], t[i][3], t[i][4], t[i][5], y[0], y[1], y[2], y[3]);
    AT_COL(t[i + 1][0], t[i + 1][1], t[i + 1][2], t[i + 1][3], t[i + 1][4], t[i + 1][5], y[4], y[5], y[6], y[7]);
    for (int j = 0; j < 8; j++) y[j] = vaddq_f32(y[j], bv);
    if (g->act == ACT32_SILU) silu8_f32x4(y);
    else if (g->act == ACT32_RELU)
      for (int j = 0; j < 8; j++) y[j] = vmaxq_f32(y[j], vdupq_n_f32(0));
    int nr = ny - i < 2 ? ny - i : 2;
    for (int r = 0; r < nr; r++) {
      float *o = out + ((size_t)(4 * ty + i + r + g->opad) * Wop + 4 * tx + g->opad) * g->ldo;
      float32x4_t *yr = y + 4 * r;
      if (res) {
        const float *rp = res + ((size_t)(4 * ty + i + r + g->rh) * (g->Wo + 2 * g->rh) + 4 * tx + g->rh) * g->rld;
        for (int j = 0; j < nx; j++) yr[j] = vaddq_f32(yr[j], vld1q_f32(rp + (size_t)j * g->rld));
      }
      for (int j = 0; j < nx; j++) vst1q_f32(o + (size_t)j * g->ldo, yr[j]);
    }
  }
}

void wino43_strip(const wino43_cfg *g, const float *xp, const float *U, const float *bz, const float *bias, float *out,
                  int strip, float *scr) {
  const int C = g->C, h = g->halo, W = g->W, H = g->H, Wp = W + 2 * h;
  const int TW = (g->Wo + 3) / 4, TH = (g->Ho + 3) / 4, T = TW * TH;
  const int s0 = strip * g->S, ns = T - s0 < g->S ? T - s0 : g->S;
  const size_t vxs = plane_v(g), mxs = plane_m(g), rs = (size_t)Wp * C;
  float *V = scr, *Mb = V + 36 * vxs, *patch = Mb + 36 * mxs;
  for (int s = 0; s < ns; s++) {
    int ty = (s0 + s) / TW, tx = (s0 + s) % TW;
    int y0 = 4 * ty - 1 + h, x0 = 4 * tx - 1 + h; /* patch origin in the haloed buffer */
    if (y0 >= 0 && x0 >= 0 && y0 + 6 <= H + 2 * h && x0 + 6 <= Wp) {
      const float *p = xp + ((size_t)y0 * Wp + x0) * C;
      for (int c = 0; c < C; c += 4) in_tile(p, rs, C, c, V + (size_t)s * C, vxs);
    } else { /* edge tile: zero-filled copy of the part inside the buffer */
      memset(patch, 0, sizeof(float) * 36 * C);
      for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
          int y = y0 + i, x = x0 + j;
          if (y >= 0 && y < H + 2 * h && x >= 0 && x < Wp) memcpy(patch + (size_t)(i * 6 + j) * C, xp + ((size_t)y * Wp + x) * C, sizeof(float) * C);
        }
      for (int c = 0; c < C; c += 4) in_tile(patch, (size_t)6 * C, C, c, V + (size_t)s * C, vxs);
    }
  }
  const int npan = g->Np / 8, nbp = g->Nb / 8;
  for (int p0 = 0; p0 < npan; p0 += nbp) {
    int p1 = p0 + nbp < npan ? p0 + nbp : npan, nb = (p1 - p0) * 8;
    { /* one 8-row tile of transform rows per point and panel for each 8 tiles of the strip (the runtime uses S 8) */
      const long pfd = (long)g->Np * C * 4, lda_b = (long)C * 4, n4 = C / 4;
      (void)bz; /* the zero bias is the +0.0 in pt_tile */
      for (int s8 = 0; s8 < ns; s8 += 8)
        for (int xi = 0; xi < 36; xi++) {
          const float *A = V + (size_t)xi * vxs + (size_t)s8 * C, *w = U + ((size_t)xi * npan + p0) * C * 8;
          float *M = Mb + (size_t)xi * mxs + (size_t)s8 * g->Nb;
          for (int p = p0; p < p1; p++, w += (size_t)C * 8, M += 8) pt_tile(A, lda_b, w, n4, pfd, M, g->Nb);
        }
    }
    for (int s = 0; s < ns; s++) {
      int ty = (s0 + s) / TW, tx = (s0 + s) % TW;
      int ny = g->Ho - 4 * ty < 4 ? g->Ho - 4 * ty : 4, nx = g->Wo - 4 * tx < 4 ? g->Wo - 4 * tx : 4;
      for (int n = 0; n < nb; n += 4)
        out_tile(g, Mb + (size_t)s * g->Nb + n, mxs, vld1q_f32(bias + p0 * 8 + n), ty, tx, ny, nx,
                 g->res ? g->res + (size_t)p0 * 8 + n : NULL, out + (size_t)p0 * 8 + n);
    }
  }
}
