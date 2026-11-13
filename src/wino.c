// Winograd F(2x2,3x3) int8 for the stride-1 FusedMBConv expand convs (b1.1-3, b2.1-3: 2.36 GMAC -> 1.05 GMAC).
// v2 (integer output transform): V = B^T d B is quantised with ONE asymmetric scale per tile (over all 16 positions x
// C), U = G g G^T with ONE symmetric scale per output channel (over 16 x C). Then every position's GEMM produces raw
// int32 sums in the same units, the output transform A^T M A (+-1 coefficients) runs on int32, and only the 4 outputs
// per tile are dequantised: y = sa*(sw*Y + zp*sw*sum_xi coef*wsum) + b, then SiLU -> fp16.
// Oracle (the retired hand-written engine's fp32 reference, mode 9): 49.4 dB vs 49.1 dB direct on dev256 (mode 7,
// per-position scales: 49.4 dB).
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "act.h"
#include "kern.h"

typedef struct {
  int C, K;
  qweights u[16];     /* int8 U per position; u[xi].scale[k] = shared s_w[k] */
  float *sw;          /* [K] */
  float *zw[4];       /* [4 outputs][K]: s_w[k] * sum_xi coef(o,xi) * wsum(xi,k) */
  float *bias;
} wino_w;

/* coefficient of position (i,j) for output (r,c): a[r][i] * a[c][j], A^T = [[1,1,1,0],[0,1,-1,-1]] */
static const int AT[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};

void *wino_pack(const float *w /* OHWI [K][3][3][C] */, const float *bias, int K, int C) {
  wino_w *ww = calloc(1, sizeof *ww);
  ww->C = C; ww->K = K;
  float *U = malloc(sizeof(float) * 16 * (size_t)K * C);
  for (int k = 0; k < K; k++)
    for (int c = 0; c < C; c++) {
      float g[3][3], tmp[4][3];
      for (int r = 0; r < 3; r++)
        for (int cc = 0; cc < 3; cc++) g[r][cc] = w[(((size_t)k * 3 + r) * 3 + cc) * C + c];
      for (int cc = 0; cc < 3; cc++) {
        tmp[0][cc] = g[0][cc]; tmp[1][cc] = 0.5f * (g[0][cc] + g[1][cc] + g[2][cc]);
        tmp[2][cc] = 0.5f * (g[0][cc] - g[1][cc] + g[2][cc]); tmp[3][cc] = g[2][cc];
      }
      for (int r = 0; r < 4; r++) {
        float u[4] = {tmp[r][0], 0.5f * (tmp[r][0] + tmp[r][1] + tmp[r][2]), 0.5f * (tmp[r][0] - tmp[r][1] + tmp[r][2]),
                      tmp[r][2]};
        for (int cc = 0; cc < 4; cc++) U[((size_t)(r * 4 + cc) * K + k) * C + c] = u[cc];
      }
    }
  ww->sw = malloc(sizeof(float) * K);
  for (int o = 0; o < 4; o++) ww->zw[o] = calloc(K, sizeof(float));
  int Kp = (C + 15) & ~15;
  int32_t *wsum = calloc(16 * (size_t)K, sizeof(int32_t));
  for (int xi = 0; xi < 16; xi++) {
    qweights *q = &ww->u[xi];
    q->N = K; q->K = C; q->Kp = Kp;
    q->w = al64((size_t)K * Kp + 64);
    memset(q->w, 0, (size_t)K * Kp);
    q->scale = malloc(sizeof(float) * K); q->swsum = calloc(K, sizeof(float)); q->bias = calloc(K, sizeof(float));
  }
  for (int k = 0; k < K; k++) {
    float mx = 0;
    for (int xi = 0; xi < 16; xi++)
      for (int c = 0; c < C; c++) mx = fmaxf(mx, fabsf(U[((size_t)xi * K + k) * C + c]));
    float s = mx > 0 ? mx / 127.f : 1.f;
    ww->sw[k] = s;
    for (int xi = 0; xi < 16; xi++) {
      qweights *q = &ww->u[xi];
      q->scale[k] = s;
      int nb = k / 16, nl = k % 16;
      int32_t sum = 0;
      for (int c = 0; c < C; c++) {
        int v = (int)nearbyintf(U[((size_t)xi * K + k) * C + c] / s);
        v = v > 127 ? 127 : (v < -127 ? -127 : v);
        sum += v;
        q->w[(size_t)nb * 16 * Kp + (size_t)(c / 4) * 64 + nl * 4 + (c % 4)] = (int8_t)v;
      }
      wsum[xi * K + k] = sum;
    }
    for (int r = 0; r < 2; r++)
      for (int cc = 0; cc < 2; cc++) {
        long z = 0;
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++) z += (long)AT[r][i] * AT[cc][j] * wsum[(i * 4 + j) * K + k];
        ww->zw[r * 2 + cc][k] = s * (float)z;
      }
  }
  free(wsum);
  free(U);
  ww->bias = malloc(sizeof(float) * K);
  memcpy(ww->bias, bias, sizeof(float) * K);
  return ww;
}

void wino_free(void *p) {
  wino_w *ww = p;
  if (!ww) return;
  for (int xi = 0; xi < 16; xi++) qw_free(&ww->u[xi]);
  free(ww->sw);
  for (int o = 0; o < 4; o++) free(ww->zw[o]);
  free(ww->bias);
  free(ww);
}

#define CHUNK 32 /* the scratch layout is for 32 tiles per GEMM call; the chunk used is WINO_CHUNK (8, kern.h) */

static inline int8x8_t q8x8(float32x4_t a, float32x4_t b, float32x4_t inv, int32x4_t zofs) {
  const float32x4_t magic = vdupq_n_f32(12582912.0f);
  int32x4_t qa = vsubq_s32(vreinterpretq_s32_f32(vfmaq_f32(magic, a, inv)), zofs);
  int32x4_t qb = vsubq_s32(vreinterpretq_s32_f32(vfmaq_f32(magic, b, inv)), zofs);
  qa = vminq_s32(vmaxq_s32(qa, vdupq_n_s32(-128)), vdupq_n_s32(127));
  qb = vminq_s32(vmaxq_s32(qb, vdupq_n_s32(-128)), vdupq_n_s32(127));
  return vmovn_s16(vcombine_s16(vmovn_s32(qa), vmovn_s32(qb)));
}

/* bytes of the padded fp32 copy of the input (the head of the scratch) and of one worker's tile work area */
size_t wino_xpad_bytes(int H, int W, int C) { return sizeof(float) * (size_t)(H + 3) * (2 * ((W + 1) / 2) + 2) * C; }
size_t wino_work_bytes(int C, int K) {
  return 16 * (size_t)CHUNK * C + sizeof(rowq) * CHUNK + sizeof(int32_t) * 16 * (size_t)CHUNK * K + sizeof(float) * 16 * C + 256;
}
/* x [H][W][C] -> xp: zero-padded copy [2 th + 2][2 tw + 2][C] (one pixel on the top / left, the rest on the far sides) */
void wino_f23_pad(const float *x, int H, int W, int C, float *xp) {
  const int th = (H + 1) / 2, tw = (W + 1) / 2;
  const int Wp = 2 * tw + 2, Hp = 2 * th + 2;
  for (int y = 0; y < Hp; y++) {
    float *row = xp + (size_t)y * Wp * C;
    if (y >= 1 && y <= H) {
      memset(row, 0, sizeof(float) * C);
      memcpy(row + C, x + (size_t)(y - 1) * W * C, sizeof(float) * (size_t)W * C);
      memset(row + (size_t)(W + 1) * C, 0, sizeof(float) * (size_t)(Wp - W - 1) * C);
    } else {
      memset(row, 0, sizeof(float) * (size_t)Wp * C);
    }
  }
}

/* tiles [t_begin, t_end) (t_begin a multiple of WINO_CHUNK) from the padded copy xp, with a work area of
 * wino_work_bytes(C, K) bytes of its own: each tile is quantised, multiplied and written on its own, so any split of
 * the tiles over threads gives the same bytes. */
void wino_f23_tiles(const void *wp, const float *xp, int H, int W, f16 *out, void *work,
                    void (*gemm)(const int8_t *, int, const rowq *, int, int, const qweights *, const epi *),
                    int t_begin, int t_end) {
  const wino_w *ww = wp;
  const int C = ww->C, K = ww->K;
  const int tw = (W + 1) / 2;
  const int Wp = 2 * tw + 2;
  char *sp = work;
  int8_t *A8 = (int8_t *)sp; sp += 16 * (size_t)CHUNK * C;
  rowq *rq = (rowq *)sp; sp += sizeof(rowq) * CHUNK;
  int32_t *Mb = (int32_t *)(((uintptr_t)sp + 63) & ~(uintptr_t)63); sp = (char *)Mb + sizeof(int32_t) * 16 * (size_t)CHUNK * K;
  float *Vt = (float *)(((uintptr_t)sp + 15) & ~(uintptr_t)15);
  const int CH = WINO_CHUNK;
  for (int t0 = t_begin; t0 < t_end; t0 += CH) {
    const int Mt = t_end - t0 < CH ? t_end - t0 : CH;
    /* 1. input transform (fp32) with fused per-tile min/max, then one quantisation per tile */
    for (int i = 0; i < Mt; i++) {
      int t = t0 + i, ti = t / tw, tj = t - ti * tw;
      const float *base = xp + ((size_t)(2 * ti) * Wp + 2 * tj) * C;
      float32x4_t mn = vdupq_n_f32(0), mx = vdupq_n_f32(0);
      for (int c = 0; c < C; c += 4) {
        float32x4_t d[4][4];
        for (int r = 0; r < 4; r++)
          for (int cc = 0; cc < 4; cc++) d[r][cc] = vld1q_f32(base + ((size_t)r * Wp + cc) * C + c);
        float32x4_t tm[4][4];
        for (int cc = 0; cc < 4; cc++) {
          tm[0][cc] = vsubq_f32(d[0][cc], d[2][cc]); tm[1][cc] = vaddq_f32(d[1][cc], d[2][cc]);
          tm[2][cc] = vsubq_f32(d[2][cc], d[1][cc]); tm[3][cc] = vsubq_f32(d[1][cc], d[3][cc]);
        }
        for (int r = 0; r < 4; r++) {
          float32x4_t v0 = vsubq_f32(tm[r][0], tm[r][2]), v1 = vaddq_f32(tm[r][1], tm[r][2]);
          float32x4_t v2 = vsubq_f32(tm[r][2], tm[r][1]), v3 = vsubq_f32(tm[r][1], tm[r][3]);
          mn = vminq_f32(mn, vminq_f32(vminq_f32(v0, v1), vminq_f32(v2, v3)));
          mx = vmaxq_f32(mx, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
          vst1q_f32(Vt + (r * 4 + 0) * C + c, v0); vst1q_f32(Vt + (r * 4 + 1) * C + c, v1);
          vst1q_f32(Vt + (r * 4 + 2) * C + c, v2); vst1q_f32(Vt + (r * 4 + 3) * C + c, v3);
        }
      }
      float fmn = vminvq_f32(mn), fmx = vmaxvq_f32(mx);
      float s = (fmx - fmn) > 0 ? (fmx - fmn) / 255.f : 1.f;
      if (s < 1e-30f) s = 1e-30f;
      float inv = 1.f / s, zp = nearbyintf(fmn * inv) + 128.f;
      rq[i].scale = s; rq[i].zp = zp;
      float32x4_t vinv = vdupq_n_f32(inv);
      int32x4_t zofs = vdupq_n_s32(0x4B400000 + (int32_t)zp);
      for (int xi = 0; xi < 16; xi++) {
        const float *v = Vt + xi * C;
        int8_t *o = A8 + ((size_t)xi * CHUNK + i) * C;
        for (int c = 0; c < C; c += 8) vst1_s8(o + c, q8x8(vld1q_f32(v + c), vld1q_f32(v + c + 4), vinv, zofs));
      }
    }
    /* 2. 16 GEMMs, raw int32 accumulators Mb[xi][i][k] */
    for (int xi = 0; xi < 16; xi++) {
      /* tile-major layout Mb[i][xi][k] (ldo 16K): the output transform reads one contiguous 16xK block per tile
       * (the position-major layout aliased all 16 streams onto 2 L1 sets: 24.6 KB stride) */
      epi e = {ACT_RAW, NULL, (float *)(Mb + (size_t)xi * K), 16 * K, 0, 0, NULL};
      gemm(A8 + (size_t)xi * CHUNK * C, C, rq, 0, Mt, &ww->u[xi], &e);
    }
    /* 3. integer output transform, dequantise 4 outputs per tile, bias, SiLU -> fp16 */
    for (int i = 0; i < Mt; i++) {
      int t = t0 + i, ti = t / tw, tj = t - ti * tw;
      int oy = 2 * ti, ox = 2 * tj;
      int ny = oy + 1 < H ? 2 : 1, nx = ox + 1 < W ? 2 : 1;
      float sa = rq[i].scale, zs = rq[i].scale * rq[i].zp;
      const int32_t *mrow = Mb + (size_t)i * 16 * K;
      const size_t xs = (size_t)K;
      for (int k = 0; k < K; k += 8) {
        float16x8_t hv[4];
        for (int h = 0; h < 2; h++) {
          int kk = k + 4 * h;
          int32x4_t m[4][4];
          for (int xi = 0; xi < 16; xi++) m[xi / 4][xi % 4] = vld1q_s32(mrow + xi * xs + kk);
          int32x4_t s0[4], s1[4];
          for (int cc = 0; cc < 4; cc++) {
            s0[cc] = vaddq_s32(vaddq_s32(m[0][cc], m[1][cc]), m[2][cc]);
            s1[cc] = vsubq_s32(vsubq_s32(m[1][cc], m[2][cc]), m[3][cc]);
          }
          int32x4_t Y[4] = {vaddq_s32(vaddq_s32(s0[0], s0[1]), s0[2]), vsubq_s32(vsubq_s32(s0[1], s0[2]), s0[3]),
                            vaddq_s32(vaddq_s32(s1[0], s1[1]), s1[2]), vsubq_s32(vsubq_s32(s1[1], s1[2]), s1[3])};
          float32x4_t swv = vld1q_f32(ww->sw + kk), bb = vld1q_f32(ww->bias + kk);
          for (int o = 0; o < 4; o++) {
            float32x4_t v = vfmaq_n_f32(vfmaq_n_f32(bb, vmulq_f32(vcvtq_f32_s32(Y[o]), swv), sa), vld1q_f32(ww->zw[o] + kk), zs);
            float16x4_t hh = vcvt_f16_f32(v);
            hv[o] = h ? vcombine_f16(vget_low_f16(hv[o]), hh) : vcombine_f16(hh, hh);
          }
        }
        silu4_h(hv);
        for (int r = 0; r < ny; r++)
          for (int cc = 0; cc < nx; cc++) vst1q_f16(out + ((size_t)(oy + r) * W + ox + cc) * K + k, hv[r * 2 + cc]);
      }
    }
  }
}
