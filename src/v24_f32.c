// BirdNET v2.4 fast engine, portable fp32 path (plain ARMv8-A NEON: Cortex-A72 / Pi 4; no fp16 arithmetic, no
// dotprod). Same graph as v24_fast.c (the fp16 path) with every stage in fp32: front end (v24_fe.c, exact FFT),
// every conv / GEMM through gemm_f32.c (8x8 FMLA .4s micro-kernel, fused bias / SiLU / ReLU / residual epilogue),
// depthwise 3x3 fp32 + SiLU + per-channel sums, SE fp32, head fp32 GEMV straight from the mmap'd weights.
// Activations NHWC fp32, channel stride = channels rounded up to 8 (pad channels are kept at 0).
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_f32.h"
#include "v24_kern.h"
#include "v24_fe.h"
#include "v24_internal.h"

#define R8(x) (((x) + 7) & ~7)
#define C0_H 98  /* padded image for c0: 1 + 96 + 1 rows */
#define C0_W 518 /* 3 + 511 + 4 columns */

typedef struct {
  fpack pw, pwl;
  const float *dw_w, *dw_b; /* [9][cmid], [cmid] (mmap views) */
  float *se2t;              /* SE expand weights transposed [rd][cmid] */
  int ld_in, ld_out;        /* channel strides */
} f32block;

struct b24_f32 {
  const b24_model *m;
  b24_fe *fe;
  fpack c0, p0, pc;
  f32block blk[MAX_BLOCKS];
  float *img, *imgp, *pin, *X, *Y, *HP, *D, *pcin, *pcout;
  float *chsum;
  int prof;
  double t[16], tl[MAX_BLOCKS][4];
  int nprof;
};

enum { P_FE, P_C0, P_P0, P_PW, P_DW, P_SE, P_PWL, P_PC, P_HEAD, P_N };
static const char *pnames[P_N] = {"front end", "c0 4x8 conv", "pool+p0", "pw 1x1", "dw 3x3", "SE", "pwl 1x1",
                                  "post+pc 3x3", "head"};

/* +64 B slack: the GEMM K loop reads 16 B past the last A row; zeroed */
static float *f32a(size_t n) {
  size_t bytes = (n * 4 + 127) & ~(size_t)63;
  float *p = aligned_alloc(64, bytes);
  if (p) memset(p, 0, bytes);
  return p;
}

b24_f32 *b24_f32_new(const b24_model *m) {
  b24_f32 *f = calloc(1, sizeof *f);
  if (!f) return NULL;
  f->m = m;
  f->fe = b24_fe_new(m);
  /* c0: OHWI [24][4][8][2] = [24][64] */
  fp_pack(&f->c0, m->c0_w, m->c0_b, 24, 64, 64);
  fp_pack(&f->p0, m->p0_w, m->p0_b, 24, 48, 48);
  int ld = R8(24);
  for (int i = 0; i < m->n_blocks; i++) {
    const b24_block *b = &m->blocks[i];
    f32block *fb = &f->blk[i];
    fb->ld_in = ld;
    fp_pack(&fb->pw, b->pw_w, b->pw_b, b->cmid, b->cin, R8(b->cin));
    fp_pack(&fb->pwl, b->pwl_w, b->pwl_b, b->cout, b->cmid, R8(b->cmid));
    fb->dw_w = b->dw_w;
    fb->dw_b = b->dw_b;
    if (b->se_rd) {
      fb->se2t = malloc(sizeof(float) * b->se_rd * b->cmid);
      for (int c = 0; c < b->cmid; c++)
        for (int r = 0; r < b->se_rd; r++) fb->se2t[(size_t)r * b->cmid + c] = b->se2_w[(size_t)c * b->se_rd + r];
    }
    ld = R8(b->cout);
    fb->ld_out = ld;
  }
  /* pc: OHWI [1024][3][3][192] = [1024][1728] */
  fp_pack(&f->pc, m->pc_w, m->pc_b, FEAT_C, 1728, 1728);
  f->img = f32a((size_t)NMEL * NFR * 2);
  f->imgp = f32a((size_t)C0_H * C0_W * 2);
  f->pin = f32a((size_t)48 * 128 * 48);
  size_t act = (size_t)48 * 128 * 48;
  f->X = f32a(act);
  f->Y = f32a(act);
  size_t hp = 0, dd = 0;
  for (int i = 0; i < m->n_blocks; i++) {
    const b24_block *b = &m->blocks[i];
    size_t a = (size_t)(b->hin + 2) * (b->win + 2) * b->cmid, d = (size_t)b->hout * b->wout * b->cmid;
    if (a > hp) hp = a;
    if (d > dd) dd = d;
  }
  f->HP = f32a(hp);
  f->D = f32a(dd);
  f->pcin = f32a((size_t)8 * 1728);
  f->pcout = f32a((size_t)8 * FEAT_C);
  f->chsum = malloc(sizeof(float) * 4096);

  const char *p = getenv("BN24_PROF");
  f->prof = p ? atoi(p) : 0;
  return f;
}

void b24_f32_free(b24_f32 *f) {
  if (!f) return;
  if (f->prof && f->nprof) {
    double tot = 0;
    for (int i = 0; i < P_N; i++) tot += f->t[i];
    fprintf(stderr, "profile over %d windows (ms/window):\n", f->nprof);
    for (int i = 0; i < P_N; i++) fprintf(stderr, "  %-14s %7.3f\n", pnames[i], 1e3 * f->t[i] / f->nprof);
    fprintf(stderr, "  %-14s %7.3f\n", "total", 1e3 * tot / f->nprof);
    if (f->prof > 1)
      for (int i = 0; i < f->m->n_blocks; i++) {
        const b24_block *b = &f->m->blocks[i];
        double mp = 1e-6 * b->hin * b->win * b->cin * b->cmid, ml = 1e-6 * b->hout * b->wout * b->cmid * b->cout;
        double tp = 1e3 * f->tl[i][0] / f->nprof, td = 1e3 * f->tl[i][1] / f->nprof, ts = 1e3 * f->tl[i][2] / f->nprof,
               tw = 1e3 * f->tl[i][3] / f->nprof;
        fprintf(stderr, "  %-5s pw %6.3f ms (%5.1f GMAC/s)  dw %6.3f  se %6.3f  pwl %6.3f ms (%5.1f GMAC/s)\n", b->tag, tp,
                mp / tp, td, ts, tw, ml / tw);
      }
  }
  b24_fe_free(f->fe);
  fp_free(&f->c0); fp_free(&f->p0); fp_free(&f->pc);
  for (int i = 0; i < f->m->n_blocks; i++) { fp_free(&f->blk[i].pw); fp_free(&f->blk[i].pwl); free(f->blk[i].se2t); }
  free(f->img); free(f->imgp); free(f->pin); free(f->X); free(f->Y); free(f->HP); free(f->D);
  free(f->pcin); free(f->pcout); free(f->chsum);
  free(f);
}

const char *b24_f32_precision(const b24_f32 *f) { (void)f; return "fp32"; }

#define TICK(i)                                                                    \
  do {                                                                             \
    if (f->prof) { double _n = b24_now(); f->t[i] += _n - t0; lastdt = _n - t0; t0 = _n; } \
  } while (0)

/* c0: 4x8 conv stride 2 (SAME_UPPER: pad top 1, left 3) on the image copied into a zero border
 * [1 + 96 + 1][3 + 511 + 4][2]; the K = 64 patch of output (oy, ox) is 4 rows of 16 contiguous floats, so the GEMM
 * reads it in place (no im2col; the image buffer carries slack for the kernel's 16 B read-ahead). ReLU + max/avg pool of output pixel pairs happen in the epilogue -> p0 input. */
void v24k32_c0_pool(const float *img, float *imgp, const fpack *c0, float *pin) {
  for (int y = 0; y < 96; y++)
    memcpy(imgp + ((size_t)(y + 1) * C0_W + 3) * 2, img + (size_t)y * 511 * 2, sizeof(float) * 511 * 2);
  for (int oy = 0; oy < 48; oy++) {
    /* row ox of the GEMM = patch (oy, ox): 4 segments (ky) of 16 floats, row stride 2 px = 4 floats */
    epi32 e = {.act = ACT32_RELU_POOL2, .out = pin + (size_t)oy * 128 * 48, .ldo = 48, .pool_off = 24};
    gemm_f32_seg(imgp + (size_t)(2 * oy) * C0_W * 2, 4, 256, 4, 16, C0_W * 2, c0, &e);
  }
}
static void c0_pool(b24_f32 *f) { v24k32_c0_pool(f->img, f->imgp, &f->c0, f->pin); }

void v24k32_zero_halo(float *xp, int H, int W, int C) {
  size_t rowl = (size_t)(W + 2) * C;
  memset(xp, 0, rowl * 4);
  memset(xp + (size_t)(H + 1) * rowl, 0, rowl * 4);
  for (int y = 1; y <= H; y++) {
    memset(xp + (size_t)y * rowl, 0, (size_t)C * 4);
    memset(xp + (size_t)y * rowl + (size_t)(W + 1) * C, 0, (size_t)C * 4);
  }
}

static inline float silu_s(float x) { return x / (1.f + expf(-x)); }

/* SE: gate = sigmoid(W2 silu(W1 mean + b1) + b2); D[p][c] *= gate[c] */
/* the SE's channel mean and first FC + SiLU alone (the same instructions as v24k32_se): s1 [rd] */
void v24k32_se_fc1(int cm, int rd, const float *se1_w, const float *se1_b, const float *chsum, int P, float *s1) {
  float mean[4096] __attribute__((aligned(16)));
  const float32x4_t inv = vdupq_n_f32(1.f / (float)P);
  for (int c = 0; c < cm; c += 4) vst1q_f32(mean + c, vmulq_f32(vld1q_f32(chsum + c), inv));
  for (int r = 0; r < rd; r++) {
    const float *w = se1_w + (size_t)r * cm;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
    int c = 0;
    for (; c + 8 <= cm; c += 8) {
      a0 = vfmaq_f32(a0, vld1q_f32(w + c), vld1q_f32(mean + c));
      a1 = vfmaq_f32(a1, vld1q_f32(w + c + 4), vld1q_f32(mean + c + 4));
    }
    for (; c < cm; c += 4) a0 = vfmaq_f32(a0, vld1q_f32(w + c), vld1q_f32(mean + c));
    s1[r] = silu_s(vaddvq_f32(vaddq_f32(a0, a1)) + se1_b[r]);
  }
}

void v24k32_se(int cm, int rd, const float *se1_w, const float *se1_b, const float *se2t, const float *se2_b,
               const float *chsum, int P, float *D) {
  float mean[4096] __attribute__((aligned(16))), s1[512], g[4096] __attribute__((aligned(16)));
  const float32x4_t inv = vdupq_n_f32(1.f / (float)P);
  for (int c = 0; c < cm; c += 4) vst1q_f32(mean + c, vmulq_f32(vld1q_f32(chsum + c), inv));
  for (int r = 0; r < rd; r++) {
    const float *w = se1_w + (size_t)r * cm;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
    int c = 0;
    for (; c + 8 <= cm; c += 8) {
      a0 = vfmaq_f32(a0, vld1q_f32(w + c), vld1q_f32(mean + c));
      a1 = vfmaq_f32(a1, vld1q_f32(w + c + 4), vld1q_f32(mean + c + 4));
    }
    for (; c < cm; c += 4) a0 = vfmaq_f32(a0, vld1q_f32(w + c), vld1q_f32(mean + c));
    s1[r] = silu_s(vaddvq_f32(vaddq_f32(a0, a1)) + se1_b[r]);
  }
  memcpy(g, se2_b, sizeof(float) * cm);
  for (int r = 0; r < rd; r++) { /* r outer: se2t rows are streamed */
    const float *w = se2t + (size_t)r * cm;
    for (int c = 0; c < cm; c += 8) {
      vst1q_f32(g + c, vfmaq_n_f32(vld1q_f32(g + c), vld1q_f32(w + c), s1[r]));
      vst1q_f32(g + c + 4, vfmaq_n_f32(vld1q_f32(g + c + 4), vld1q_f32(w + c + 4), s1[r]));
    }
  }
  for (int c = 0; c < cm; c += 4) vst1q_f32(g + c, sigmoid_f32x4(vld1q_f32(g + c)));
  for (int p = 0; p < P; p++) {
    float *d = D + (size_t)p * cm;
    for (int c = 0; c < cm; c += 8) {
      vst1q_f32(d + c, vmulq_f32(vld1q_f32(d + c), vld1q_f32(g + c)));
      vst1q_f32(d + c + 4, vmulq_f32(vld1q_f32(d + c + 4), vld1q_f32(g + c + 4)));
    }
  }
}

/* head: logits[o] = b[o] + sum_k feat[k] W[o][k]; fp32 GEMV over the mmap'd [N][1024] weights (DRAM-bound) */
void v24k32_head(const float *W, const float *bias, int N, const float *feat, float *out) {
  int o = 0;
  for (; o + 4 <= N; o += 4) {
    const float *w0 = W + (size_t)o * FEAT_C, *w1 = w0 + FEAT_C, *w2 = w1 + FEAT_C, *w3 = w2 + FEAT_C;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0, c0 = a0, c1 = a0, c2 = a0, c3 = a0;
    for (int k = 0; k < FEAT_C; k += 8) {
      float32x4_t x0 = vld1q_f32(feat + k), x1 = vld1q_f32(feat + k + 4);
      a0 = vfmaq_f32(a0, vld1q_f32(w0 + k), x0); c0 = vfmaq_f32(c0, vld1q_f32(w0 + k + 4), x1);
      a1 = vfmaq_f32(a1, vld1q_f32(w1 + k), x0); c1 = vfmaq_f32(c1, vld1q_f32(w1 + k + 4), x1);
      a2 = vfmaq_f32(a2, vld1q_f32(w2 + k), x0); c2 = vfmaq_f32(c2, vld1q_f32(w2 + k + 4), x1);
      a3 = vfmaq_f32(a3, vld1q_f32(w3 + k), x0); c3 = vfmaq_f32(c3, vld1q_f32(w3 + k + 4), x1);
    }
    out[o] = vaddvq_f32(vaddq_f32(a0, c0)) + bias[o];
    out[o + 1] = vaddvq_f32(vaddq_f32(a1, c1)) + bias[o + 1];
    out[o + 2] = vaddvq_f32(vaddq_f32(a2, c2)) + bias[o + 2];
    out[o + 3] = vaddvq_f32(vaddq_f32(a3, c3)) + bias[o + 3];
  }
  for (; o < N; o++) {
    const float *w = W + (size_t)o * FEAT_C;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
    for (int k = 0; k < FEAT_C; k += 8) {
      a0 = vfmaq_f32(a0, vld1q_f32(w + k), vld1q_f32(feat + k));
      a1 = vfmaq_f32(a1, vld1q_f32(w + k + 4), vld1q_f32(feat + k + 4));
    }
    out[o] = vaddvq_f32(vaddq_f32(a0, a1)) + bias[o];
  }
}

/* post BN + ReLU -> [3][8][192], im2col 3x3 valid -> 6 x 1728, pc GEMM + ReLU, mean over the 6 positions */
void v24k32_postpc(const float *X, const float *post_s, const float *post_b, float *pcin, float *pcout, const fpack *pc,
                   float *feat) {
  /* post BN + ReLU -> [3][8][192], im2col 3x3 valid -> 6 x 1728 */
  float post[24 * 192] __attribute__((aligned(16)));
  const float32x4_t z = vdupq_n_f32(0);
  for (int p = 0; p < 24; p++)
    for (int c = 0; c < 192; c += 4) {
      float32x4_t v = vfmaq_f32(vld1q_f32(post_b + c), vld1q_f32(X + p * 192 + c), vld1q_f32(post_s + c));
      vst1q_f32(post + p * 192 + c, vmaxq_f32(v, z));
    }
  for (int ox = 0; ox < 6; ox++)
    for (int ky = 0; ky < 3; ky++)
      for (int kx = 0; kx < 3; kx++)
        memcpy(pcin + (size_t)ox * 1728 + (ky * 3 + kx) * 192, post + (ky * 8 + ox + kx) * 192, 192 * 4);
  epi32 e3 = {.act = ACT32_RELU, .out = pcout, .ldo = FEAT_C};
  gemm_f32(pcin, 1728, 6, pc, &e3);
  for (int c = 0; c < FEAT_C; c += 4) {
    float32x4_t s = vdupq_n_f32(0);
    for (int ox = 0; ox < 6; ox++) s = vaddq_f32(s, vld1q_f32(pcout + ox * FEAT_C + c));
    vst1q_f32(feat + c, vmulq_n_f32(s, 1.f / 6.f));
  }
}

int b24_f32_infer(b24_f32 *f, const float *pcm, float *logits, float *emb) {
  const b24_model *m = f->m;
  double t0 = f->prof ? b24_now() : 0, lastdt = 0;
  (void)lastdt;
  b24_fe_run(f->fe, pcm, f->img);
  TICK(P_FE);
  c0_pool(f);
  TICK(P_C0);
  epi32 e0 = {.act = ACT32_NONE, .out = f->X, .ldo = 24};
  gemm_f32(f->pin, 48, 48 * 128, &f->p0, &e0);
  TICK(P_P0);
  float *X = f->X, *Y = f->Y;
  for (int i = 0; i < m->n_blocks; i++) {
    const b24_block *b = &m->blocks[i];
    const f32block *fb = &f->blk[i];
    int H = b->hin, W = b->win, Ho = b->hout, Wo = b->wout, cm = b->cmid;
    v24k32_zero_halo(f->HP, H, W, cm);
    epi32 e = {.act = ACT32_SILU, .out = f->HP, .ldo = cm, .ow = W, .opad = 1};
    gemm_f32(X, fb->ld_in, H * W, &fb->pw, &e);
    TICK(P_PW);
    f->tl[i][0] += lastdt;
    dw3x3_f32(f->HP, H, W, cm, b->stride, fb->dw_w, fb->dw_b, f->D, Ho, Wo, f->chsum);
    TICK(P_DW);
    f->tl[i][1] += lastdt;
    if (b->se_rd) v24k32_se(b->cmid, b->se_rd, b->se1_w, b->se1_b, fb->se2t, b->se2_b, f->chsum, Ho * Wo, f->D);
    TICK(P_SE);
    f->tl[i][2] += lastdt;
    epi32 e2 = {.act = ACT32_NONE, .out = Y, .ldo = fb->ld_out, .res = b->skip ? X : NULL, .ldr = fb->ld_in};
    gemm_f32(f->D, cm, Ho * Wo, &fb->pwl, &e2);
    TICK(P_PWL);
    f->tl[i][3] += lastdt;
    float *t = X; X = Y; Y = t;
  }
  float feat[FEAT_C] __attribute__((aligned(16)));
  v24k32_postpc(X, m->post_s, m->post_b, f->pcin, f->pcout, &f->pc, feat);
  if (emb) memcpy(emb, feat, sizeof feat);
  TICK(P_PC);
  v24k32_head(m->head_w, m->head_b, m->n_classes, feat, logits);
  TICK(P_HEAD);
  f->nprof++;
  return 0;
}
