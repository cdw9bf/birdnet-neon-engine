// BirdNET v2.4 fast engine, fp16 path (Cortex-A76: ARMv8.2 fp16 arithmetic).
// Front end fp32 (v24_fe.c, exact FFT); every conv / GEMM fp16 with fp32 flushes for long K (gemm_f16.c);
// depthwise 3x3 fp16 + SiLU (shared kern_f16.c kernel); SE fp32; residual stream fp16; head fp16 GEMV with fp32
// flushes. Activations NHWC fp16, channel stride = channels rounded up to 16 (pad channels are kept at 0).
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemm_f16.h"
#include "v24_kern.h"
#include "kern.h"
#include "v24_fe.h"
#include "v24_internal.h"

#define R8(x) (((x) + 7) & ~7)
#define C0_H 98  /* padded fp16 image for c0: 1 + 96 + 1 rows */
#define C0_W 518 /* 3 + 511 + 4 columns */

typedef struct {
  hpack pw, pwl;
  f16 *dw_w, *dw_b;  /* [9][cmid], [cmid] */
  float *se2t;       /* SE expand weights transposed [rd][cmid] */
  int ld_in, ld_out; /* channel strides */
} fblock;

struct b24_fast {
  const b24_model *m;
  b24_fe *fe;
  hpack c0, p0, pc;
  fblock blk[MAX_BLOCKS];
  f16 *head_w; /* [n_classes][1024] */
  float *head_b;
  float *img;
  f16 *img16, *pin, *X, *Y, *HP, *D, *pcin, *pcout;
  const f16 **rows; /* c0: [4][256] row pointers into img16 */
  float *chsum;
  int prof;
  double t[16], tl[MAX_BLOCKS][4];
  int nprof;
};

enum { P_FE, P_C0, P_P0, P_PW, P_DW, P_SE, P_PWL, P_PC, P_HEAD, P_N };
static const char *pnames[P_N] = {"front end", "c0 4x8 conv", "pool+p0", "pw 1x1", "dw 3x3", "SE", "pwl 1x1",
                                  "post+pc 3x3", "head"};

/* +64 B slack: the GEMM K loop reads 16 B past the last A row */
static f16 *h16(size_t n) { return aligned_alloc(64, (n * 2 + 127) & ~(size_t)63); }

b24_fast *b24_fast_new(const b24_model *m, int mode) {
  b24_fast *f = calloc(1, sizeof *f);
  if (!f) return NULL;
  f->m = m;
  f->fe = b24_fe_new(m);
  /* c0: OHWI [24][4][8][2] = [24][64] */
  hp_pack(&f->c0, m->c0_w, m->c0_b, 24, 64, 64);
  hp_pack(&f->p0, m->p0_w, m->p0_b, 24, 48, 48);
  int ld = R8(24);
  for (int i = 0; i < m->n_blocks; i++) {
    const b24_block *b = &m->blocks[i];
    fblock *fb = &f->blk[i];
    fb->ld_in = ld;
    hp_pack(&fb->pw, b->pw_w, b->pw_b, b->cmid, b->cin, R8(b->cin));
    hp_pack(&fb->pwl, b->pwl_w, b->pwl_b, b->cout, b->cmid, R8(b->cmid));
    fb->dw_w = h16((size_t)9 * b->cmid);
    fb->dw_b = h16(b->cmid);
    for (int k = 0; k < 9 * b->cmid; k++) fb->dw_w[k] = (f16)b->dw_w[k];
    for (int c = 0; c < b->cmid; c++) fb->dw_b[c] = (f16)b->dw_b[c];
    if (b->se_rd) {
      fb->se2t = malloc(sizeof(float) * b->se_rd * b->cmid);
      for (int c = 0; c < b->cmid; c++)
        for (int r = 0; r < b->se_rd; r++) fb->se2t[(size_t)r * b->cmid + c] = b->se2_w[(size_t)c * b->se_rd + r];
    }
    ld = R8(b->cout);
    fb->ld_out = ld;
  }
  /* pc: OHWI [1024][3][3][192] = [1024][1728] */
  hp_pack(&f->pc, m->pc_w, m->pc_b, FEAT_C, 1728, 1728);
  f->head_w = h16((size_t)m->n_classes * FEAT_C);
  for (size_t i = 0; i < (size_t)m->n_classes * FEAT_C; i++) f->head_w[i] = (f16)m->head_w[i];
  f->head_b = malloc(sizeof(float) * m->n_classes);
  memcpy(f->head_b, m->head_b, sizeof(float) * m->n_classes);

  f->img = aligned_alloc(64, sizeof(float) * NMEL * NFR * 2);
  f->img16 = h16((size_t)C0_H * C0_W * 2);
  memset(f->img16, 0, (size_t)C0_H * C0_W * 2 * 2);
  f->rows = malloc(sizeof(f16 *) * 4 * 256);
  f->pin = h16((size_t)48 * 128 * 48);
  size_t act = (size_t)48 * 128 * 48;
  f->X = h16(act);
  f->Y = h16(act);
  size_t hp = 0, dd = 0;
  for (int i = 0; i < m->n_blocks; i++) {
    const b24_block *b = &m->blocks[i];
    size_t a = (size_t)(b->hin + 2) * (b->win + 2) * b->cmid, d = (size_t)b->hout * b->wout * b->cmid;
    if (a > hp) hp = a;
    if (d > dd) dd = d;
  }
  f->HP = h16(hp);
  f->D = h16(dd);
  f->pcin = h16((size_t)8 * 1728);
  f->pcout = h16((size_t)8 * FEAT_C);
  memset(f->pcin, 0, (size_t)8 * 1728 * 2);
  f->chsum = malloc(sizeof(float) * 4096);

  const char *p = getenv("BN24_PROF");
  f->prof = p ? atoi(p) : 0;
  return f;
}

void b24_fast_free(b24_fast *f) {
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
  hp_free(&f->c0); hp_free(&f->p0); hp_free(&f->pc);
  for (int i = 0; i < f->m->n_blocks; i++) {
    hp_free(&f->blk[i].pw); hp_free(&f->blk[i].pwl); free(f->blk[i].dw_w); free(f->blk[i].dw_b); free(f->blk[i].se2t);
  }
  free(f->head_w); free(f->head_b); free(f->img); free(f->img16); free(f->rows); free(f->pin); free(f->X); free(f->Y);
  free(f->HP); free(f->D); free(f->pcin); free(f->pcout); free(f->chsum);
  free(f);
}

const char *b24_fast_precision(const b24_fast *f) { (void)f; return "fp16"; }

#define TICK(i)                                                                    \
  do {                                                                             \
    if (f->prof) { double _n = b24_now(); f->t[i] += _n - t0; lastdt = _n - t0; t0 = _n; } \
  } while (0)

/* c0: 4x8 conv stride 2 (SAME_UPPER: pad top 1, left 3) on the image converted to fp16 with a zero border
 * [1 + 96 + 1][3 + 511 + 4][2]; the K = 64 patch of output (oy, ox) is 4 rows of 16 contiguous halfs, so the GEMM
 * reads it in place (no im2col). ReLU + max/avg pool of output pixel pairs happen in the epilogue -> p0 input. */
void v24k16_c0_pool(const float *img, f16 *img16, const f16 **rows, const hpack *c0, f16 *pin) {
  for (int y = 0; y < 96; y++) {
    const float *src = img + (size_t)y * 511 * 2;
    f16 *dst = img16 + ((size_t)(y + 1) * C0_W + 3) * 2;
    int i = 0;
    for (; i + 8 <= 511 * 2; i += 8)
      vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(vld1q_f32(src + i)), vcvt_f16_f32(vld1q_f32(src + i + 4))));
    for (; i < 511 * 2; i++) dst[i] = (f16)src[i];
  }
  for (int oy = 0; oy < 48; oy++) {
    for (int ky = 0; ky < 4; ky++)
      for (int ox = 0; ox < 256; ox++) rows[ky * 256 + ox] = img16 + ((size_t)(2 * oy + ky) * C0_W + 2 * ox) * 2;
    epi16 e = {.act = ACT16_RELU_POOL2, .out = pin + (size_t)oy * 128 * 48, .ldo = 48, .pool_off = 24};
    gemm_f16_rows(rows, 4, 16, 256, c0, &e);
  }
}
static void c0_pool(b24_fast *f) { v24k16_c0_pool(f->img, f->img16, f->rows, &f->c0, f->pin); }

void v24k16_zero_halo(f16 *xp, int H, int W, int C) {
  size_t rowl = (size_t)(W + 2) * C;
  memset(xp, 0, rowl * 2);
  memset(xp + (size_t)(H + 1) * rowl, 0, rowl * 2);
  for (int y = 1; y <= H; y++) {
    memset(xp + (size_t)y * rowl, 0, (size_t)C * 2);
    memset(xp + (size_t)y * rowl + (size_t)(W + 1) * C, 0, (size_t)C * 2);
  }
}

static inline float silu_s(float x) { return x / (1.f + expf(-x)); }

/* SE: gate = sigmoid(W2 silu(W1 mean + b1) + b2) (fp32); D[p][c] *= gate[c] (fp16) */
/* sigmoid(a) = 1 / (1 + 2^(-a log2 e)); 2^t by a degree-6 polynomial (rel. err ~1e-7), t clamped to [-126, 126] */
static inline float32x4_t sigm4(float32x4_t a) {
  float32x4_t t = vmulq_n_f32(a, -1.44269504f);
  t = vminq_f32(vmaxq_f32(t, vdupq_n_f32(-126.f)), vdupq_n_f32(126.f));
  float32x4_t n = vrndnq_f32(t), f = vsubq_f32(t, n);
  float32x4_t p = vdupq_n_f32(1.5403530e-4f);
  p = vfmaq_f32(vdupq_n_f32(1.3333558e-3f), p, f);
  p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), p, f);
  p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, f);
  p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(1.f), p, f);
  float32x4_t e = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(vcvtq_s32_f32(n), 23)));
  return vdivq_f32(vdupq_n_f32(1.f), vaddq_f32(vdupq_n_f32(1.f), e));
}
/* the SE gate vector g16 [cm] (fp16) from the channel sums: mean, fc + SiLU, fc, sigmoid */
void v24k16_se_gate(int cm, int rd, const float *se1_w, const float *se1_b, const float *se2t, const float *se2_b,
                    const float *chsum, int P, f16 *g16) {
  float mean[4096] __attribute__((aligned(16))), s1[512];
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
  for (int c = 0; c < cm; c += 8) {
    float32x4_t a0 = vld1q_f32(se2_b + c), a1 = vld1q_f32(se2_b + c + 4);
    for (int r = 0; r < rd; r++) {
      a0 = vfmaq_n_f32(a0, vld1q_f32(se2t + (size_t)r * cm + c), s1[r]);
      a1 = vfmaq_n_f32(a1, vld1q_f32(se2t + (size_t)r * cm + c + 4), s1[r]);
    }
    vst1q_f16(g16 + c, vcombine_f16(vcvt_f16_f32(sigm4(a0)), vcvt_f16_f32(sigm4(a1))));
  }
}
/* pixels [p0, p1) of D [P][cm] scaled by the gate, in place (each pixel on its own) */
void v24k16_se_scale(int cm, const f16 *g16, f16 *D, int p0, int p1) {
  for (int p = p0; p < p1; p++) {
    f16 *d = D + (size_t)p * cm;
    for (int c = 0; c < cm; c += 8) vst1q_f16(d + c, vmulq_f16(vld1q_f16(d + c), vld1q_f16(g16 + c)));
  }
}
void v24k16_se(int cm, int rd, const float *se1_w, const float *se1_b, const float *se2t, const float *se2_b,
               const float *chsum, int P, f16 *D) {
  f16 g16[4096] __attribute__((aligned(16)));
  v24k16_se_gate(cm, rd, se1_w, se1_b, se2t, se2_b, chsum, P, g16);
  v24k16_se_scale(cm, g16, D, 0, P);
}

/* head: logits[o] = b[o] + sum_k feat[k] W[o][k]; fp16 FMLA, fp32 flush every 128 k */
void v24k16_head(const f16 *W, const float *bias, int N, const float *feat, float *out) {
  f16 x[FEAT_C] __attribute__((aligned(16)));
  for (int k = 0; k < FEAT_C; k += 8)
    vst1q_f16(x + k, vcombine_f16(vcvt_f16_f32(vld1q_f32(feat + k)), vcvt_f16_f32(vld1q_f32(feat + k + 4))));
  int o = 0;
  for (; o + 4 <= N; o += 4) {
    const f16 *w0 = W + (size_t)o * FEAT_C, *w1 = w0 + FEAT_C, *w2 = w1 + FEAT_C, *w3 = w2 + FEAT_C;
    float32x4_t s[4][2];
    for (int r = 0; r < 4; r++) s[r][0] = s[r][1] = vdupq_n_f32(0);
    for (int k0 = 0; k0 < FEAT_C; k0 += 128) {
      float16x8_t a0 = vdupq_n_f16(0), a1 = a0, a2 = a0, a3 = a0;
      for (int k = k0; k < k0 + 128; k += 8) {
        float16x8_t xv = vld1q_f16(x + k);
        a0 = vfmaq_f16(a0, vld1q_f16(w0 + k), xv);
        a1 = vfmaq_f16(a1, vld1q_f16(w1 + k), xv);
        a2 = vfmaq_f16(a2, vld1q_f16(w2 + k), xv);
        a3 = vfmaq_f16(a3, vld1q_f16(w3 + k), xv);
      }
      float16x8_t a[4] = {a0, a1, a2, a3};
      for (int r = 0; r < 4; r++) {
        s[r][0] = vaddq_f32(s[r][0], vcvt_f32_f16(vget_low_f16(a[r])));
        s[r][1] = vaddq_f32(s[r][1], vcvt_high_f32_f16(a[r]));
      }
    }
    for (int r = 0; r < 4; r++) out[o + r] = vaddvq_f32(vaddq_f32(s[r][0], s[r][1])) + bias[o + r];
  }
  for (; o < N; o++) {
    const f16 *w = W + (size_t)o * FEAT_C;
    float s = 0;
    for (int k = 0; k < FEAT_C; k++) s += (float)w[k] * (float)x[k];
    out[o] = s + bias[o];
  }
}

/* post BN + ReLU (fp32) -> [3][8][192] fp16, im2col 3x3 valid -> 6 x 1728, pc GEMM + ReLU, mean over the 6
 * positions -> feat [1024] fp32; -2 if a feature is not finite (fp16 overflow: the caller falls back) */
int v24k16_postpc(const f16 *X, const float *post_s, const float *post_b, f16 *pcin, f16 *pcout, const hpack *pc,
                  float *feat) {
  /* post BN + ReLU (fp32) -> [3][8][192] fp16, im2col 3x3 valid -> 6 x 1728 */
  f16 post[24 * 192] __attribute__((aligned(16)));
  for (int p = 0; p < 24; p++)
    for (int c = 0; c < 192; c += 4) {
      float32x4_t v = vcvt_f32_f16(vld1_f16(X + p * 192 + c));
      v = vfmaq_f32(vld1q_f32(post_b + c), v, vld1q_f32(post_s + c));
      vst1_f16(post + p * 192 + c, vcvt_f16_f32(vmaxq_f32(v, vdupq_n_f32(0))));
    }
  for (int ox = 0; ox < 6; ox++)
    for (int ky = 0; ky < 3; ky++)
      for (int kx = 0; kx < 3; kx++)
        memcpy(pcin + (size_t)ox * 1728 + (ky * 3 + kx) * 192, post + (ky * 8 + ox + kx) * 192, 192 * 2);
  epi16 e3 = {.act = ACT16_RELU, .out = pcout, .ldo = FEAT_C};
  gemm_f16(pcin, 1728, 6, pc, &e3);
  for (int c = 0; c < FEAT_C; c += 4) {
    float32x4_t s = vdupq_n_f32(0);
    for (int ox = 0; ox < 6; ox++) s = vaddq_f32(s, vcvt_f32_f16(vld1_f16(pcout + ox * FEAT_C + c)));
    vst1q_f32(feat + c, vmulq_n_f32(s, 1.f / 6.f));
  }
  /* fp16 overflow guard: an out-of-range window gives Inf/NaN features -> report an error (the caller falls back) */
  uint32x4_t bad = vdupq_n_u32(0);
  for (int c = 0; c < FEAT_C; c += 4) {
    float32x4_t v = vld1q_f32(feat + c);
    bad = vorrq_u32(bad, vmvnq_u32(vcleq_f32(vabsq_f32(v), vdupq_n_f32(3.0e38f))));
  }
  if (vmaxvq_u32(bad)) return -2;
  return 0;
}

int b24_fast_infer(b24_fast *f, const float *pcm, float *logits, float *emb) {
  const b24_model *m = f->m;
  double t0 = f->prof ? b24_now() : 0, lastdt = 0;
  (void)lastdt;
  b24_fe_run(f->fe, pcm, f->img);
  TICK(P_FE);
  c0_pool(f);
  TICK(P_C0);
  epi16 e0 = {.act = ACT16_NONE, .out = f->X, .ldo = 24};
  gemm_f16(f->pin, 48, 48 * 128, &f->p0, &e0);
  TICK(P_P0);
  f16 *X = f->X, *Y = f->Y;
  for (int i = 0; i < m->n_blocks; i++) {
    const b24_block *b = &m->blocks[i];
    const fblock *fb = &f->blk[i];
    int H = b->hin, W = b->win, Ho = b->hout, Wo = b->wout, cm = b->cmid;
    v24k16_zero_halo(f->HP, H, W, cm);
    epi16 e = {.act = ACT16_SILU, .out = f->HP, .ldo = cm, .ow = W, .opad = 1};
    gemm_f16(X, fb->ld_in, H * W, &fb->pw, &e);
    TICK(P_PW);
    f->tl[i][0] += lastdt;
    dw3x3_f16(f->HP, H, W, cm, b->stride, 1, 1, fb->dw_w, fb->dw_b, f->D, Ho, Wo, f->chsum);
    TICK(P_DW);
    f->tl[i][1] += lastdt;
    if (b->se_rd) v24k16_se(b->cmid, b->se_rd, b->se1_w, b->se1_b, fb->se2t, b->se2_b, f->chsum, Ho * Wo, f->D);
    TICK(P_SE);
    f->tl[i][2] += lastdt;
    epi16 e2 = {.act = ACT16_NONE, .out = Y, .ldo = fb->ld_out, .res = b->skip ? X : NULL, .ldr = fb->ld_in};
    gemm_f16(f->D, cm, Ho * Wo, &fb->pwl, &e2);
    TICK(P_PWL);
    f->tl[i][3] += lastdt;
    f16 *t = X; X = Y; Y = t;
  }
  float feat[FEAT_C] __attribute__((aligned(16)));
  if (v24k16_postpc(X, m->post_s, m->post_b, f->pcin, f->pcout, &f->pc, feat)) return -2;
  if (emb) memcpy(emb, feat, sizeof feat);
  TICK(P_PC);
  v24k16_head(f->head_w, f->head_b, m->n_classes, feat, logits);
  TICK(P_HEAD);
  f->nprof++;
  return 0;
}
