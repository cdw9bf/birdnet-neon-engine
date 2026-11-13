// v2.4 front end, fp32 NEON: normalisation, two spectrograms (n_fft 2048 hop 278 and n_fft 1024 hop 280, 511 frames,
// periodic Hann from the weights file), REAL part of the DFT via a half-size complex FFT (4 frames per float32x4
// lane), sparse mel, square, pow (exp2/log2 polynomials), mel-axis reversal and the spectrogram BN folded into the
// output write. Output: image [96][511][2] fp32 (NHWC). Exact cos (matches TFLite FP32, not the ONNX dense matrix).
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fft_tab.h"
#include "gemm_f32.h"
#include "v24_fe.h"
#include "v24_internal.h"

#define AI static inline __attribute__((always_inline))
#ifdef FE_PROF
double fe_prof[4];
#define FP(i) do { double _n = b24_now(); fe_prof[i] += _n - _t; _t = _n; } while (0)
#else
#define FP(i) do { } while (0)
#endif

typedef struct {
  int N, M, hop, lg;        /* n_fft, M = N/2 complex points, log2 M */
  int klo, khi;             /* bins needed by the mel filters */
  float *tw_re, *tw_im;     /* e^{-2 pi i k / M}, k < M/2 */
  float *pc, *ps;           /* cos, sin(2 pi k / N) for the real split, k in [klo, khi) */
  uint16_t *rev;            /* bit reversal of [0, M) */
  float *win;               /* [N] */
  int mel_off[NMEL + 1];    /* sparse mel: entries [mel_off[j], mel_off[j+1]) */
  uint16_t *mel_k;
  float *mel_w;
  float pw2;                /* 2 * pow exponent: out = |m|^(2p) */
  float bn_s, bn_b;
  int ch;
} fe_spec;

struct b24_fe {
  fe_spec s[2];
  float *y;                 /* normalised samples */
  float32x4_t *re, *im, *R; /* work: [M], [M], [khi] */
  float32x4_t P[2][NMEL];   /* per-group outputs of both spectrograms (interleaved store) */
};

static void spec_init(fe_spec *s, const b24_model *m, int which) {
  s->ch = which;
  s->N = which ? N2 : N1;
  s->hop = which ? HOP2 : HOP1;
  s->M = s->N / 2;
  s->lg = 0;
  while ((1 << s->lg) < s->M) s->lg++;
  const float *mel = which ? m->mel2 : m->mel1;
  const int *lo = which ? m->mel2_lo : m->mel1_lo, *hi = which ? m->mel2_hi : m->mel1_hi;
  int bins = which ? B2 : B1;
  s->klo = bins; s->khi = 0;
  for (int j = 0; j < NMEL; j++)
    if (hi[j] > lo[j]) { if (lo[j] < s->klo) s->klo = lo[j]; if (hi[j] > s->khi) s->khi = hi[j]; }
  s->tw_re = malloc(sizeof(float) * s->M / 2);
  s->tw_im = malloc(sizeof(float) * s->M / 2);
  fft_twiddles(s->tw_re, s->tw_im, s->M);
  s->pc = malloc(sizeof(float) * s->khi);
  s->ps = malloc(sizeof(float) * s->khi);
  for (int k = 0; k < s->khi; k++) {
    double a = 2 * M_PI * k / s->N;
    s->pc[k] = (float)cos(a); s->ps[k] = (float)sin(a);
  }
  s->rev = malloc(sizeof(uint16_t) * s->M);
  for (int i = 0; i < s->M; i++) s->rev[i] = (uint16_t)fft_bitrev(i, s->lg);
  s->win = malloc(sizeof(float) * s->N);
  memcpy(s->win, which ? m->win2 : m->win1, sizeof(float) * s->N);
  int nnz = 0;
  for (int j = 0; j < NMEL; j++) nnz += hi[j] - lo[j];
  s->mel_k = malloc(sizeof(uint16_t) * (nnz + 1));
  s->mel_w = malloc(sizeof(float) * (nnz + 1));
  nnz = 0;
  for (int j = 0; j < NMEL; j++) {
    s->mel_off[j] = nnz;
    for (int k = lo[j]; k < hi[j]; k++) { s->mel_k[nnz] = (uint16_t)k; s->mel_w[nnz] = mel[(size_t)j * bins + k]; nnz++; }
  }
  s->mel_off[NMEL] = nnz;
  s->pw2 = 2.f * (which ? m->pow2[0] : m->pow1[0]);
  s->bn_s = m->fe_bn_s[which];
  s->bn_b = m->fe_bn_b[which];
}

b24_fe *b24_fe_new(const b24_model *m) {
  b24_fe *f = calloc(1, sizeof *f);
  spec_init(&f->s[0], m, 0);
  spec_init(&f->s[1], m, 1);
  f->y = aligned_alloc(64, sizeof(float) * (NSAMP + 64));
  f->re = aligned_alloc(64, sizeof(float32x4_t) * N1 / 2);
  f->im = aligned_alloc(64, sizeof(float32x4_t) * N1 / 2);
  f->R = aligned_alloc(64, sizeof(float32x4_t) * 1028);
  return f;
}

void b24_fe_free(b24_fe *f) {
  if (!f) return;
  for (int i = 0; i < 2; i++) {
    fe_spec *s = &f->s[i];
    free(s->tw_re); free(s->tw_im); free(s->pc); free(s->ps); free(s->rev); free(s->win); free(s->mel_k); free(s->mel_w);
  }
  free(f->y); free(f->re); free(f->im); free(f->R); free(f);
}

/* |x|^e for x >= 0 via exp2(e * log2 x), log2 by log2_acc_f32x4 (gemm_f32.h); x < 1e-30 -> treated as 1e-30
 * (contributes ~1e-6). exp2_acc is gemm_f32.h's exp2_f32x4 without the upper clamp at 126 (the same polynomial);
 * it stays a function of its own so that this front end's instructions are the gated ones. */
AI float32x4_t exp2_acc(float32x4_t t) {
  t = vmaxq_f32(t, vdupq_n_f32(-126.f));
  float32x4_t n = vrndnq_f32(t);
  float32x4_t f = vsubq_f32(t, n); /* [-0.5, 0.5] */
  /* 2^f, degree 6 Taylor in f*ln2 (|err| < 2e-8 rel on [-.5,.5]) */
  float32x4_t p = vdupq_n_f32(1.5403530393381606e-4f);
  p = vfmaq_f32(vdupq_n_f32(1.3333558146428443e-3f), p, f);
  p = vfmaq_f32(vdupq_n_f32(9.6181291076284772e-3f), p, f);
  p = vfmaq_f32(vdupq_n_f32(5.5504108664821580e-2f), p, f);
  p = vfmaq_f32(vdupq_n_f32(2.4022650695910071e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(6.9314718055994531e-1f), p, f);
  p = vfmaq_f32(vdupq_n_f32(1.f), p, f);
  int32x4_t ni = vcvtq_s32_f32(n);
  return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(ni, 23)));
}

/* 4 frames (lanes) of one spectrogram starting at frame t0 (frames >= NFR are computed from a clamped start and
 * not written) -> img rows */
static void spec_group(b24_fe *F, const fe_spec *s, int t0) {
  const int M = s->M;
  float32x4_t *re = F->re, *im = F->im;
#ifdef FE_PROF
  double _t = b24_now();
#endif
  int st[4];
  for (int q = 0; q < 4; q++) st[q] = (t0 + q < NFR ? t0 + q : NFR - 1) * s->hop;
  /* load z[n] = w[2n] x[2n] + i w[2n+1] x[2n+1] for 4 frames, 4 n at a time (vld2 + 4x4 transpose) */
  for (int n = 0; n < M; n += 4) {
    float32x4x2_t v0 = vld2q_f32(F->y + st[0] + 2 * n), v1 = vld2q_f32(F->y + st[1] + 2 * n),
                  v2 = vld2q_f32(F->y + st[2] + 2 * n), v3 = vld2q_f32(F->y + st[3] + 2 * n);
    float32x4x2_t wv = vld2q_f32(s->win + 2 * n);
    for (int part = 0; part < 2; part++) {
      float32x4_t a = vmulq_f32(v0.val[part], wv.val[part]), b = vmulq_f32(v1.val[part], wv.val[part]),
                  c = vmulq_f32(v2.val[part], wv.val[part]), d = vmulq_f32(v3.val[part], wv.val[part]);
      float32x4_t ab0 = vtrn1q_f32(a, b), ab1 = vtrn2q_f32(a, b), cd0 = vtrn1q_f32(c, d), cd1 = vtrn2q_f32(c, d);
      float32x4_t r0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(ab0), vreinterpretq_f64_f32(cd0)));
      float32x4_t r1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(ab1), vreinterpretq_f64_f32(cd1)));
      float32x4_t r2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(ab0), vreinterpretq_f64_f32(cd0)));
      float32x4_t r3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(ab1), vreinterpretq_f64_f32(cd1)));
      float32x4_t *dst = part ? im : re;
      dst[s->rev[n]] = r0; dst[s->rev[n + 1]] = r1; dst[s->rev[n + 2]] = r2; dst[s->rev[n + 3]] = r3;
    }
  }
  FP(0);
  fft_dit_x4(re, im, M, s->tw_re, s->tw_im);
  FP(1);
  /* real part of X_k = 1/2 [Zr_k + Zr_{M-k} + c (Zi_k + Zi_{M-k}) - s (Zr_k - Zr_{M-k})] */
  float32x4_t *R = F->R;
  const float32x4_t half = vdupq_n_f32(0.5f);
  for (int k = s->klo; k < s->khi; k++) {
    int k2 = k == 0 ? 0 : M - k;
    float32x4_t zr = re[k], zi = im[k], yr = re[k2], yi = im[k2];
    float32x4_t v = vaddq_f32(zr, yr);
    v = vfmaq_n_f32(v, vaddq_f32(zi, yi), s->pc[k]);
    v = vfmsq_n_f32(v, vsubq_f32(zr, yr), s->ps[k]);
    R[k] = vmulq_f32(v, half);
  }
  FP(2);
  /* sparse mel, square, pow, BN, write reversed mel rows */
  const float32x4_t pw2 = vdupq_n_f32(s->pw2), tiny = vdupq_n_f32(1e-30f);
  float32x4_t *P = F->P[s->ch];
  for (int j = 0; j < NMEL; j++) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    int e0 = s->mel_off[j], e1 = s->mel_off[j + 1], e = e0;
    for (; e + 2 <= e1; e += 2) {
      a0 = vfmaq_n_f32(a0, R[s->mel_k[e]], s->mel_w[e]);
      a1 = vfmaq_n_f32(a1, R[s->mel_k[e + 1]], s->mel_w[e + 1]);
    }
    if (e < e1) a0 = vfmaq_n_f32(a0, R[s->mel_k[e]], s->mel_w[e]);
    P[j] = vmaxq_f32(vabsq_f32(vaddq_f32(a0, a1)), tiny);
  }
  /* |m|^(2p) and the BN, 4 independent chains per step (the log2/exp2 Horner chains are latency-bound) */
  const float32x4_t bs = vdupq_n_f32(s->bn_s), bb = vdupq_n_f32(s->bn_b);
  for (int j = 0; j < NMEL; j += 4) {
    float32x4_t l0 = log2_acc_f32x4(P[j]), l1 = log2_acc_f32x4(P[j + 1]), l2 = log2_acc_f32x4(P[j + 2]), l3 = log2_acc_f32x4(P[j + 3]);
    float32x4_t p0 = exp2_acc(vmulq_f32(pw2, l0)), p1 = exp2_acc(vmulq_f32(pw2, l1)), p2 = exp2_acc(vmulq_f32(pw2, l2)),
                p3 = exp2_acc(vmulq_f32(pw2, l3));
    P[j] = vfmaq_f32(bb, p0, bs); P[j + 1] = vfmaq_f32(bb, p1, bs);
    P[j + 2] = vfmaq_f32(bb, p2, bs); P[j + 3] = vfmaq_f32(bb, p3, bs);
  }
  FP(3);
}

void b24_fe_run(b24_fe *F, const float *pcm, float *img) {
  float32x4_t mn4 = vld1q_f32(pcm), mx4 = mn4;
  for (int i = 0; i < NSAMP; i += 4) {
    float32x4_t v = vld1q_f32(pcm + i);
    mn4 = vminq_f32(mn4, v); mx4 = vmaxq_f32(mx4, v);
  }
  float mn = vminvq_f32(mn4);
  /* the graph: d = x - mn; mx = max(d); y = (d / (mx + 1e-6) - 0.5) * 2 (fp32, same op order) */
  float mx = vmaxvq_f32(mx4) - mn;
  float den = mx + 1e-6f;
  const float32x4_t vmn = vdupq_n_f32(mn), vden = vdupq_n_f32(den), h = vdupq_n_f32(0.5f), two = vdupq_n_f32(2.f);
  for (int i = 0; i < NSAMP; i += 4) {
    float32x4_t d = vsubq_f32(vld1q_f32(pcm + i), vmn);
    vst1q_f32(F->y + i, vmulq_f32(vsubq_f32(vdivq_f32(d, vden), h), two));
  }
  memset(F->y + NSAMP, 0, 64 * sizeof(float));
  for (int t0 = 0; t0 < NFR; t0 += 4) {
    spec_group(F, &F->s[0], t0);
    spec_group(F, &F->s[1], t0);
    int nq = NFR - t0 < 4 ? NFR - t0 : 4;
    for (int j = 0; j < NMEL; j++) {
      float *row = img + ((size_t)(NMEL - 1 - j) * NFR + t0) * 2;
      if (nq == 4) {
        float32x4x2_t v = {{F->P[0][j], F->P[1][j]}};
        vst2q_f32(row, v);
      } else {
        float a[4], b[4];
        vst1q_f32(a, F->P[0][j]); vst1q_f32(b, F->P[1][j]);
        for (int q = 0; q < nq; q++) { row[2 * q] = a[q]; row[2 * q + 1] = b[q]; }
      }
    }
  }
}
