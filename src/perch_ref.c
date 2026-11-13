// fp32 reference ("oracle") for Perch v2: a direct, readable implementation of the ONNX graph in NHWC (front end with
// an exact double FFT, EfficientNet-B3 body, ProtoPNet head), plus an optional fake-quantisation layer (env PERCH_Q)
// for the precision study. Not optimised (on macOS the GEMMs use Accelerate).
//
// PERCH_Q: comma list of key=mode. key = <layer>.a|.w, <class>.a|.w or all.a|all.w, plus store=f16.
//   classes: stem ex dw se pj top proto      layers: stem m<i>.ex m<i>.dw m<i>.se m<i>.pj top proto
//   activation (GEMM input) modes: f32 f16 h16 (the fp16 kernel's arithmetic: fp16 FMA, K blocks <= PERCH_FLUSH (128)
//     summed in fp32; for dw: fp16 FMA over the taps) i8row (per-row asym) i8tens (per-tensor asym) i8sym (per-row
//     symmetric) i8tsym (per-tensor symmetric)
//   weight modes: f32 f16 i8 (per-output-channel symmetric) i8t (per-tensor symmetric)
//   store=f16: every tensor between layers is rounded to fp16 (the fast engine's activation storage)
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif
#include "perch_internal.h"

/* ---------------- fake quantisation config ---------------- */
enum { Q_F32 = 0, Q_F16, Q_H16, Q_I8ROW, Q_I8TENS, Q_I8SYM, Q_I8TSYM, Q_I8, Q_I8T };
typedef struct { char key[40]; int mode; } qent;
static qent g_q[128];
static int g_nq = -1, g_store16 = 0, g_flush = 128;

static int qmode_of(const char *v) {
  static const char *names[] = {"f32", "f16", "h16", "i8row", "i8tens", "i8sym", "i8tsym", "i8", "i8t"};
  for (int i = 0; i < 9; i++) if (!strcmp(v, names[i])) return i;
  fprintf(stderr, "PERCH_Q: unknown mode %s\n", v);
  exit(2);
}

static void q_parse_once(void) {
  g_nq = 0;
  const char *f = getenv("PERCH_FLUSH");
  if (f) g_flush = atoi(f);
  const char *e = getenv("PERCH_Q");
  if (!e) return;
  char buf[4096];
  snprintf(buf, sizeof buf, "%s", e);
  for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
    char *eq = strchr(t, '=');
    if (!eq) continue;
    *eq = 0;
    if (!strcmp(t, "store")) { g_store16 = !strcmp(eq + 1, "f16"); continue; }
    if (g_nq < 128) { snprintf(g_q[g_nq].key, sizeof g_q[g_nq].key, "%s", t); g_q[g_nq].mode = qmode_of(eq + 1); g_nq++; }
  }
}
static pthread_once_t g_q_once = PTHREAD_ONCE_INIT;
static void q_parse(void) { pthread_once(&g_q_once, q_parse_once); }

static int q_lookup(const char *layer, const char *cls, char aw) {
  q_parse();
  char k1[48], k2[48], k3[8];
  snprintf(k1, sizeof k1, "%s.%c", layer, aw);
  snprintf(k2, sizeof k2, "%s.%c", cls, aw);
  snprintf(k3, sizeof k3, "all.%c", aw);
  for (int i = 0; i < g_nq; i++) if (!strcmp(g_q[i].key, k1)) return g_q[i].mode;
  for (int i = 0; i < g_nq; i++) if (!strcmp(g_q[i].key, k2)) return g_q[i].mode;
  for (int i = 0; i < g_nq; i++) if (!strcmp(g_q[i].key, k3)) return g_q[i].mode;
  return Q_F32;
}

static inline float r16(float x) { return (float)(__fp16)x; }
static void store_round(float *x, size_t n) {
  if (!g_store16) return;
  for (size_t i = 0; i < n; i++) x[i] = r16(x[i]);
}

static void asym_params(float mn, float mx, float *s, float *zp) {
  if (mn > 0) mn = 0;
  if (mx < 0) mx = 0;
  float sc = (mx - mn) / 255.f;
  if (!(sc > 1e-12f)) sc = 1e-12f;
  *s = sc;
  *zp = -128.f - roundf(mn / sc);
}
static inline float fq_asym(float x, float s, float zp) {
  float q = roundf(x / s) + zp;
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (q - zp) * s;
}
static inline float fq_sym(float x, float s) {
  float q = roundf(x / s);
  if (q > 127) q = 127;
  if (q < -127) q = -127;
  return q * s;
}

/* quantise a GEMM input matrix A [M][K] in place */
static void fq_act(float *A, int M, int K, int mode) {
  size_t n = (size_t)M * K;
  if (mode == Q_F32) return;
  if (mode == Q_F16 || mode == Q_H16) { for (size_t i = 0; i < n; i++) A[i] = r16(A[i]); return; }
  if (mode == Q_I8TENS || mode == Q_I8TSYM) {
    float mn = A[0], mx = A[0];
    for (size_t i = 0; i < n; i++) { if (A[i] < mn) mn = A[i]; if (A[i] > mx) mx = A[i]; }
    if (mode == Q_I8TSYM) {
      float s = fmaxf(fabsf(mn), fabsf(mx)) / 127.f;
      if (!(s > 1e-12f)) s = 1e-12f;
      for (size_t i = 0; i < n; i++) A[i] = fq_sym(A[i], s);
      return;
    }
    float s, zp;
    asym_params(mn, mx, &s, &zp);
    for (size_t i = 0; i < n; i++) A[i] = fq_asym(A[i], s, zp);
    return;
  }
  for (int m = 0; m < M; m++) {
    float *a = A + (size_t)m * K;
    float mn = a[0], mx = a[0];
    for (int k = 0; k < K; k++) { if (a[k] < mn) mn = a[k]; if (a[k] > mx) mx = a[k]; }
    if (mode == Q_I8ROW) {
      float s, zp;
      asym_params(mn, mx, &s, &zp);
      for (int k = 0; k < K; k++) a[k] = fq_asym(a[k], s, zp);
    } else { /* Q_I8SYM */
      float s = fmaxf(fabsf(mn), fabsf(mx)) / 127.f;
      if (!(s > 1e-12f)) s = 1e-12f;
      for (int k = 0; k < K; k++) a[k] = fq_sym(a[k], s);
    }
  }
}

/* weights W [N][K] -> quantised copy (caller frees) */
static float *fq_w(const float *W, int N, int K, int mode) {
  float *o = malloc(sizeof(float) * (size_t)N * K);
  memcpy(o, W, sizeof(float) * (size_t)N * K);
  if (mode == Q_F16) for (size_t i = 0; i < (size_t)N * K; i++) o[i] = r16(o[i]);
  if (mode == Q_I8 || mode == Q_I8T) {
    float tm = 0;
    if (mode == Q_I8T) for (size_t i = 0; i < (size_t)N * K; i++) tm = fmaxf(tm, fabsf(o[i]));
    for (int n = 0; n < N; n++) {
      float am = tm;
      if (mode == Q_I8) for (int k = 0; k < K; k++) am = fmaxf(am, fabsf(o[(size_t)n * K + k]));
      float s = am / 127.f;
      if (!(s > 0)) continue;
      for (int k = 0; k < K; k++) o[(size_t)n * K + k] = fq_sym(o[(size_t)n * K + k], s);
    }
  }
  return o;
}

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#include <arm_fp16.h>
static inline __fp16 fma16(__fp16 acc, __fp16 a, __fp16 w) { return vfmah_f16(acc, a, w); }
#else /* the fp32 product of two fp16 values is exact; one rounding of the double sum to fp16 */
static inline __fp16 fma16(__fp16 acc, __fp16 a, __fp16 w) { return (__fp16)((double)acc + (double)a * (double)w); }
#endif

/* ---------------- primitives ---------------- */
static inline float swish(float x) { return x / (1.f + expf(-x)); }
static inline float sigm(float x) { return 1.f / (1.f + expf(-x)); }

/* C[M][N] = A[M][K] * W[N][K]^T (+ b) */
static void gemm_nt(const float *A, int M, int K, const float *W, int N, const float *b, float *C) {
#ifdef __APPLE__
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.f, A, K, W, K, 0.f, C, N);
  if (b) for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) C[(size_t)m * N + n] += b[n];
#else
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      const float *a = A + (size_t)m * K, *w = W + (size_t)n * K;
      float s = 0;
      for (int k = 0; k < K; k++) s += a[k] * w[k];
      C[(size_t)m * N + n] = s + (b ? b[n] : 0.f);
    }
#endif
}

/* the fp16 GEMM kernel's arithmetic (gemm_f16.c): K padded to 8, nblk balanced blocks of <= g_flush steps, fp16 FMA
 * (k ascending) per block; one block: fp16 bias add; several: fp32 sum of the blocks + fp32 bias -> fp16 */
static void gemm_h16(const float *A, int M, int K, const float *W, int N, const float *b, float *C) {
  int Kp = (K + 7) & ~7, nblk = (Kp + g_flush - 1) / g_flush, bs = ((Kp + nblk - 1) / nblk + 7) & ~7;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      const float *a = A + (size_t)m * K, *w = W + (size_t)n * K;
      float tot = 0;
      __fp16 acc = 0;
      for (int kb = 0; kb < K; kb += bs) {
        acc = 0;
        for (int k = kb; k < kb + bs && k < K; k++) acc = fma16(acc, (__fp16)a[k], (__fp16)w[k]);
        tot += (float)acc;
      }
      float bb = b ? b[n] : 0.f;
      C[(size_t)m * N + n] = nblk == 1 ? (float)(__fp16)((float)acc + (float)(__fp16)bb) : (float)(__fp16)(tot + bb);
    }
}

/* quantised GEMM layer (A is copied; the quantiser mutates it) */
static void qgemm(const char *layer, const char *cls, const float *A, int M, int K, const float *W, int N,
                  const float *b, float *C) {
  int am = q_lookup(layer, cls, 'a'), wm = q_lookup(layer, cls, 'w');
  float *Aq = malloc(sizeof(float) * (size_t)M * K);
  memcpy(Aq, A, sizeof(float) * (size_t)M * K);
  fq_act(Aq, M, K, am);
  float *Wq = (wm == Q_F32) ? (float *)W : fq_w(W, N, K, wm);
  if (am == Q_H16) gemm_h16(Aq, M, K, Wq, N, b, C);
  else gemm_nt(Aq, M, K, Wq, N, b, C);
  if (Wq != W) free(Wq);
  free(Aq);
}

/* ---------------- front end ---------------- */
/* exact N-point DFT of a real frame (double, radix-2): re/im of bins [0, nb) */
static void dft_fft(const double *x, int N, double *re_out, double *im_out, int nb) {
  double *re = malloc(sizeof(double) * N), *im = malloc(sizeof(double) * N);
  int lg = 0;
  while ((1 << lg) < N) lg++;
  for (int i = 0; i < N; i++) {
    int r = 0;
    for (int b = 0; b < lg; b++) r |= ((i >> b) & 1) << (lg - 1 - b);
    re[r] = x[i]; im[r] = 0;
  }
  for (int len = 2; len <= N; len <<= 1) {
    double a = -2 * M_PI / len;
    for (int i = 0; i < N; i += len)
      for (int j = 0; j < len / 2; j++) {
        double wr = cos(a * j), wi = sin(a * j);
        double ur = re[i + j], ui = im[i + j];
        double vr = re[i + j + len / 2] * wr - im[i + j + len / 2] * wi;
        double vi = re[i + j + len / 2] * wi + im[i + j + len / 2] * wr;
        re[i + j] = ur + vr; im[i + j] = ui + vi;
        re[i + j + len / 2] = ur - vr; im[i + j + len / 2] = ui - vi;
      }
  }
  for (int k = 0; k < nb; k++) { re_out[k] = re[k]; im_out[k] = im[k]; }
  free(re); free(im);
}

void pc_ref_frontend(const pc_model *m, const float *pcm, float *spec) {
  double f[PC_NFFT], re[PC_BINS], im[PC_BINS], mag[PC_BINS];
  for (int t = 0; t < PC_NFR; t++) {
    for (int n = 0; n < PC_NFFT; n++) {
      int i = t * PC_HOP - PC_PAD + n;
      f[n] = (n < PC_FRAME && i >= 0 && i < PC_SAMPLES) ? (double)(pcm[i] * m->win[n]) : 0.0;
    }
    dft_fft(f, PC_NFFT, re, im, PC_BINS);
    for (int k = 0; k < PC_BINS; k++) mag[k] = sqrt(re[k] * re[k] + im[k] * im[k]);
    for (int j = 0; j < PC_NMEL; j++) {
      double a = 0;
      for (int k = m->mel_lo[j]; k < m->mel_hi[j]; k++) a += mag[k] * m->mel[(size_t)j * PC_BINS + k];
      float v = (float)a;
      spec[t * PC_NMEL + j] = m->log_scale * logf(v > m->log_floor ? v : m->log_floor);
    }
  }
}

/* ---------------- body ---------------- */
struct pc_ref {
  const pc_model *m;
  float *a, *b, *c, *col;
};

#define MAXT ((size_t)249 * 63 * 144)

pc_ref *pc_ref_new(const pc_model *m) {
  pc_ref *r = calloc(1, sizeof *r);
  r->m = m;
  r->a = malloc(sizeof(float) * MAXT);
  r->b = malloc(sizeof(float) * MAXT);
  r->c = malloc(sizeof(float) * MAXT);
  size_t colsz = (size_t)PC_CELLS * m->n_proto;
  if (colsz < MAXT) colsz = MAXT;
  r->col = malloc(sizeof(float) * colsz);
  return r;
}
void pc_ref_free(pc_ref *r) {
  if (!r) return;
  free(r->a); free(r->b); free(r->c); free(r->col); free(r);
}

/* depthwise kxk, stride s, pads (pt, pl) (bottom/right implied by the output size) + swish; [H][W][C] -> [Ho][Wo][C].
 * h16: fp16 FMA over the taps (ky, kx ascending) starting from the fp16 bias (the fast engine's dw kernel). */
static void dwconv(const char *layer, const float *x, int H, int W, int C, int k, int s, int pt, int pl,
                   const float *w, const float *bias, float *out, int Ho, int Wo) {
  int am = q_lookup(layer, "dw", 'a'), wm = q_lookup(layer, "dw", 'w');
  const float *xin = x;
  float *xq = NULL;
  if (am != Q_F32 && am != Q_H16) {
    xq = malloc(sizeof(float) * (size_t)H * W * C);
    memcpy(xq, x, sizeof(float) * (size_t)H * W * C);
    fq_act(xq, H * W, C, am);
    xin = xq;
  }
  float *wq = (wm == Q_F32) ? (float *)w : fq_w(w, k * k, C, wm);
  for (int oy = 0; oy < Ho; oy++)
    for (int ox = 0; ox < Wo; ox++)
      for (int c = 0; c < C; c++) {
        float acc = bias[c];
        __fp16 h = (__fp16)bias[c];
        for (int ky = 0; ky < k; ky++) {
          int iy = oy * s - pt + ky;
          if (iy < 0 || iy >= H) continue;
          for (int kx = 0; kx < k; kx++) {
            int ix = ox * s - pl + kx;
            if (ix < 0 || ix >= W) continue;
            float xv = xin[((size_t)iy * W + ix) * C + c], wv = wq[(ky * k + kx) * C + c];
            if (am == Q_H16) h = fma16(h, (__fp16)xv, (__fp16)wv);
            else acc += wv * xv;
          }
        }
        out[((size_t)oy * Wo + ox) * C + c] = swish(am == Q_H16 ? (float)h : acc);
      }
  if (wq != w) free(wq);
  free(xq);
}

/* ProtoPNet head on the normalised cells xh [64][1536]: m_j = max_p xh_p . P_j, logit_c = sum_q w_cq m_{c ppc + q} + b_c.
 * The similarity GEMM is done in chunks of prototypes (the full 64 x 59180 matrix is 15 MB; fine, but chunking keeps
 * the h16 emulation cache-friendly). */
static void proto_head(pc_ref *r, const float *xh, float *logits) {
  const pc_model *m = r->m;
  int am = q_lookup("proto", "proto", 'a'), wm = q_lookup("proto", "proto", 'w');
  float *xq = malloc(sizeof(float) * PC_CELLS * PC_EMB);
  memcpy(xq, xh, sizeof(float) * PC_CELLS * PC_EMB);
  fq_act(xq, PC_CELLS, PC_EMB, am);
  float *mx = malloc(sizeof(float) * m->n_proto);
  const int CH = 4096;
  for (int j0 = 0; j0 < m->n_proto; j0 += CH) {
    int nj = m->n_proto - j0 < CH ? m->n_proto - j0 : CH;
    const float *Pw = m->proto + (size_t)j0 * PC_EMB;
    float *Pq = (wm == Q_F32) ? (float *)Pw : fq_w(Pw, nj, PC_EMB, wm);
    float *S = r->col; /* [64][nj] */
    if (am == Q_H16) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
      /* vectorised over 8 prototypes, same per-element arithmetic as gemm_h16 (fp16 FMA, k ascending, blocks) */
      int Kp = PC_EMB, nblk = (Kp + g_flush - 1) / g_flush, bs = ((Kp + nblk - 1) / nblk + 7) & ~7;
      __fp16 *xh16 = malloc(sizeof(__fp16) * PC_CELLS * PC_EMB), *pt16 = malloc(sizeof(__fp16) * (size_t)PC_EMB * 8);
      for (int i = 0; i < PC_CELLS * PC_EMB; i++) xh16[i] = (__fp16)xq[i];
      for (int j = 0; j < nj; j += 8) {
        int nn = nj - j < 8 ? nj - j : 8;
        for (int k = 0; k < PC_EMB; k++)
          for (int q = 0; q < 8; q++) pt16[k * 8 + q] = (__fp16)(q < nn ? Pq[(size_t)(j + q) * PC_EMB + k] : 0.f);
        for (int p = 0; p < PC_CELLS; p++) {
          float32x4_t t0 = vdupq_n_f32(0), t1 = t0;
          float16x8_t acc = vdupq_n_f16(0);
          for (int kb = 0; kb < PC_EMB; kb += bs) {
            acc = vdupq_n_f16(0);
            for (int k = kb; k < kb + bs && k < PC_EMB; k++)
              acc = vfmaq_n_f16(acc, vld1q_f16(pt16 + k * 8), xh16[p * PC_EMB + k]);
            t0 = vaddq_f32(t0, vcvt_f32_f16(vget_low_f16(acc)));
            t1 = vaddq_f32(t1, vcvt_high_f32_f16(acc));
          }
          float o[8];
          if (nblk == 1) { vst1q_f32(o, vcvt_f32_f16(vget_low_f16(acc))); vst1q_f32(o + 4, vcvt_high_f32_f16(acc)); }
          else {
            float16x8_t h = vcombine_f16(vcvt_f16_f32(t0), vcvt_f16_f32(t1));
            vst1q_f32(o, vcvt_f32_f16(vget_low_f16(h))); vst1q_f32(o + 4, vcvt_high_f32_f16(h));
          }
          for (int q = 0; q < nn; q++) S[(size_t)p * nj + j + q] = o[q];
        }
      }
      free(xh16); free(pt16);
#else
      gemm_h16(xq, PC_CELLS, PC_EMB, Pq, nj, NULL, S);
#endif
    } else
      gemm_nt(xq, PC_CELLS, PC_EMB, Pq, nj, NULL, S);
    for (int j = 0; j < nj; j++) {
      float v = S[j];
      for (int p = 1; p < PC_CELLS; p++) if (S[(size_t)p * nj + j] > v) v = S[(size_t)p * nj + j];
      mx[j0 + j] = v;
    }
    if (Pq != Pw) free(Pq);
  }
  for (int c = 0; c < m->n_classes; c++) {
    float s = 0;
    for (int q = 0; q < m->ppc; q++) s += mx[c * m->ppc + q] * m->cls_w[c * m->ppc + q];
    logits[c] = s + m->cls_b[c];
  }
  free(mx); free(xq);
}

int pc_ref_infer(pc_ref *r, const float *pcm, float *logits, float *emb) {
  const pc_model *m = r->m;
  q_parse();
  float *spec = r->c; /* [500][128] */
  pc_ref_frontend(m, pcm, spec);
  /* stem 3x3 stride 2 VALID, 1 -> 40, + swish: im2col [249*63][9] */
  int H = (PC_NFR - 3) / 2 + 1, W = (PC_NMEL - 3) / 2 + 1;
  float *col = r->col;
  for (int oy = 0; oy < H; oy++)
    for (int ox = 0; ox < W; ox++)
      for (int ky = 0; ky < 3; ky++)
        for (int kx = 0; kx < 3; kx++) col[((size_t)oy * W + ox) * 9 + ky * 3 + kx] = spec[(2 * oy + ky) * PC_NMEL + 2 * ox + kx];
  qgemm("stem", "stem", col, H * W, 9, m->stem_w, m->stem_c, m->stem_b, r->a);
  for (size_t i = 0; i < (size_t)H * W * m->stem_c; i++) r->a[i] = swish(r->a[i]);
  store_round(r->a, (size_t)H * W * m->stem_c);
  float *x = r->a, *h = r->b, *d = r->c;
  char nm[32];
  for (int i = 0; i < m->n_blocks; i++) {
    const pc_block *b = &m->blocks[i];
    int Ho = b->hout, Wo = b->wout, ce = b->cexp;
    H = b->hin; W = b->win;
    const float *dwin = x;
    if (b->has_ex) {
      snprintf(nm, sizeof nm, "m%d.ex", i);
      qgemm(nm, "ex", x, H * W, b->cin, b->ex_w, ce, b->ex_b, h);
      for (size_t k = 0; k < (size_t)H * W * ce; k++) h[k] = swish(h[k]);
      store_round(h, (size_t)H * W * ce);
      dwin = h;
    }
    snprintf(nm, sizeof nm, "m%d.dw", i);
    dwconv(nm, dwin, H, W, ce, b->k, b->stride, b->pt, b->pl, b->dw_w, b->dw_b, d, Ho, Wo);
    store_round(d, (size_t)Ho * Wo * ce);
    {
      float mean[4096], s1[512], s2[4096];
      for (int c = 0; c < ce; c++) mean[c] = 0;
      for (int p = 0; p < Ho * Wo; p++) for (int c = 0; c < ce; c++) mean[c] += d[(size_t)p * ce + c];
      for (int c = 0; c < ce; c++) mean[c] /= (float)(Ho * Wo);
      snprintf(nm, sizeof nm, "m%d.se", i);
      qgemm(nm, "se", mean, 1, ce, b->se1_w, b->se_rd, b->se1_b, s1);
      for (int k = 0; k < b->se_rd; k++) s1[k] = swish(s1[k]);
      qgemm(nm, "se", s1, 1, b->se_rd, b->se2_w, ce, b->se2_b, s2);
      for (int c = 0; c < ce; c++) s2[c] = sigm(s2[c]);
      for (int p = 0; p < Ho * Wo; p++) for (int c = 0; c < ce; c++) d[(size_t)p * ce + c] *= s2[c];
      store_round(d, (size_t)Ho * Wo * ce);
    }
    snprintf(nm, sizeof nm, "m%d.pj", i);
    float *o = (dwin == h) ? h : h; /* h is free here in both cases (the dw input was x or h, consumed) */
    qgemm(nm, "pj", d, Ho * Wo, ce, b->pj_w, b->cout, b->pj_b, o);
    if (b->skip) for (size_t k = 0; k < (size_t)Ho * Wo * b->cout; k++) o[k] += x[k];
    store_round(o, (size_t)Ho * Wo * b->cout);
    float *t = x; x = o; h = t;
  }
  /* top conv 1x1 384 -> 1536 + swish = spatial embedding [64][1536]; embedding = mean */
  float *se = h;
  qgemm("top", "top", x, PC_CELLS, 384, m->top_w, PC_EMB, m->top_b, se);
  for (size_t k = 0; k < (size_t)PC_CELLS * PC_EMB; k++) se[k] = swish(se[k]);
  store_round(se, (size_t)PC_CELLS * PC_EMB);
  if (emb) {
    for (int c = 0; c < PC_EMB; c++) {
      float s = 0;
      for (int p = 0; p < PC_CELLS; p++) s += se[p * PC_EMB + c];
      emb[c] = s / (float)PC_CELLS;
    }
  }
  /* normalise each cell: x / (||x|| + eps) */
  float *xh = d;
  for (int p = 0; p < PC_CELLS; p++) {
    double ss = 0;
    for (int c = 0; c < PC_EMB; c++) ss += (double)se[p * PC_EMB + c] * se[p * PC_EMB + c];
    float den = (float)sqrt(ss) + m->norm_eps;
    for (int c = 0; c < PC_EMB; c++) xh[p * PC_EMB + c] = se[p * PC_EMB + c] / den;
  }
  proto_head(r, xh, logits);
  return 0;
}
