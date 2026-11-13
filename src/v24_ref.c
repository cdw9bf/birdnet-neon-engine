// fp32 reference ("oracle") for BirdNET v2.4: a direct, readable implementation of the ONNX graph in NHWC, plus an
// optional fake-quantisation layer (env BN24_Q) used for the precision study. Not optimised (on macOS the GEMMs use
// Accelerate so that sweeps over a few hundred windows are quick).
//
// BN24_Q: comma list of key=mode. key = <layer name>.a|.w, <class>.a|.w or all.a|all.w, plus store=f16 and fe=fft|dft.
//   classes: c0 p0 pw dw se pwl pc head      layer names: c0 p0 b{s}.{i}.{pw|dw|pwl|se} pc head
//   activation (GEMM input) modes: f32 f16 i8row (per-pixel asym, dynamic) i8tens (per-tensor asym, dynamic)
//                                  i8sym (per-pixel symmetric)  i8c (per-input-channel asym, dynamic, over the tensor)
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
#include "v24_internal.h"

/* ---------------- fake quantisation config ---------------- */
enum { Q_F32 = 0, Q_F16, Q_I8ROW, Q_I8TENS, Q_I8SYM, Q_I8C, Q_I8, Q_I8T, Q_H16 };
typedef struct { char key[40]; int mode; } qent;
static qent g_q[128];
static int g_nq = -1, g_store16 = 0, g_fe_fft = 0;

static int qmode_of(const char *v) {
  static const char *names[] = {"f32", "f16", "i8row", "i8tens", "i8sym", "i8c", "i8", "i8t", "h16"};
  for (int i = 0; i < 9; i++) if (!strcmp(v, names[i])) return i;
  fprintf(stderr, "BN24_Q: unknown mode %s\n", v);
  exit(2);
}

static void q_parse_once(void);
static pthread_once_t g_q_once = PTHREAD_ONCE_INIT;
/* the fake-quantisation config is parsed once per process (thread-safe) */
static void q_parse(void) { pthread_once(&g_q_once, q_parse_once); }
static void q_parse_once(void) {
  g_nq = 0;
  const char *e = getenv("BN24_Q");
  if (!e) return;
  char buf[4096];
  snprintf(buf, sizeof buf, "%s", e);
  for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
    char *eq = strchr(t, '=');
    if (!eq) continue;
    *eq = 0;
    if (!strcmp(t, "store")) { g_store16 = !strcmp(eq + 1, "f16"); continue; }
    if (!strcmp(t, "fe")) { g_fe_fft = !strcmp(eq + 1, "fft"); continue; }
    if (g_nq < 128) { snprintf(g_q[g_nq].key, sizeof g_q[g_nq].key, "%s", t); g_q[g_nq].mode = qmode_of(eq + 1); g_nq++; }
  }
}

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

/* quantise a GEMM input matrix A [M][K] (in place) */
static void fq_act(float *A, int M, int K, int mode) {
  if (mode == Q_F32) return;
  if (mode == Q_F16 || mode == Q_H16) { for (size_t i = 0; i < (size_t)M * K; i++) A[i] = r16(A[i]); return; }
  if (mode == Q_I8TENS) {
    float mn = A[0], mx = A[0];
    for (size_t i = 0; i < (size_t)M * K; i++) { if (A[i] < mn) mn = A[i]; if (A[i] > mx) mx = A[i]; }
    float s, zp;
    asym_params(mn, mx, &s, &zp);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = fq_asym(A[i], s, zp);
    return;
  }
  if (mode == Q_I8C) {
    for (int k = 0; k < K; k++) {
      float mn = A[k], mx = A[k];
      for (int m = 0; m < M; m++) { float v = A[(size_t)m * K + k]; if (v < mn) mn = v; if (v > mx) mx = v; }
      float s, zp;
      asym_params(mn, mx, &s, &zp);
      for (int m = 0; m < M; m++) A[(size_t)m * K + k] = fq_asym(A[(size_t)m * K + k], s, zp);
    }
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
      float am = fmaxf(fabsf(mn), fabsf(mx)) / 127.f;
      if (!(am > 1e-12f)) am = 1e-12f;
      for (int k = 0; k < K; k++) { float q = roundf(a[k] / am); if (q > 127) q = 127; if (q < -127) q = -127; a[k] = q * am; }
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
      for (int k = 0; k < K; k++) o[(size_t)n * K + k] = roundf(o[(size_t)n * K + k] / s) * s;
    }
  }
  return o;
}

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#include <arm_fp16.h>
static inline __fp16 fma16(__fp16 acc, __fp16 a, __fp16 w) { return vfmah_f16(acc, a, w); }
#else /* exact: the fp32 product of two fp16 values is exact, so double-rounding only via the fp32 add */
static inline __fp16 fma16(__fp16 acc, __fp16 a, __fp16 w) { return (__fp16)((double)acc + (double)a * (double)w); }
#endif

/* ---------------- primitives ---------------- */
static inline float silu(float x) { return x / (1.f + expf(-x)); }
static inline float sigm(float x) { return 1.f / (1.f + expf(-x)); }

/* C[M][N] = A[M][K] * W[N][K]^T + b */
static void gemm_nt(const float *A, int M, int K, const float *W, int N, const float *b, float *C) {
#ifdef __APPLE__
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.f, A, K, W, K, 0.f, C, N);
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) C[(size_t)m * N + n] += b[n];
#else
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      const float *a = A + (size_t)m * K, *w = W + (size_t)n * K;
      float s = 0;
      for (int k = 0; k < K; k++) s += a[k] * w[k];
      C[(size_t)m * N + n] = s + b[n];
    }
#endif
}

/* quantised GEMM layer: copies A (the quantiser mutates), quantises weights */
static void qgemm(const char *layer, const char *cls, const float *A, int M, int K, const float *W, int N,
                  const float *b, float *C) {
  int am = q_lookup(layer, cls, 'a'), wm = q_lookup(layer, cls, 'w');
  float *Aq = malloc(sizeof(float) * (size_t)M * K);
  memcpy(Aq, A, sizeof(float) * (size_t)M * K);
  fq_act(Aq, M, K, am);
  float *Wq = (wm == Q_F32) ? (float *)W : fq_w(W, N, K, wm);
  if (am == Q_H16) {
    /* the fp16 GEMM kernel's arithmetic: K (padded to 8) split into nblk balanced blocks of <= BN24_FLUSH steps
       (default 128, gemm_f16.h FLUSH_K); fp16 FMA accumulation (one rounding per FMA, k ascending) per block;
       one block: fp16 bias add; several: fp32 sum of the blocks + fp32 bias, then rounded to fp16 */
    static int fmax_ = -1;
    if (fmax_ < 0) { const char *e = getenv("BN24_FLUSH"); fmax_ = e ? atoi(e) : 128; }
    int Kp = (K + 7) & ~7, nblk = (Kp + fmax_ - 1) / fmax_, bs = ((Kp + nblk - 1) / nblk + 7) & ~7;
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        const float *a = Aq + (size_t)m * K, *w = Wq + (size_t)n * K;
        float tot = 0;
        __fp16 acc = 0;
        for (int kb = 0; kb < K; kb += bs) {
          acc = 0;
          for (int k = kb; k < kb + bs && k < K; k++) acc = fma16(acc, (__fp16)a[k], (__fp16)w[k]);
          tot += (float)acc;
        }
        C[(size_t)m * N + n] = nblk == 1 ? (float)(__fp16)((float)acc + (float)(__fp16)b[n]) : (float)(__fp16)(tot + b[n]);
      }
  } else
    gemm_nt(Aq, M, K, Wq, N, b, C);
  if (Wq != W) free(Wq);
  free(Aq);
}

/* ---------------- front end ---------------- */
/* exact real part of the N-point DFT of a real frame (double, radix-2), bins [0, nb) */
static void dft_cos_fft(const double *x, int N, double *re_out, int nb) {
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
  for (int k = 0; k < nb; k++) re_out[k] = re[k];
  free(re); free(im);
}

void b24_ref_frontend(const b24_model *m, const float *pcm, float *img, int use_fft) {
  /* normalisation exactly as the graph (fp32) */
  float mn = pcm[0];
  for (int i = 1; i < NSAMP; i++) if (pcm[i] < mn) mn = pcm[i];
  float mx = pcm[0] - mn;
  float *y = malloc(sizeof(float) * NSAMP);
  for (int i = 0; i < NSAMP; i++) { y[i] = pcm[i] - mn; if (y[i] > mx) mx = y[i]; }
  float den = mx + 1e-6f;
  for (int i = 0; i < NSAMP; i++) y[i] = (y[i] / den - 0.5f) * 2.f;
  for (int s = 0; s < 2; s++) {
    int N = s ? N2 : N1, hop = s ? HOP2 : HOP1, nb = s ? B2 : B1;
    const float *win = s ? m->win2 : m->win1, *D = s ? m->dft2 : m->dft1, *mel = s ? m->mel2 : m->mel1;
    const int *lo = s ? m->mel2_lo : m->mel1_lo, *hi = s ? m->mel2_hi : m->mel1_hi;
    float pw = s ? m->pow2[0] : m->pow1[0];
    double *f = malloc(sizeof(double) * N), *R = malloc(sizeof(double) * nb);
    for (int t = 0; t < NFR; t++) {
      for (int n = 0; n < N; n++) f[n] = (double)(y[t * hop + n] * win[n]);
      if (use_fft) dft_cos_fft(f, N, R, nb);
      else {
        int khi = 0;
        for (int j = 0; j < NMEL; j++) if (hi[j] > khi) khi = hi[j];
        for (int k = 0; k < nb; k++) R[k] = 0;
        for (int n = 0; n < N; n++) {
          const float *d = D + (size_t)n * nb;
          for (int k = 0; k < khi; k++) R[k] += f[n] * d[k];
        }
      }
      for (int j = 0; j < NMEL; j++) {
        double a = 0;
        for (int k = lo[j]; k < hi[j]; k++) a += R[k] * mel[(size_t)j * nb + k];
        float v = (float)a;
        v = powf(v * v, pw);
        img[((size_t)(NMEL - 1 - j) * NFR + t) * 2 + s] = v * m->fe_bn_s[s] + m->fe_bn_b[s];
      }
    }
    free(f); free(R);
  }
  free(y);
}

/* ---------------- body ---------------- */
struct b24_ref {
  const b24_model *m;
  float *a, *b, *c, *col;
};

#define MAXT (48 * 256 * 72)

b24_ref *b24_ref_new(const b24_model *m) {
  b24_ref *r = calloc(1, sizeof *r);
  r->m = m;
  r->a = malloc(sizeof(float) * MAXT);
  r->b = malloc(sizeof(float) * MAXT);
  r->c = malloc(sizeof(float) * MAXT);
  r->col = malloc(sizeof(float) * MAXT * 2);
  return r;
}
void b24_ref_free(b24_ref *r) {
  if (!r) return;
  free(r->a); free(r->b); free(r->c); free(r->col); free(r);
}

/* depthwise 3x3, pad 1, stride s, + SiLU; in [H][W][C] -> out [Ho][Wo][C] */
static void dw3x3(const char *layer, const float *x, int H, int W, int C, int s, const float *w, const float *bias,
                  float *out, int Ho, int Wo) {
  int am = q_lookup(layer, "dw", 'a'), wm = q_lookup(layer, "dw", 'w');
  const float *xin = x;
  float *xq = NULL;
  if (am != Q_F32) {
    xq = malloc(sizeof(float) * (size_t)H * W * C);
    memcpy(xq, x, sizeof(float) * (size_t)H * W * C);
    fq_act(xq, H * W, C, am);
    xin = xq;
  }
  float *wq = (wm == Q_F32) ? (float *)w : fq_w(w, 9, C, wm); /* per-tap scales for i8 (exploration only) */
  for (int oy = 0; oy < Ho; oy++)
    for (int ox = 0; ox < Wo; ox++)
      for (int c = 0; c < C; c++) {
        float acc = bias[c];
        for (int ky = 0; ky < 3; ky++) {
          int iy = oy * s - 1 + ky;
          if (iy < 0 || iy >= H) continue;
          for (int kx = 0; kx < 3; kx++) {
            int ix = ox * s - 1 + kx;
            if (ix < 0 || ix >= W) continue;
            acc += wq[(ky * 3 + kx) * C + c] * xin[((size_t)iy * W + ix) * C + c];
          }
        }
        out[((size_t)oy * Wo + ox) * C + c] = silu(acc);
      }
  if (wq != w) free(wq);
  free(xq);
}

int b24_ref_infer(b24_ref *r, const float *pcm, float *logits, float *emb) {
  const b24_model *m = r->m;
  q_parse();
  float *img = r->col; /* [96][511][2] */
  b24_ref_frontend(m, pcm, img, g_fe_fft);
  /* c0: 4x8 stride 2, SAME_UPPER (pad top 1, left 3) as im2col [48*256][4*8*2] */
  float *col = r->col + 96 * 511 * 2;
  for (int oy = 0; oy < 48; oy++)
    for (int ox = 0; ox < 256; ox++) {
      float *cp = col + ((size_t)oy * 256 + ox) * 64;
      for (int ky = 0; ky < 4; ky++)
        for (int kx = 0; kx < 8; kx++) {
          int iy = 2 * oy - 1 + ky, ix = 2 * ox - 3 + kx;
          for (int c = 0; c < 2; c++)
            cp[(ky * 8 + kx) * 2 + c] = (iy < 0 || iy >= 96 || ix < 0 || ix >= 511) ? 0.f : img[((size_t)iy * 511 + ix) * 2 + c];
        }
    }
  qgemm("c0", "c0", col, 48 * 256, 64, m->c0_w, 24, m->c0_b, r->a);
  for (size_t i = 0; i < 48 * 256 * 24; i++) r->a[i] = fmaxf(r->a[i], 0.f);
  store_round(r->a, 48 * 256 * 24);
  /* pools along W: [48][128][48] = [max 24 | avg 24] */
  for (int y = 0; y < 48; y++)
    for (int x = 0; x < 128; x++)
      for (int c = 0; c < 24; c++) {
        float u = r->a[((size_t)y * 256 + 2 * x) * 24 + c], v = r->a[((size_t)y * 256 + 2 * x + 1) * 24 + c];
        r->b[((size_t)y * 128 + x) * 48 + c] = fmaxf(u, v);
        r->b[((size_t)y * 128 + x) * 48 + 24 + c] = (u + v) * 0.5f;
      }
  qgemm("p0", "p0", r->b, 48 * 128, 48, m->p0_w, 24, m->p0_b, r->a);
  store_round(r->a, 48 * 128 * 24);
  float *x = r->a, *h = r->b, *d = r->c;
  char nm[32];
  for (int i = 0; i < m->n_blocks; i++) {
    const b24_block *b = &m->blocks[i];
    int H = b->hin, W = b->win, Ho = b->hout, Wo = b->wout, cm = b->cmid;
    snprintf(nm, sizeof nm, "%s.pw", b->tag);
    qgemm(nm, "pw", x, H * W, b->cin, b->pw_w, cm, b->pw_b, h);
    for (size_t k = 0; k < (size_t)H * W * cm; k++) h[k] = silu(h[k]);
    store_round(h, (size_t)H * W * cm);
    snprintf(nm, sizeof nm, "%s.dw", b->tag);
    dw3x3(nm, h, H, W, cm, b->stride, b->dw_w, b->dw_b, d, Ho, Wo);
    store_round(d, (size_t)Ho * Wo * cm);
    if (b->se_rd) {
      float mean[4096], s1[512], s2[4096];
      for (int c = 0; c < cm; c++) mean[c] = 0;
      for (int p = 0; p < Ho * Wo; p++) for (int c = 0; c < cm; c++) mean[c] += d[(size_t)p * cm + c];
      for (int c = 0; c < cm; c++) mean[c] /= (float)(Ho * Wo);
      snprintf(nm, sizeof nm, "%s.se", b->tag);
      qgemm(nm, "se", mean, 1, cm, b->se1_w, b->se_rd, b->se1_b, s1);
      for (int k = 0; k < b->se_rd; k++) s1[k] = silu(s1[k]);
      qgemm(nm, "se", s1, 1, b->se_rd, b->se2_w, cm, b->se2_b, s2);
      for (int c = 0; c < cm; c++) s2[c] = sigm(s2[c]);
      for (int p = 0; p < Ho * Wo; p++) for (int c = 0; c < cm; c++) d[(size_t)p * cm + c] *= s2[c];
      store_round(d, (size_t)Ho * Wo * cm);
    }
    snprintf(nm, sizeof nm, "%s.pwl", b->tag);
    qgemm(nm, "pwl", d, Ho * Wo, cm, b->pwl_w, b->cout, b->pwl_b, h);
    if (b->skip) for (size_t k = 0; k < (size_t)Ho * Wo * b->cout; k++) h[k] += x[k];
    store_round(h, (size_t)Ho * Wo * b->cout);
    float *t = x; x = h; h = t;
  }
  /* post BN + ReLU, 3x3 valid conv 192->1024 + ReLU, mean over the 1x6 map */
  for (int p = 0; p < 24; p++) for (int c = 0; c < 192; c++) {
    float v = x[p * 192 + c] * m->post_s[c] + m->post_b[c];
    x[p * 192 + c] = v > 0 ? v : 0;
  }
  store_round(x, 24 * 192);
  for (int ox = 0; ox < 6; ox++)
    for (int ky = 0; ky < 3; ky++)
      for (int kx = 0; kx < 3; kx++)
        memcpy(r->col + (size_t)ox * 1728 + (ky * 3 + kx) * 192, x + (ky * 8 + ox + kx) * 192, 192 * sizeof(float));
  float *pc = h;
  qgemm("pc", "pc", r->col, 6, 1728, m->pc_w, FEAT_C, m->pc_b, pc);
  float feat[FEAT_C];
  for (int c = 0; c < FEAT_C; c++) {
    float s = 0;
    for (int ox = 0; ox < 6; ox++) s += fmaxf(pc[ox * FEAT_C + c], 0.f);
    feat[c] = s / 6.f;
  }
  if (emb) memcpy(emb, feat, sizeof feat);
  qgemm("head", "head", feat, 1, FEAT_C, m->head_w, m->n_classes, m->head_b, logits);
  return 0;
}
