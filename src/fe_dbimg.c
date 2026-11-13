// dB-image front end (fp32, baseline ARMv8-A): centre padding (reflect | zero) + window + N-point real FFT (an N/2-point
// complex FFT of the even/odd samples, 4 frames per float32x4 lane: no shuffles) + power + sparse mel + Clip floor +
// logf + up to three scalar ops + top-db floor + mel-row selection + min/max normalisation + half-pixel bilinear
// resize + per-channel ops, written NHWC into a zero-haloed image. The arithmetic follows the retired v3 engine's
// front end (frontend.c, removed with that engine) and, where the graph divides, divides (ONNX Runtime's order); every size and constant is data.
#include <arm_neon.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fe_dbimg.h"
#include "fft_tab.h"
#include "rt.h"

struct fe_db {
  fe_db_cfg g;
  int NC, F;              /* complex FFT size N/2, bins N/2 + 1 */
  float *tw_re, *tw_im;   /* e^{-2 pi i k / NC}, k < NC/2 */
  float *sp_re, *sp_im;   /* e^{-2 pi i k / N}, k <= NC */
  int *rev;
  float *win;
  int *mel_off, *mel_k, klo, khi;
  float *mel_w;
  int *rows;
  float *chv;             /* [2][C] */
  int *ry1, *ry2, *rx1, *rx2;
  float *dy1, *dy2, *dx1, *dx2;
  float32x4_t *re, *im, *pw;
  float *db, *sp;         /* [nmel][nfr], [R][nfr] */
};

void fe_db_free(fe_db *f) {
  if (!f) return;
  void *ps[] = {f->tw_re, f->tw_im, f->sp_re, f->sp_im, f->rev, f->win, f->mel_off, f->mel_k, f->mel_w, f->rows, f->chv,
                f->ry1, f->ry2, f->rx1, f->rx2, f->dy1, f->dy2, f->dx1, f->dx2, f->re, f->im, f->pw, f->db, f->sp};
  for (size_t i = 0; i < sizeof ps / sizeof *ps; i++) free(ps[i]);
  free(f);
}

/* ONNX Runtime's linear half_pixel coordinate set-up for one axis (upsample: in = (out + 0.5) / scale - 0.5) */
static void axis_setup(int nin, int nout, int *i1, int *i2, float *d1, float *d2) {
  const float scale = (float)nout / (float)nin;
  for (int o = 0; o < nout; o++) {
    float x = ((float)o + 0.5f) / scale - 0.5f;
    if (x > (float)(nin - 1)) x = (float)(nin - 1);
    if (x < 0.f) x = 0.f;
    int a = (int)x < nin - 1 ? (int)x : nin - 1;
    int b = a + 1 < nin - 1 ? a + 1 : nin - 1;
    i1[o] = a; i2[o] = b;
    d1[o] = fabsf(x - (float)a); d2[o] = fabsf(x - (float)b);
    if (a == b) { d1[o] = 0.5f; d2[o] = 0.5f; }
  }
}

fe_db *fe_db_new(const fe_db_cfg *g, const float *window, const float *mel, const int32_t *rows, const float *chv) {
  fe_db *f = calloc(1, sizeof *f);
  if (!f) return NULL;
  f->g = *g;
  const int N = g->nfft, NC = N / 2, F = NC + 1;
  f->NC = NC; f->F = F;
  f->tw_re = rt_zalloc64_tail(sizeof(float) * NC / 2); f->tw_im = rt_zalloc64_tail(sizeof(float) * NC / 2);
  f->sp_re = rt_zalloc64_tail(sizeof(float) * (NC + 1)); f->sp_im = rt_zalloc64_tail(sizeof(float) * (NC + 1));
  f->rev = rt_zalloc64_tail(sizeof(int) * NC); f->win = rt_zalloc64_tail(sizeof(float) * N);
  f->mel_off = rt_zalloc64_tail(sizeof(int) * (g->nmel + 1));
  int nnz = 0;
  for (size_t i = 0; i < (size_t)g->nmel * F; i++) nnz += mel[i] != 0.f;
  f->mel_k = rt_zalloc64_tail(sizeof(int) * (nnz + 1)); f->mel_w = rt_zalloc64_tail(sizeof(float) * (nnz + 1));
  f->rows = rt_zalloc64_tail(sizeof(int) * g->R); f->chv = rt_zalloc64_tail(sizeof(float) * 2 * g->C);
  f->ry1 = rt_zalloc64_tail(sizeof(int) * g->Ho); f->ry2 = rt_zalloc64_tail(sizeof(int) * g->Ho); f->dy1 = rt_zalloc64_tail(4 * g->Ho); f->dy2 = rt_zalloc64_tail(4 * g->Ho);
  f->rx1 = rt_zalloc64_tail(sizeof(int) * g->Wo); f->rx2 = rt_zalloc64_tail(sizeof(int) * g->Wo); f->dx1 = rt_zalloc64_tail(4 * g->Wo); f->dx2 = rt_zalloc64_tail(4 * g->Wo);
  f->re = rt_zalloc64_tail(sizeof(float32x4_t) * NC); f->im = rt_zalloc64_tail(sizeof(float32x4_t) * NC); f->pw = rt_zalloc64_tail(sizeof(float32x4_t) * F);
  f->db = rt_zalloc64_tail(sizeof(float) * g->nmel * g->nfr); f->sp = rt_zalloc64_tail(sizeof(float) * g->R * g->nfr);
  if (!f->tw_re || !f->tw_im || !f->sp_re || !f->sp_im || !f->rev || !f->win || !f->mel_off || !f->mel_k || !f->mel_w ||
      !f->rows || !f->chv || !f->ry1 || !f->ry2 || !f->dy1 || !f->dy2 || !f->rx1 || !f->rx2 || !f->dx1 || !f->dx2 ||
      !f->re || !f->im || !f->pw || !f->db || !f->sp) {
    fe_db_free(f);
    return NULL;
  }
  fft_twiddles(f->tw_re, f->tw_im, NC);
  for (int k = 0; k <= NC; k++) {
    double a = -2 * M_PI * k / N;
    f->sp_re[k] = (float)cos(a); f->sp_im[k] = (float)sin(a);
  }
  int bits = 0;
  while ((1 << bits) < NC) bits++;
  for (int i = 0; i < NC; i++) f->rev[i] = fft_bitrev(i, bits);
  memcpy(f->win, window, sizeof(float) * N);
  int q = 0, klo = F, khi = 0;
  for (int j = 0; j < g->nmel; j++) {
    f->mel_off[j] = q;
    for (int k = 0; k < F; k++) {
      float w = mel[(size_t)j * F + k];
      if (w != 0.f) {
        f->mel_k[q] = k; f->mel_w[q] = w; q++;
        if (k < klo) klo = k;
        if (k + 1 > khi) khi = k + 1;
      }
    }
  }
  f->mel_off[g->nmel] = q;
  f->klo = klo < khi ? klo : 0; f->khi = khi;
  for (int r = 0; r < g->R; r++) f->rows[r] = rows[r];
  memcpy(f->chv, chv, sizeof(float) * 2 * g->C);
  axis_setup(g->R, g->Ho, f->ry1, f->ry2, f->dy1, f->dy2);
  axis_setup(g->nfr, g->Wo, f->rx1, f->rx2, f->dx1, f->dx2);
  return f;
}

static inline int src_index(int i, int n, int mode) { /* -1: a zero sample */
  if (i >= 0 && i < n) return i;
  if (mode != 1) return -1;
  if (i < 0) i = -i;
  if (i >= n) i = 2 * n - 2 - i;
  return i >= 0 && i < n ? i : -1;
}

static void fft4(const fe_db *t, float32x4_t *re, float32x4_t *im) {
  const int NC = t->NC;
  for (int len = 2, tws = NC / 2; len <= NC; len <<= 1, tws >>= 1) {
    int h = len / 2;
    for (int i = 0; i < NC; i += len) {
      for (int k = 0; k < h; k++) {
        float wr = t->tw_re[k * tws], wi = t->tw_im[k * tws];
        float32x4_t br = re[i + k + h], bi = im[i + k + h];
        float32x4_t vr = vfmsq_n_f32(vmulq_n_f32(br, wr), bi, wi);
        float32x4_t vi = vfmaq_n_f32(vmulq_n_f32(bi, wr), br, wi);
        float32x4_t ur = re[i + k], ui = im[i + k];
        re[i + k] = vaddq_f32(ur, vr); im[i + k] = vaddq_f32(ui, vi);
        re[i + k + h] = vsubq_f32(ur, vr); im[i + k + h] = vsubq_f32(ui, vi);
      }
    }
  }
}

static inline float post_op(float v, int code, float c) {
  switch (code) {
  case 1: return v * c;
  case 2: return v / c;
  case 3: return v + c;
  case 4: return v - c;
  default: return v;
  }
}

/* frames [t0, t1) in groups of 4 (t0 % 4 == 0): dB rows into f->db */
void fe_db_frames(fe_db *f, const float *pcm, int t0, int t1, float32x4_t *re, float32x4_t *im, float32x4_t *pw) {
  const fe_db_cfg *g = &f->g;
  const int N = g->nfft, NC = f->NC, nfr = g->nfr;
  for (int tb = t0; tb < t1; tb += 4) {
    int interior = 1, base[4];
    for (int k = 0; k < 4; k++) {
      int tf = tb + k < nfr ? tb + k : nfr - 1;
      base[k] = tf * g->hop - g->pad;
      if (base[k] < 0 || base[k] + N > g->nsamp) interior = 0;
    }
    if (interior) {
      for (int n = 0; n < NC; n += 4) {
        float32x4x2_t w2 = vld2q_f32(f->win + 2 * n);
        float32x4_t ev[4], od[4];
        for (int k = 0; k < 4; k++) {
          float32x4x2_t v = vld2q_f32(pcm + base[k] + 2 * n);
          ev[k] = vmulq_f32(v.val[0], w2.val[0]);
          od[k] = vmulq_f32(v.val[1], w2.val[1]);
        }
        float32x4_t e01l = vzip1q_f32(ev[0], ev[1]), e01h = vzip2q_f32(ev[0], ev[1]);
        float32x4_t e23l = vzip1q_f32(ev[2], ev[3]), e23h = vzip2q_f32(ev[2], ev[3]);
        float32x4_t o01l = vzip1q_f32(od[0], od[1]), o01h = vzip2q_f32(od[0], od[1]);
        float32x4_t o23l = vzip1q_f32(od[2], od[3]), o23h = vzip2q_f32(od[2], od[3]);
#define Z1(a, b) vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)))
#define Z2(a, b) vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)))
        re[f->rev[n]] = Z1(e01l, e23l); re[f->rev[n + 1]] = Z2(e01l, e23l);
        re[f->rev[n + 2]] = Z1(e01h, e23h); re[f->rev[n + 3]] = Z2(e01h, e23h);
        im[f->rev[n]] = Z1(o01l, o23l); im[f->rev[n + 1]] = Z2(o01l, o23l);
        im[f->rev[n + 2]] = Z1(o01h, o23h); im[f->rev[n + 3]] = Z2(o01h, o23h);
#undef Z1
#undef Z2
      }
    } else {
      for (int n = 0; n < NC; n++) {
        float a[4], b[4];
        for (int k = 0; k < 4; k++) {
          int i0 = src_index(base[k] + 2 * n, g->nsamp, g->padmode), i1 = src_index(base[k] + 2 * n + 1, g->nsamp, g->padmode);
          a[k] = (i0 < 0 ? 0.f : pcm[i0]) * f->win[2 * n];
          b[k] = (i1 < 0 ? 0.f : pcm[i1]) * f->win[2 * n + 1];
        }
        re[f->rev[n]] = vld1q_f32(a); im[f->rev[n]] = vld1q_f32(b);
      }
    }
    fft4(f, re, im);
    for (int k = f->klo; k < f->khi; k++) {
      int k1 = k & (NC - 1), k2 = (NC - k) & (NC - 1);
      float32x4_t zr = re[k1], zi = im[k1], cr = re[k2], ci = vnegq_f32(im[k2]);
      float32x4_t er = vmulq_n_f32(vaddq_f32(zr, cr), 0.5f), ei = vmulq_n_f32(vaddq_f32(zi, ci), 0.5f);
      float32x4_t dr = vsubq_f32(zr, cr), di = vsubq_f32(zi, ci);
      float32x4_t or_ = vmulq_n_f32(di, 0.5f), oi = vmulq_n_f32(dr, -0.5f);
      float wr = f->sp_re[k], wi = f->sp_im[k];
      float32x4_t xr = vaddq_f32(er, vfmsq_n_f32(vmulq_n_f32(or_, wr), oi, wi));
      float32x4_t xi = vaddq_f32(ei, vfmaq_n_f32(vmulq_n_f32(oi, wr), or_, wi));
      pw[k] = vaddq_f32(vmulq_f32(xr, xr), vmulq_f32(xi, xi)); /* re^2 + im^2 as the graph adds them */
    }
    for (int j = 0; j < g->nmel; j++) {
      float32x4_t acc = vdupq_n_f32(0);
      for (int q = f->mel_off[j]; q < f->mel_off[j + 1]; q++) acc = vfmaq_n_f32(acc, pw[f->mel_k[q]], f->mel_w[q]);
      float v[4];
      vst1q_f32(v, acc);
      for (int k = 0; k < 4 && tb + k < nfr; k++) {
        float x = v[k] < g->floor ? g->floor : v[k];
        float y = logf(x);
        for (int u = 0; u < 3; u++) y = post_op(y, g->post[u], g->postv[u]);
        f->db[(size_t)j * nfr + tb + k] = y;
      }
    }
  }
}

/* everything after the dB rows: top-db, rows, min/max, resize, channels -> img (zero-haloed NHWC, channel stride ld) */
void fe_db_image(fe_db *f, float *img) {
  const fe_db_cfg *g = &f->g;
  const int nfr = g->nfr, R = g->R, h = g->halo, ld = g->ld, Wp = g->Wo + 2 * h;
  float *db = f->db, *sp = f->sp;
  float fl = -FLT_MAX;
  if (g->has_top_db) {
    float mx = -FLT_MAX;
    for (size_t i = 0; i < (size_t)g->nmel * nfr; i++) mx = fmaxf(mx, db[i]);
    fl = mx - g->top_db;
  }
  float mn = FLT_MAX, mx = -FLT_MAX;
  for (int r = 0; r < R; r++) {
    const float *s = db + (size_t)f->rows[r] * nfr;
    float *d = sp + (size_t)r * nfr;
    for (int t = 0; t < nfr; t++) {
      float v = g->has_top_db ? fmaxf(s[t], fl) : s[t];
      d[t] = v;
      mn = fminf(mn, v); mx = fmaxf(mx, v);
    }
  }
  if (g->norm) {
    float den = (mx - mn) + g->eps;
    for (size_t i = 0; i < (size_t)R * nfr; i++) sp[i] = g->norm == 2 ? (mx - sp[i]) / den : (sp[i] - mn) / den;
  }
  if (h) rt_zero_halo(img, g->Ho, g->Wo, ld, h, 4);
  const float *c0 = f->chv, *c1 = f->chv + g->C;
  for (int oy = 0; oy < g->Ho; oy++) {
    const float *r1 = sp + (size_t)f->ry1[oy] * nfr, *r2 = sp + (size_t)f->ry2[oy] * nfr;
    const float dy1 = f->dy1[oy], dy2 = f->dy2[oy];
    float *o = img + ((size_t)(oy + h) * Wp + h) * ld;
    for (int ox = 0; ox < g->Wo; ox++) {
      int x1 = f->rx1[ox], x2 = f->rx2[ox];
      float dx1 = f->dx1[ox], dx2 = f->dx2[ox];
      float v = dx2 * dy2 * r1[x1] + dx1 * dy2 * r1[x2] + dx2 * dy1 * r2[x1] + dx1 * dy1 * r2[x2];
      float *px = o + (size_t)ox * ld;
      for (int c = 0; c < g->C; c++) px[c] = post_op(post_op(v, g->chan[0], c0[c]), g->chan[1], c1[c]);
      for (int c = g->C; c < ld; c++) px[c] = 0.f;
    }
  }
}

void fe_db_run(fe_db *f, const float *pcm, float *img) {
  fe_db_frames(f, pcm, 0, f->g.nfr, f->re, f->im, f->pw);
  fe_db_image(f, img);
}
const fe_db_cfg *fe_db_config(const fe_db *f) { return &f->g; }
size_t fe_db_work_bytes(const fe_db *f) { return sizeof(float32x4_t) * (2 * (size_t)f->NC + f->F); }
