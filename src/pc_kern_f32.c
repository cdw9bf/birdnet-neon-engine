// fp32 kernels for compiled models (baseline ARMv8-A objects): framed log-mel front end (exact real FFT, 4 frames
// per float32x4 lane), im2col, depthwise kxk, padding copy, mean, L2 normalisation, prototype logits.
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fft_tab.h"
#include "pc_kern.h"
#include "rt.h"

#define AI static inline __attribute__((always_inline))

struct pc_logmel {
  int nsamp, frame, hop, pad, nfft, M, lg, nmel, nfr, klo, khi;
  float *y;                    /* signal with pad zeros before and >= nfft zeros after */
  float *win;                  /* [nfft], zero beyond frame */
  float *tw_re, *tw_im;        /* e^{-2 pi i k / M}, k < M/2 */
  float *pc, *ps;              /* cos, sin(2 pi k / nfft), k in [0, khi) */
  uint16_t *rev;
  int *mel_off;                /* sparse mel [nmel + 1] */
  uint16_t *mel_k;
  float *mel_w;
  float floor_, scale;
  float32x4_t *re, *im, *mag, *P;
};

pc_logmel *pc_logmel_new(int nsamp, int frame, int hop, int pad, int nfft, int nmel, const float *window,
                         const float *mel, float floor_, float scale) {
  if (nfft < 16 || nfft > 2048 || (nfft & (nfft - 1)) || frame > nfft || hop <= 0 || pad < 0 || nmel < 1 || nmel > 512)
    return NULL;
  pc_logmel *f = calloc(1, sizeof *f);
  if (!f) return NULL;
  f->nsamp = nsamp; f->frame = frame; f->hop = hop; f->pad = pad; f->nfft = nfft; f->M = nfft / 2; f->nmel = nmel;
  f->nfr = (nsamp + 2 * pad - frame) / hop + 1;
  while ((1 << f->lg) < f->M) f->lg++;
  int bins = nfft / 2 + 1;
  f->klo = bins; f->khi = 0;
  int nnz = 0;
  for (int j = 0; j < nmel; j++)
    for (int k = 0; k < bins; k++)
      if (mel[(size_t)j * bins + k] != 0.f) { nnz++; if (k < f->klo) f->klo = k; if (k + 1 > f->khi) f->khi = k + 1; }
  if (f->khi == 0) f->klo = 0;
  size_t ylen = (size_t)(f->nfr - 1) * hop + nfft + 64;
  f->y = calloc(ylen > (size_t)nsamp + 2 * pad + 64 ? ylen : (size_t)nsamp + 2 * pad + 64, sizeof(float));
  f->win = calloc(nfft, sizeof(float));
  f->tw_re = malloc(sizeof(float) * f->M / 2); f->tw_im = malloc(sizeof(float) * f->M / 2);
  f->pc = malloc(sizeof(float) * (f->khi + 1)); f->ps = malloc(sizeof(float) * (f->khi + 1));
  f->rev = malloc(sizeof(uint16_t) * f->M);
  f->mel_off = malloc(sizeof(int) * (nmel + 1));
  f->mel_k = malloc(sizeof(uint16_t) * (nnz + 1)); f->mel_w = malloc(sizeof(float) * (nnz + 1));
  f->re = rt_al64(sizeof(float32x4_t) * f->M); f->im = rt_al64(sizeof(float32x4_t) * f->M);
  f->mag = rt_al64(sizeof(float32x4_t) * (bins + 1)); f->P = rt_al64(sizeof(float32x4_t) * nmel);
  if (!f->y || !f->win || !f->tw_re || !f->tw_im || !f->pc || !f->ps || !f->rev || !f->mel_off || !f->mel_k || !f->mel_w ||
      !f->re || !f->im || !f->mag || !f->P) { pc_logmel_free(f); return NULL; }
  memcpy(f->win, window, sizeof(float) * frame);
  fft_twiddles(f->tw_re, f->tw_im, f->M);
  for (int k = 0; k < f->khi; k++) {
    double a = 2 * M_PI * k / nfft;
    f->pc[k] = (float)cos(a); f->ps[k] = (float)sin(a);
  }
  for (int i = 0; i < f->M; i++) f->rev[i] = (uint16_t)fft_bitrev(i, f->lg);
  nnz = 0;
  for (int j = 0; j < nmel; j++) {
    f->mel_off[j] = nnz;
    for (int k = 0; k < bins; k++)
      if (mel[(size_t)j * bins + k] != 0.f) { f->mel_k[nnz] = (uint16_t)k; f->mel_w[nnz] = mel[(size_t)j * bins + k]; nnz++; }
  }
  f->mel_off[nmel] = nnz;
  f->floor_ = floor_; f->scale = scale;
  return f;
}

void pc_logmel_free(pc_logmel *f) {
  if (!f) return;
  free(f->y); free(f->win); free(f->tw_re); free(f->tw_im); free(f->pc); free(f->ps); free(f->rev); free(f->mel_off);
  free(f->mel_k); free(f->mel_w); free(f->re); free(f->im); free(f->mag); free(f->P); free(f);
}
int pc_logmel_frames(const pc_logmel *f) { return f->nfr; }

/* ln x for x > 0: log2_acc_f32x4 (gemm_f32.h) times ln 2 */
AI float32x4_t ln_acc(float32x4_t x) { return vmulq_n_f32(log2_acc_f32x4(x), 0.69314718f); }

/* 4 frames (lanes) starting at frame t0 (frames >= nfr use a clamped start and are not written) */
static void group(pc_logmel *F, int t0, float *spec) {
  const int M = F->M;
  float32x4_t *re = F->re, *im = F->im;
  int st[4];
  for (int q = 0; q < 4; q++) st[q] = (t0 + q < F->nfr ? t0 + q : F->nfr - 1) * F->hop;
  /* z[n] = w[2n] x[2n] + i w[2n+1] x[2n+1]; the window is zero past the frame, so z = 0 for 2n >= frame */
  int nz = (F->frame + 1) / 2;
  for (int n = 0; n < M; n += 4) {
    if (n >= nz) {
      for (int q = 0; q < 4; q++) { re[F->rev[n + q]] = vdupq_n_f32(0); im[F->rev[n + q]] = vdupq_n_f32(0); }
      continue;
    }
    float32x4x2_t v0 = vld2q_f32(F->y + st[0] + 2 * n), v1 = vld2q_f32(F->y + st[1] + 2 * n),
                  v2 = vld2q_f32(F->y + st[2] + 2 * n), v3 = vld2q_f32(F->y + st[3] + 2 * n);
    float32x4x2_t wv = vld2q_f32(F->win + 2 * n);
    for (int part = 0; part < 2; part++) {
      float32x4_t a = vmulq_f32(v0.val[part], wv.val[part]), b = vmulq_f32(v1.val[part], wv.val[part]),
                  c = vmulq_f32(v2.val[part], wv.val[part]), d = vmulq_f32(v3.val[part], wv.val[part]);
      float32x4_t ab0 = vtrn1q_f32(a, b), ab1 = vtrn2q_f32(a, b), cd0 = vtrn1q_f32(c, d), cd1 = vtrn2q_f32(c, d);
      float32x4_t r0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(ab0), vreinterpretq_f64_f32(cd0)));
      float32x4_t r1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(ab1), vreinterpretq_f64_f32(cd1)));
      float32x4_t r2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(ab0), vreinterpretq_f64_f32(cd0)));
      float32x4_t r3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(ab1), vreinterpretq_f64_f32(cd1)));
      float32x4_t *dst = part ? im : re;
      dst[F->rev[n]] = r0; dst[F->rev[n + 1]] = r1; dst[F->rev[n + 2]] = r2; dst[F->rev[n + 3]] = r3;
    }
  }
  fft_dit_x4(re, im, M, F->tw_re, F->tw_im);
  /* real-input split: A = Z_k, B = conj(Z_{M-k}); X_k = (A + B)/2 + e^{-2 pi i k/N} (A - B)/(2i); |X_k| */
  const float32x4_t hf = vdupq_n_f32(0.5f);
  for (int k = F->klo; k < F->khi; k++) {
    int k2 = (k == 0 || k == M) ? 0 : M - k;
    int ka = k == M ? 0 : k;
    float32x4_t Ar = re[ka], Ai = im[ka], Br = re[k2], Bi = vnegq_f32(im[k2]);
    float32x4_t Er = vmulq_f32(vaddq_f32(Ar, Br), hf), Ei = vmulq_f32(vaddq_f32(Ai, Bi), hf);
    float32x4_t Or = vmulq_f32(vsubq_f32(Ai, Bi), hf), Oi = vmulq_f32(vsubq_f32(Br, Ar), hf); /* (A-B)/(2i) */
    /* W^k = c - i s: W^k O = (c Or + s Oi, c Oi - s Or) */
    float c = F->pc[k], s = F->ps[k];
    float32x4_t Xr = vfmaq_n_f32(vfmaq_n_f32(Er, Or, c), Oi, s);
    float32x4_t Xi = vfmsq_n_f32(vfmaq_n_f32(Ei, Oi, c), Or, s);
    F->mag[k] = vsqrtq_f32(vfmaq_f32(vmulq_f32(Xr, Xr), Xi, Xi));
  }
  const float32x4_t fl = vdupq_n_f32(F->floor_);
  for (int j = 0; j < F->nmel; j++) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    int e = F->mel_off[j], e1 = F->mel_off[j + 1];
    for (; e + 2 <= e1; e += 2) {
      a0 = vfmaq_n_f32(a0, F->mag[F->mel_k[e]], F->mel_w[e]);
      a1 = vfmaq_n_f32(a1, F->mag[F->mel_k[e + 1]], F->mel_w[e + 1]);
    }
    if (e < e1) a0 = vfmaq_n_f32(a0, F->mag[F->mel_k[e]], F->mel_w[e]);
    F->P[j] = vmulq_n_f32(ln_acc(vmaxq_f32(vaddq_f32(a0, a1), fl)), F->scale);
  }
  int nq = F->nfr - t0 < 4 ? F->nfr - t0 : 4;
  for (int j = 0; j < F->nmel; j++) {
    float v[4];
    vst1q_f32(v, F->P[j]);
    for (int q = 0; q < nq; q++) spec[(size_t)(t0 + q) * F->nmel + j] = v[q];
  }
}

void pc_logmel_run(pc_logmel *F, const float *pcm, float *spec) {
  memcpy(F->y + F->pad, pcm, sizeof(float) * F->nsamp);
  for (int t0 = 0; t0 < F->nfr; t0 += 4) group(F, t0, spec);
}

void pc_im2col1(const float *img, int H, int W, int k, int s, int Ho, int Wo, float *col, int Kp) {
  (void)H;
  for (int oy = 0; oy < Ho; oy++)
    for (int ox = 0; ox < Wo; ox++) {
      float *c = col + ((size_t)oy * Wo + ox) * Kp;
      int q = 0;
      for (int ky = 0; ky < k; ky++)
        for (int kx = 0; kx < k; kx++) c[q++] = img[(size_t)(oy * s + ky) * W + ox * s + kx];
      for (; q < Kp; q++) c[q] = 0.f;
    }
}

void pc_dwk_f32(const float *xp, int H, int W, int C, int k, int s, int pt, int pl, const float *w, const float *b,
                int act, float *out, int Ho, int Wo, float *chsum) {
  pc_dwk_f32_cr(xp, H, W, C, k, s, pt, pl, w, b, act, out, Ho, Wo, chsum, 0, C);
}
/* Staged form of the depthwise k x k conv: one 16-channel group at a time over the whole image, its k input rows
 * copied into a contiguous ring ([k][Wp][16] floats) and its k*k weight vectors into a local table. The pixel-major
 * loop below touches, per 4 outputs, k*k weight lines and 4 k*k input lines spread over buffers far larger than L1.
 * Here the taps read L1, and with 16 channels = one 64-byte line per pixel every input and output line of the big
 * buffers is touched by one group only. Each output is the same chain of operations (bias, even / odd tap sums in tap
 * order, add, activation; the single-chain form for the last Wo % 4 pixels of a row) and each channel sum adds the
 * same values in the same order: bit-identical. */
#define DWK_ST_G 16
#define DWK_ST_BYTES 49152 /* ring limit (k * Wp * 64 bytes) */
#define DWK_ST_PF 8 /* pixels ahead for the input-copy and output-line prefetches (4 and 16 were slower: DEADENDS D29) */
static inline __attribute__((always_inline)) void dwk_staged_k(const float *xp, int W, int C, const int k, const int s,
                                                               int pt, int pl, const float *w, const float *b, int act,
                                                               float *out, int Ho, int Wo, float *chsum, int c0, int c1) {
  const int h = (k - 1) / 2, Wp = W + 2 * h, kk = k * k;
  float ring[DWK_ST_BYTES / 4] __attribute__((aligned(64)));
  float wl[49 * DWK_ST_G] __attribute__((aligned(64)));
  const size_t rowf = (size_t)Wp * DWK_ST_G; /* floats per staged row */
  const float32x4_t z = vdupq_n_f32(0);
  for (int c = c0; c < c1; c += DWK_ST_G) {
    const int gw = c1 - c < DWK_ST_G ? c1 - c : DWK_ST_G; /* channels of this group (a multiple of 4) */
    for (int t = 0; t < kk; t++)
      for (int g = 0; g < gw; g += 4) vst1q_f32(wl + DWK_ST_G * t + g, vld1q_f32(w + (size_t)t * C + c + g));
    float32x4_t cs[DWK_ST_G / 4] = {z, z, z, z};
    int staged = 0; /* input rows [0, staged) of the padded image have been copied (row iy lives in slot iy % k) */
    for (int oy = 0; oy < Ho; oy++) {
      const int iy0 = oy * s - pt + h;
      for (; staged < iy0 + k; staged++) {
        const float *src = xp + (size_t)staged * Wp * C + c;
        float *dst = ring + (size_t)(staged % k) * rowf;
        for (int x = 0; x < Wp; x++) {
          __builtin_prefetch(src + (size_t)(x + DWK_ST_PF) * C, 0, 3); /* one line every C floats: no hardware stream */
          for (int g = 0; g < gw; g += 4) vst1q_f32(dst + DWK_ST_G * x + g, vld1q_f32(src + (size_t)x * C + g));
        }
      }
      const float *rows[7];
      for (int ky = 0; ky < k; ky++) rows[ky] = ring + (size_t)((iy0 + ky) % k) * rowf;
      float *o = out + (size_t)oy * Wo * C + c;
      const size_t d = (size_t)s * DWK_ST_G;
      int ox = 0;
      for (; ox + 4 <= Wo; ox += 4) {
        const size_t x0 = (size_t)(ox * s - pl + h) * DWK_ST_G;
        for (int q = 0; q < 4; q++) __builtin_prefetch(o + (size_t)(ox + DWK_ST_PF + q) * C, 1, 3);
        for (int g = 0; g < gw; g += 4) {
          const float32x4_t bb = vld1q_f32(b + c + g);
          float32x4_t a0 = bb, a1 = bb, a2 = bb, a3 = bb, e0 = z, e1 = z, e2 = z, e3 = z;
          int t = 0;
#pragma GCC unroll 7
          for (int ky = 0; ky < k; ky++) {
            const float *r = rows[ky] + x0 + g;
#pragma GCC unroll 7
            for (int kx = 0; kx < k; kx++, t++) {
              const float32x4_t wv = vld1q_f32(wl + DWK_ST_G * t + g);
              const float *q = r + (size_t)kx * DWK_ST_G;
              if (t & 1) {
                e0 = vfmaq_f32(e0, vld1q_f32(q), wv); e1 = vfmaq_f32(e1, vld1q_f32(q + d), wv);
                e2 = vfmaq_f32(e2, vld1q_f32(q + 2 * d), wv); e3 = vfmaq_f32(e3, vld1q_f32(q + 3 * d), wv);
              } else {
                a0 = vfmaq_f32(a0, vld1q_f32(q), wv); a1 = vfmaq_f32(a1, vld1q_f32(q + d), wv);
                a2 = vfmaq_f32(a2, vld1q_f32(q + 2 * d), wv); a3 = vfmaq_f32(a3, vld1q_f32(q + 3 * d), wv);
              }
            }
          }
          float32x4_t v[4] = {vaddq_f32(a0, e0), vaddq_f32(a1, e1), vaddq_f32(a2, e2), vaddq_f32(a3, e3)};
          if (act == 1) silu4_f32x4(v);
          for (int q = 0; q < 4; q++) vst1q_f32(o + (size_t)(ox + q) * C + g, v[q]);
          cs[g / 4] = vaddq_f32(cs[g / 4], vaddq_f32(vaddq_f32(v[0], v[1]), vaddq_f32(v[2], v[3])));
        }
      }
      for (; ox < Wo; ox++) {
        const size_t x0 = (size_t)(ox * s - pl + h) * DWK_ST_G;
        for (int g = 0; g < gw; g += 4) {
          float32x4_t a = vld1q_f32(b + c + g);
          int t = 0;
          for (int ky = 0; ky < k; ky++)
            for (int kx = 0; kx < k; kx++, t++)
              a = vfmaq_f32(a, vld1q_f32(rows[ky] + x0 + (size_t)kx * DWK_ST_G + g), vld1q_f32(wl + DWK_ST_G * t + g));
          if (act == 1) a = silu_f32x4(a);
          vst1q_f32(o + (size_t)ox * C + g, a);
          cs[g / 4] = vaddq_f32(cs[g / 4], a);
        }
      }
    }
    for (int g = 0; g < gw; g += 4) vst1q_f32(chsum + c + g, cs[g / 4]);
  }
}

/* k and the stride as compile-time constants: the tap loops unroll and the even / odd choice needs no branch */
#define DWK_ST(K, S) \
  static __attribute__((noinline)) void dwk_staged_##K##_##S(const float *xp, int W, int C, int pt, int pl, const float *w, \
                                                             const float *b, int act, float *out, int Ho, int Wo,          \
                                                             float *chsum, int c0, int c1) {                               \
    dwk_staged_k(xp, W, C, K, S, pt, pl, w, b, act, out, Ho, Wo, chsum, c0, c1);                                           \
  }
DWK_ST(3, 1) DWK_ST(3, 2) DWK_ST(5, 1) DWK_ST(5, 2) DWK_ST(7, 1) DWK_ST(7, 2)
static void dwk_staged_cr(const float *xp, int W, int C, int k, int s, int pt, int pl, const float *w, const float *b,
                          int act, float *out, int Ho, int Wo, float *chsum, int c0, int c1) {
  switch (10 * k + s) {
  case 31: dwk_staged_3_1(xp, W, C, pt, pl, w, b, act, out, Ho, Wo, chsum, c0, c1); break;
  case 32: dwk_staged_3_2(xp, W, C, pt, pl, w, b, act, out, Ho, Wo, chsum, c0, c1); break;
  case 51: dwk_staged_5_1(xp, W, C, pt, pl, w, b, act, out, Ho, Wo, chsum, c0, c1); break;
  case 52: dwk_staged_5_2(xp, W, C, pt, pl, w, b, act, out, Ho, Wo, chsum, c0, c1); break;
  case 71: dwk_staged_7_1(xp, W, C, pt, pl, w, b, act, out, Ho, Wo, chsum, c0, c1); break;
  default: dwk_staged_7_2(xp, W, C, pt, pl, w, b, act, out, Ho, Wo, chsum, c0, c1);
  }
}

void pc_dwk_f32_cr(const float *xp, int H, int W, int C, int k, int s, int pt, int pl, const float *w, const float *b,
                   int act, float *out, int Ho, int Wo, float *chsum, int c0, int c1) {
  (void)H;
  const int h = (k - 1) / 2, Wp = W + 2 * h;
  if ((k == 3 || k == 5 || k == 7) && (s == 1 || s == 2) && (size_t)k * Wp * DWK_ST_G * 4 <= DWK_ST_BYTES && c0 % 4 == 0 && (c1 - c0) % 4 == 0) {
    dwk_staged_cr(xp, W, C, k, s, pt, pl, w, b, act, out, Ho, Wo, chsum, c0, c1);
    return;
  }
  for (int c = c0; c < c1; c++) chsum[c] = 0.f;
  for (int oy = 0; oy < Ho; oy++) {
    int ox = 0;
    /* four output pixels per pass, 4 channels at a time, two partial sums each (even / odd taps): 8 FMA chains */
    for (; ox + 4 <= Wo; ox += 4) {
      const float *x0 = xp + ((size_t)(oy * s - pt + h) * Wp + (ox * s - pl + h)) * C;
      const size_t d = (size_t)s * C;
      float *o = out + ((size_t)oy * Wo + ox) * C;
      for (int c = c0; c < c1; c += 4) {
        float32x4_t bb = vld1q_f32(b + c), z = vdupq_n_f32(0);
        float32x4_t a0 = bb, a1 = bb, a2 = bb, a3 = bb, e0 = z, e1 = z, e2 = z, e3 = z;
        int t = 0;
        for (int ky = 0; ky < k; ky++) {
          const float *r = x0 + (size_t)ky * Wp * C + c;
          const float *wr = w + (size_t)ky * k * C + c;
          for (int kx = 0; kx < k; kx++, t++) {
            float32x4_t wv = vld1q_f32(wr + (size_t)kx * C);
            const float *q = r + (size_t)kx * C;
            if (t & 1) {
              e0 = vfmaq_f32(e0, vld1q_f32(q), wv); e1 = vfmaq_f32(e1, vld1q_f32(q + d), wv);
              e2 = vfmaq_f32(e2, vld1q_f32(q + 2 * d), wv); e3 = vfmaq_f32(e3, vld1q_f32(q + 3 * d), wv);
            } else {
              a0 = vfmaq_f32(a0, vld1q_f32(q), wv); a1 = vfmaq_f32(a1, vld1q_f32(q + d), wv);
              a2 = vfmaq_f32(a2, vld1q_f32(q + 2 * d), wv); a3 = vfmaq_f32(a3, vld1q_f32(q + 3 * d), wv);
            }
          }
        }
        float32x4_t v[4] = {vaddq_f32(a0, e0), vaddq_f32(a1, e1), vaddq_f32(a2, e2), vaddq_f32(a3, e3)};
        if (act == 1) silu4_f32x4(v);
        for (int q = 0; q < 4; q++) vst1q_f32(o + (size_t)q * C + c, v[q]);
        vst1q_f32(chsum + c, vaddq_f32(vld1q_f32(chsum + c), vaddq_f32(vaddq_f32(v[0], v[1]), vaddq_f32(v[2], v[3]))));
      }
    }
    for (; ox < Wo; ox++) {
      const float *x0 = xp + ((size_t)(oy * s - pt + h) * Wp + (ox * s - pl + h)) * C;
      float *o = out + ((size_t)oy * Wo + ox) * C;
      for (int c = c0; c < c1; c += 8) {
        float32x4_t a0 = vld1q_f32(b + c), a1 = vld1q_f32(b + c + 4);
        for (int ky = 0; ky < k; ky++) {
          const float *xr = x0 + (size_t)ky * Wp * C + c;
          const float *wr = w + (size_t)ky * k * C + c;
          for (int kx = 0; kx < k; kx++) {
            a0 = vfmaq_f32(a0, vld1q_f32(xr + (size_t)kx * C), vld1q_f32(wr + (size_t)kx * C));
            a1 = vfmaq_f32(a1, vld1q_f32(xr + (size_t)kx * C + 4), vld1q_f32(wr + (size_t)kx * C + 4));
          }
        }
        if (act == 1) { a0 = silu_f32x4(a0); a1 = silu_f32x4(a1); }
        vst1q_f32(o + c, a0); vst1q_f32(o + c + 4, a1);
        vst1q_f32(chsum + c, vaddq_f32(vld1q_f32(chsum + c), a0));
        vst1q_f32(chsum + c + 4, vaddq_f32(vld1q_f32(chsum + c + 4), a1));
      }
    }
  }
}

void pc_pad_f32(const float *x, int H, int W, int C, int ld, int h, float *xp) {
  for (int y = 0; y < H; y++)
    for (int xx = 0; xx < W; xx++)
      memcpy(xp + ((size_t)(y + h) * (W + 2 * h) + xx + h) * C, x + ((size_t)y * W + xx) * ld, sizeof(float) * C);
}

void pc_mean_f32(const float *x, int P, int C, int ld, float *out) {
  for (int c = 0; c < C; c += 4) {
    float32x4_t s = vdupq_n_f32(0);
    for (int p = 0; p < P; p++) s = vaddq_f32(s, vld1q_f32(x + (size_t)p * ld + c));
    vst1q_f32(out + c, vdivq_f32(s, vdupq_n_f32((float)P)));
  }
}

void pc_l2norm_f32(const float *x, int P, int C, int ld, float eps, float *a) {
  for (int p = 0; p < P; p++) {
    const float *r = x + (size_t)p * ld;
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    for (int c = 0; c < C; c += 8) {
      float32x4_t u = vld1q_f32(r + c), v = vld1q_f32(r + c + 4);
      s0 = vfmaq_f32(s0, u, u); s1 = vfmaq_f32(s1, v, v);
    }
    float den = sqrtf(vaddvq_f32(vaddq_f32(s0, s1))) + eps;
    float32x4_t d = vdupq_n_f32(den);
    for (int c = 0; c < C; c += 4) vst1q_f32(a + (size_t)p * C + c, vdivq_f32(vld1q_f32(r + c), d));
  }
}

int pc_proto_logits(const float *colmax, const float *w, const float *b, int nc, int ppc, float *logits) {
  int bad = 0;
  for (int c = 0; c < nc; c++) {
    float s = 0;
    for (int q = 0; q < ppc; q++) s += colmax[(size_t)c * ppc + q] * w[(size_t)c * ppc + q];
    float v = s + b[c];
    logits[c] = v;
    bad |= !(fabsf(v) <= 3.0e38f);
  }
  return bad ? -2 : 0;
}

/* column maxima for prototypes stored in fp16 (weight panels [Np/8][K][8], converted with FCVTL in the K loop; plain
 * ARMv8-A) and fp32 cells a [P][K]: colmax[n] = max_p a[p] . w[n], fp32 arithmetic. P % 8 == 0, K % 4 == 0. */
void pc_proto_colmax_f32h(const float *a, int P, int K, const uint16_t *w16, int Np, float *colmax) {
  for (int n0 = 0; n0 < Np; n0 += 8) {
    const __fp16 *wp = (const __fp16 *)w16 + (size_t)(n0 / 8) * K * 8;
    float32x4_t m0 = vdupq_n_f32(-INFINITY), m1 = m0;
    for (int p0 = 0; p0 < P; p0 += 8) {
      const float *ar = a + (size_t)p0 * K;
      float32x4_t c[8][2];
      for (int r = 0; r < 8; r++) c[r][0] = c[r][1] = vdupq_n_f32(0);
      for (int k = 0; k < K; k += 4) {
        float32x4_t av[8];
        for (int r = 0; r < 8; r++) av[r] = vld1q_f32(ar + (size_t)r * K + k);
#define KSTEP(L)                                                                                                    \
  do {                                                                                                              \
    float16x8_t h = vld1q_f16(wp + (size_t)(k + (L)) * 8);                                                          \
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(h)), hi = vcvt_high_f32_f16(h);                                      \
    for (int r = 0; r < 8; r++) {                                                                                   \
      c[r][0] = vfmaq_laneq_f32(c[r][0], lo, av[r], L);                                                             \
      c[r][1] = vfmaq_laneq_f32(c[r][1], hi, av[r], L);                                                             \
    }                                                                                                               \
  } while (0)
        KSTEP(0); KSTEP(1); KSTEP(2); KSTEP(3);
#undef KSTEP
      }
      for (int r = 0; r < 8; r++) { m0 = vmaxq_f32(m0, c[r][0]); m1 = vmaxq_f32(m1, c[r][1]); }
    }
    vst1q_f32(colmax + n0, m0);
    vst1q_f32(colmax + n0 + 4, m1);
  }
}
