// Shared by the front ends' radix-2 FFTs: the table set-up (v24_fe.c, pc_kern_f32.c, fe_dbimg.c; init time) and the
// 4-lane transform of the two front ends that use the same one (v24_fe.c, pc_kern_f32.c).
#ifndef BNN_FFT_TAB_H
#define BNN_FFT_TAB_H
#include <arm_neon.h>
#include <math.h>

/* twiddles e^{-2 pi i k / M} for k < M / 2, computed in double and rounded once */
static inline void fft_twiddles(float *re, float *im, int M) {
  for (int k = 0; k < M / 2; k++) {
    double a = -2 * M_PI * k / M;
    re[k] = (float)cos(a); im[k] = (float)sin(a);
  }
}
/* i with its low `bits` bits reversed */
static inline int fft_bitrev(int i, int bits) {
  int r = 0;
  for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
  return r;
}

/* Radix-2 DIT FFT of M complex points in place, 4 independent transforms in the 4 lanes (input in bit-reversed
 * order; tw = fft_twiddles(M)). Always inlined into its two callers (v24_fe.c, pc_kern_f32.c), which load and
 * unpack differently. */
static inline __attribute__((always_inline)) void fft_dit_x4(float32x4_t *re, float32x4_t *im, const int M,
                                                             const float *tw_re, const float *tw_im) {
  /* radix-2 DIT; first two stages fused without twiddle multiplies */
  for (int i = 0; i < M; i += 4) {
    float32x4_t ar = re[i], ai = im[i], br = re[i + 1], bi = im[i + 1], cr = re[i + 2], ci = im[i + 2], dr = re[i + 3],
                di = im[i + 3];
    float32x4_t s0r = vaddq_f32(ar, br), s0i = vaddq_f32(ai, bi), d0r = vsubq_f32(ar, br), d0i = vsubq_f32(ai, bi);
    float32x4_t s1r = vaddq_f32(cr, dr), s1i = vaddq_f32(ci, di), d1r = vsubq_f32(cr, dr), d1i = vsubq_f32(ci, di);
    re[i] = vaddq_f32(s0r, s1r); im[i] = vaddq_f32(s0i, s1i);
    re[i + 2] = vsubq_f32(s0r, s1r); im[i + 2] = vsubq_f32(s0i, s1i);
    /* d1 * (-i) = d1i - i d1r */
    re[i + 1] = vaddq_f32(d0r, d1i); im[i + 1] = vsubq_f32(d0i, d1r);
    re[i + 3] = vsubq_f32(d0r, d1i); im[i + 3] = vaddq_f32(d0i, d1r);
  }
  /* remaining levels two at a time (radix-2^2 passes: half the loads/stores of radix-2), last level alone if odd */
  int len = 8;
  for (; 2 * len <= M; len <<= 2) {
    int h = len / 2, st1 = M / len, st2 = M / (2 * len);
    for (int i = 0; i < M; i += 2 * len)
      for (int j = 0; j < h; j++) {
        float w1r = tw_re[j * st1], w1i = tw_im[j * st1], w2r = tw_re[j * st2], w2i = tw_im[j * st2];
        float32x4_t *pr = re + i + j, *pi = im + i + j;
        float32x4_t ar = pr[0], ai = pi[0], br = pr[h], bi = pi[h], cr = pr[2 * h], ci = pi[2 * h], dr = pr[3 * h],
                    di = pi[3 * h];
        /* level 1 (len): (a,b), (c,d) with w1 */
        float32x4_t tr = vfmsq_n_f32(vmulq_n_f32(br, w1r), bi, w1i), ti = vfmaq_n_f32(vmulq_n_f32(br, w1i), bi, w1r);
        float32x4_t a1r = vaddq_f32(ar, tr), a1i = vaddq_f32(ai, ti), b1r = vsubq_f32(ar, tr), b1i = vsubq_f32(ai, ti);
        tr = vfmsq_n_f32(vmulq_n_f32(dr, w1r), di, w1i); ti = vfmaq_n_f32(vmulq_n_f32(dr, w1i), di, w1r);
        float32x4_t c1r = vaddq_f32(cr, tr), c1i = vaddq_f32(ci, ti), d1r = vsubq_f32(cr, tr), d1i = vsubq_f32(ci, ti);
        /* level 2 (2 len): (a1,c1) with w2, (b1,d1) with w3 = -i w2 */
        tr = vfmsq_n_f32(vmulq_n_f32(c1r, w2r), c1i, w2i); ti = vfmaq_n_f32(vmulq_n_f32(c1r, w2i), c1i, w2r);
        pr[0] = vaddq_f32(a1r, tr); pi[0] = vaddq_f32(a1i, ti);
        pr[2 * h] = vsubq_f32(a1r, tr); pi[2 * h] = vsubq_f32(a1i, ti);
        float32x4_t ur = vfmsq_n_f32(vmulq_n_f32(d1r, w2r), d1i, w2i), ui = vfmaq_n_f32(vmulq_n_f32(d1r, w2i), d1i, w2r);
        /* (-i) * u = ui - i ur */
        pr[h] = vaddq_f32(b1r, ui); pi[h] = vsubq_f32(b1i, ur);
        pr[3 * h] = vsubq_f32(b1r, ui); pi[3 * h] = vaddq_f32(b1i, ur);
      }
  }
  for (; len <= M; len <<= 1) {
    int half = len / 2, step = M / len;
    for (int i = 0; i < M; i += len)
      for (int j = 0; j < half; j++) {
        float wr = tw_re[j * step], wi = tw_im[j * step];
        float32x4_t xr = re[i + j + half], xi = im[i + j + half];
        float32x4_t vr = vfmsq_n_f32(vmulq_n_f32(xr, wr), xi, wi);
        float32x4_t vi = vfmaq_n_f32(vmulq_n_f32(xr, wi), xi, wr);
        float32x4_t ur = re[i + j], ui = im[i + j];
        re[i + j] = vaddq_f32(ur, vr); im[i + j] = vaddq_f32(ui, vi);
        re[i + j + half] = vsubq_f32(ur, vr); im[i + j + half] = vsubq_f32(ui, vi);
      }
  }
}
#endif
