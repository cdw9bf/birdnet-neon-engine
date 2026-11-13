/* fp32 Winograd F(4x4, 3x3), stride 1, pad 1 (wino_f32.c) */
#ifndef WINO_F32_H
#define WINO_F32_H
#include <stddef.h>
typedef struct {
  int C, N, Np;        /* input channels (= the input's ld, % 4 == 0), outputs, outputs rounded up to 8 */
  int H, W, halo;      /* input (zero halo >= 1, channel stride C) */
  int Ho, Wo, act, ldo, opad; /* output (= H x W), ACT32_*, channel stride, halo */
  int S, Nb;           /* tiles per strip, output channels per block (multiple of 8) */
  const float *res;    /* optional residual (added after the activation): H x W pixels, halo rh, channel stride rld */
  int rh, rld;
} wino43_cfg;
/* scratch floats per worker */
size_t wino43_scratch_floats(const wino43_cfg *g);
/* one strip of S tiles (row-major tile order); U [36][Np/8][C][8], bz zeros [Np + 8], bias [Np] */
void wino43_strip(const wino43_cfg *g, const float *xp, const float *U, const float *bz, const float *bias, float *out,
                  int strip, float *scr);
#endif
