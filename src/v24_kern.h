// BirdNET v2.4 stage kernels shared by the hand-written engines (v24_fast.c, v24_f32.c) and the package runtime
// (rt_ops_*.c). Fixed v2.4 geometry: image [96][511][2], c0 4x8 s2 -> pool -> [48][128][48], post [3][8][192] ->
// pc 3x3 valid -> 6 x 1024 -> mean, dense head over K = 1024.
#ifndef V24_KERN_H
#define V24_KERN_H
#include "gemm_f16.h"
#include "gemm_f32.h"

/* fp16 path (Cortex-A76 objects) */
void v24k16_c0_pool(const float *img, f16 *img16 /*[98][518][2], zero border*/, const f16 **rows /*[4*256]*/,
                    const hpack *c0, f16 *pin /*[48][128][48]*/);
void v24k16_zero_halo(f16 *xp, int H, int W, int C);
void v24k16_se(int cm, int rd, const float *se1_w, const float *se1_b, const float *se2t, const float *se2_b,
               const float *chsum, int P, f16 *D);
/* the same in two steps (threads): the gate vector g16 [cm], then pixels [p0, p1) scaled in place */
void v24k16_se_gate(int cm, int rd, const float *se1_w, const float *se1_b, const float *se2t, const float *se2_b,
                    const float *chsum, int P, f16 *g16);
void v24k16_se_scale(int cm, const f16 *g16, f16 *D, int p0, int p1);
int v24k16_postpc(const f16 *X, const float *post_s, const float *post_b, f16 *pcin, f16 *pcout, const hpack *pc,
                  float *feat);
void v24k16_head(const f16 *W, const float *bias, int N, const float *feat, float *out);
/* fp32 path (baseline ARMv8-A objects) */
void v24k32_c0_pool(const float *img, float *imgp /*[98][518][2], zero border*/, const fpack *c0, float *pin);
void v24k32_zero_halo(float *xp, int H, int W, int C);
void v24k32_se_fc1(int cm, int rd, const float *se1_w, const float *se1_b, const float *chsum, int P, float *s1);
void v24k32_se(int cm, int rd, const float *se1_w, const float *se1_b, const float *se2t, const float *se2_b,
               const float *chsum, int P, float *D);
void v24k32_postpc(const float *X, const float *post_s, const float *post_b, float *pcin, float *pcout, const fpack *pc,
                   float *feat);
void v24k32_head(const float *W, const float *bias, int N, const float *feat, float *out);
#endif
