// General dense layers on a vector (custom-classifier heads: Dense K -> N, optional ReLU, any K and N). Baseline
// ARMv8-A objects: fp32 accumulation; the fp16-weight form converts weights with FCVTL (an ARMv8.0 instruction), so it
// runs on every AArch64 CPU. The fixed-width 1024 head kernels (v24k32_head / v24k16_head) stay as they are: packages
// that use them must keep scoring bit-identically.
#include <arm_neon.h>
#include <stddef.h>

#include "gemv.h"

static inline float act_f(float v, int act) { return act == 2 && v < 0.f ? 0.f : v; } /* RT_ACT_RELU */

/* out[n] = act(b[n] + sum_k W[n][k] x[k]), W row-major [N][K] fp32 */
void bnn_gemv_f32(const float *W, const float *b, int N, int K, const float *x, int act, float *out) {
  int K4 = K & ~3, K8 = K & ~7;
  for (int n = 0; n < N; n++) {
    const float *w = W + (size_t)n * K;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
    int k = 0;
    for (; k < K8; k += 8) {
      a0 = vfmaq_f32(a0, vld1q_f32(w + k), vld1q_f32(x + k));
      a1 = vfmaq_f32(a1, vld1q_f32(w + k + 4), vld1q_f32(x + k + 4));
    }
    for (; k < K4; k += 4) a0 = vfmaq_f32(a0, vld1q_f32(w + k), vld1q_f32(x + k));
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; k < K; k++) s += w[k] * x[k];
    out[n] = act_f(s + b[n], act);
  }
}

/* the same with fp16-stored weights (half the memory traffic), fp32 arithmetic */
void bnn_gemv_h16(const __fp16 *W, const float *b, int N, int K, const float *x, int act, float *out) {
  int K4 = K & ~3, K8 = K & ~7;
  for (int n = 0; n < N; n++) {
    const __fp16 *w = W + (size_t)n * K;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
    int k = 0;
    for (; k < K8; k += 8) {
      float16x8_t wv = vld1q_f16(w + k);
      a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(wv)), vld1q_f32(x + k));
      a1 = vfmaq_f32(a1, vcvt_high_f32_f16(wv), vld1q_f32(x + k + 4));
    }
    for (; k < K4; k += 4) a0 = vfmaq_f32(a0, vcvt_f32_f16(vld1_f16(w + k)), vld1q_f32(x + k));
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; k < K; k++) s += (float)w[k] * x[k];
    out[n] = act_f(s + b[n], act);
  }
}
