/* general dense layers on a vector (gemv.c): any K and N, optional ReLU; fp32 or fp16-stored weights, fp32 math */
#ifndef BNN_GEMV_H
#define BNN_GEMV_H
void bnn_gemv_f32(const float *W, const float *b, int N, int K, const float *x, int act, float *out);
void bnn_gemv_h16(const __fp16 *W, const float *b, int N, int K, const float *x, int act, float *out);
#endif
