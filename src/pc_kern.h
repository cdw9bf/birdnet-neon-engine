// Kernels for compiled models beyond the BirdNET v2.4 geometry (first user: Perch v2): framed log-mel front end,
// im2col stem, depthwise kxk with asymmetric pads, padding copy, spatial mean, prototype-head helpers.
// *_f32 functions are baseline ARMv8-A objects (pc_kern_f32.c), *_f16 Cortex-A76 objects (pc_kern_f16.c).
#ifndef PC_KERN_H
#define PC_KERN_H
#include <stddef.h>
#include <stdint.h>

#include "gemm_f16.h"
#include "gemm_f32.h"

/* framed log-mel: frames of `frame` samples every `hop` (signal zero-padded by `pad` both sides), window [frame],
 * zero pad to nfft (power of two, <= 2048), |rFFT|, mel [nmel][nfft/2+1] (sparse), max(floor), log, * scale ->
 * spec [nfr][nmel] */
typedef struct pc_logmel pc_logmel;
pc_logmel *pc_logmel_new(int nsamp, int frame, int hop, int pad, int nfft, int nmel, const float *window,
                         const float *mel, float floor, float scale);
void pc_logmel_run(pc_logmel *f, const float *pcm, float *spec);
void pc_logmel_free(pc_logmel *f);
int pc_logmel_frames(const pc_logmel *f);

/* im2col of a 1-channel image [H][W] for a k x k conv with stride s, VALID: rows [Ho*Wo][Kp] (K = k*k, zero tail) */
void pc_im2col1(const float *img, int H, int W, int k, int s, int Ho, int Wo, float *col, int Kp);

/* depthwise k x k (k odd), stride s, pads (pt, pl) on an input with a symmetric zero halo h = (k-1)/2 (pt, pl <= h),
 * + act (0 none, 1 SiLU), out [Ho][Wo][C] (C % 8 == 0), chsum[c] = sum of the outputs */
void pc_dwk_f32(const float *xp, int H, int W, int C, int k, int s, int pt, int pl, const float *w /*[k*k][C]*/,
                const float *b, int act, float *out, int Ho, int Wo, float *chsum);
void pc_dwk_f32_cr(const float *xp, int H, int W, int C, int k, int s, int pt, int pl, const float *w, const float *b,
                   int act, float *out, int Ho, int Wo, float *chsum, int c0, int c1); /* channels [c0, c1), 8-aligned */
void pc_dwk_f16(const f16 *xp, int H, int W, int C, int k, int s, int pt, int pl, const f16 *w, const f16 *b, int act,
                f16 *out, int Ho, int Wo, float *chsum);
void pc_dwk_f16_cr(const f16 *xp, int H, int W, int C, int k, int s, int pt, int pl, const f16 *w, const f16 *b, int act,
                   f16 *out, int Ho, int Wo, float *chsum, int c0, int c1); /* channels [c0, c1), 8-aligned */
/* copy [H][W] pixels (row stride ld channels, C used) into a zero-haloed [(H+2h)][(W+2h)][C] buffer (interior only) */
void pc_pad_f32(const float *x, int H, int W, int C, int ld, int h, float *xp);
void pc_pad_f16(const f16 *x, int H, int W, int C, int ld, int h, f16 *xp);
/* fp32 [M][C] (stride ld) -> fp16 into a zero-haloed buffer (halo h, width W, channel stride ldo) */
void pc_cvt_pad_f16(const float *x, int M, int W, int C, int ld, int h, int ldo, f16 *xp);
/* mean over P pixels (row stride ld) -> out [C] fp32 */
void pc_mean_f32(const float *x, int P, int C, int ld, float *out);
void pc_mean_f16(const f16 *x, int P, int C, int ld, float *out);
/* x / (||x|| + eps) per pixel (fp32 norm) -> a [P][C] */
void pc_l2norm_f32(const float *x, int P, int C, int ld, float eps, float *a);
void pc_l2norm_f16(const f16 *x, int P, int C, int ld, float eps, f16 *a);
/* int8 SDOT prototype head (pc_proto_s8.c, A76 + dotprod): per-cell symmetric int8 quantisation of the cells
 * (packed [P/8][K/4][8][4], scales of the normalised cells) and the column maxima over the cells for prototypes
 * packed [Np/12][K/4][12][4] with per-prototype scales sw; P % 8 == 0, K % 8 == 0, Np % 12 == 0 */
void pc_quant_cells_f16(const f16 *x, int P, int C, int ld, float eps, int8_t *qa, float *sx);
void pc_proto_colmax_s8(const int8_t *qa, const float *sx, int P, int K, const int8_t *qw, const float *sw, int Np,
                        float *colmax);
void pc_proto_colmax_s8_range(const int8_t *qa, const float *sx, int P, int K, const int8_t *qw, const float *sw,
                              float *colmax, int n_begin, int n_end); /* prototypes [n_begin, n_end), multiples of 12 */
/* the same head for CPUs without dotprod (proto_s8n.c, baseline ARMv8-A): cells fp32 -> qa [P][Kp] row-major int8
 * in [-127, 127] (never -128; zero beyond C; C % 8 == 0, 0 < C <= Kp) with the scales of the normalised cells; column
 * maxima of panels [p0, p1) for prototypes packed [Np/4][Kp/16][4][16] (P % 4 == 0, Kp % 16 == 0). n4 (or NULL): one
 * byte per panel, nonzero = every int16 lane's 4 weights over two 16-byte steps have sum |w| <= 258, so 4 products
 * per lane are exact for activations in [-127, 127] (with n4 flags the activations must not hold -128). For equal
 * int8 inputs the maxima equal pc_proto_colmax_s8's. */
void pc_quant_cells_f32(const float *x, int P, int C, int ld, float eps, int8_t *qa, int Kp, float *sx);
/* the first flagged panel whose weights break the n4 rule (a lane's 4 weights per 32-byte step sum to > 258), or -1:
 * the loader runs it once so a package cannot make the n4 K loop wrap */
int pc_proto_n4_check(const int8_t *qw, int Kp, const uint8_t *n4, int npanels);
void pc_proto_colmax_s8n(const int8_t *qa, int Kp, const float *sx, int P, const int8_t *qw, const float *sw,
                         const uint8_t *n4, int p0, int p1, float *colmax);
/* fp32 arithmetic on prototypes stored in fp16 panels [Np/8][K][8] (FCVTL; baseline ARMv8-A): colmax [Np] */
void pc_proto_colmax_f32h(const float *a, int P, int K, const uint16_t *w16, int Np, float *colmax);
/* logits[c] = b[c] + sum_q w[c][q] colmax[c * ppc + q]; returns -2 if a logit is not finite */
int pc_proto_logits(const float *colmax, const float *w, const float *b, int nc, int ppc, float *logits);
#endif
