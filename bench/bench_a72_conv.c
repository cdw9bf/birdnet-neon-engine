// Cortex-A72 / A53 calibration for a BirdNET v3.0 NEON path without dotprod or fp16 arithmetic (experiment
// H002; summaries in results/v3_older_hw/). Standalone: links only src/gemm_f32.c (the engine's tuned 8x8 fp32
// FMLA kernel, unchanged); the int8 kernel and the Winograd transforms below are benchmark code, not engine code.
//
//   gcc -O3 -mcpu=cortex-a72 -std=gnu11 -o bench_a72_conv bench/bench_a72_conv.c src/gemm_f32.c -lm -lpthread
//   taskset -c 3 ./bench_a72_conv 1      (argument: threads for the multi-thread rows)
//
// 1. Peak instruction throughput: FMLA .4s; the int8 widening pattern SMULL(8b) + SMLAL2(16b) -> int16 + SADALP ->
//    int32 (16 MACs per 3 instructions, operands in [-127, 127] so the int16 pair sum cannot overflow); SMLAL .4s by
//    element on int16 operands (4 MACs per instruction: the classic non-dotprod int8 GEMM form); MLA .8h int16.
// 2. The representative dense 3x3 conv of BirdNET v3.0, b1.1.exp (FusedMBConv expansion, 48 -> 192 channels, 56 x 117
//    output, stride 1, pad 1; 543 MMAC = 8.8 % of the model; b1.1-b1.3 are 26.5 %):
//    fp32 direct (gemm_f32_seg: 3 patch-row segments on the padded NHWC input, no im2col) with and without SiLU;
//    fp32 im2col + gemm_f32; int8 direct with the SMULL/SMLAL2/SADALP 4x4 kernel (segments, per-tensor scale) and
//    fp32 dequant (+ SiLU) epilogue; fp32 Winograd F(2x2,3x3) and F(4x4,3x3): input transform + 16 / 36 GEMMs +
//    output transform (+ bias + SiLU), transforms vectorized over 4 channels, not tuned; int8 Winograd F(2,3) GEMMs.
// 3. 1x1 conv shapes (49 % of the model's MACs): fp32 vs int8 GEMM.
// 4. The same direct conv split over T threads (rows of the output), fp32 and int8.
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/gemm_f32.h"

#define INL static inline __attribute__((always_inline))
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *amalloc(size_t n) { void *p = aligned_alloc(64, (n + 127) & ~(size_t)63); memset(p, 0, (n + 127) & ~(size_t)63); return p; }
static float *frnd(size_t n, float s) {
  float *p = amalloc(n * 4);
  for (size_t i = 0; i < n; i++) p[i] = s * (float)(rand() / (double)RAND_MAX - 0.5);
  return p;
}
static int8_t *irnd(size_t n) {
  int8_t *p = amalloc(n);
  for (size_t i = 0; i < n; i++) p[i] = (int8_t)(rand() % 255 - 127);
  return p;
}
static int g_reps = 10;
#define TIME_BEST(var, ...)                                                                    \
  do {                                                                                          \
    __VA_ARGS__;                                                                                \
    double b_ = 1e9;                                                                            \
    for (int r_ = 0; r_ < g_reps; r_++) { double t_ = now_s(); __VA_ARGS__; t_ = now_s() - t_; if (t_ < b_) b_ = t_; } \
    var = b_;                                                                                   \
  } while (0)

/* ---------------------------------------------------------------- 1. peaks */
#define REP8(X) X(16, 24) X(17, 25) X(18, 26) X(19, 27) X(20, 28) X(21, 29) X(22, 30) X(23, 31)
static double peak_fmla(long n) { /* 24 independent FMLA .4s per iteration = 96 MACs */
  double t = now_s();
  __asm__ volatile("1:\n"
#define F1(a, b) "fmla v" #a ".4s, v0.4s, v1.4s\n fmla v" #b ".4s, v0.4s, v2.4s\n"
                   REP8(F1)
#undef F1
                   "fmla v8.4s, v0.4s, v1.4s\n fmla v9.4s, v0.4s, v1.4s\n fmla v10.4s, v0.4s, v1.4s\n fmla v11.4s, v0.4s, v1.4s\n"
                   "fmla v12.4s, v0.4s, v1.4s\n fmla v13.4s, v0.4s, v1.4s\n fmla v14.4s, v0.4s, v1.4s\n fmla v15.4s, v0.4s, v1.4s\n"
                   "subs %[n], %[n], #1\n b.ne 1b\n"
                   : [n] "+r"(n) : : "v0", "v1", "v2", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
                     "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc");
  return now_s() - t;
}
static double peak_smull(long n) { /* 8 x (SMULL 8b + SMLAL2 16b + SADALP) per iteration = 128 MACs */
  double t = now_s();
  __asm__ volatile("1:\n"
#define S1(a, b) "smull v" #a ".8h, v0.8b, v1.8b\n"
                   REP8(S1)
#undef S1
#define S2(a, b) "smlal2 v" #a ".8h, v0.16b, v1.16b\n"
                   REP8(S2)
#undef S2
#define S3(a, b) "sadalp v" #b ".4s, v" #a ".8h\n"
                   REP8(S3)
#undef S3
                   "subs %[n], %[n], #1\n b.ne 1b\n"
                   : [n] "+r"(n) : : "v0", "v1", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                     "v27", "v28", "v29", "v30", "v31", "cc");
  return now_s() - t;
}
static double peak_smlal_lane(long n) { /* 16 SMLAL(2) .4s by element per iteration = 64 MACs */
  double t = now_s();
  __asm__ volatile("1:\n"
#define L1(a, b) "smlal v" #a ".4s, v0.4h, v1.h[0]\n smlal2 v" #b ".4s, v0.8h, v1.h[1]\n"
                   REP8(L1)
#undef L1
                   "subs %[n], %[n], #1\n b.ne 1b\n"
                   : [n] "+r"(n) : : "v0", "v1", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                     "v27", "v28", "v29", "v30", "v31", "cc");
  return now_s() - t;
}
static double peak_mla16(long n) { /* 16 MLA .8h per iteration = 128 MACs (int16 accumulate: overflows; peak only) */
  double t = now_s();
  __asm__ volatile("1:\n"
#define M1(a, b) "mla v" #a ".8h, v0.8h, v1.8h\n mla v" #b ".8h, v0.8h, v1.8h\n"
                   REP8(M1)
#undef M1
                   "subs %[n], %[n], #1\n b.ne 1b\n"
                   : [n] "+r"(n) : : "v0", "v1", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                     "v27", "v28", "v29", "v30", "v31", "cc");
  return now_s() - t;
}

/* ---------------------------------------------------------------- int8 GEMM: SMULL / SMLAL2 / SADALP, 4x4 tile */
typedef struct {
  int N, Kp;
  int8_t *w;   /* [N/4][Kp/16][4][16] */
  float *sw;   /* per output channel scale */
  float *bias;
} i8pack;
static void i8_pack(i8pack *p, const int8_t *w /*[N][Kp]*/, int N, int Kp) {
  p->N = N; p->Kp = Kp;
  p->w = amalloc((size_t)N * Kp);
  p->sw = frnd(N, 0.01f); p->bias = frnd(N, 0.1f);
  for (int n = 0; n < N; n++)
    for (int k = 0; k < Kp; k++) p->w[(((size_t)(n / 4) * (Kp / 16) + k / 16) * 4 + n % 4) * 16 + k % 16] = w[(size_t)n * Kp + k];
}
/* A rows: row m, segment s at A + m * lda + s * segstride (bytes), nseg * segk = Kp, segk % 16 == 0.
 * C[m][n] = act(acc * sa * sw[n] + bias[n]) as fp32, row stride ldc. M % 4 == 0 (rows beyond are computed). */
typedef struct { const int8_t *A; long lda, segstride; int nseg, segk, m0, m1; const i8pack *W; float sa; float *C; int ldc; int act; } i8job;
INL int32x4_t hsum4(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d) {
  return vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d));
}
#define PAIR(acc, a, w) acc = vpadalq_s16(acc, vmlal_high_s8(vmull_s8(vget_low_s8(a), vget_low_s8(w)), a, w))
static void i8_gemm(const i8job *j) {
  const i8pack *W = j->W;
  const int nkb = W->Kp / 16, kbs = j->segk / 16;
  for (int m = j->m0; m < j->m1; m += 4) {
    const int8_t *ar[4];
    for (int i = 0; i < 4; i++) ar[i] = j->A + (long)(m + i) * j->lda;
    for (int nb = 0; nb < W->N / 4; nb++) {
      const int8_t *wp = W->w + (size_t)nb * nkb * 64;
      int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00;
      int32x4_t c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00, c33 = c00;
      for (int s = 0; s < j->nseg; s++) {
        const long so = s * j->segstride;
        for (int kb = 0; kb < kbs; kb++, wp += 64) {
          int8x16_t a0 = vld1q_s8(ar[0] + so + kb * 16), a1 = vld1q_s8(ar[1] + so + kb * 16);
          int8x16_t a2 = vld1q_s8(ar[2] + so + kb * 16), a3 = vld1q_s8(ar[3] + so + kb * 16);
          int8x16_t w0 = vld1q_s8(wp), w1 = vld1q_s8(wp + 16), w2 = vld1q_s8(wp + 32), w3 = vld1q_s8(wp + 48);
          PAIR(c00, a0, w0); PAIR(c01, a0, w1); PAIR(c02, a0, w2); PAIR(c03, a0, w3);
          PAIR(c10, a1, w0); PAIR(c11, a1, w1); PAIR(c12, a1, w2); PAIR(c13, a1, w3);
          PAIR(c20, a2, w0); PAIR(c21, a2, w1); PAIR(c22, a2, w2); PAIR(c23, a2, w3);
          PAIR(c30, a3, w0); PAIR(c31, a3, w1); PAIR(c32, a3, w2); PAIR(c33, a3, w3);
        }
      }
      int32x4_t r[4] = {hsum4(c00, c01, c02, c03), hsum4(c10, c11, c12, c13), hsum4(c20, c21, c22, c23),
                        hsum4(c30, c31, c32, c33)};
      float32x4_t sc = vmulq_n_f32(vld1q_f32(W->sw + nb * 4), j->sa), bi = vld1q_f32(W->bias + nb * 4);
      for (int i = 0; i < 4; i++) {
        float32x4_t x = vfmaq_f32(bi, vcvtq_f32_s32(r[i]), sc);
        if (j->act) x = silu_f32x4(x);
        vst1q_f32(j->C + (size_t)(m + i) * j->ldc + nb * 4, x);
      }
    }
  }
}

/* ---------------------------------------------------------------- Winograd transforms (fp32, NHWC, 4 channels per vector) */
static const float BT2[16] = {1, 0, -1, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, 0, -1};
static const float AT2[8] = {1, 1, 1, 0, 0, 1, -1, -1};
static const float BT4[36] = {4, 0, -5, 0, 1, 0, 0, -4, -4, 1, 1, 0, 0, 4, -4, -1, 1, 0,
                              0, -2, -1, 2, 1, 0, 0, 2, -1, -2, 1, 0, 0, 4, 0, -5, 0, 1};
static const float AT4[24] = {1, 1, 1, 1, 1, 0, 0, 1, -1, 2, -2, 0, 0, 1, 1, 4, 4, 0, 0, 1, -1, 8, -8, 1};
INL float32x4_t lin(const float *T, int r, int C, const float32x4_t *v, int st) {
  float32x4_t s = vdupq_n_f32(0);
  for (int c = 0; c < C; c++) {
    float k = T[r * C + c];
    if (k == 1.f) s = vaddq_f32(s, v[c * st]);
    else if (k == -1.f) s = vsubq_f32(s, v[c * st]);
    else if (k != 0.f) s = vfmaq_n_f32(s, v[c * st], k);
  }
  return s;
}
/* input transform: padded input xp [Hp][Wp][C] -> V[a*a][tiles][C]; tile (ty, tx) reads rows ty*o.., cols tx*o.. */
#define WINO_IN(NAME, A, O, BT)                                                                                   \
  static void NAME(const float *xp, int Wp, int C, int TY, int TX, float *V) {                                   \
    const size_t ntile = (size_t)TY * TX;                                                                        \
    for (int ty = 0; ty < TY; ty++)                                                                              \
      for (int tx = 0; tx < TX; tx++)                                                                            \
        for (int c = 0; c < C; c += 4) {                                                                         \
          float32x4_t d[A * A], t[A * A];                                                                        \
          for (int i = 0; i < A; i++)                                                                            \
            for (int k = 0; k < A; k++) d[i * A + k] = vld1q_f32(xp + ((size_t)(ty * O + i) * Wp + tx * O + k) * C + c); \
          for (int i = 0; i < A; i++)                                                                            \
            for (int k = 0; k < A; k++) t[i * A + k] = lin(BT, i, A, d + k, A);                                   \
          for (int i = 0; i < A; i++)                                                                            \
            for (int k = 0; k < A; k++)                                                                          \
              vst1q_f32(V + ((size_t)(i * A + k) * ntile + ty * TX + tx) * C + c, lin(BT, k, A, t + i * A, 1));  \
        }                                                                                                        \
  }
WINO_IN(wino_in2, 4, 2, BT2)
WINO_IN(wino_in4, 6, 4, BT4)
/* output transform + bias + SiLU: Mm[a*a][tiles][N] -> y [H][W][N] (edge tiles clipped) */
#define WINO_OUT(NAME, A, O, AT)                                                                                  \
  static void NAME(const float *Mm, int TY, int TX, int N, const float *bias, float *y, int H, int W) {         \
    const size_t ntile = (size_t)TY * TX;                                                                        \
    for (int ty = 0; ty < TY; ty++)                                                                              \
      for (int tx = 0; tx < TX; tx++)                                                                            \
        for (int n = 0; n < N; n += 4) {                                                                         \
          float32x4_t m[A * A], t[O * A];                                                                        \
          for (int p = 0; p < A * A; p++) m[p] = vld1q_f32(Mm + ((size_t)p * ntile + ty * TX + tx) * N + n);     \
          for (int i = 0; i < O; i++)                                                                            \
            for (int k = 0; k < A; k++) t[i * A + k] = lin(AT, i, A, m + k, A);                                   \
          float32x4_t b = vld1q_f32(bias + n);                                                                   \
          for (int i = 0; i < O; i++)                                                                            \
            for (int k = 0; k < O; k++) {                                                                        \
              int yy = ty * O + i, xx = tx * O + k;                                                              \
              float32x4_t v = silu_f32x4(vaddq_f32(b, lin(AT, k, A, t + i * A, 1)));                             \
              if (yy < H && xx < W) vst1q_f32(y + ((size_t)yy * W + xx) * N + n, v);                             \
            }                                                                                                    \
        }                                                                                                        \
  }
WINO_OUT(wino_out2, 4, 2, AT2)
WINO_OUT(wino_out4, 6, 4, AT4)

/* ---------------------------------------------------------------- threads */
typedef struct { int kind; const float *A; int lda, M; const fpack *P; epi32 e; int nseg, segk; long segstride; i8job ij; } job;
static void *run_job(void *p) {
  job *j = p;
  if (j->kind == 0) gemm_f32_seg(j->A, j->lda, j->M, j->nseg, j->segk, j->segstride, j->P, &j->e);
  else i8_gemm(&j->ij);
  return NULL;
}
static void run_threads(job *jobs, int T) {
  pthread_t th[16];
  for (int i = 1; i < T; i++) pthread_create(&th[i], NULL, run_job, &jobs[i]);
  run_job(&jobs[0]);
  for (int i = 1; i < T; i++) pthread_join(th[i], NULL);
}

static void row(const char *tag, double s, double mmac) {
  printf("  %-58s %8.2f ms  %6.2f GMAC/s (model-MAC basis: %.1f MMAC)\n", tag, s * 1e3, mmac / s * 1e-3, mmac);
}

int main(int argc, char **argv) {
  int T = argc > 1 ? atoi(argv[1]) : 1;
  if (T < 1 || T > 16) T = 1;
  if (argc > 2) g_reps = atoi(argv[2]);
  printf("bench_a72_conv: threads for section 4 = %d, best of %d reps\n", T, g_reps);

  /* 1. peaks */
  long n = 20000000;
  double t;
  printf("1. peak throughput (1 core; GMAC/s; per-cycle figures assume 1.5 GHz)\n");
  peak_fmla(n / 10); t = peak_fmla(n); printf("  FMLA .4s (24 acc)                 %6.2f GMAC/s  %.2f MAC/cycle\n", 96.0 * n / t * 1e-9, 96.0 * n / t / 1.5e9);
  t = peak_smull(n); printf("  SMULL+SMLAL2+SADALP (int8->int32)  %6.2f GMAC/s  %.2f MAC/cycle\n", 128.0 * n / t * 1e-9, 128.0 * n / t / 1.5e9);
  t = peak_smlal_lane(n); printf("  SMLAL .4s by element (int16)       %6.2f GMAC/s  %.2f MAC/cycle\n", 64.0 * n / t * 1e-9, 64.0 * n / t / 1.5e9);
  t = peak_mla16(n); printf("  MLA .8h (int16, peak only)         %6.2f GMAC/s  %.2f MAC/cycle\n", 128.0 * n / t * 1e-9, 128.0 * n / t / 1.5e9);

  /* 2. b1.1.exp: 48 -> 192, 3x3 s1 p1, 56 x 117 */
  const int H = 56, W = 117, C = 48, N = 192, Wp = W + 2, Hp = H + 2, K = 9 * C;
  const double mmac = (double)H * W * N * K * 1e-6;
  float *xp = frnd((size_t)(Hp + 4) * Wp * C + 64, 2.f);
  memset(xp, 0, sizeof(float) * Wp * C);
  float *wt = frnd((size_t)N * K, 0.1f), *bias = frnd(N, 0.1f);
  /* weights in [ky][kx][c] order to match the segment layout (3 rows of 3 pixels x C contiguous) */
  fpack P; fp_pack(&P, wt, bias, N, K, K);
  const int Mseg = H * Wp - 2; /* output computed on the padded width; 2 junk pixels per row */
  float *ys = amalloc((size_t)(Mseg + 16) * N * 4);
  epi32 e = {.act = ACT32_NONE, .out = ys, .ldo = N};
  printf("2. b1.1.exp dense 3x3, 48 -> 192, 56 x 117 (%.1f MMAC)\n", mmac);
  TIME_BEST(t, gemm_f32_seg(xp, C, Mseg, 3, 3 * C, (long)Wp * C, &P, &e)); row("fp32 direct (gemm_f32_seg, no act)", t, mmac);
  e.act = ACT32_SILU;
  TIME_BEST(t, gemm_f32_seg(xp, C, Mseg, 3, 3 * C, (long)Wp * C, &P, &e)); row("fp32 direct + SiLU", t, mmac);
  double t_f32_direct = t;
  /* im2col + gemm */
  const int M = H * W;
  float *col = amalloc((size_t)(M + 16) * K * 4);
  TIME_BEST(t, {
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++)
        for (int ky = 0; ky < 3; ky++) memcpy(col + ((size_t)y * W + x) * K + ky * 3 * C, xp + ((size_t)(y + ky) * Wp + x) * C, 3 * C * 4);
  });
  row("fp32 im2col alone", t, 0.001); double t_im2col = t;
  float *yc = amalloc((size_t)(M + 16) * N * 4);
  epi32 e2 = {.act = ACT32_SILU, .out = yc, .ldo = N};
  TIME_BEST(t, gemm_f32(col, K, M, &P, &e2)); row("fp32 gemm on im2col + SiLU (excl. im2col)", t, mmac);
  printf("  -> im2col + gemm total %.2f ms\n", (t + t_im2col) * 1e3);

  /* int8 direct */
  int8_t *xq = irnd((size_t)(Hp + 4) * Wp * C + 64);
  int8_t *wq = irnd((size_t)N * K);
  i8pack Q; i8_pack(&Q, wq, N, K);
  float *yq = amalloc((size_t)(Mseg + 16) * N * 4);
  i8job ij = {.A = xq, .lda = C, .segstride = (long)Wp * C, .nseg = 3, .segk = 3 * C, .m0 = 0, .m1 = Mseg & ~3,
              .W = &Q, .sa = 0.02f, .C = yq, .ldc = N, .act = 0};
  TIME_BEST(t, i8_gemm(&ij)); row("int8 direct SMULL/SMLAL2/SADALP 4x4, dequant", t, mmac);
  ij.act = 1;
  TIME_BEST(t, i8_gemm(&ij)); row("int8 direct + dequant + SiLU", t, mmac);
  double t_i8_direct = t;
  /* activation quantization pass (fp32 NHWC input -> int8, per-tensor absmax known): cost per input element */
  TIME_BEST(t, {
    float32x4_t inv = vdupq_n_f32(127.f / 2.f);
    for (size_t i = 0; i < (size_t)Hp * Wp * C; i += 8) {
      int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + i), inv)), q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp + i + 4), inv));
      vst1_s8(xq + i, vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1))));
    }
  });
  row("int8 input quantization pass (58 x 119 x 48)", t, 0.001);

  /* Winograd F(2,3) fp32 */
  {
    const int TY = H / 2, TX = (W + 1) / 2;
    const size_t nt = (size_t)TY * TX;
    float *V = amalloc(16 * (nt + 16) * C * 4), *Mm = amalloc(16 * (nt + 16) * N * 4), *y = amalloc((size_t)H * W * N * 4);
    fpack U[16];
    for (int p = 0; p < 16; p++) { float *u = frnd((size_t)N * C, 0.1f); fp_pack(&U[p], u, NULL, N, C, C); free(u); }
    double ti, tg, to;
    TIME_BEST(ti, wino_in2(xp, Wp, C, TY, TX, V));
    TIME_BEST(tg, for (int p = 0; p < 16; p++) { epi32 ee = {.act = ACT32_NONE, .out = Mm + p * nt * N, .ldo = N}; gemm_f32(V + p * nt * C, C, nt, &U[p], &ee); });
    TIME_BEST(to, wino_out2(Mm, TY, TX, N, bias, y, H, W));
    double wm = 16.0 * nt * C * N * 1e-6;
    row("fp32 F(2,3): input transform", ti, 0.001);
    row("fp32 F(2,3): 16 GEMMs (K = 48)", tg, wm);
    row("fp32 F(2,3): output transform + bias + SiLU", to, 0.001);
    printf("  -> F(2,3) total %.2f ms = %.2f x the fp32 direct conv (%.1f model GMAC/s)\n", (ti + tg + to) * 1e3,
           t_f32_direct / (ti + tg + to), mmac / (ti + tg + to) * 1e-3);
    /* int8 F(2,3) GEMMs (transform-domain int8, 16 GEMMs, K = 48) */
    int8_t *Vq = irnd(16 * (nt + 16) * C);
    i8pack UQ[16];
    for (int p = 0; p < 16; p++) { int8_t *u = irnd((size_t)N * C); i8_pack(&UQ[p], u, N, C); free(u); }
    TIME_BEST(tg, for (int p = 0; p < 16; p++) {
      i8job jj = {.A = Vq + p * nt * C, .lda = C, .segstride = 0, .nseg = 1, .segk = C, .m0 = 0, .m1 = (int)(nt & ~3),
                  .W = &UQ[p], .sa = 0.02f, .C = Mm + p * nt * N, .ldc = N, .act = 0};
      i8_gemm(&jj);
    });
    row("int8 F(2,3): 16 GEMMs (K = 48), fp32 out", tg, wm);
    printf("  -> int8 F(2,3) est. total (fp32 transforms + int8 GEMMs + a quant pass ~ input transform) %.2f ms\n",
           (2 * ti + tg + to) * 1e3);
  }
  /* Winograd F(4,3) fp32 */
  {
    const int TY = H / 4, TX = (W + 3) / 4;
    const size_t nt = (size_t)TY * TX;
    float *xp4 = frnd((size_t)(TY * 4 + 2 + 4) * (TX * 4 + 2) * C + 64, 2.f);
    float *V = amalloc(36 * (nt + 16) * C * 4), *Mm = amalloc(36 * (nt + 16) * N * 4), *y = amalloc((size_t)H * W * N * 4);
    fpack U[36];
    for (int p = 0; p < 36; p++) { float *u = frnd((size_t)N * C, 0.1f); fp_pack(&U[p], u, NULL, N, C, C); free(u); }
    double ti, tg, to;
    TIME_BEST(ti, wino_in4(xp4, TX * 4 + 2, C, TY, TX, V));
    TIME_BEST(tg, for (int p = 0; p < 36; p++) { epi32 ee = {.act = ACT32_NONE, .out = Mm + p * nt * N, .ldo = N}; gemm_f32(V + p * nt * C, C, nt, &U[p], &ee); });
    TIME_BEST(to, wino_out4(Mm, TY, TX, N, bias, y, H, W));
    double wm = 36.0 * nt * C * N * 1e-6;
    row("fp32 F(4,3): input transform", ti, 0.001);
    row("fp32 F(4,3): 36 GEMMs (K = 48)", tg, wm);
    row("fp32 F(4,3): output transform + bias + SiLU", to, 0.001);
    printf("  -> F(4,3) total %.2f ms = %.2f x the fp32 direct conv (%.1f model GMAC/s)\n", (ti + tg + to) * 1e3,
           t_f32_direct / (ti + tg + to), mmac / (ti + tg + to) * 1e-3);
  }

  /* 3. 1x1 shapes */
  printf("3. 1x1 conv GEMMs (M pixels, K in, N out), + SiLU epilogue\n");
  int shapes[][3] = {{6552, 192, 48}, {1652, 256, 64}, {420, 160, 960}, {420, 960, 160}, {105, 256, 1280}};
  const char *names[] = {"b1.1.pwl", "b2.1.pwl", "b4.1.pw", "b4.1.pwl", "conv_head"};
  for (int s = 0; s < 5; s++) {
    int m = shapes[s][0], k = shapes[s][1], nn = shapes[s][2];
    double mm = (double)m * k * nn * 1e-6;
    float *A = frnd((size_t)(m + 16) * k, 1.f), *w = frnd((size_t)nn * k, 0.1f), *b = frnd(nn, 0.1f), *Cc = amalloc((size_t)(m + 16) * nn * 4);
    fpack Pp; fp_pack(&Pp, w, b, nn, k, k);
    epi32 ee = {.act = ACT32_SILU, .out = Cc, .ldo = nn};
    char tag[96];
    double tf, tq;
    TIME_BEST(tf, gemm_f32(A, k, m, &Pp, &ee));
    int8_t *Aq = irnd((size_t)(m + 16) * k), *wq2 = irnd((size_t)nn * k);
    i8pack Qq; i8_pack(&Qq, wq2, nn, k);
    i8job jj = {.A = Aq, .lda = k, .segstride = 0, .nseg = 1, .segk = k, .m0 = 0, .m1 = (m + 3) & ~3, .W = &Qq, .sa = 0.02f, .C = Cc, .ldc = nn, .act = 1};
    TIME_BEST(tq, i8_gemm(&jj));
    snprintf(tag, sizeof tag, "%-9s M %4d K %4d N %4d fp32", names[s], m, k, nn); row(tag, tf, mm);
    snprintf(tag, sizeof tag, "%-9s M %4d K %4d N %4d int8 (x%.2f)", names[s], m, k, nn, tf / tq); row(tag, tq, mm);
    fp_free(&Pp); free(A); free(w); free(b); free(Cc); free(Aq); free(wq2);
  }

  /* 4. threads: the direct conv split by output rows */
  if (T > 1) {
    printf("4. b1.1.exp direct conv on %d threads (rows split; wall time)\n", T);
    job jobs[16];
    int per = ((Mseg / T) + 7) & ~7;
    for (int i = 0; i < T; i++) {
      int m0 = i * per, m1 = (i == T - 1) ? Mseg : (i + 1) * per;
      jobs[i] = (job){.kind = 0, .A = xp + (size_t)m0 * C, .lda = C, .M = m1 - m0, .P = &P,
                      .e = {.act = ACT32_SILU, .out = ys + (size_t)m0 * N, .ldo = N}, .nseg = 3, .segk = 3 * C, .segstride = (long)Wp * C};
    }
    TIME_BEST(t, run_threads(jobs, T)); row("fp32 direct + SiLU", t, mmac);
    printf("  -> speed-up %.2f x on %d threads (efficiency %.0f %%)\n", t_f32_direct / t, T, 100 * t_f32_direct / t / T);
    for (int i = 0; i < T; i++) {
      int m0 = i * per, m1 = (i == T - 1) ? (Mseg & ~3) : (i + 1) * per;
      jobs[i] = (job){.kind = 1, .ij = ij};
      jobs[i].ij.m0 = m0; jobs[i].ij.m1 = m1;
    }
    TIME_BEST(t, run_threads(jobs, T)); row("int8 direct + dequant + SiLU", t, mmac);
    printf("  -> speed-up %.2f x on %d threads (efficiency %.0f %%)\n", t_i8_direct / t, T, 100 * t_i8_direct / t / T);
  }
  return 0;
}
