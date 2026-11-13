// Package runtime, dotprod + fp16 family (Cortex-A76 class): the mixed-precision plan of dense-conv models such as
// BirdNET v3.0, as registry kernels. Stem in fp32, 24-channel 3x3 convs in fp16 (Winograd F(2,3)), dense 3x3 convs in
// int8 (Winograd F(2,3) with per-tile dynamic quantisation, or im2col with a per-tensor quantisation), 1x1 convs and
// heads in int8 with per-row dynamic activations (SDOT), depthwise in fp16, SE / GeM / attention pooling in fp32.
// Every op wraps the kernels of gemm_s8.c, wino.c, wino_s0.c and kern_f16.c with the arithmetic and call order of the
// retired hand-written BirdNET v3.0 engine, so a compiled package reproduces that engine's numbers from the image
// onwards (docs/GATES.md, K2B-G1).
// Weights arrive as fp32 tensors in the layouts those kernels' packers take; the ops pack them at init.
#include <arm_neon.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "act.h"
#include "kern.h"
#include "rt_ops.h"

#define COLROWS 256 /* im2col rows per GEMM call */
#define MAXW 64     /* threads of a context (rt_set_threads clamps to this) */

/* Threads: every op below splits only over independent outputs (image rows, GEMM rows, Winograd tiles, im2col
 * chunks), each computed by the same code as on one thread, so the bytes do not depend on the thread count.
 * Work areas are per worker, allocated on first use by the calling thread. */
static void *worker_area(void **slot, size_t bytes) {
  if (!*slot) *slot = rt_al64(bytes);
  if (*slot) memset(*slot, 0, bytes);
  return *slot;
}

/* TF "same" padding: output size and the low pad */
static void tf_same(int in, int k, int s, int *out, int *pad_lo) {
  int o = (in + s - 1) / s, tot = (o - 1) * s + k - in;
  *out = o;
  *pad_lo = tot < 0 ? 0 : tot / 2;
}
/* tensor argument i: fp32 with exactly `elems` elements */
static int need_f32_t(rt_ctx *c, rt_op *op, int i, size_t elems) {
  if (rt_need_t(c, op, i, elems * 4, 0)) return RT_E_PARAM;
  const rt_tensor *t = rt_tensor_at(c->pkg, op->t[i], NULL);
  if (t->dtype != RT_F32) return rt_bad(c, op, "tensor argument %d: expected fp32, found %s", i, rt_dtype_name(t->dtype));
  return 0;
}

/* -------- STEM3: b = [img f32 [H][W][3] (no halo, stride 3), out f16 (halo 1) [Ho][Wo][24]];
 *          t = [w OHWI [24][3][3][3], b [24]]; p = [H W Ho Wo]. H and W even: TF-same padding of the stride-2 conv is
 *          then bottom / right only, the case the kernel implements. */
static int stem3_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 2, 4)) return RT_E_PARAM;
  const int *p = op->p;
  RT_REQ(c, op, p[0] >= 4 && p[1] >= 4 && p[0] % 2 == 0 && p[1] % 2 == 0 && p[2] == p[0] / 2 && p[3] == p[1] / 2);
  if (rt_need_haloed(c, op, 0, RT_F32, p[0], p[1], 3, 0) || rt_need_haloed(c, op, 1, RT_F16, p[2], p[3], 24, 1)) return RT_E_PARAM;
  return need_f32_t(c, op, 0, 24 * 27) || need_f32_t(c, op, 1, 24) ? RT_E_PARAM : 0;
}
static int stem3_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  f16 *out = rt_b(c, op, 1);
  rt_zero_halo(out, p[2], p[3], 24, 1, sizeof(f16));
  stem_conv_f32_h(rt_b(c, op, 0), p[0], p[1], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), out);
  return 0;
}

/* -------- CONV3C24: b = [x f16 (halo 1) [H][W][24], out likewise]; the input is also the residual;
 *          t = [w OHWI [24][3][3][24], b [24]]; p = [H W] */
typedef struct { struct wino_s0_t *w; f16 *V[MAXW], *M[MAXW]; } c24_st;
typedef struct { const c24_st *s; const f16 *x; f16 *out; int H, W, per; } c24_job;
static void c24_task(void *arg, int k, int w) {
  const c24_job *j = arg;
  int t0 = k * j->per, t1 = rt_chunk_end(k, j->per, j->H / 2);
  conv3x3_s1_wino_f16_c24_rows(j->x, j->H, j->W, j->s->w, j->out, j->x, j->s->V[w], j->s->M[w], t0, t1);
}
static int c24_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 2, 2)) return RT_E_PARAM;
  const int *p = op->p;
  RT_REQ(c, op, p[0] >= 2 && p[1] >= 2 && p[0] % 2 == 0 && p[1] % 2 == 0); /* 2x2 output tiles */
  if (rt_need_haloed(c, op, 0, RT_F16, p[0], p[1], 24, 1) || rt_need_haloed(c, op, 1, RT_F16, p[0], p[1], 24, 1)) return RT_E_PARAM;
  if (need_f32_t(c, op, 0, 24 * 9 * 24) || need_f32_t(c, op, 1, 24)) return RT_E_PARAM;
  c24_st *s = calloc(1, sizeof *s);
  op->state = s;
  if (!s) return rt_oom(c, op, sizeof *s);
  s->w = wino_s0_pack(rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), p[1]);
  rt_release_t(c, op, 0);
  return s->w ? 0 : rt_oom(c, op, 16 * 24 * 24 * 2);
}
static int c24_run(rt_ctx *c, rt_op *op) {
  c24_st *s = op->state;
  const f16 *x = rt_b(c, op, 0);
  f16 *out = rt_b(c, op, 1);
  int H = op->p[0], W = op->p[1], T = rt_threads(c);
  size_t n = wino_s0_scratch_elems(s->w) * sizeof(f16);
  rt_zero_halo(out, H, W, 24, 1, sizeof(f16));
  /* worker 0 uses the scratch packed with the weights (V[0] / M[0] stay NULL); the others get their own, V zeroed
   * at allocation (its padding tiles are never written) */
  for (int w = 1; w < T; w++)
    if ((!s->V[w] && !worker_area((void **)&s->V[w], n)) || (!s->M[w] && !worker_area((void **)&s->M[w], n))) return RT_E_OOM;
  c24_job j = {s, x, out, H, W, chunk_rows(H / 2, T, 1)};
  rt_par(c, rt_nchunks(H / 2, j.per), c24_task, &j);
  return 0;
}
static void c24_fini(rt_op *op) {
  c24_st *s = op->state;
  if (!s) return;
  wino_s0_free(s->w);
  for (int w = 0; w < MAXW; w++) { free(s->V[w]); free(s->M[w]); }
  free(s);
}

/* -------- UNPAD: b = [x f16 (halo 1) [H][W][C], out f32 [H*W][C]]; p = [H W C] */
static int unpad_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 0, 3)) return RT_E_PARAM;
  const int *p = op->p;
  RT_REQ(c, op, p[0] > 0 && p[1] > 0 && p[2] > 0 && (p[1] * p[2]) % 8 == 0);
  if (rt_need_haloed(c, op, 0, RT_F16, p[0], p[1], p[2], 1)) return RT_E_PARAM;
  return rt_need_rows(c, op, 1, RT_F32, (size_t)p[0] * p[1], p[2]);
}
static int unpad_run(rt_ctx *c, rt_op *op) {
  const int H = op->p[0], W = op->p[1], C = op->p[2];
  const f16 *x = rt_b(c, op, 0);
  float *o = rt_b(c, op, 1);
  for (int y = 0; y < H; y++)
    for (int i = 0; i < W * C; i += 8) {
      float16x8_t h = vld1q_f16(x + ((size_t)(y + 1) * (W + 2) + 1) * C + i);
      vst1q_f32(o + (size_t)y * W * C + i, vcvt_f32_f16(vget_low_f16(h)));
      vst1q_f32(o + (size_t)y * W * C + i + 4, vcvt_high_f32_f16(h));
    }
  return 0;
}

/* -------- WINO_S8: b = [x f32 [H*W][C], out f16 [H*W][K]]; t = [w OHWI [K][3][3][C], b [K]]; p = [H W C K] */
typedef struct { void *w; float *xp; void *work[MAXW]; size_t nwork; } wino_st;
typedef struct { const wino_st *s; int H, W, T, per; f16 *out; } wino_job;
static void wino_task(void *arg, int k, int w) {
  const wino_job *j = arg;
  int t0 = k * j->per, t1 = rt_chunk_end(k, j->per, j->T);
  wino_f23_tiles(j->s->w, j->s->xp, j->H, j->W, j->out, j->s->work[w], gemm_s8, t0, t1);
}
static int wino_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 2, 4)) return RT_E_PARAM;
  const int *p = op->p;
  int H = p[0], W = p[1], C = p[2], K = p[3];
  RT_REQ(c, op, H > 0 && W > 0 && C > 0 && C % 8 == 0 && C <= 2048 && K > 0 && K % 16 == 0 && K <= 2048);
  if (rt_need_rows(c, op, 0, RT_F32, (size_t)H * W, C) || rt_need_rows(c, op, 1, RT_F16, (size_t)H * W, K)) return RT_E_PARAM;
  if (need_f32_t(c, op, 0, (size_t)K * 9 * C) || need_f32_t(c, op, 1, (size_t)K)) return RT_E_PARAM;
  wino_st *s = calloc(1, sizeof *s);
  op->state = s;
  if (!s) return rt_oom(c, op, sizeof *s);
  s->w = wino_pack(rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), K, C);
  rt_release_t(c, op, 0);
  s->xp = rt_zalloc64(wino_xpad_bytes(H, W, C));
  s->nwork = wino_work_bytes(C, K);
  return s->w && s->xp ? 0 : rt_oom(c, op, wino_xpad_bytes(H, W, C));
}
static int wino_run(rt_ctx *c, rt_op *op) {
  wino_st *s = op->state;
  const int *p = op->p;
  int H = p[0], W = p[1], T = rt_threads(c), tiles = ((H + 1) / 2) * ((W + 1) / 2);
  for (int w = 0; w < T; w++)
    if (!s->work[w] && !worker_area(&s->work[w], s->nwork)) return RT_E_OOM;
  wino_f23_pad(rt_b(c, op, 0), H, W, p[2], s->xp);
  wino_job j = {s, H, W, tiles, chunk_rows(tiles, T, WINO_CHUNK), rt_b(c, op, 1)};
  rt_par(c, rt_nchunks(tiles, j.per), wino_task, &j);
  return 0;
}
static void wino_fini(rt_op *op) {
  wino_st *s = op->state;
  if (!s) return;
  wino_free(s->w);
  free(s->xp);
  for (int w = 0; w < MAXW; w++) free(s->work[w]);
  free(s);
}

/* -------- QCONV3: b = [x f32 [H*W][C], out f16 [Ho*Wo][K]]; t = [w OHWI [K][3][3][C], b [K]]; p = [H W C K s Ho Wo].
 * Per-tensor asymmetric int8 input in a padded copy (halo = the zero point), im2col in chunks, SDOT GEMM + SiLU. */
typedef struct { qweights q; int8_t *xp; void *col[MAXW]; int pt, pl; /* TF-same low pads */ } qc_st;
typedef struct { const qc_st *st; const int *p; rowq qt; f16 *out; } qc_job;
static void qconv_task(void *arg, int k, int w) {
  const qc_job *j = arg;
  const int *p = j->p;
  int K = p[3], M = p[5] * p[6], p0 = k * COLROWS, np = rt_chunk_end(k, COLROWS, M) - p0;
  int8_t *col = j->st->col[w];
  im2col_s8(j->st->xp, p[0], p[1], p[2], p[4], j->st->pt, j->st->pl, p[6], p0, np, col, j->st->q.Kp);
  epi e = {.act = ACT_SILU, .out16 = j->out + (size_t)p0 * K, .ldo = K};
  gemm_s8(col, j->st->q.Kp, &j->qt, 1, np, &j->st->q, &e);
}
static int qconv_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 2, 7)) return RT_E_PARAM;
  const int *p = op->p;
  int H = p[0], W = p[1], C = p[2], K = p[3], s = p[4], Ho, Wo, pt, pl;
  RT_REQ(c, op, H > 0 && W > 0 && C > 0 && C % 8 == 0 && K > 0 && K % 16 == 0 && (s == 1 || s == 2));
  tf_same(H, 3, s, &Ho, &pt);
  tf_same(W, 3, s, &Wo, &pl);
  RT_REQ(c, op, p[5] == Ho && p[6] == Wo); /* TF-same geometry */
  if (rt_need_rows(c, op, 0, RT_F32, (size_t)H * W, C) || rt_need_rows(c, op, 1, RT_F16, (size_t)Ho * Wo, K)) return RT_E_PARAM;
  if (need_f32_t(c, op, 0, (size_t)K * 9 * C) || need_f32_t(c, op, 1, (size_t)K)) return RT_E_PARAM;
  qc_st *st = calloc(1, sizeof *st);
  op->state = st;
  if (!st) return rt_oom(c, op, sizeof *st);
  st->pt = pt;
  st->pl = pl;
  qw_pack(&st->q, rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), K, 9 * C);
  rt_release_t(c, op, 0);
  size_t nx = (size_t)(H + 2) * (W + 2) * C + 64;
  st->xp = rt_zalloc64(nx);
  return st->q.w && st->xp ? 0 : rt_oom(c, op, nx);
}
static int qconv_run(rt_ctx *c, rt_op *op) {
  qc_st *st = op->state;
  const int *p = op->p;
  int H = p[0], W = p[1], M = p[5] * p[6];
  qc_job j = {.st = st, .p = p, .out = rt_b(c, op, 1)};
  for (int w = 0; w < rt_threads(c); w++)
    if (!st->col[w] && !worker_area(&st->col[w], (size_t)COLROWS * st->q.Kp + 64)) return RT_E_OOM;
  quant_tensor_pad_f32(rt_b(c, op, 0), H, W, p[2], st->xp, &j.qt);
  rt_par(c, rt_nchunks(M, COLROWS), qconv_task, &j);
  return 0;
}
static void qconv_fini(rt_op *op) {
  qc_st *st = op->state;
  if (!st) return;
  if (st->q.w) qw_free(&st->q);
  free(st->xp);
  for (int w = 0; w < MAXW; w++) free(st->col[w]);
  free(st);
}

/* -------- QPW: b = [in, out, a8 int8 [M][Kp], rq f32 [2 M], res f32 | -1, gate f16 [K] | -1]; t = [w [N][K], b [N]];
 *          p = [N K M in_dt ldx act out_dt ldo ow opad].
 * in_dt RT_F32: fp32 rows; RT_F16: fp16 rows, each channel first multiplied by the gate (an SE) when one is given.
 * out_dt RT_F16: fp16, zero-haloed by opad (ow = the image width) when opad > 0; RT_F32: fp32 rows + residual. */
static int qpw_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 6, 2, 10)) return RT_E_PARAM;
  const int *p = op->p;
  int N = p[0], K = p[1], M = p[2], idt = p[3], ldx = p[4], act = p[5], odt = p[6], ldo = p[7], ow = p[8], opad = p[9];
  int Kp = (K + 15) & ~15;
  RT_REQ(c, op, N > 0 && N % 16 == 0 && K > 0 && K % 8 == 0 && M > 0 && ldx >= K && ldo >= N);
  RT_REQ(c, op, (idt == RT_F32 || idt == RT_F16) && (odt == RT_F32 || odt == RT_F16));
  RT_REQ(c, op, act == ACT_NONE || act == ACT_SILU || (act == ACT_RELU && odt == RT_F32));
  RT_REQ(c, op, opad == 0 || (opad == 1 && odt == RT_F16 && ow > 0 && M % ow == 0 && ldo == N));
  RT_REQ(c, op, op->b[4] < 0 || (odt == RT_F32 && opad == 0)); /* residual: fp32 rows */
  RT_REQ(c, op, op->b[5] < 0 || idt == RT_F16);                /* gate: fp16 input */
  if (rt_need_nohalo(c, op, 0, idt, (size_t)M * ldx)) return RT_E_PARAM;
  if (opad) {
    if (rt_need_haloed(c, op, 1, RT_F16, M / ow, ow, ldo, 1)) return RT_E_PARAM;
  } else if (rt_need_rows(c, op, 1, odt, (size_t)M, ldo))
    return RT_E_PARAM;
  if (rt_need(c, op, 2, RT_I8, (size_t)(M + 4) * Kp) || rt_need(c, op, 3, RT_F32, (size_t)2 * (M + 4))) return RT_E_PARAM;
  if (op->b[4] >= 0 && rt_need(c, op, 4, RT_F32, (size_t)M * ldo)) return RT_E_PARAM;
  if (op->b[5] >= 0 && rt_need(c, op, 5, RT_F16, (size_t)K)) return RT_E_PARAM;
  if (need_f32_t(c, op, 0, (size_t)N * K) || need_f32_t(c, op, 1, (size_t)N)) return RT_E_PARAM;
  qweights *q = calloc(1, sizeof *q);
  op->state = q;
  if (!q) return rt_oom(c, op, sizeof *q);
  qw_pack(q, rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), N, K);
  rt_release_t(c, op, 0);
  return q->w ? 0 : rt_oom(c, op, (size_t)N * Kp);
}
/* rows [m0, m1) of a GEMM with per-row quantised inputs: quantise them (when x is given), then multiply and store.
 * e describes the whole output; m0 is a multiple of the image width when the output is zero-haloed. */
typedef struct {
  const qweights *q;
  int8_t *a;
  rowq *rq;
  epi e;
  int M, per;
  const void *x; /* rows to quantise first (NULL: a / rq are ready) */
  int x_dt, ldx, K;
  const f16 *gate;
} rows_job;
static void rows_range(const rows_job *j, int m0, int m1) {
  int Kp = j->q->Kp;
  int8_t *a = j->a + (size_t)m0 * Kp;
  if (j->x && j->x_dt == RT_F32) quant_rows_f32((const float *)j->x + (size_t)m0 * j->ldx, j->ldx, m1 - m0, j->K, a, Kp, Kp, j->rq + m0);
  else if (j->x) quant_rows_f16((const f16 *)j->x + (size_t)m0 * j->ldx, j->ldx, m1 - m0, j->K, j->gate, a, Kp, Kp, j->rq + m0);
  epi e = j->e;
  size_t skip = e.opad ? (size_t)(m0 / e.ow) * (e.ow + 2 * e.opad) * e.ldo : (size_t)m0 * e.ldo;
  if (e.out16) e.out16 += skip;
  if (e.out32) e.out32 += skip;
  if (e.res32) e.res32 += (size_t)m0 * e.ldo;
  gemm_s8(a, Kp, j->rq + m0, 0, m1 - m0, j->q, &e);
}
static void rows_task(void *arg, int k, int w) {
  const rows_job *j = arg;
  int m0 = k * j->per;
  rows_range(j, m0, rt_chunk_end(k, j->per, j->M));
}
/* output channels [n0, n1) (multiples of 16) of every row, the rows already quantised: a view of the packed weights */
static void cols_range(const rows_job *j, int n0, int n1) {
  qweights v = *j->q;
  v.N = n1 - n0;
  v.w += (size_t)(n0 / 16) * 16 * v.Kp;
  v.scale += n0; v.swsum += n0; v.bias += n0;
  epi e = j->e;
  if (e.out16) e.out16 += n0;
  if (e.out32) e.out32 += n0;
  if (e.res32) e.res32 += n0;
  gemm_s8(j->a, v.Kp, j->rq, 0, j->M, &v, &e);
}
static void cols_task(void *arg, int k, int w) {
  const rows_job *j = arg;
  int n0 = k * j->per;
  cols_range(j, n0, rt_chunk_end(k, j->per, j->q->N));
}
/* A GEMM over threads. Many rows (activation maps): chunks of rows, each quantised and multiplied by its thread;
 * unit = rows a chunk boundary must be a multiple of (the image width for a zero-haloed output, else 4). Few rows
 * (late stages, heads): the rows are quantised first and the output channels are split, so each thread streams its
 * own part of the weights once. Either way every output element is computed by the same code as on one thread. */
static void rows_par(rt_ctx *c, rows_job *j, int unit) {
  int T = rt_threads(c), N = j->q->N;
  if (T < 2) { rows_range(j, 0, j->M); return; }
  if (j->M >= 64 * T && j->M >= 2 * unit) {
    j->per = chunk_rows(j->M, T, unit);
    rt_par(c, rt_nchunks(j->M, j->per), rows_task, j);
    return;
  }
  if (N < 32) { rows_range(j, 0, j->M); return; }
  if (j->x) { /* quantise every row once, on this thread */
    int Kp = j->q->Kp;
    if (j->x_dt == RT_F32) quant_rows_f32(j->x, j->ldx, j->M, j->K, j->a, Kp, Kp, j->rq);
    else quant_rows_f16(j->x, j->ldx, j->M, j->K, j->gate, j->a, Kp, Kp, j->rq);
    j->x = NULL;
  }
  j->per = chunk_rows(N, T, 16);
  rt_par(c, rt_nchunks(N, j->per), cols_task, j);
}
static int qpw_run(rt_ctx *c, rt_op *op) {
  const qweights *q = op->state;
  const int *p = op->p;
  int M = p[2];
  rows_job j = {.q = q, .a = rt_b(c, op, 2), .rq = rt_b(c, op, 3), .M = M, .x = rt_b(c, op, 0), .x_dt = p[3], .ldx = p[4],
                .K = p[1], .gate = rt_b(c, op, 5),
                .e = {.act = p[5], .ldo = p[7], .ow = p[8], .opad = p[9], .res32 = rt_b(c, op, 4)}};
  if (p[6] == RT_F16) {
    j.e.out16 = rt_b(c, op, 1);
    if (p[9]) rt_zero_halo(j.e.out16, M / p[8], p[8], p[7], 1, sizeof(f16));
  } else
    j.e.out32 = rt_b(c, op, 1);
  rows_par(c, &j, p[9] ? p[8] : 4);
  return 0;
}
static void qw_fini(rt_op *op) {
  qweights *q = op->state;
  if (!q) return;
  if (q->w) qw_free(q);
  free(q);
}

/* -------- SE_GATE16: b = [chsum f32 [C] (scaled to means in place), gate f16 [C]];
 *          t = [reduce w [rd][C], reduce b [rd], expand w [C][rd], expand b [C]] (fp32); p = [C rd P].
 * The weights are held in fp16 (they are streamed once per window); the arithmetic is fp32. */
typedef struct { f16 *wr, *wt; float *g, *tmp; const float *br, *be; } seg_st;
static int seg_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 4, 3)) return RT_E_PARAM;
  int C = op->p[0], rd = op->p[1];
  RT_REQ(c, op, C > 0 && C % 8 == 0 && rd > 0 && op->p[2] > 0);
  if (rt_need(c, op, 0, RT_F32, (size_t)C) || rt_need(c, op, 1, RT_F16, (size_t)C)) return RT_E_PARAM;
  if (need_f32_t(c, op, 0, (size_t)rd * C) || need_f32_t(c, op, 1, (size_t)rd) || need_f32_t(c, op, 2, (size_t)C * rd) ||
      need_f32_t(c, op, 3, (size_t)C))
    return RT_E_PARAM;
  seg_st *s = calloc(1, sizeof *s);
  op->state = s;
  if (!s) return rt_oom(c, op, sizeof *s);
  size_t n = (size_t)C * rd;
  s->wr = rt_zalloc64(sizeof(f16) * n);
  s->wt = rt_zalloc64(sizeof(f16) * n);
  s->g = rt_zalloc64(sizeof(float) * C);
  s->tmp = rt_zalloc64(sizeof(float) * rd);
  if (!s->wr || !s->wt || !s->g || !s->tmp) return rt_oom(c, op, 4 * n);
  const float *wr = rt_t(c, op, 0, NULL), *we = rt_t(c, op, 2, NULL);
  for (int ch = 0; ch < C; ch++)
    for (int r = 0; r < rd; r++) {
      s->wt[(size_t)r * C + ch] = (f16)we[(size_t)ch * rd + r];
      s->wr[(size_t)r * C + ch] = (f16)wr[(size_t)r * C + ch];
    }
  s->br = rt_t(c, op, 1, NULL);
  s->be = rt_t(c, op, 3, NULL);
  rt_release_t(c, op, 0);
  rt_release_t(c, op, 2);
  return 0;
}
static int seg_run(rt_ctx *c, rt_op *op) {
  const seg_st *s = op->state;
  int cm = op->p[0], rd = op->p[1];
  float *chsum = rt_b(c, op, 0), *g = s->g;
  f16 *gate16 = rt_b(c, op, 1);
  float inv = 1.f / op->p[2];
  for (int ch = 0; ch < cm; ch++) chsum[ch] *= inv;
  for (int r = 0; r < rd; r++) {
    const f16 *wr = s->wr + (size_t)r * cm;
    float32x4_t acc = vdupq_n_f32(0), acc2 = acc;
    for (int ch = 0; ch < cm; ch += 8) {
      float16x8_t h = vld1q_f16(wr + ch);
      acc = vfmaq_f32(acc, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(chsum + ch));
      acc2 = vfmaq_f32(acc2, vcvt_high_f32_f16(h), vld1q_f32(chsum + ch + 4));
    }
    float v = vaddvq_f32(vaddq_f32(acc, acc2)) + s->br[r];
    s->tmp[r] = v / (1.f + expf(-v));
  }
  memcpy(g, s->be, sizeof(float) * cm);
  for (int r = 0; r < rd; r++) {
    float t = s->tmp[r];
    const f16 *w2 = s->wt + (size_t)r * cm;
    for (int ch = 0; ch < cm; ch += 8) {
      float16x8_t h = vld1q_f16(w2 + ch);
      vst1q_f32(g + ch, vfmaq_n_f32(vld1q_f32(g + ch), vcvt_f32_f16(vget_low_f16(h)), t));
      vst1q_f32(g + ch + 4, vfmaq_n_f32(vld1q_f32(g + ch + 4), vcvt_high_f32_f16(h), t));
    }
  }
  for (int ch = 0; ch < cm; ch += 4) { /* sigmoid: exp2 polynomial + reciprocal estimate, two Newton steps */
    float32x4_t v = vld1q_f32(g + ch);
    float32x4_t e = exp2_p5_f32(vminq_f32(vmaxq_f32(vmulq_n_f32(v, -1.44269504f), vdupq_n_f32(-126.f)), vdupq_n_f32(126.f)));
    float32x4_t d = vaddq_f32(e, vdupq_n_f32(1.f));
    float32x4_t r0 = vrecpeq_f32(d);
    r0 = vmulq_f32(r0, vrecpsq_f32(d, r0));
    r0 = vmulq_f32(r0, vrecpsq_f32(d, r0));
    vst1q_f32(g + ch, r0);
  }
  for (int ch = 0; ch < cm; ch += 8)
    vst1q_f16(gate16 + ch, vcombine_f16(vcvt_f16_f32(vld1q_f32(g + ch)), vcvt_f16_f32(vld1q_f32(g + ch + 4))));
  return 0;
}
static void seg_fini(rt_op *op) {
  seg_st *s = op->state;
  if (!s) return;
  free(s->wr); free(s->wt); free(s->g); free(s->tmp);
  free(s);
}

/* -------- GEM_A: b = [F f32 [M][C], e f32 [C]]; t = [consts {eps, p}]; p = [M C].
 * e[c] = (mean over rows of max(F, eps)^p)^(1/p), the inner power as exp2(p log2 x) in vectors */
static int gem_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 1, 2)) return RT_E_PARAM;
  int M = op->p[0], C = op->p[1];
  RT_REQ(c, op, M > 0 && C > 0 && C % 4 == 0);
  if (rt_need_rows(c, op, 0, RT_F32, (size_t)M, C) || rt_need(c, op, 1, RT_F32, (size_t)C) || need_f32_t(c, op, 0, 2)) return RT_E_PARAM;
  op->state = rt_zalloc64(sizeof(float) * C);
  return op->state ? 0 : rt_oom(c, op, sizeof(float) * C);
}
static int gem_run(rt_ctx *c, rt_op *op) {
  int M = op->p[0], C = op->p[1];
  const float *F = rt_b(c, op, 0), *k = rt_t(c, op, 0, NULL);
  float *e = rt_b(c, op, 1), *acc = op->state, p = k[1];
  memset(acc, 0, sizeof(float) * C);
  float32x4_t vp = vdupq_n_f32(p), veps = vdupq_n_f32(k[0]);
  for (int i = 0; i < M; i++) {
    const float *row = F + (size_t)i * C;
    for (int ch = 0; ch < C; ch += 4) {
      float32x4_t v = vmaxq_f32(vld1q_f32(row + ch), veps);
      vst1q_f32(acc + ch, vaddq_f32(vld1q_f32(acc + ch), exp2_p5_f32(vmaxq_f32(vmulq_f32(vp, log2_f32(v)), vdupq_n_f32(-126.f)))));
    }
  }
  for (int ch = 0; ch < C; ch++) e[ch] = powf(acc[ch] / M, 1.0f / p);
  return 0;
}

/* -------- HEADS_S8: b = [F f32 [H*W][C], e f32 [C], logits f32 [ncp], a8 int8, rq f32];
 *          t = [lin w [nc][C], lin b, then for the head over H and the head over W: fc w [hid][C], fc b, att w [nc][hid],
 *               att b, cla w [nc][hid], cla b]; p = [H W C hid nc].
 * logits = lin(e) + sum over the two heads of the attention-weighted class scores: per time step t the mean of F over
 * the other axis, max-pool 3 + average-pool 3 (zero-padded, weights 0.33333334), fc + ReLU, attention clipped to
 * [-10, 10] and soft-maxed over t, class scores weighted by it. All GEMMs int8 SDOT with per-row dynamic inputs;
 * the class dimension is padded to 16 (ncp). */
typedef struct {
  qweights lin, fc[2], att[2], cla[2];
  float *s, *pooled, *h, *att_o, *cla_o;
} heads_st;
static int pack_padded(qweights *q, const float *w, const float *b, int nc, int ncp, int K, float *tw, float *tb) {
  memset(tw, 0, sizeof(float) * (size_t)ncp * K);
  memset(tb, 0, sizeof(float) * ncp);
  memcpy(tw, w, sizeof(float) * (size_t)nc * K);
  memcpy(tb, b, sizeof(float) * nc);
  qw_pack(q, tw, tb, ncp, K);
  return q->w ? 0 : -1;
}
static int heads_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 5, 14, 5)) return RT_E_PARAM;
  const int *p = op->p;
  int H = p[0], W = p[1], C = p[2], hid = p[3], nc = p[4], ncp = (nc + 15) & ~15, L = H > W ? H : W;
  RT_REQ(c, op, H > 0 && W > 0 && C > 0 && C % 16 == 0 && hid > 0 && hid % 16 == 0 && nc > 0);
  int kq = C > hid ? C : hid;
  if (rt_need_rows(c, op, 0, RT_F32, (size_t)H * W, C) || rt_need(c, op, 1, RT_F32, (size_t)C) || rt_need(c, op, 2, RT_F32, (size_t)ncp) ||
      rt_need(c, op, 3, RT_I8, (size_t)(L + 4) * kq) || rt_need(c, op, 4, RT_F32, (size_t)2 * (L + 4)))
    return RT_E_PARAM;
  if (need_f32_t(c, op, 0, (size_t)nc * C) || need_f32_t(c, op, 1, (size_t)nc)) return RT_E_PARAM;
  for (int i = 0; i < 2; i++) {
    int t = 2 + 6 * i;
    if (need_f32_t(c, op, t, (size_t)hid * C) || need_f32_t(c, op, t + 1, (size_t)hid) ||
        need_f32_t(c, op, t + 2, (size_t)nc * hid) || need_f32_t(c, op, t + 3, (size_t)nc) ||
        need_f32_t(c, op, t + 4, (size_t)nc * hid) || need_f32_t(c, op, t + 5, (size_t)nc))
      return RT_E_PARAM;
  }
  heads_st *s = calloc(1, sizeof *s);
  op->state = s;
  if (!s) return rt_oom(c, op, sizeof *s);
  size_t nw = (size_t)ncp * kq;
  float *tw = calloc(nw, sizeof(float)), *tb = calloc(ncp, sizeof(float));
  int bad = !tw || !tb;
  /* each fp32 matrix is released from the mapping as soon as it is packed, so the peak holds one at a time */
  bad = bad || pack_padded(&s->lin, rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), nc, ncp, C, tw, tb);
  rt_release_t(c, op, 0);
  for (int i = 0; i < 2 && !bad; i++) {
    int t = 2 + 6 * i;
    qw_pack(&s->fc[i], rt_t(c, op, t, NULL), rt_t(c, op, t + 1, NULL), hid, C);
    rt_release_t(c, op, t);
    bad = !s->fc[i].w || pack_padded(&s->att[i], rt_t(c, op, t + 2, NULL), rt_t(c, op, t + 3, NULL), nc, ncp, hid, tw, tb);
    rt_release_t(c, op, t + 2);
    bad = bad || pack_padded(&s->cla[i], rt_t(c, op, t + 4, NULL), rt_t(c, op, t + 5, NULL), nc, ncp, hid, tw, tb);
    rt_release_t(c, op, t + 4);
  }
  free(tw);
  free(tb);
  s->s = rt_zalloc64(sizeof(float) * (L + 4) * C);
  s->pooled = rt_zalloc64(sizeof(float) * (L + 4) * C);
  s->h = rt_zalloc64(sizeof(float) * (L + 4) * hid);
  s->att_o = rt_zalloc64(sizeof(float) * (size_t)(L + 4) * ncp);
  s->cla_o = rt_zalloc64(sizeof(float) * (size_t)(L + 4) * ncp);
  if (bad || !s->s || !s->pooled || !s->h || !s->att_o || !s->cla_o) return rt_oom(c, op, nw * 4);
  return 0;
}
static int heads_run(rt_ctx *c, rt_op *op) {
  const heads_st *st = op->state;
  const int *p = op->p;
  const int H = p[0], W = p[1], C = p[2], hid = p[3], nc = p[4], ncp = st->lin.N;
  const float *F = rt_b(c, op, 0), *e_ = rt_b(c, op, 1);
  float *logits = rt_b(c, op, 2);
  int8_t *a8 = rt_b(c, op, 3);
  rowq *rq = rt_b(c, op, 4);
  quant_rows_f32(e_, C, 1, C, a8, st->lin.Kp, st->lin.Kp, rq);
  epi el = {.act = ACT_NONE, .out32 = logits, .ldo = ncp};
  gemm_s8(a8, st->lin.Kp, rq, 0, 1, &st->lin, &el);
  for (int i = 0; i < 2; i++) {
    int L = i == 0 ? H : W;
    float *s = st->s, *pooled = st->pooled, *h = st->h, *att = st->att_o, *cla = st->cla_o;
    for (int t = 0; t < L; t++)
      for (int ch = 0; ch < C; ch++) {
        float a2 = 0;
        if (i == 0) { for (int u = 0; u < W; u++) a2 += F[(size_t)(t * W + u) * C + ch]; s[t * C + ch] = a2 / W; }
        else { for (int u = 0; u < H; u++) a2 += F[(size_t)(u * W + t) * C + ch]; s[t * C + ch] = a2 / H; }
      }
    for (int t = 0; t < L; t++)
      for (int ch = 0; ch < C; ch++) {
        float mx = -FLT_MAX, sum = 0;
        for (int d = -1; d <= 1; d++) {
          int u = t + d;
          if (u < 0 || u >= L) continue;
          float v = s[u * C + ch];
          mx = fmaxf(mx, v); sum += v * 0.33333334f;
        }
        pooled[t * C + ch] = mx + sum;
      }
    quant_rows_f32(pooled, C, L, C, a8, st->fc[i].Kp, st->fc[i].Kp, rq);
    rows_job jf = {.q = &st->fc[i], .a = a8, .rq = rq, .M = L, .e = {.act = ACT_RELU, .out32 = h, .ldo = hid}};
    rows_par(c, &jf, 4);
    quant_rows_f32(h, hid, L, hid, a8, st->att[i].Kp, st->att[i].Kp, rq);
    rows_job ja = {.q = &st->att[i], .a = a8, .rq = rq, .M = L, .e = {.act = ACT_NONE, .out32 = att, .ldo = ncp}};
    rows_par(c, &ja, 4);
    rows_job jc = {.q = &st->cla[i], .a = a8, .rq = rq, .M = L, .e = {.act = ACT_NONE, .out32 = cla, .ldo = ncp}};
    rows_par(c, &jc, 4);
    for (int k = 0; k < nc; k++) {
      float mx = -FLT_MAX;
      for (int t = 0; t < L; t++) {
        float a2 = att[(size_t)t * ncp + k];
        a2 = a2 < -10.f ? -10.f : (a2 > 10.f ? 10.f : a2);
        att[(size_t)t * ncp + k] = a2;
        mx = fmaxf(mx, a2);
      }
      float z = 0, acc = 0;
      for (int t = 0; t < L; t++) { float ev = expf(att[(size_t)t * ncp + k] - mx); z += ev; acc += ev * cla[(size_t)t * ncp + k]; }
      logits[k] += acc / z;
    }
  }
  return 0;
}
static void heads_fini(rt_op *op) {
  heads_st *s = op->state;
  if (!s) return;
  if (s->lin.w) qw_free(&s->lin);
  for (int i = 0; i < 2; i++) {
    if (s->fc[i].w) qw_free(&s->fc[i]);
    if (s->att[i].w) qw_free(&s->att[i]);
    if (s->cla[i].w) qw_free(&s->cla[i]);
  }
  free(s->s); free(s->pooled); free(s->h); free(s->att_o); free(s->cla_o);
  free(s);
}

const rt_kernel rt_k_dot[RT_OP_N] = {
    [RT_OP_STEM3] = {stem3_init, stem3_run, NULL, "stem3_f32to16", 0x2, 0},
    [RT_OP_CONV3C24] = {c24_init, c24_run, c24_fini, "conv3c24_wino16", 0x2, 0},
    [RT_OP_UNPAD] = {unpad_init, unpad_run, NULL, "unpad_f16to32", 0x2, 0},
    [RT_OP_WINO_S8] = {wino_init, wino_run, wino_fini, "wino3_s8", 0x2, 0},
    [RT_OP_QCONV3] = {qconv_init, qconv_run, qconv_fini, "qconv3_s8", 0x2, 0},
    [RT_OP_QPW] = {qpw_init, qpw_run, qw_fini, "qpw_s8", 0xe, 0x30},
    [RT_OP_SE_GATE16] = {seg_init, seg_run, seg_fini, "se_gate16", 0x3, 0},
    [RT_OP_GEM_A] = {gem_init, gem_run, rt_free_state, "gem_a", 0x2, 0},
    [RT_OP_HEADS_S8] = {heads_init, heads_run, heads_fini, "heads_s8", 0x1c, 0},
};
