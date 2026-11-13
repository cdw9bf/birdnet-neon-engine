// Registry kernels with fp16 arithmetic (Cortex-A76 objects: ARMv8.2 fp16 [+ dotprod]). Thin wrappers over the
// hand-tuned code (gemm_f16.c, kern_f16.c, v24_fast.c stage kernels). Opcode = base | 1 << 8.
#include <stdlib.h>
#include <string.h>

#include "gemm_f16.h"
#include "gemv.h"
#include "kern.h"
#include "pc_kern.h"
#include "rt_ops.h"
#include "v24_internal.h"
#include "v24_kern.h"

/* packed fp16 GEMM weights: t = [w (hpack panels), b32, b16]; p = [N, K, Kp, Np] at offset po */
static int hpack_view(rt_ctx *c, rt_op *op, int ti, int po, hpack *h) {
  if (op->ni < po + 4)
    return rt_bad(c, op, "GEMM shape [N K Kp Np] expected at int params %d..%d, the op has %d ints", po, po + 3, op->ni);
  h->N = op->p[po]; h->K = op->p[po + 1]; h->Kp = op->p[po + 2]; h->Np = op->p[po + 3];
  if (h->Np != ((h->N + 7) & ~7) || h->Kp < h->K || h->Kp % 8)
    return rt_bad(c, op, "GEMM shape N %d K %d Kp %d Np %d: expected Np = N rounded up to 8, Kp >= K, Kp %% 8 == 0",
                  h->N, h->K, h->Kp, h->Np);
  if (rt_need_t(c, op, ti, (size_t)h->Np * h->Kp * 2, 1) || rt_need_t(c, op, ti + 1, (size_t)(h->Np + 16) * 4, 1) ||
      rt_need_t(c, op, ti + 2, (size_t)(h->Np + 16) * 2, 1))
    return RT_E_PARAM;
  h->w = (f16 *)rt_t(c, op, ti, NULL); h->b32 = (float *)rt_t(c, op, ti + 1, NULL); h->b16 = (f16 *)rt_t(c, op, ti + 2, NULL);
  return 0;
}

/* -------- C0POOL f16: b = [img f32, pin f16 [48*128][48]]; t = [w b32 b16]; p = [N K Kp Np] */
typedef struct { hpack h; f16 *img16; const f16 **rows; } c0_st;
static int c0_init(rt_ctx *c, rt_op *op) {
  c0_st *s = calloc(1, sizeof *s);
  if (!s) return rt_oom(c, op, sizeof *s);
  op->state = s;
  if (rt_need_args(c, op, 2, 3, 4) || hpack_view(c, op, 0, 0, &s->h)) return RT_E_PARAM;
  RT_REQ(c, op, s->h.N == 24 && s->h.K == 64 && s->h.Kp == 64);
  if (rt_need(c, op, 0, RT_F32, (size_t)96 * 511 * 2) || rt_need(c, op, 1, RT_F16, (size_t)48 * 128 * 48)) return RT_E_PARAM;
  s->img16 = rt_zalloc64_tail((size_t)98 * 518 * 2 * 2);
  s->rows = malloc(sizeof(f16 *) * 4 * 256);
  return s->img16 && s->rows ? 0 : rt_oom(c, op, (size_t)98 * 518 * 2 * 2 + sizeof(f16 *) * 4 * 256);
}
static int c0_run(rt_ctx *c, rt_op *op) {
  c0_st *s = op->state;
  v24k16_c0_pool(rt_b(c, op, 0), s->img16, s->rows, &s->h, rt_b(c, op, 1));
  return 0;
}
static void c0_fini(rt_op *op) {
  c0_st *s = op->state;
  if (s) { free(s->img16); free(s->rows); }
  free(s);
}

/* -------- PW f16: b = [in, out, res|-1]; t = [w b32 b16]; p = [N K Kp Np, M, lda, act, ldo, ow, opad, ldr] */
static int pw_init(rt_ctx *c, rt_op *op) {
  hpack *h = calloc(1, sizeof *h);
  op->state = h;
  if (!h) return rt_oom(c, op, sizeof *h);
  if (rt_need_args(c, op, 3, 3, 11) || hpack_view(c, op, 0, 0, h)) return RT_E_PARAM;
  return rt_pw_check(c, op, RT_F16, h->Kp, h->Np);
}
typedef struct { const f16 *A; int lda, M; const hpack *W; const epi16 *e; int mb; } pw_job;
static void pw_task(void *arg, int k, int worker) {
  const pw_job *j = arg;
  int m0 = k * j->mb, m1 = rt_chunk_end(k, j->mb, j->M);
  gemm_f16_range(j->A, j->lda, j->M, j->W, j->e, m0, m1);
}
static int pw_run(rt_ctx *c, rt_op *op) {
  const hpack *h = op->state;
  const int *p = op->p;
  f16 *out = rt_b(c, op, 1);
  const rt_bufdesc *o = rt_bd(c, op, 1);
  if (p[9]) rt_zero_halo(out, (int)o->H, (int)o->W, (int)o->ld, p[9], 2);
  epi16 e = {.act = p[6], .out = out, .ldo = p[7], .ow = p[8], .opad = p[9], .res = rt_b(c, op, 2), .ldr = p[10]};
  int T = rt_threads(c), M = p[4];
  if (T > 1 && M >= 16) { /* rows in chunks of whole 8-row tiles (about 4 chunks per thread, taken dynamically) */
    pw_job j = {rt_b(c, op, 0), p[5], M, h, &e, chunk_rows(M, T, 8)};
    rt_par(c, rt_nchunks(M, j.mb), pw_task, &j);
  } else
    gemm_f16(rt_b(c, op, 0), p[5], M, h, &e);
  return 0;
}

/* -------- DW3 f16: b = [xp (halo 1), out, chsum f32]; t = [w f16 [9][C], b f16 [C]]; p = [H W C s Ho Wo pt pl] */
static int dw3_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 3, 2, 8)) return RT_E_PARAM;
  RT_REQ(c, op, op->p[2] <= 2048); /* C: the fp16 kernel's channel-sum scratch */
  return rt_dw3_check(c, op, RT_F16, op->p[6], op->p[7]);
}
/* depthwise and SE splits: 32 channels = 64 B of every fp16 pixel; SE scaling in pixel chunks */
#define CH16_CHUNK 32
#define SE16_PIX 256
typedef struct { rt_ctx *c; rt_op *op; const f16 *g; } op16_job;
static void dw3_task(void *a, int k, int w) {
  const op16_job *j = a;
  const int *p = j->op->p;
  int c0 = k * CH16_CHUNK, c1 = rt_chunk_end(k, CH16_CHUNK, p[2]);
  dw3x3_f16_cr(rt_b(j->c, j->op, 0), p[0], p[1], p[2], p[3], p[6], p[7], rt_t(j->c, j->op, 0, NULL),
               rt_t(j->c, j->op, 1, NULL), rt_b(j->c, j->op, 1), p[4], p[5], rt_b(j->c, j->op, 2), c0, c1);
}
static void dwk_task(void *a, int k, int w) {
  const op16_job *j = a;
  const int *p = j->op->p;
  int c0 = k * CH16_CHUNK, c1 = rt_chunk_end(k, CH16_CHUNK, p[2]);
  pc_dwk_f16_cr(rt_b(j->c, j->op, 0), p[0], p[1], p[2], p[3], p[4], p[5], p[6], rt_t(j->c, j->op, 0, NULL),
                rt_t(j->c, j->op, 1, NULL), p[9], rt_b(j->c, j->op, 1), p[7], p[8], rt_b(j->c, j->op, 2), c0, c1);
}
static void se_task(void *a, int k, int w) {
  const op16_job *j = a;
  int P = j->op->p[2], p0 = k * SE16_PIX, p1 = rt_chunk_end(k, SE16_PIX, P);
  v24k16_se_scale(j->op->p[0], j->g, rt_b(j->c, j->op, 0), p0, p1);
}
static int dw3_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  if (rt_threads(c) > 1 && p[2] >= 2 * CH16_CHUNK) {
    op16_job j = {c, op, NULL};
    rt_par(c, rt_nchunks(p[2], CH16_CHUNK), dw3_task, &j);
    return 0;
  }
  dw3x3_f16(rt_b(c, op, 0), p[0], p[1], p[2], p[3], p[6], p[7], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL),
            rt_b(c, op, 1), p[4], p[5], rt_b(c, op, 2));
  return 0;
}

/* -------- SE f16: b = [D f16 (in place), chsum]; t = [se1_w, se1_b, se2t, se2_b] (fp32); p = [C rd P] */
static int se_init(rt_ctx *c, rt_op *op) { return rt_se_check(c, op, RT_F16, 0); }
static int se_run(rt_ctx *c, rt_op *op) {
  if (rt_threads(c) > 1 && op->p[2] >= 2 * SE16_PIX) {
    f16 g16[4096] __attribute__((aligned(16)));
    v24k16_se_gate(op->p[0], op->p[1], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), rt_t(c, op, 2, NULL),
                   rt_t(c, op, 3, NULL), rt_b(c, op, 1), op->p[2], g16);
    op16_job j = {c, op, g16};
    rt_par(c, rt_nchunks(op->p[2], SE16_PIX), se_task, &j);
    return 0;
  }
  v24k16_se(op->p[0], op->p[1], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), rt_t(c, op, 2, NULL), rt_t(c, op, 3, NULL),
            rt_b(c, op, 1), op->p[2], rt_b(c, op, 0));
  return 0;
}

/* -------- POSTPC f16 (v2.4): b = [X f16 [24][192], feat f32 [1024], pcin f16 [8][1728], pcout f16 [8][1024]];
 *          t = [post_s, post_b, pc w b32 b16]; p = [N K Kp Np] */
static int postpc_init(rt_ctx *c, rt_op *op) {
  hpack *h = calloc(1, sizeof *h);
  op->state = h;
  if (!h) return rt_oom(c, op, sizeof *h);
  if (rt_need_args(c, op, 4, 5, 4) || hpack_view(c, op, 2, 0, h)) return RT_E_PARAM;
  RT_REQ(c, op, h->N == FEAT_C && h->K == 1728); /* the v2.4 post conv: 192 ch x 3x3 -> 1024 */
  return rt_postpc_check(c, op, RT_F16);
}
static int postpc_run(rt_ctx *c, rt_op *op) {
  return v24k16_postpc(rt_b(c, op, 0), rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), rt_b(c, op, 2), rt_b(c, op, 3),
                       op->state, rt_b(c, op, 1));
}

/* -------- DENSE f16: b = [feat f32 [1024], out [N]]; t = [w f16 [N][1024], b f32 [N]]; p = [N] */
static int dense_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 2, 1)) return RT_E_PARAM;
  RT_REQ(c, op, op->p[0] > 0);
  if (rt_need(c, op, 0, RT_F32, FEAT_C) || rt_need(c, op, 1, RT_F32, (size_t)op->p[0])) return RT_E_PARAM;
  return rt_need_t(c, op, 0, (size_t)op->p[0] * FEAT_C * 2, 0) || rt_need_t(c, op, 1, (size_t)op->p[0] * 4, 0) ? RT_E_PARAM : 0;
}
static int dense_run(rt_ctx *c, rt_op *op) {
  v24k16_head(rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), op->p[0], rt_b(c, op, 0), rt_b(c, op, 1));
  return 0;
}

/* GEMV with fp16-stored weights, fp32 math: b = [x f32 [K], out f32 [N]]; t = [w f16 [N][K], b f32 [N]]; p = [N K act] */
static int gemv_h_init(rt_ctx *c, rt_op *op) { return rt_gemv_check(c, op, 2); }
static int gemv_h_run(rt_ctx *c, rt_op *op) {
  bnn_gemv_h16(rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), op->p[0], op->p[1], rt_b(c, op, 0), op->p[2], rt_b(c, op, 1));
  return 0;
}

/* -------- STEM, fp32 math (baseline gemm_f32), fp16 padded output: b = [img, out f16, col f32, tmp f32 [M][Np]] */
static int stem_init(rt_ctx *c, rt_op *op) {
  fpack *f = calloc(1, sizeof *f);
  op->state = f;
  if (!f) return rt_oom(c, op, sizeof *f);
  if (rt_need_args(c, op, 4, -1, -1) || rt_stem_f32_check(c, op, f) || rt_need(c, op, 1, RT_F16, 1)) return RT_E_PARAM;
  return rt_need(c, op, 3, RT_F32, (size_t)op->p[6] * op->p[7] * f->Np);
}
static int stem_run(rt_ctx *c, rt_op *op) {
  const fpack *f = op->state;
  const int *p = op->p;
  float *tmp = rt_b(c, op, 3);
  rt_stem_f32_tmp(c, op, f, tmp, f->Np);
  if (p[12]) rt_zero_halo(rt_b(c, op, 1), p[6], p[7], p[11], p[12], 2);
  pc_cvt_pad_f16(tmp, p[6] * p[7], p[7], f->N, f->Np, p[12], p[11], rt_b(c, op, 1));
  return 0;
}

static int dwk_init(rt_ctx *c, rt_op *op) { return rt_dwk_check(c, op, 2); }
static int dwk_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  if (rt_threads(c) > 1 && p[2] >= 2 * CH16_CHUNK) {
    op16_job j = {c, op, NULL};
    rt_par(c, rt_nchunks(p[2], CH16_CHUNK), dwk_task, &j);
    return 0;
  }
  pc_dwk_f16(rt_b(c, op, 0), p[0], p[1], p[2], p[3], p[4], p[5], p[6], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), p[9],
             rt_b(c, op, 1), p[7], p[8], rt_b(c, op, 2));
  return 0;
}
static int pad_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  rt_zero_halo(rt_b(c, op, 1), p[0], p[1], p[2], p[4], 2);
  pc_pad_f16(rt_b(c, op, 0), p[0], p[1], p[2], p[3], p[4], rt_b(c, op, 1));
  return 0;
}
static int mean_run(rt_ctx *c, rt_op *op) {
  pc_mean_f16(rt_b(c, op, 0), op->p[0], op->p[1], op->p[2], rt_b(c, op, 1));
  return 0;
}
/* prototype heads over threads: chunks of PROTO_PAN weight panels (each panel's column maxima are independent) */
#define PROTO_PAN 32
typedef struct {
  const f16 *a; int lda, P; const hpack *W; float *cm; int n;                    /* fp16: n 16-column panels */
  const int8_t *qa; const float *sx; int K; const int8_t *qw; const float *sw;   /* int8: n 12-column panels */
} proto_job;
static void proto_task(void *arg, int k, int worker) {
  const proto_job *j = arg;
  int n0 = k * PROTO_PAN, n1 = rt_chunk_end(k, PROTO_PAN, j->n);
  if (j->qa) pc_proto_colmax_s8_range(j->qa, j->sx, j->P, j->K, j->qw, j->sw, j->cm, n0 * 12, n1 * 12);
  else gemm_f16_colmax_range(j->a, j->lda, j->P, j->W, j->cm, n0, n1);
}

/* PROTO f16: prototypes as hpack (w b32 b16), cells normalised in fp32 then rounded to fp16, fp16 GEMM with fp32
 * flushes, max over cells in fp32 */
static int proto_init(rt_ctx *c, rt_op *op) {
  hpack *h = calloc(1, sizeof *h);
  op->state = h;
  if (!h) return rt_oom(c, op, sizeof *h);
  if (rt_proto_geom(c, op, 6, 9, 1 << 20) || hpack_view(c, op, 0, 0, h) || rt_proto_check(c, op, 3)) return RT_E_PARAM;
  return 0;
}
static int proto_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  const float *eps = rt_t(c, op, 5, NULL);
  f16 *a = rt_b(c, op, 2);
  float *cm = rt_b(c, op, 3);
  pc_l2norm_f16(rt_b(c, op, 0), p[4], p[5], p[6], eps[0], a);
  const hpack *h = op->state;
  int npan = h->Np / 16 + (h->Np % 16 != 0);
  if (rt_threads(c) > 1 && npan >= 2 * PROTO_PAN) {
    proto_job j = {.a = a, .lda = p[5], .P = p[4], .W = h, .cm = cm, .n = npan};
    rt_par(c, rt_nchunks(npan, PROTO_PAN), proto_task, &j);
  } else
    gemm_f16_colmax(a, p[5], p[4], h, cm);
  return pc_proto_logits(cm, rt_t(c, op, 3, NULL), rt_t(c, op, 4, NULL), p[7], p[8], rt_b(c, op, 1));
}

/* PROTO s8 (dotprod): b = [x f16 [P][ld], logits f32 [nc], qa int8 [P][K], colmax f32 [Np12]];
 * t = [qw int8 [Np12/12][K/4][12][4], sw f32 [Np12], cls w, cls b, eps]; p = [N K Np12, P C ld nc ppc] */
static int proto8_init(rt_ctx *c, rt_op *op) {
  /* K <= 2^17 = 131,072: an int32 sum of K products, each at most 128 x 127 = 16,256 in magnitude (activations in
   * [-128, 127], weights in [-127, 127]), cannot overflow (131,072 x 16,256 < 2^31) */
  if (rt_proto_geom(c, op, 5, 8, 1 << 17)) return RT_E_PARAM;
  const int *p = op->p;
  int N = p[0], K = p[1], Np = p[2], P = p[3];
  RT_REQ(c, op, K % 8 == 0);
  RT_REQ(c, op, P % 8 == 0 && Np % 12 == 0 && Np >= N);
  if (rt_need_t(c, op, 0, (size_t)Np * K, 1) || rt_need_t(c, op, 1, (size_t)Np * 4, 1) || rt_proto_io(c, op, 2, RT_F16))
    return RT_E_PARAM;
  if (rt_need(c, op, 2, RT_I8, (size_t)P * K) || rt_need(c, op, 3, RT_F32, (size_t)Np)) return RT_E_PARAM;
  op->state = malloc(sizeof(float) * P);
  return op->state ? 0 : rt_oom(c, op, sizeof(float) * P);
}
static int proto8_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  const float *eps = rt_t(c, op, 4, NULL);
  int8_t *qa = rt_b(c, op, 2);
  float *cm = rt_b(c, op, 3), *sx = op->state;
  pc_quant_cells_f16(rt_b(c, op, 0), p[3], p[4], p[5], eps[0], qa, sx);
  int npan = p[2] / 12;
  if (rt_threads(c) > 1 && npan >= 2 * PROTO_PAN) {
    proto_job j = {.P = p[3], .cm = cm, .n = npan, .qa = qa, .sx = sx, .K = p[1], .qw = rt_t(c, op, 0, NULL), .sw = rt_t(c, op, 1, NULL)};
    rt_par(c, rt_nchunks(npan, PROTO_PAN), proto_task, &j);
  } else
    pc_proto_colmax_s8(qa, sx, p[3], p[1], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), p[2], cm);
  return pc_proto_logits(cm, rt_t(c, op, 2, NULL), rt_t(c, op, 3, NULL), p[6], p[7], rt_b(c, op, 1));
}

/* variant 2 of an opcode below the dotprod family's own (rt_ops_dot.c): the prototype head with int8 SDOT GEMMs */
const rt_kernel rt_k_f16_s8[RT_OP_N] = {
    [RT_OP_PROTO] = {proto8_init, proto8_run, rt_free_state, "proto_s8", 0xe, 0},
};

const rt_kernel rt_k_f16[RT_OP_N] = {
    [RT_OP_C0POOL] = {c0_init, c0_run, c0_fini, "c0pool_f16", 0x2, 0},
    [RT_OP_PW] = {pw_init, pw_run, rt_free_state, "pw_f16", 0x2, 0x4},
    [RT_OP_DW3] = {dw3_init, dw3_run, NULL, "dw3_f16", 0x6, 0},
    [RT_OP_SE] = {se_init, se_run, NULL, "se_f16", 0x1, 0},
    [RT_OP_POSTPC] = {postpc_init, postpc_run, rt_free_state, "postpc_f16", 0xe, 0},
    [RT_OP_DENSE] = {dense_init, dense_run, NULL, "dense_f16", 0x2, 0},
    [RT_OP_STEM] = {stem_init, stem_run, rt_free_state, "stem_f32to16", 0xe, 0},
    [RT_OP_DWK] = {dwk_init, dwk_run, NULL, "dwk_f16", 0x6, 0},
    [RT_OP_PAD] = {rt_pad_check, pad_run, NULL, "pad_f16", 0x2, 0},
    [RT_OP_MEAN] = {rt_mean_check, mean_run, NULL, "mean_f16", 0x2, 0},
    [RT_OP_PROTO] = {proto_init, proto_run, rt_free_state, "proto_f16", 0xe, 0},
    [RT_OP_GEMV] = {gemv_h_init, gemv_h_run, NULL, "gemv_h16", 0x2, 0},
};
