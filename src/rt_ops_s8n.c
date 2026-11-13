// Registry kernels of the int8 path for CPUs without dotprod (baseline ARMv8-A objects): 1x1 conv / GEMM, the SED
// attention head and the dense head with per-channel int8 weights and per-row dynamic int8 activations (gemm_s8n.c).
// Element-wise work, pooling, softmax and every reduction stay fp32. Threads split rows or 4-channel panels: each
// output is computed by the same instructions whatever the split (bit-identical for every thread count).
#include <stdlib.h>
#include <string.h>

#include "gemm_s8n.h"
#include "pc_kern.h"
#include "rt_ops.h"

/* s8pack from 4 tensors at ti (w, sw, colsum, b) and [N K Kp Np] at int params po */
static int s8_view(rt_ctx *c, rt_op *op, int ti, int po, s8pack *f) {
  if (op->ni < po + 4) return rt_bad(c, op, "int8 GEMM shape [N K Kp Np] expected at int params %d..%d", po, po + 3);
  f->N = op->p[po]; f->K = op->p[po + 1]; f->Kp = op->p[po + 2]; f->Np = op->p[po + 3];
  if (f->N <= 0 || f->K <= 0 || f->Np != ((f->N + 7) & ~7) || f->Kp != ((f->K + 15) & ~15))
    return rt_bad(c, op, "int8 GEMM shape N %d K %d Kp %d Np %d: expected Np = N rounded up to 8, Kp = K rounded up to 16",
                  f->N, f->K, f->Kp, f->Np);
  if (rt_need_t(c, op, ti, (size_t)f->Np * f->Kp, 0) || rt_need_t(c, op, ti + 1, (size_t)f->Np * 4, 1) ||
      rt_need_t(c, op, ti + 2, (size_t)f->Np * 4, 1) || rt_need_t(c, op, ti + 3, (size_t)f->Np * 4, 1))
    return RT_E_PARAM;
  f->w = rt_t(c, op, ti, NULL); f->sw = rt_t(c, op, ti + 1, NULL); f->colsum = rt_t(c, op, ti + 2, NULL);
  f->b = rt_t(c, op, ti + 3, NULL);
  return 0;
}

/* ---- a threaded int8 GEMM over q rows (row blocks of 32, or 4-channel panel groups when rows are few) */
typedef struct { const int8_t *q; int Kp; const float *sa; const int32_t *za; int M; const s8pack *W; const epi32 *e; int byrows, per; } s8_job;
#define S8_ROWS 32
static void s8_task(void *a, int k, int w) {
  const s8_job *j = a;
  int npan = j->W->Np / 4;
  if (j->byrows) {
    int m0 = k * S8_ROWS, m1 = rt_chunk_end(k, S8_ROWS, j->M);
    s8n_gemm_range(j->q, j->Kp, j->sa, j->za, j->M, j->W, j->e, m0, m1, 0, npan);
  } else {
    int p0 = k * j->per, p1 = rt_chunk_end(k, j->per, npan);
    s8n_gemm_range(j->q, j->Kp, j->sa, j->za, j->M, j->W, j->e, 0, j->M, p0, p1);
  }
}
static void s8_gemm(const rt_ctx *c, const int8_t *q, const float *sa, const int32_t *za, int M, const s8pack *W, const epi32 *e) {
  int T = rt_threads(c), npan = W->Np / 4;
  s8_job j = {q, W->Kp, sa, za, M, W, e, 1, 1};
  int nblk = rt_nchunks(M, S8_ROWS);
  if (T == 1) { s8n_gemm_range(q, W->Kp, sa, za, M, W, e, 0, M, 0, npan); return; }
  if (nblk < 2 * T) {
    j.byrows = 0;
    j.per = rt_nchunks(npan, 4 * T);
    rt_par(c, rt_nchunks(npan, j.per), s8_task, &j);
  } else
    rt_par(c, nblk, s8_task, &j);
}
/* ---- quantize + GEMM of a row range in one task: the chunk's int8 rows are still in L1/L2 when its panels run, and
 * one fork/join per layer instead of two. Row chunks are 4-aligned so no tile straddles two chunks; at T threads the
 * rows split into 3T nearly equal chunks (a fixed 32-row chunk left 1 of 8 slots idle at M = 420: 14 chunks on 4 threads)
 * and small-M layers keep the panel split, after a quantization pass in 2T chunks. */
typedef struct { const float *x; int ld, K; const float *gate; int8_t *q; float *sa; int32_t *za; int M, n; const s8pack *W; const epi32 *e; int gemm, per; } qg_job;
static void qg_rows(const qg_job *j, int k, int *m0, int *m1) {
  *m0 = (int)(((long)j->M * k / j->n) & ~3L);
  *m1 = k + 1 == j->n ? j->M : (int)(((long)j->M * (k + 1) / j->n) & ~3L);
}
static void qg_task(void *a, int k, int w) {
  const qg_job *j = a;
  int m0, m1;
  qg_rows(j, k, &m0, &m1);
  if (m1 <= m0) return;
  if (j->gate) s8n_quant_rows_gate(j->x, j->ld, m0, m1, j->K, j->W->Kp, j->gate, j->q, j->sa, j->za);
  else s8n_quant_rows(j->x, j->ld, m0, m1, j->K, j->W->Kp, j->q, j->sa, j->za);
  if (j->gemm) s8n_gemm_range(j->q, j->W->Kp, j->sa, j->za, j->M, j->W, j->e, m0, m1, 0, j->W->Np / 4);
}
static void pan_task(void *a, int k, int w) {
  const qg_job *j = a;
  int npan = j->W->Np / 4, p0 = k * j->per, p1 = rt_chunk_end(k, j->per, npan);
  s8n_gemm_range(j->q, j->W->Kp, j->sa, j->za, j->M, j->W, j->e, 0, j->M, p0, p1);
}
static void quant_gemm(const rt_ctx *c, const float *x, int ld, int M, const float *gate, int8_t *q, float *sa, int32_t *za,
                       const s8pack *W, const epi32 *e) {
  int T = rt_threads(c), npan = W->Np / 4;
  qg_job j = {x, ld, W->K, gate, q, sa, za, M, 1, W, e, 1, 1};
  if (T == 1) { qg_task(&j, 0, 0); return; }
  if (rt_nchunks(M, S8_ROWS) < 2 * T) {
    j.gemm = 0; j.n = 2 * T;
    rt_par(c, j.n, qg_task, &j);
    j.per = rt_nchunks(npan, 4 * T);
    rt_par(c, rt_nchunks(npan, j.per), pan_task, &j);
    return;
  }
  j.n = 3 * T;
  rt_par(c, j.n, qg_task, &j);
}

/* -------- PW_S8: b = [in f32 [M][lda], out, res | -1, q i8 [M4][Kp], sa f32 [M4], za i32 [M4] (, gate f32 [K] | -1)];
 *          t = [w, sw, colsum, b]; p = [N K Kp Np, M, lda, act, ldo, ow, opad, ldr] */
static int pw8_init(rt_ctx *c, rt_op *op) {
  s8pack *f = calloc(1, sizeof *f);
  op->state = f;
  if (!f) return rt_oom(c, op, sizeof *f);
  if (rt_need_args(c, op, op->nb == 7 ? 7 : 6, 4, 11) || s8_view(c, op, 0, 0, f)) return RT_E_PARAM; /* the gate is optional */
  if (op->nb == 7 && op->b[6] >= 0) {
    RT_REQ(c, op, f->K % 4 == 0 && f->K <= 4096);
    if (rt_need(c, op, 6, RT_F32, (size_t)f->K)) return RT_E_PARAM;
  }
  const int *p = op->p;
  int M = p[4], lda = p[5], act = p[6], ldo = p[7], opad = p[9], ldr = p[10];
  size_t M4 = ((size_t)M + 3) & ~(size_t)3;
  RT_REQ(c, op, M > 0 && lda >= f->K && act >= 0 && act <= 2 && ldo >= f->Np && opad >= 0 && opad <= 3);
  const rt_bufdesc *r = rt_bd(c, op, 2);
  if (rt_need_nohalo(c, op, 0, RT_F32, (size_t)M * lda) || rt_pw_out_check(c, op, RT_F32)) return RT_E_PARAM;
  if (r) {
    RT_REQ(c, op, ldr >= f->Np && r->dtype == RT_F32 && r->ld == (uint32_t)ldr);
    if (r->halo) {
      RT_REQ(c, op, (size_t)r->H * r->W == (size_t)M && r->W > 0);
      if (rt_need(c, op, 2, RT_F32, (size_t)(r->H + 2 * r->halo) * (r->W + 2 * r->halo) * ldr)) return RT_E_PARAM;
    } else if (rt_need(c, op, 2, RT_F32, (size_t)M * ldr))
      return RT_E_PARAM;
  }
  return rt_need(c, op, 3, RT_I8, M4 * f->Kp) || rt_need(c, op, 4, RT_F32, M4) || rt_need(c, op, 5, RT_I32, M4) ? RT_E_PARAM : 0;
}
static int pw8_run(rt_ctx *c, rt_op *op) {
  const s8pack *f = op->state;
  const int *p = op->p;
  int M = p[4];
  float *out = rt_b(c, op, 1);
  const rt_bufdesc *o = rt_bd(c, op, 1), *r = rt_bd(c, op, 2);
  if (p[9]) rt_zero_halo(out, (int)o->H, (int)o->W, (int)o->ld, p[9], 4);
  int8_t *q = rt_b(c, op, 3);
  float *sa = rt_b(c, op, 4);
  int32_t *za = rt_b(c, op, 5);
  const float *gate = op->nb == 7 && op->b[6] >= 0 ? rt_b(c, op, 6) : NULL;
  epi32 e = {.act = p[6], .out = out, .ldo = p[7], .ow = p[8], .opad = p[9], .res = rt_b(c, op, 2), .ldr = p[10]};
  if (r && r->halo) { e.rpad = (int)r->halo; e.rw = (int)r->W; }
  quant_gemm(c, rt_b(c, op, 0), p[5], M, gate, q, sa, za, f, &e);
  return 0;
}

/* -------- GEMV_S8: b = [x f32 [K], out f32 [N], q i8 [4][Kp], sa f32 [4], za i32 [4]]; t = [w sw colsum b];
 *          p = [N K Kp Np, act] */
static int gemv8_init(rt_ctx *c, rt_op *op) {
  s8pack *f = calloc(1, sizeof *f);
  op->state = f;
  if (!f) return rt_oom(c, op, sizeof *f);
  if (rt_need_args(c, op, 5, 4, 5) || s8_view(c, op, 0, 0, f)) return RT_E_PARAM;
  RT_REQ(c, op, op->p[4] == RT_ACT_NONE || op->p[4] == RT_ACT_RELU);
  return rt_need(c, op, 0, RT_F32, (size_t)f->K) || rt_need(c, op, 1, RT_F32, (size_t)f->Np) ||
                 rt_need(c, op, 2, RT_I8, (size_t)4 * f->Kp) || rt_need(c, op, 3, RT_F32, 4) || rt_need(c, op, 4, RT_I32, 4)
             ? RT_E_PARAM
             : 0;
}
static int gemv8_run(rt_ctx *c, rt_op *op) {
  const s8pack *f = op->state;
  int8_t *q = rt_b(c, op, 2);
  float *sa = rt_b(c, op, 3);
  int32_t *za = rt_b(c, op, 4);
  s8n_quant_rows(rt_b(c, op, 0), f->K, 0, 1, f->K, f->Kp, q, sa, za);
  epi32 e = {.act = op->p[4], .out = rt_b(c, op, 1), .ldo = f->Np};
  s8_gemm(c, q, sa, za, 1, f, &e);
  return 0;
}

/* -------- SED_S8: the SED head (rt_ops_f32x.c) with int8 fc / att / cla GEMMs.
 *   b = [x, out, s [L8][Cp], pooled [L8][Cp], h [L8][hid_p], att [L8][NCp], cla [L8][NCp], q i8 [L8][Kq], sa f32 [L8],
 *        za i32 [L8]]; t = [dw, db, fc (w sw cs b), att (4), cla (4), consts]; p as SED */
typedef struct { s8pack fc, at, cl; int Cp; } sed8_st;
static int sed8_init(rt_ctx *c, rt_op *op) {
  sed8_st *s = calloc(1, sizeof *s);
  op->state = s;
  if (!s) return rt_oom(c, op, sizeof *s);
  if (rt_need_args(c, op, 10, 15, 21) || s8_view(c, op, 2, 0, &s->fc) || s8_view(c, op, 6, 4, &s->at) ||
      s8_view(c, op, 10, 8, &s->cl))
    return RT_E_PARAM;
  s->Cp = (op->p[15] + 7) & ~7;
  if (rt_sed_geom(c, op, 14, s->Cp)) return RT_E_PARAM;
  size_t L8 = (size_t)((op->p[17] + 7) & ~7), Kq = (size_t)(s->fc.Kp > s->at.Kp ? s->fc.Kp : s->at.Kp);
  return rt_need(c, op, 7, RT_I8, L8 * Kq) || rt_need(c, op, 8, RT_F32, L8) || rt_need(c, op, 9, RT_I32, L8) ? RT_E_PARAM : 0;
}
static void sed8_pool_task(void *a, int k, int w) {
  const op_job *j = a;
  int C = j->op->p[15], c0 = k * CH_CHUNK, c1 = rt_chunk_end(k, CH_CHUNK, C);
  rt_sed_pool(j->c, j->op, ((const sed8_st *)j->op->state)->Cp, c0, c1);
}
static void sed8_out_task(void *a, int k, int w) {
  const op_job *j = a;
  const sed8_st *st = j->op->state;
  int k0 = k * SED_CLS, k1 = rt_chunk_end(k, SED_CLS, st->at.N);
  rt_sed_out(j->c, j->op, st->at.Np, st->cl.Np, 14, k0, k1);
}
static int sed8_run(rt_ctx *c, rt_op *op) {
  const sed8_st *st = op->state;
  const int *p = op->p;
  int C = p[15], L = p[17], Cp = st->Cp;
  float *s = rt_b(c, op, 2), *pl = rt_b(c, op, 3), *h = rt_b(c, op, 4), *sa = rt_b(c, op, 8);
  int8_t *q = rt_b(c, op, 7);
  int32_t *za = rt_b(c, op, 9);
  for (int t = 0; t < L; t++) {
    memset(s + (size_t)t * Cp + C, 0, sizeof(float) * (Cp - C));
    memset(pl + (size_t)t * Cp + C, 0, sizeof(float) * (Cp - C));
  }
  op_job j = {c, op};
  rt_par(c, rt_nchunks(C, CH_CHUNK), sed8_pool_task, &j);
  s8n_quant_rows(pl, Cp, 0, L, C, st->fc.Kp, q, sa, za);
  epi32 ef = {.act = ACT32_RELU, .out = h, .ldo = st->fc.Np};
  s8_gemm(c, q, sa, za, L, &st->fc, &ef);
  s8n_quant_rows(h, st->fc.Np, 0, L, st->fc.N, st->at.Kp, q, sa, za);
  epi32 ea = {.act = ACT32_NONE, .out = rt_b(c, op, 5), .ldo = st->at.Np};
  s8_gemm(c, q, sa, za, L, &st->at, &ea);
  epi32 ec = {.act = ACT32_NONE, .out = rt_b(c, op, 6), .ldo = st->cl.Np};
  s8_gemm(c, q, sa, za, L, &st->cl, &ec);
  rt_par(c, rt_nchunks(st->at.N, SED_CLS), sed8_out_task, &j);
  return 0;
}

/* -------- PROTO_S8: the prototype head with int8 prototypes and per-cell int8 activations (proto_s8n.c).
 *   b = [x f32 [P][ld], logits f32 [nc], qa i8 [P][Kp], colmax f32 [Np]];
 *   t = [qw i8 [Np/4][Kp/16][4][16], sw f32 [Np], n4 i8 [Np/4], cls w [nc][ppc], cls b [nc], eps [1]];
 *   p = [N K Kp Np, P C ld nc ppc]. Threads split the 4-prototype panels (each column maximum is one panel's work). */
#define PROTO_PANELS 64
typedef struct { rt_ctx *c; rt_op *op; const float *sx; } proto8_job;
static int proto8n_init(rt_ctx *c, rt_op *op) {
  /* K <= 2^17: sizes below stay far inside int and size_t; int32 sums of K products of |q| <= 127 need
   * K <= 2^31 / 127^2 */
  if (rt_proto_geom(c, op, 6, 9, 1 << 17)) return RT_E_PARAM;
  const int *p = op->p;
  int N = p[0], K = p[1], Kp = p[2], Np = p[3], P = p[4], C = p[5];
  RT_REQ(c, op, C % 8 == 0 && Kp == ((K + 15) & ~15));
  RT_REQ(c, op, P % 4 == 0 && Np == ((N + 3) & ~3));
  if (rt_need_t(c, op, 0, (size_t)Np * Kp, 0) || rt_need_t(c, op, 1, (size_t)Np * 4, 1) || rt_need_t(c, op, 2, (size_t)Np / 4, 1) ||
      rt_proto_io(c, op, 3, RT_F32))
    return RT_E_PARAM;
  if (rt_need(c, op, 2, RT_I8, (size_t)P * Kp) || rt_need(c, op, 3, RT_F32, (size_t)Np)) return RT_E_PARAM;
  int bad = pc_proto_n4_check(rt_t(c, op, 0, NULL), Kp, rt_t(c, op, 2, NULL), Np / 4);
  if (bad >= 0)
    return rt_bad(c, op, "prototype panel %d is flagged for 4 products per int16 lane, but a lane's 4 weights sum to more "
                         "than 258 (the sums could wrap)", bad);
  op->state = malloc(sizeof(float) * (size_t)P);
  return op->state ? 0 : rt_oom(c, op, sizeof(float) * (size_t)P);
}
static void proto8n_task(void *a, int k, int w) {
  const proto8_job *j = a;
  const rt_op *op = j->op;
  const int *p = op->p;
  int npan = p[3] / 4, p0 = k * PROTO_PANELS, p1 = rt_chunk_end(k, PROTO_PANELS, npan);
  pc_proto_colmax_s8n(rt_b(j->c, op, 2), p[2], j->sx, p[4], rt_t(j->c, op, 0, NULL), rt_t(j->c, op, 1, NULL),
                      rt_t(j->c, op, 2, NULL), p0, p1, rt_b(j->c, op, 3));
}
static int proto8n_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  const float *eps = rt_t(c, op, 5, NULL);
  float *sx = op->state;
  pc_quant_cells_f32(rt_b(c, op, 0), p[4], p[5], p[6], eps[0], rt_b(c, op, 2), p[2], sx);
  proto8_job j = {c, op, sx};
  rt_par(c, rt_nchunks(p[3] / 4, PROTO_PANELS), proto8n_task, &j);
  return pc_proto_logits(rt_b(c, op, 3), rt_t(c, op, 3, NULL), rt_t(c, op, 4, NULL), p[7], p[8], rt_b(c, op, 1));
}

const rt_kernel rt_k_s8n[RT_OP_N] = {
    [RT_OP_PW_S8] = {pw8_init, pw8_run, rt_free_state, "pw_s8n", 0x3a, 0x44},
    [RT_OP_GEMV_S8] = {gemv8_init, gemv8_run, rt_free_state, "gemv_s8n", 0x1e, 0},
    [RT_OP_SED_S8] = {sed8_init, sed8_run, rt_free_state, "sed_s8n", 0x3fe, 0},
    [RT_OP_PROTO_S8] = {proto8n_init, proto8n_run, rt_free_state, "proto_s8n", 0xe, 0},
};
