// Package runtime, private to the registry kernel files (rt_ops_*.c): the checks, helpers and small types they share.
// Declarations and integer inlines only, so the baseline and the Cortex-A76 objects can both include it.
#ifndef BNN_RT_OPS_H
#define BNN_RT_OPS_H
#include "gemm_f32.h"
#include "rt.h"

/* rt_ops_f32.c: parameter checks shared with the fp16 variants of the same opcodes */
int rt_pw_check(rt_ctx *c, rt_op *op, int dt, int Kp, int Np);
int rt_pw_out_check(rt_ctx *c, rt_op *op, int dt);
int rt_dw3_check(rt_ctx *c, rt_op *op, int dt, int pt, int pl);
int rt_se_check(rt_ctx *c, rt_op *op, int dt, int gate);
int rt_postpc_check(rt_ctx *c, rt_op *op, int dt);
int rt_dwk_check(rt_ctx *c, rt_op *op, size_t esz);
int rt_pad_check(rt_ctx *c, rt_op *op);
int rt_mean_check(rt_ctx *c, rt_op *op);
int rt_proto_geom(rt_ctx *c, rt_op *op, int nt, int ni, int kmax);
int rt_proto_io(rt_ctx *c, rt_op *op, int ct, int dt);
int rt_proto_check(rt_ctx *c, rt_op *op, int nw);
int rt_gemv_check(rt_ctx *c, rt_op *op, size_t esz);
/* rt_ops_f32.c: an fpack over tensor arguments ti (panels), ti + 1 (bias) with [N K Kp Np] at int params po..po + 3 */
int rt_fpack_view(rt_ctx *c, rt_op *op, int ti, int po, fpack *f);
/* rt_ops_f32.c: the fp32 stem (baseline gemm_f32), also the first half of the fp16 path's stem */
int rt_stem_f32_check(rt_ctx *c, rt_op *op, fpack *f);
void rt_stem_f32_tmp(rt_ctx *c, rt_op *op, const fpack *f, float *tmp, int ldt);
/* rt_ops_f32.c: gemm_f32 split over the context's threads; bit-identical to gemm_f32 */
void rt_gemm_par(const rt_ctx *c, const float *A, int lda, int M, const fpack *W, const epi32 *e);
/* rt_ops_f32x.c: the two fp32 stages of the SED head, shared with its int8 form (rt_ops_s8n.c) */
int rt_sed_geom(rt_ctx *c, rt_op *op, int kt, int Cp);
void rt_sed_pool(rt_ctx *c, rt_op *op, int Cp, int c0, int c1);
void rt_sed_out(rt_ctx *c, rt_op *op, int ldn, int ldc, int kt, int k0, int k1);

/* rt_pkg.c: the fini of an op whose state is one malloc block */
void rt_free_state(rt_op *op);

/* The kernel tables, indexed by base opcode (an entry without a run = not in this table). rt_pkg.c lists them by
 * variant; an (opcode, variant) pair has its kernel in exactly one of them. */
extern const rt_kernel rt_k_f32[RT_OP_N];    /* rt_ops_f32.c: variant 0, opcodes 1..15 and SE_GATE (baseline objects) */
extern const rt_kernel rt_k_f32x[RT_OP_N];   /* rt_ops_f32x.c: variant 0, opcodes 16..21 (baseline objects) */
extern const rt_kernel rt_k_s8n[RT_OP_N];    /* rt_ops_s8n.c: variant 0, the int8 opcodes without dotprod (baseline objects) */
extern const rt_kernel rt_k_f16[RT_OP_N];    /* rt_ops_f16.c: variant 1, fp16 arithmetic (A76 objects) */
extern const rt_kernel rt_k_f16_s8[RT_OP_N]; /* rt_ops_f16.c: variant 2 of PROTO (A76 objects, dotprod) */
extern const rt_kernel rt_k_dot[RT_OP_N];    /* rt_ops_dot.c: variant 2, opcodes 26..34 (A76 objects, dotprod + fp16) */

/* a pool job that needs only the context and the op */
typedef struct { rt_ctx *c; rt_op *op; } op_job;
#define CH_CHUNK 16 /* channel block of the fp32 depthwise / mean / pooling splits: 64 B of each pixel, no false sharing */
#define SED_CLS 256 /* classes per chunk of the SED attention output */
/* n items in chunks of `per`: the number of chunks, and the end of chunk k (which starts at k * per) */
static inline int rt_nchunks(int n, int per) { return (n + per - 1) / per; }
static inline int rt_chunk_end(int k, int per, int n) { return k * per + per < n ? k * per + per : n; }
/* rows per chunk: about 4 chunks per thread, a multiple of `unit` rows */
static inline int chunk_rows(int M, int threads, int unit) {
  int per = (M + 4 * threads - 1) / (4 * threads);
  per = (per + unit - 1) / unit * unit;
  return per < unit ? unit : per;
}
#endif
