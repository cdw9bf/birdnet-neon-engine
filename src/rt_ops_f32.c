// Registry kernels with fp32 arithmetic (baseline ARMv8-A objects; any AArch64). Each kernel is a thin wrapper over
// the hand-tuned code (v24_fe.c, gemm_f32.c, v24_f32.c stage kernels), so a compiled package runs exactly the same
// instructions as the hand-written engine.
#include <stdlib.h>
#include <string.h>

#include "gemm_f32.h"
#include "gemv.h"
#include "pc_kern.h"
#include "rt_ops.h"
#include "v24_fe.h"
#include "v24_internal.h"
#include "v24_kern.h"

/* -------- FE_STFT2: BirdNET v2.4 front end (v24_fe.c). t: win1 win2 mel1 [96][1025] mel2 [96][513] pow1 pow2 bn_s bn_b */
static void nz_range(const float *mel, int bins, int *lo, int *hi) {
  for (int j = 0; j < NMEL; j++) {
    const float *f = mel + (size_t)j * bins;
    int l = 0, h = 0;
    for (int k = 0; k < bins; k++)
      if (f[k] != 0.f) { if (h == 0) l = k; h = k + 1; }
    lo[j] = l; hi[j] = h;
  }
}
static int fe_stft2_init(rt_ctx *c, rt_op *op) {
  b24_model m;
  memset(&m, 0, sizeof m);
  const float **dst[8] = {&m.win1, &m.win2, &m.mel1, &m.mel2, &m.pow1, &m.pow2, &m.fe_bn_s, &m.fe_bn_b};
  const size_t want[8] = {N1, N2, (size_t)NMEL * B1, (size_t)NMEL * B2, 1, 1, 2, 2};
  if (rt_need_args(c, op, 2, 8, -1)) return RT_E_PARAM;
  for (int i = 0; i < 8; i++) {
    if (rt_need_t(c, op, i, want[i] * 4, 0)) return RT_E_PARAM;
    *dst[i] = rt_t(c, op, i, NULL);
  }
  if (rt_need(c, op, 0, RT_F32, NSAMP) || rt_need(c, op, 1, RT_F32, (size_t)NMEL * NFR * 2)) return RT_E_PARAM;
  nz_range(m.mel1, B1, m.mel1_lo, m.mel1_hi);
  nz_range(m.mel2, B2, m.mel2_lo, m.mel2_hi);
  op->state = b24_fe_new(&m); /* copies window, mel (sparse), pow, BN; builds the FFT tables */
  return op->state ? 0 : rt_oom(c, op, sizeof(b24_model));
}
static int fe_stft2_run(rt_ctx *c, rt_op *op) {
  b24_fe_run(op->state, rt_b(c, op, 0), rt_b(c, op, 1));
  return 0;
}
static void fe_stft2_fini(rt_op *op) { b24_fe_free(op->state); }

/* -------- packed fp32 GEMM weights: t = [w (fpack panels), b]; p = [N, K, Kp, Np] at offset po (shared with
 * rt_ops_f32x.c) */
int rt_fpack_view(rt_ctx *c, rt_op *op, int ti, int po, fpack *f) {
  if (op->ni < po + 4)
    return rt_bad(c, op, "GEMM shape [N K Kp Np] expected at int params %d..%d, the op has %d ints", po, po + 3, op->ni);
  f->N = op->p[po]; f->K = op->p[po + 1]; f->Kp = op->p[po + 2]; f->Np = op->p[po + 3];
  if (f->N <= 0 || f->K <= 0 || f->Np != ((f->N + 7) & ~7) || f->Kp < f->K || f->Kp % 4)
    return rt_bad(c, op, "GEMM shape N %d K %d Kp %d Np %d: expected Np = N rounded up to 8, Kp >= K, Kp %% 4 == 0",
                  f->N, f->K, f->Kp, f->Np);
  if (rt_need_t(c, op, ti, (size_t)f->Np * f->Kp * 4, 1) || rt_need_t(c, op, ti + 1, (size_t)f->Np * 4, 1)) return RT_E_PARAM;
  f->w = (float *)rt_t(c, op, ti, NULL); f->b = (float *)rt_t(c, op, ti + 1, NULL);
  return 0;
}

/* -------- C0POOL: b = [img f32 [96][511][2], pin f32 [48*128][48]]; t = [w, b]; p = [N K Kp Np] */
typedef struct { fpack f; float *imgp; } c0_st;
static int c0_init(rt_ctx *c, rt_op *op) {
  c0_st *s = calloc(1, sizeof *s);
  if (!s) return rt_oom(c, op, sizeof *s);
  op->state = s;
  if (rt_need_args(c, op, 2, 2, 4) || rt_fpack_view(c, op, 0, 0, &s->f)) return RT_E_PARAM;
  RT_REQ(c, op, s->f.N == 24 && s->f.K == 64 && s->f.Kp == 64);
  if (rt_need(c, op, 0, RT_F32, (size_t)96 * 511 * 2) || rt_need(c, op, 1, RT_F32, (size_t)48 * 128 * 48)) return RT_E_PARAM;
  s->imgp = rt_zalloc64_tail(sizeof(float) * 98 * 518 * 2);
  return s->imgp ? 0 : rt_oom(c, op, sizeof(float) * 98 * 518 * 2);
}
static int c0_run(rt_ctx *c, rt_op *op) {
  c0_st *s = op->state;
  v24k32_c0_pool(rt_b(c, op, 0), s->imgp, &s->f, rt_b(c, op, 1));
  return 0;
}
static void c0_fini(rt_op *op) {
  c0_st *s = op->state;
  if (s) free(s->imgp);
  free(s);
}

/* -------- threads: a GEMM split into M blocks (enough rows) or groups of weight panels (few rows: every thread
 * reads only its own panels); every part computes its outputs exactly as the whole GEMM */
typedef struct { const float *A; int lda, M, mb, per, byrows; const fpack *W; const epi32 *e; } gemm_job;
static void gemm_task(void *a, int k, int w) {
  const gemm_job *j = a;
  int npan = j->W->Np / 8;
  if (j->byrows) {
    int m0 = k * j->mb, m1 = rt_chunk_end(k, j->mb, j->M);
    gemm_f32_range(j->A, j->lda, j->M, j->W, j->e, m0, m1, 0, npan);
  } else {
    int p0 = k * j->per, p1 = rt_chunk_end(k, j->per, npan);
    gemm_f32_range(j->A, j->lda, j->M, j->W, j->e, 0, j->M, p0, p1);
  }
}
void rt_gemm_par(const rt_ctx *c, const float *A, int lda, int M, const fpack *W, const epi32 *e) {
  int T = rt_threads(c), npan = W->Np / 8;
  gemm_job j = {A, lda, M, gemm_f32_mblock(M, W), 1, 1, W, e};
  int nblk = rt_nchunks(M, j.mb);
  if (nblk < 2 * T && npan >= 2) { /* few rows: split the weight panels */
    j.byrows = 0;
    j.per = rt_nchunks(npan, 4 * T);
    rt_par(c, rt_nchunks(npan, j.per), gemm_task, &j);
  } else
    rt_par(c, nblk, gemm_task, &j);
}

/* -------- PW: b = [in, out, res|-1]; t = [w, b]; p = [N K Kp Np, M, lda, act, ldo, ow, opad, ldr] */
/* shared with rt_ops_f16.c: the GEMM reads 8-row tiles (rows up to M rounded up to 8 must be inside the buffer);
 * the output descriptor must describe exactly the region written (zero_halo uses it) */
/* the output of a 1x1 conv (buffer 1; p[4] = M, p[7] = ldo, p[8] = ow, p[9] = opad): dtype dt, channel stride ldo,
 * halo opad, and when padded exactly M / ow x ow pixels; shared with PW_S8 (rt_ops_s8n.c) */
int rt_pw_out_check(rt_ctx *c, rt_op *op, int dt) {
  const int *p = op->p;
  int M = p[4], ldo = p[7], ow = p[8], opad = p[9];
  const rt_bufdesc *o = rt_bd(c, op, 1);
  if (!o) return rt_bad(c, op, "buffer argument 1 (output) is absent");
  if ((int)o->dtype != dt || o->ld != (uint32_t)ldo || o->halo != (uint32_t)opad)
    return rt_bad(c, op, "buffer argument 1 (buffer #%d, output): expected dtype %s ld %d halo %d, found dtype %s ld %u halo %u", op->b[1],
                  rt_dtype_name((uint32_t)dt), ldo, opad, rt_dtype_name(o->dtype), o->ld, o->halo);
  if (!opad) return rt_need(c, op, 1, dt, (size_t)M * ldo);
  RT_REQ(c, op, ow > 0 && M % ow == 0);
  if (o->W != (uint32_t)ow || (size_t)o->H * o->W != (size_t)M)
    return rt_bad(c, op, "buffer argument 1 (buffer #%d, padded output): expected %d x %d rows, found %u x %u", op->b[1], M / ow, ow, o->H, o->W);
  return rt_need(c, op, 1, dt, (size_t)(M / ow + 2 * opad) * (ow + 2 * opad) * ldo);
}
int rt_pw_check(rt_ctx *c, rt_op *op, int dt, int Kp, int Np) {
  const int *p = op->p;
  int M = p[4], lda = p[5], act = p[6], ldo = p[7], opad = p[9], ldr = p[10];
  size_t Mr = ((size_t)M + 7) & ~(size_t)7;
  RT_REQ(c, op, M > 0);
  RT_REQ(c, op, lda >= Kp);   /* input channel stride covers the packed K */
  RT_REQ(c, op, ldo >= Np);   /* output channel stride covers the packed N */
  RT_REQ(c, op, act >= 0 && act <= 2);
  RT_REQ(c, op, opad >= 0 && opad <= 3);
  if (rt_need(c, op, 0, dt, Mr * lda)) return RT_E_PARAM; /* 8-row tiles */
  if (rt_pw_out_check(c, op, dt)) return RT_E_PARAM;
  if (op->b[2] >= 0) {
    RT_REQ(c, op, ldr >= Np);
    const rt_bufdesc *r = rt_bd(c, op, 2);
    if (r->halo) { /* a residual in a zero-haloed buffer (fp32 only): the same pixel grid as the M rows */
      if (dt != RT_F32 || r->ld != (uint32_t)ldr || (size_t)r->H * r->W != (size_t)M || !r->W)
        return rt_bad(c, op, "buffer argument 2 (buffer #%d, residual): halo %u needs an fp32 buffer with ld %d and H x W = "
                      "%d rows, found %s ld %u %ux%u", op->b[2], r->halo, ldr, M, rt_dtype_name(r->dtype), r->ld, r->H, r->W);
      if (rt_need(c, op, 2, dt, (size_t)(r->H + 2 * r->halo) * (r->W + 2 * r->halo) * ldr)) return RT_E_PARAM;
    } else if (rt_need(c, op, 2, dt, (size_t)M * ldr))
      return RT_E_PARAM;
  }
  return 0;
}
/* optional b[3]: an input-channel gate f32 [K] (SE_GATE's output) applied to A as the GEMM reads it (needs K == Kp) */
typedef struct { fpack f; float *scr; size_t per; int nw; } pw_st;
static int pw_init(rt_ctx *c, rt_op *op) {
  pw_st *s = calloc(1, sizeof *s);
  op->state = s;
  if (!s) return rt_oom(c, op, sizeof *s);
  fpack *f = &s->f;
  if (rt_need_args(c, op, op->nb == 4 ? 4 : 3, 2, 11) || rt_fpack_view(c, op, 0, 0, f)) return RT_E_PARAM; /* the gate is optional */
  if (op->nb == 4 && op->b[3] >= 0) {
    RT_REQ(c, op, f->K == f->Kp);
    if (rt_need(c, op, 3, RT_F32, (size_t)f->K)) return RT_E_PARAM;
  }
  return rt_pw_check(c, op, RT_F32, f->Kp, f->Np);
}
typedef struct { const float *A, *gate; int lda, M, mb; const fpack *W; const epi32 *e; float *scr; size_t per; } pwg_job;
static void pwg_task(void *a, int k, int w) {
  const pwg_job *j = a;
  int m0 = k * j->mb, m1 = rt_chunk_end(k, j->mb, j->M);
  gemm_f32_range_gate(j->A, j->lda, j->M, j->W, j->e, m0, m1, 0, j->W->Np / 8, j->gate, j->scr + (size_t)w * j->per);
}
static int pw_gated(rt_ctx *c, pw_st *s, const float *A, int lda, int M, const float *gate, const epi32 *e) {
  const fpack *f = &s->f;
  int T = rt_threads(c), npan = f->Np / 8, mb = gemm_f32_mblock(M, f), nblk = rt_nchunks(M, mb);
  size_t M8 = ((size_t)M + 7) & ~(size_t)7; /* the GEMM reads whole 8-row tiles (rows past M: ignored) */
  size_t per = gemm_f32_gate_scratch(M, f), whole = M8 * f->Kp + 16, need = per * T > whole ? per * T : whole;
  if (s->nw < T && !(s->scr = rt_worker_scratch(s->scr, &s->nw, T, need * 4))) return RT_E_OOM;
  if (T == 1) { gemm_f32_range_gate(A, lda, M, f, e, 0, M, 0, npan, gate, s->scr); return 0; }
  if (nblk < 2 * T && npan >= 2) { /* few rows (< 2T M blocks): gate them once on the caller, then rt_gemm_par's panel split */
    for (size_t r = 0; r < M8; r++) {
      const float *x = A + r * lda;
      float *y = s->scr + r * f->Kp;
      for (int k = 0; k < f->Kp; k += 4) vst1q_f32(y + k, vmulq_f32(vld1q_f32(x + k), vld1q_f32(gate + k)));
    }
    rt_gemm_par(c, s->scr, f->Kp, M, f, e);
    return 0;
  }
  pwg_job j = {A, gate, lda, M, mb, f, e, s->scr, per};
  rt_par(c, nblk, pwg_task, &j);
  return 0;
}
static int pw_run(rt_ctx *c, rt_op *op) {
  pw_st *s = op->state;
  const fpack *f = &s->f;
  const int *p = op->p;
  float *out = rt_b(c, op, 1);
  const rt_bufdesc *o = rt_bd(c, op, 1);
  if (p[9]) rt_zero_halo(out, (int)o->H, (int)o->W, (int)o->ld, p[9], 4);
  epi32 e = {.act = p[6], .out = out, .ldo = p[7], .ow = p[8], .opad = p[9], .res = rt_b(c, op, 2), .ldr = p[10]};
  const rt_bufdesc *r = rt_bd(c, op, 2);
  if (r && r->halo) { e.rpad = (int)r->halo; e.rw = (int)r->W; }
  if (op->nb == 4 && op->b[3] >= 0) return pw_gated(c, s, rt_b(c, op, 0), p[5], p[4], rt_b(c, op, 3), &e);
  if (rt_threads(c) > 1) rt_gemm_par(c, rt_b(c, op, 0), p[5], p[4], f, &e);
  else gemm_f32(rt_b(c, op, 0), p[5], p[4], f, &e);
  return 0;
}
static void pw_fini(rt_op *op) {
  pw_st *s = op->state;
  if (s) free(s->scr);
  free(s);
}

/* -------- DW3: b = [xp (halo 1), out, chsum f32]; t = [w [9][C], b [C]]; p = [H W C s Ho Wo] (pads 1 all round) */
/* shared with rt_ops_f16.c: geometry of a 3x3 depthwise on a halo-1 input (pt, pl offsets in {0, 1}) */
int rt_dw3_check(rt_ctx *c, rt_op *op, int dt, int pt, int pl) {
  const int *p = op->p;
  int H = p[0], W = p[1], C = p[2], s = p[3], Ho = p[4], Wo = p[5];
  size_t esz = dt == RT_F16 ? 2 : 4;
  RT_REQ(c, op, H > 0 && W > 0 && Ho > 0 && Wo > 0 && C > 0);
  RT_REQ(c, op, C % 8 == 0);
  RT_REQ(c, op, s == 1 || s == 2);
  RT_REQ(c, op, pt >= 0 && pt <= 1 && pl >= 0 && pl <= 1);
  if (rt_need_t(c, op, 0, (size_t)9 * C * esz, 0) || rt_need_t(c, op, 1, (size_t)C * esz, 0)) return RT_E_PARAM;
  RT_REQ(c, op, (Ho - 1) * s - pt + 1 + 3 <= H + 2); /* the last tap stays inside the haloed input */
  RT_REQ(c, op, (Wo - 1) * s - pl + 1 + 3 <= W + 2);
  if (rt_need_haloed(c, op, 0, dt, H, W, C, 1)) return RT_E_PARAM;
  return rt_need(c, op, 1, dt, (size_t)Ho * Wo * C) || rt_need(c, op, 2, RT_F32, (size_t)C) ? RT_E_PARAM : 0;
}
static int dw3_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 3, 2, 6)) return RT_E_PARAM;
  return rt_dw3_check(c, op, RT_F32, 1, 1);
}
static void dw3_task(void *a, int k, int w) {
  const op_job *j = a;
  const int *p = j->op->p;
  int c0 = k * CH_CHUNK, c1 = rt_chunk_end(k, CH_CHUNK, p[2]);
  dw3x3_f32_cr(rt_b(j->c, j->op, 0), p[0], p[1], p[2], p[3], rt_t(j->c, j->op, 0, NULL), rt_t(j->c, j->op, 1, NULL),
               rt_b(j->c, j->op, 1), p[4], p[5], rt_b(j->c, j->op, 2), c0, c1);
}
static int dw3_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  if (rt_threads(c) > 1) {
    op_job j = {c, op};
    rt_par(c, rt_nchunks(p[2], CH_CHUNK), dw3_task, &j);
    return 0;
  }
  dw3x3_f32(rt_b(c, op, 0), p[0], p[1], p[2], p[3], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), rt_b(c, op, 1), p[4],
            p[5], rt_b(c, op, 2));
  return 0;
}

/* -------- SE: b = [D (in place), chsum]; t = [se1_w [rd][C], se1_b, se2t [rd][C], se2_b]; p = [C rd P] */
/* shared with rt_ops_f16.c and with SE_GATE (gate = 1: buffer 0 is the channel sums f32 [C] and buffer 1 the gate
 * f32 [C], instead of the activation [P][C] of dtype dt and the channel sums) */
int rt_se_check(rt_ctx *c, rt_op *op, int dt, int gate) {
  if (rt_need_args(c, op, 2, 4, 3)) return RT_E_PARAM;
  int C = op->p[0], rd = op->p[1], P = op->p[2];
  RT_REQ(c, op, C > 0 && C % 8 == 0 && C <= 4096);
  RT_REQ(c, op, rd > 0 && rd <= 512);
  RT_REQ(c, op, P > 0);
  const size_t want[4] = {(size_t)rd * C, (size_t)rd, (size_t)rd * C, (size_t)C};
  for (int i = 0; i < 4; i++)
    if (rt_need_t(c, op, i, want[i] * 4, 0)) return RT_E_PARAM;
  return rt_need(c, op, 0, dt, gate ? (size_t)C : (size_t)P * C) || rt_need(c, op, 1, RT_F32, (size_t)C) ? RT_E_PARAM : 0;
}
static int se_init(rt_ctx *c, rt_op *op) { return rt_se_check(c, op, RT_F32, 0); }
/* SE with threads: mean and the first FC on the caller (as v24k32_se), the second FC, the sigmoid and the gating
 * multiply split over channel blocks (per channel the same instructions in the same order) */
/* the second FC and the sigmoid of an SE gate for channels [c0, c1) (multiples of 8), from the first FC's outputs
 * s1 [rd]: g[0 .. c1 - c0). Per channel the instructions of v24k32_se: from the bias, one FMA per row r, sigmoid. */
static inline void se_gate_range(const float *se2t, const float *se2_b, const float *s1, int rd, int cm, int c0, int c1,
                                 float *g) {
  memcpy(g, se2_b + c0, sizeof(float) * (c1 - c0));
  for (int r = 0; r < rd; r++) {
    const float *wr = se2t + (size_t)r * cm + c0;
    for (int c = 0; c < c1 - c0; c += 8) {
      vst1q_f32(g + c, vfmaq_n_f32(vld1q_f32(g + c), vld1q_f32(wr + c), s1[r]));
      vst1q_f32(g + c + 4, vfmaq_n_f32(vld1q_f32(g + c + 4), vld1q_f32(wr + c + 4), s1[r]));
    }
  }
  for (int c = 0; c < c1 - c0; c += 4) vst1q_f32(g + c, sigmoid_f32x4(vld1q_f32(g + c)));
}
typedef struct { rt_ctx *c; rt_op *op; const float *s1; } se_job;
#define SE_CHUNK 64
static void se_task(void *a, int k, int w) {
  const se_job *j = a;
  int cm = j->op->p[0], rd = j->op->p[1], P = j->op->p[2];
  int c0 = k * SE_CHUNK, c1 = rt_chunk_end(k, SE_CHUNK, cm);
  float g[SE_CHUNK] __attribute__((aligned(16)));
  se_gate_range(rt_t(j->c, j->op, 2, NULL), rt_t(j->c, j->op, 3, NULL), j->s1, rd, cm, c0, c1, g);
  float *D = rt_b(j->c, j->op, 0);
  for (int p = 0; p < P; p++) {
    float *d = D + (size_t)p * cm + c0;
    for (int c = 0; c < c1 - c0; c += 8) {
      vst1q_f32(d + c, vmulq_f32(vld1q_f32(d + c), vld1q_f32(g + c)));
      vst1q_f32(d + c + 4, vmulq_f32(vld1q_f32(d + c + 4), vld1q_f32(g + c + 4)));
    }
  }
}
static int se_run(rt_ctx *c, rt_op *op) {
  if (rt_threads(c) > 1) {
    float s1[512];
    v24k32_se_fc1(op->p[0], op->p[1], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), rt_b(c, op, 1), op->p[2], s1);
    se_job j = {c, op, s1};
    rt_par(c, rt_nchunks(op->p[0], SE_CHUNK), se_task, &j);
    return 0;
  }
  v24k32_se(op->p[0], op->p[1], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), rt_t(c, op, 2, NULL), rt_t(c, op, 3, NULL),
            rt_b(c, op, 1), op->p[2], rt_b(c, op, 0));
  return 0;
}

/* -------- SE_GATE: b = [chsum f32 [C], gate f32 [C]]; t = [se1_w [rd][C], se1_b, se2t [rd][C], se2_b]; p = [C rd P].
 * The gate of an SE block whose output feeds only a 1x1 conv: that conv multiplies its input by the gate as it reads
 * it (PW b[3], PW_S8 b[6]), so the block's activation is not rewritten in place. The gate is computed with the same
 * instructions as SE's (fc1 on the mean, then fc2 row by row from the bias, sigmoid): bit-identical. */
static int seg_init(rt_ctx *c, rt_op *op) { return rt_se_check(c, op, RT_F32, 1); }
static int seg_run(rt_ctx *c, rt_op *op) {
  int cm = op->p[0], rd = op->p[1];
  float s1[512];
  v24k32_se_fc1(cm, rd, rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), rt_b(c, op, 0), op->p[2], s1);
  se_gate_range(rt_t(c, op, 2, NULL), rt_t(c, op, 3, NULL), s1, rd, cm, 0, cm, rt_b(c, op, 1));
  return 0;
}

/* -------- POSTPC (v2.4): b = [X [24][192], feat [1024], pcin [8][1728], pcout [8][1024]];
 *          t = [post_s, post_b, pc_w, pc_b]; p = [N K Kp Np] */
int rt_postpc_check(rt_ctx *c, rt_op *op, int dt) {
  if (rt_need_t(c, op, 0, 192 * 4, 1) || rt_need_t(c, op, 1, 192 * 4, 1)) return RT_E_PARAM;
  return rt_need(c, op, 0, dt, 24 * 192) || rt_need(c, op, 1, RT_F32, FEAT_C) || rt_need(c, op, 2, dt, 8 * 1728) ||
                 rt_need(c, op, 3, dt, 8 * FEAT_C)
             ? RT_E_PARAM
             : 0;
}
static int postpc_init(rt_ctx *c, rt_op *op) {
  fpack *f = calloc(1, sizeof *f);
  op->state = f;
  if (!f) return rt_oom(c, op, sizeof *f);
  if (rt_need_args(c, op, 4, 4, 4) || rt_fpack_view(c, op, 2, 0, f)) return RT_E_PARAM;
  RT_REQ(c, op, f->N == FEAT_C && f->K == 1728); /* the v2.4 post conv: 192 ch x 3x3 -> 1024 */
  return rt_postpc_check(c, op, RT_F32);
}
static int postpc_run(rt_ctx *c, rt_op *op) {
  v24k32_postpc(rt_b(c, op, 0), rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), rt_b(c, op, 2), rt_b(c, op, 3), op->state,
                rt_b(c, op, 1));
  return 0;
}

/* -------- DENSE: b = [feat [1024], out [N]]; t = [w [N][1024] f32, b [N]]; p = [N] */
static int dense_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 2, 1)) return RT_E_PARAM;
  RT_REQ(c, op, op->p[0] > 0);
  if (rt_need(c, op, 0, RT_F32, FEAT_C) || rt_need(c, op, 1, RT_F32, (size_t)op->p[0])) return RT_E_PARAM;
  return rt_need_t(c, op, 0, (size_t)op->p[0] * FEAT_C * 4, 0) || rt_need_t(c, op, 1, (size_t)op->p[0] * 4, 0) ? RT_E_PARAM : 0;
}
static int dense_run(rt_ctx *c, rt_op *op) {
  v24k32_head(rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), op->p[0], rt_b(c, op, 0), rt_b(c, op, 1));
  return 0;
}

/* -------- FE_LOGMEL: b = [pcm, spec f32 [nfr][nmel]]; t = [window [frame], mel [nmel][nfft/2+1], consts {floor,
 *          scale}]; p = [nsamp frame hop pad nfft nmel] */
static int logmel_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 3, 6)) return RT_E_PARAM;
  const int *p = op->p;
  RT_REQ(c, op, p[0] > 0 && p[1] > 0 && p[2] > 0 && p[3] >= 0 && p[4] >= p[1] && p[5] > 0);
  if (rt_need_t(c, op, 0, (size_t)p[1] * 4, 0) || rt_need_t(c, op, 1, (size_t)p[5] * (p[4] / 2 + 1) * 4, 0) ||
      rt_need_t(c, op, 2, 8, 0))
    return RT_E_PARAM;
  const float *w = rt_t(c, op, 0, NULL), *mel = rt_t(c, op, 1, NULL), *k = rt_t(c, op, 2, NULL);
  if (rt_need(c, op, 0, RT_F32, (size_t)p[0]) || rt_need(c, op, 1, RT_F32, 1)) return RT_E_PARAM;
  /* the FFT's limits (pc_kern_f32.c pc_logmel_new): nfft a power of two in 16..2048, frame <= nfft, nmel <= 512 */
  RT_REQ(c, op, (p[4] & (p[4] - 1)) == 0 && p[4] >= 16 && p[4] <= 2048); /* nfft */
  RT_REQ(c, op, p[1] <= p[4] && p[5] <= 512);                             /* frame, nmel */
  pc_logmel *f = pc_logmel_new(p[0], p[1], p[2], p[3], p[4], p[5], w, mel, k[0], k[1]);
  op->state = f;
  if (!f) return rt_oom(c, op, (size_t)p[4] * 16);
  return rt_need(c, op, 1, RT_F32, (size_t)pc_logmel_frames(f) * p[5]);
}
static int logmel_run(rt_ctx *c, rt_op *op) {
  pc_logmel_run(op->state, rt_b(c, op, 0), rt_b(c, op, 1));
  return 0;
}
static void logmel_fini(rt_op *op) { pc_logmel_free(op->state); }

/* -------- STEM (fp32 math): b = [img f32 [H][W], out (halo opad, channel stride ldo), col f32 [R8(M)][Kp]];
 *          t = [w, b]; p = [N K Kp Np, H W Ho Wo k s act ldo opad] */
/* also the first half of the fp16 path's stem (fp32 math, fp16 padded output): rt_ops_f16.c */
int rt_stem_f32_check(rt_ctx *c, rt_op *op, fpack *f) {
  if (rt_need_args(c, op, -1, 2, 13) || rt_fpack_view(c, op, 0, 0, f)) return RT_E_PARAM;
  const int *p = op->p;
  int H = p[4], W = p[5], Ho = p[6], Wo = p[7], k = p[8], s = p[9], ldo = p[11], h = p[12];
  RT_REQ(c, op, k >= 1 && s >= 1 && Ho > 0 && Wo > 0);
  RT_REQ(c, op, (Ho - 1) * s + k <= H && (Wo - 1) * s + k <= W); /* VALID conv */
  RT_REQ(c, op, f->K == k * k);
  RT_REQ(c, op, ldo >= f->Np); /* the GEMM stores whole 8-channel panels */
  RT_REQ(c, op, h >= 0 && h <= 3);
  RT_REQ(c, op, p[10] >= 0 && p[10] <= 2);
  if (rt_need(c, op, 0, RT_F32, (size_t)H * W) || rt_need(c, op, 2, RT_F32, (size_t)((Ho * Wo + 7) & ~7) * f->Kp))
    return RT_E_PARAM;
  const rt_bufdesc *o = rt_bd(c, op, 1);
  if (!o) return rt_bad(c, op, "buffer argument 1 (output) is absent");
  if (o->halo != (uint32_t)h || o->ld != (uint32_t)ldo || (h && (o->W != (uint32_t)Wo || o->H != (uint32_t)Ho)))
    return rt_bad(c, op, "buffer argument 1 (buffer #%d, output): expected %d x %d, halo %d, ld %d; found %u x %u, halo %u, ld %u", op->b[1],
                  Ho, Wo, h, ldo, o->H, o->W, o->halo, o->ld);
  return rt_need(c, op, 1, (int)o->dtype, (size_t)(Ho + 2 * h) * (Wo + 2 * h) * ldo);
}
static int stem_init(rt_ctx *c, rt_op *op) {
  fpack *f = calloc(1, sizeof *f);
  op->state = f;
  if (!f) return rt_oom(c, op, sizeof *f);
  if (rt_need_args(c, op, 3, -1, -1) || rt_stem_f32_check(c, op, f)) return RT_E_PARAM;
  return rt_need(c, op, 1, RT_F32, 1);
}
static int stem_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  const fpack *f = op->state;
  float *col = rt_b(c, op, 2);
  pc_im2col1(rt_b(c, op, 0), p[4], p[5], p[8], p[9], p[6], p[7], col, f->Kp);
  if (p[12]) rt_zero_halo(rt_b(c, op, 1), p[6], p[7], p[11], p[12], 4);
  epi32 e = {.act = p[10], .out = rt_b(c, op, 1), .ldo = p[11], .ow = p[7], .opad = p[12]};
  rt_gemm_par(c, col, f->Kp, p[6] * p[7], f, &e); /* gemm_f32 over the context's threads: the same outputs */
  return 0;
}
void rt_stem_f32_tmp(rt_ctx *c, rt_op *op, const fpack *f, float *tmp, int ldt) {
  const int *p = op->p;
  float *col = rt_b(c, op, 2);
  pc_im2col1(rt_b(c, op, 0), p[4], p[5], p[8], p[9], p[6], p[7], col, f->Kp);
  epi32 e = {.act = p[10], .out = tmp, .ldo = ldt};
  gemm_f32(col, f->Kp, p[6] * p[7], f, &e);
}

/* -------- DWK: b = [xp (halo (k-1)/2), out [Ho][Wo][C], chsum]; t = [w [k*k][C], b]; p = [H W C k s pt pl Ho Wo act] */
int rt_dwk_check(rt_ctx *c, rt_op *op, size_t esz) {
  if (rt_need_args(c, op, 3, 2, 10)) return RT_E_PARAM;
  const int *p = op->p;
  int H = p[0], W = p[1], C = p[2], k = p[3], s = p[4], pt = p[5], pl = p[6], Ho = p[7], Wo = p[8], h = (k - 1) / 2;
  RT_REQ(c, op, H > 0 && W > 0 && Ho > 0 && Wo > 0 && C > 0);
  RT_REQ(c, op, C % 8 == 0);
  RT_REQ(c, op, k % 2 == 1 && k >= 1 && k <= 7);
  RT_REQ(c, op, s >= 1 && s <= 2);
  RT_REQ(c, op, pt >= 0 && pt <= h && pl >= 0 && pl <= h);
  RT_REQ(c, op, p[9] >= 0 && p[9] <= 1);
  if (rt_need_t(c, op, 0, (size_t)k * k * C * esz, 0) || rt_need_t(c, op, 1, (size_t)C * esz, 0)) return RT_E_PARAM;
  /* the last tap of the last output stays inside the haloed input */
  RT_REQ(c, op, (Ho - 1) * s - pt + k <= H + 2 * h - h);
  RT_REQ(c, op, (Wo - 1) * s - pl + k <= W + 2 * h - h);
  int dt = esz == 2 ? RT_F16 : RT_F32;
  if (rt_need_haloed(c, op, 0, dt, H, W, C, h)) return RT_E_PARAM;
  return rt_need(c, op, 1, dt, (size_t)Ho * Wo * C) || rt_need(c, op, 2, RT_F32, (size_t)C) ? RT_E_PARAM : 0;
}
static int dwk_init(rt_ctx *c, rt_op *op) { return rt_dwk_check(c, op, 4); }
static void dwk_task(void *a, int k, int w) {
  const op_job *j = a;
  const int *p = j->op->p;
  int c0 = k * CH_CHUNK, c1 = rt_chunk_end(k, CH_CHUNK, p[2]);
  pc_dwk_f32_cr(rt_b(j->c, j->op, 0), p[0], p[1], p[2], p[3], p[4], p[5], p[6], rt_t(j->c, j->op, 0, NULL),
                rt_t(j->c, j->op, 1, NULL), p[9], rt_b(j->c, j->op, 1), p[7], p[8], rt_b(j->c, j->op, 2), c0, c1);
}
static int dwk_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  if (rt_threads(c) > 1) {
    op_job j = {c, op};
    rt_par(c, rt_nchunks(p[2], CH_CHUNK), dwk_task, &j);
    return 0;
  }
  pc_dwk_f32(rt_b(c, op, 0), p[0], p[1], p[2], p[3], p[4], p[5], p[6], rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), p[9],
             rt_b(c, op, 1), p[7], p[8], rt_b(c, op, 2));
  return 0;
}

/* -------- PAD: b = [x, xp (halo h)]; p = [H W C ld h] */
int rt_pad_check(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, -1, 5)) return RT_E_PARAM;
  const int *p = op->p;
  const rt_bufdesc *x = rt_bd(c, op, 0), *o = rt_bd(c, op, 1);
  if (!x || !o) return rt_bad(c, op, "buffer argument %d is absent", x ? 1 : 0);
  RT_REQ(c, op, p[0] > 0 && p[1] > 0 && p[2] > 0);
  RT_REQ(c, op, p[4] >= 1 && p[4] <= 3);
  RT_REQ(c, op, p[3] >= p[2]);
  if (x->dtype != o->dtype)
    return rt_bad(c, op, "buffer argument 0 (buffer #%d, input) is %s but buffer argument 1 (buffer #%d, output) is %s", op->b[0], rt_dtype_name(x->dtype), op->b[1],
                  rt_dtype_name(o->dtype));
  if (x->halo) return rt_bad(c, op, "buffer argument 0 (buffer #%d, input): expected no halo, found halo %u", op->b[0], x->halo);
  if (o->ld != (uint32_t)p[2] || o->halo != (uint32_t)p[4])
    return rt_bad(c, op, "buffer argument 1 (buffer #%d, output): expected ld %d halo %d, found ld %u halo %u", op->b[1], p[2], p[4], o->ld, o->halo);
  if (rt_need(c, op, 0, (int)x->dtype, (size_t)p[0] * p[1] * p[3]) ||
      rt_need(c, op, 1, (int)o->dtype, (size_t)(p[0] + 2 * p[4]) * (p[1] + 2 * p[4]) * p[2]))
    return RT_E_PARAM;
  return 0;
}
static int pad_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  rt_zero_halo(rt_b(c, op, 1), p[0], p[1], p[2], p[4], 4);
  pc_pad_f32(rt_b(c, op, 0), p[0], p[1], p[2], p[3], p[4], rt_b(c, op, 1));
  return 0;
}

/* -------- MEAN: b = [x [P][ld], out f32 [C]]; p = [P C ld] */
int rt_mean_check(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, -1, 3)) return RT_E_PARAM;
  RT_REQ(c, op, op->p[0] > 0 && op->p[1] > 0);
  RT_REQ(c, op, op->p[1] % 4 == 0);
  RT_REQ(c, op, op->p[2] >= op->p[1]);
  const rt_bufdesc *x = rt_bd(c, op, 0);
  if (!x) return rt_bad(c, op, "buffer argument 0 (input) is absent");
  if (x->halo) return rt_bad(c, op, "buffer argument 0 (buffer #%d, input): expected no halo, found halo %u", op->b[0], x->halo);
  if (rt_need(c, op, 0, (int)x->dtype, (size_t)op->p[0] * op->p[2]) || rt_need(c, op, 1, RT_F32, (size_t)op->p[1]))
    return RT_E_PARAM;
  return 0;
}
static int mean_run(rt_ctx *c, rt_op *op) {
  pc_mean_f32(rt_b(c, op, 0), op->p[0], op->p[1], op->p[2], rt_b(c, op, 1));
  return 0;
}

/* -------- PROTO: b = [x [P][ld], logits f32 [nc], a [P][C] scratch, colmax f32 [Np]];
 *          t = [proto w, b (fpack | hpack w b32 b16), cls w [nc][ppc], cls b [nc], eps [1]];
 *          p = [N K Kp Np, P C ld nc ppc] */
/* The geometry every prototype head shares (the four forms: PROTO fp32 / fp16, PROTO_F32H, the dotprod int8 head,
 * PROTO_S8): an op record of 4 buffers, nt tensors and ni ints, p = [N K ...] with [P C ld nc ppc] as its last five
 * ints; kmax bounds K. N and K are checked here, before a caller rounds or multiplies them (the pack views round
 * N up). */
int rt_proto_geom(rt_ctx *c, rt_op *op, int nt, int ni, int kmax) {
  if (rt_need_args(c, op, 4, nt, ni)) return RT_E_PARAM;
  const int *p = op->p, *q = p + ni - 5;
  int N = p[0], K = p[1], P = q[0], C = q[1], ld = q[2], nc = q[3], ppc = q[4];
  RT_REQ(c, op, P > 0 && P <= 65536 && nc > 0 && ppc >= 1);
  RT_REQ(c, op, N > 0 && N <= (1 << 26) && K > 0 && K <= kmax); /* a negative K or C must not reach the kernels */
  RT_REQ(c, op, K == C);              /* prototype length = cell channels */
  RT_REQ(c, op, ld >= C);
  RT_REQ(c, op, (size_t)nc * ppc == (size_t)N); /* prototypes per class x classes = N */
  return 0;
}
/* ... and what they read and write alike, after rt_proto_geom: the class tensors at tensor arguments ct, ct + 1,
 * ct + 2 (cls w [N], cls b [nc], eps [1]), the unpadded cells (buffer 0, dtype dt, [P][ld]) and the logits (buffer 1) */
int rt_proto_io(rt_ctx *c, rt_op *op, int ct, int dt) {
  const int *p = op->p, *q = p + op->ni - 5;
  int N = p[0], P = q[0], ld = q[2], nc = q[3];
  if (rt_need_t(c, op, ct, (size_t)N * 4, 0) || rt_need_t(c, op, ct + 1, (size_t)nc * 4, 0) || rt_need_t(c, op, ct + 2, 4, 0))
    return RT_E_PARAM;
  if (rt_need_nohalo(c, op, 0, dt, (size_t)P * ld)) return RT_E_PARAM;
  return rt_need(c, op, 1, RT_F32, (size_t)nc);
}
/* PROTO fp32 / fp16 after rt_proto_geom and the pack view (nw = the view's tensors): Kp, the 8-row / 8-channel
 * tiles, then tensors and buffers */
int rt_proto_check(rt_ctx *c, rt_op *op, int nw) {
  const int *p = op->p;
  int K = p[1], P = p[4], C = p[5], dt = nw == 3 ? RT_F16 : RT_F32;
  RT_REQ(c, op, p[2] == K); /* Kp: the kernels read K floats per row of the K-stride scratch */
  RT_REQ(c, op, C % 8 == 0 && P % 8 == 0);
  if (rt_proto_io(c, op, nw, dt)) return RT_E_PARAM;
  return rt_need(c, op, 2, dt, (size_t)P * C) || rt_need(c, op, 3, RT_F32, (size_t)p[3]) ? RT_E_PARAM : 0;
}
static int proto_init(rt_ctx *c, rt_op *op) {
  fpack *f = calloc(1, sizeof *f);
  op->state = f;
  if (!f) return rt_oom(c, op, sizeof *f);
  if (rt_proto_geom(c, op, 5, 9, 1 << 20) || rt_fpack_view(c, op, 0, 0, f) || rt_proto_check(c, op, 2)) return RT_E_PARAM;
  return 0;
}
static int proto_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  const float *eps = rt_t(c, op, 4, NULL);
  float *a = rt_b(c, op, 2), *cm = rt_b(c, op, 3);
  pc_l2norm_f32(rt_b(c, op, 0), p[4], p[5], p[6], eps[0], a);
  gemm_f32_colmax(a, p[5], p[4], op->state, cm);
  return pc_proto_logits(cm, rt_t(c, op, 2, NULL), rt_t(c, op, 3, NULL), p[7], p[8], rt_b(c, op, 1));
}

/* PROTO_F32H: PROTO with fp16-stored prototypes, fp32 math (an opcode of its own, not a variant of PROTO):
 * t = [w16 [Np/8][K][8] f16, cls w, cls b, eps]; p = [N K Np, P C ld nc ppc] */
static int proto_h_init(rt_ctx *c, rt_op *op) {
  if (rt_proto_geom(c, op, 4, 8, 1 << 20)) return RT_E_PARAM;
  const int *p = op->p;
  int N = p[0], K = p[1], Np = p[2], P = p[3], C = p[4];
  RT_REQ(c, op, K % 4 == 0);
  RT_REQ(c, op, P % 8 == 0 && Np % 8 == 0 && Np >= N);
  if (rt_need_t(c, op, 0, (size_t)Np * K * 2, 1) || rt_proto_io(c, op, 1, RT_F32)) return RT_E_PARAM;
  return rt_need(c, op, 2, RT_F32, (size_t)P * C) || rt_need(c, op, 3, RT_F32, (size_t)Np) ? RT_E_PARAM : 0;
}
static int proto_h_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  const float *eps = rt_t(c, op, 3, NULL);
  float *a = rt_b(c, op, 2), *cm = rt_b(c, op, 3);
  pc_l2norm_f32(rt_b(c, op, 0), p[3], p[4], p[5], eps[0], a);
  pc_proto_colmax_f32h(a, p[3], p[1], rt_t(c, op, 0, NULL), p[2], cm);
  return pc_proto_logits(cm, rt_t(c, op, 1, NULL), rt_t(c, op, 2, NULL), p[6], p[7], rt_b(c, op, 1));
}

/* -------- GEMV: b = [x f32 [K], out f32 [N]]; t = [w f32 [N][K], b f32 [N]]; p = [N K act]; reads x[0..K) only */
int rt_gemv_check(rt_ctx *c, rt_op *op, size_t esz) {
  if (rt_need_args(c, op, 2, 2, 3)) return RT_E_PARAM;
  int N = op->p[0], K = op->p[1];
  RT_REQ(c, op, N > 0 && K > 0);
  RT_REQ(c, op, op->p[2] == RT_ACT_NONE || op->p[2] == RT_ACT_RELU);
  if (rt_need_t(c, op, 0, (size_t)N * K * esz, 0) || rt_need_t(c, op, 1, (size_t)N * 4, 0)) return RT_E_PARAM;
  return rt_need(c, op, 0, RT_F32, (size_t)K) || rt_need(c, op, 1, RT_F32, (size_t)N) ? RT_E_PARAM : 0;
}
static int gemv_init(rt_ctx *c, rt_op *op) { return rt_gemv_check(c, op, 4); }
#define GEMV_CHUNK 128
static void gemv_task(void *a, int k, int w) {
  const op_job *j = a;
  const int *p = j->op->p;
  int n0 = k * GEMV_CHUNK, n = rt_chunk_end(k, GEMV_CHUNK, p[0]) - n0;
  const float *W = rt_t(j->c, j->op, 0, NULL), *b = rt_t(j->c, j->op, 1, NULL);
  bnn_gemv_f32(W + (size_t)n0 * p[1], b + n0, n, p[1], rt_b(j->c, j->op, 0), p[2], (float *)rt_b(j->c, j->op, 1) + n0);
}
static int gemv_run(rt_ctx *c, rt_op *op) {
  if (rt_threads(c) > 1) {
    op_job j = {c, op};
    rt_par(c, rt_nchunks(op->p[0], GEMV_CHUNK), gemv_task, &j);
    return 0;
  }
  bnn_gemv_f32(rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), op->p[0], op->p[1], rt_b(c, op, 0), op->p[2], rt_b(c, op, 1));
  return 0;
}

const rt_kernel rt_k_f32[RT_OP_N] = {
    [RT_OP_FE_STFT2] = {fe_stft2_init, fe_stft2_run, fe_stft2_fini, "fe_stft2", 0x2, 0},
    [RT_OP_C0POOL] = {c0_init, c0_run, c0_fini, "c0pool_f32", 0x2, 0},
    [RT_OP_PW] = {pw_init, pw_run, pw_fini, "pw_f32", 0x2, 0xc},
    [RT_OP_SE_GATE] = {seg_init, seg_run, NULL, "se_gate_f32", 0x2, 0},
    [RT_OP_DW3] = {dw3_init, dw3_run, NULL, "dw3_f32", 0x6, 0},
    [RT_OP_SE] = {se_init, se_run, NULL, "se_f32", 0x1, 0},
    [RT_OP_POSTPC] = {postpc_init, postpc_run, rt_free_state, "postpc_f32", 0xe, 0},
    [RT_OP_DENSE] = {dense_init, dense_run, NULL, "dense_f32", 0x2, 0},
    [RT_OP_FE_LOGMEL] = {logmel_init, logmel_run, logmel_fini, "fe_logmel", 0x2, 0},
    [RT_OP_STEM] = {stem_init, stem_run, rt_free_state, "stem_f32", 0x6, 0},
    [RT_OP_DWK] = {dwk_init, dwk_run, NULL, "dwk_f32", 0x6, 0},
    [RT_OP_PAD] = {rt_pad_check, pad_run, NULL, "pad_f32", 0x2, 0},
    [RT_OP_MEAN] = {rt_mean_check, mean_run, NULL, "mean_f32", 0x2, 0},
    [RT_OP_PROTO] = {proto_init, proto_run, rt_free_state, "proto_f32", 0xe, 0},
    [RT_OP_PROTO_F32H] = {proto_h_init, proto_h_run, NULL, "proto_f32h", 0xe, 0},
    [RT_OP_GEMV] = {gemv_init, gemv_run, NULL, "gemv_f32", 0x2, 0},
};
