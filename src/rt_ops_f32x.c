// Registry kernels with fp32 arithmetic, opcodes 16.. (baseline ARMv8-A objects; any AArch64): the dB-image front end,
// dense k x k convolution, GeM pooling, the SED attention head and vector addition. Every size is data; init checks
// every buffer, tensor and parameter the run touches.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fe_dbimg.h"
#include "gemm_f32.h"
#include "rt_ops.h"
#include "wino_f32.h"

typedef struct { fe_db *f; void *work; int nw; } fe_st;

/* -------- FE_DBIMG: b = [pcm, img (halo h, ld)]; t = [window [N], mel [nmel][N/2+1], rows i32 [R], consts
 *          {floor, top_db, eps, post[3], chan0[C], chan1[C]}]; p = [nsamp N hop pad padmode nfr nmel R has_top_db norm
 *          post0 post1 post2 resize Ho Wo C chan0 chan1 ld h] */
static int fe_db_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 4, 21)) return RT_E_PARAM;
  const int *p = op->p;
  fe_db_cfg g = {.nsamp = p[0], .nfft = p[1], .hop = p[2], .pad = p[3], .padmode = p[4], .nfr = p[5], .nmel = p[6], .R = p[7],
                 .has_top_db = p[8], .norm = p[9], .post = {p[10], p[11], p[12]}, .Ho = p[14], .Wo = p[15], .C = p[16],
                 .chan = {p[17], p[18]}, .ld = p[19], .halo = p[20]};
  int N = g.nfft, F = N / 2 + 1;
  RT_REQ(c, op, N >= 64 && N <= 8192 && (N & (N - 1)) == 0);
  RT_REQ(c, op, g.hop > 0 && g.pad >= 0 && g.padmode >= 0 && g.padmode <= 2 && (g.padmode || !g.pad));
  RT_REQ(c, op, g.padmode != 1 || g.pad < g.nsamp);
  RT_REQ(c, op, g.nfr > 0 && g.nfr == (g.nsamp + 2 * g.pad - N) / g.hop + 1 && g.nsamp + 2 * g.pad >= N);
  RT_REQ(c, op, g.nmel > 0 && g.R > 0 && g.nmel <= 4096);
  RT_REQ(c, op, g.has_top_db == 0 || g.has_top_db == 1);
  RT_REQ(c, op, g.norm >= 0 && g.norm <= 2);
  RT_REQ(c, op, p[10] >= 0 && p[10] <= 4 && p[11] >= 0 && p[11] <= 4 && p[12] >= 0 && p[12] <= 4 && p[13] == 1);
  RT_REQ(c, op, g.Ho > 0 && g.Wo > 0 && g.C > 0 && g.C <= 64 && g.ld >= g.C && g.halo >= 0 && g.halo <= 3);
  RT_REQ(c, op, g.chan[0] >= 0 && g.chan[0] <= 4 && g.chan[1] >= 0 && g.chan[1] <= 4);
  if (rt_need_t(c, op, 0, (size_t)N * 4, 0) || rt_need_t(c, op, 1, (size_t)g.nmel * F * 4, 0) ||
      rt_need_t(c, op, 2, (size_t)g.R * 4, 0) || rt_need_t(c, op, 3, (size_t)(6 + 2 * g.C) * 4, 0))
    return RT_E_PARAM;
  const int32_t *rows = rt_t(c, op, 2, NULL);
  for (int r = 0; r < g.R; r++)
    if (rows[r] < 0 || rows[r] >= g.nmel) return rt_bad(c, op, "row table entry %d = %d outside 0..%d", r, rows[r], g.nmel - 1);
  const float *k = rt_t(c, op, 3, NULL);
  g.floor = k[0]; g.top_db = k[1]; g.eps = k[2]; g.postv[0] = k[3]; g.postv[1] = k[4]; g.postv[2] = k[5];
  if (rt_need(c, op, 0, RT_F32, (size_t)g.nsamp)) return RT_E_PARAM;
  const rt_bufdesc *o = rt_bd(c, op, 1);
  if (!o) return rt_bad(c, op, "buffer argument 1 (image) is absent");
  if (o->halo != (uint32_t)g.halo || o->ld != (uint32_t)g.ld || o->H != (uint32_t)g.Ho || o->W != (uint32_t)g.Wo)
    return rt_bad(c, op, "buffer argument 1 (buffer #%d, image): expected %d x %d, ld %d, halo %d; found %u x %u, ld %u, halo %u",
                  op->b[1], g.Ho, g.Wo, g.ld, g.halo, o->H, o->W, o->ld, o->halo);
  if (rt_need(c, op, 1, RT_F32, (size_t)(g.Ho + 2 * g.halo) * (g.Wo + 2 * g.halo) * g.ld)) return RT_E_PARAM;
  fe_st *s = calloc(1, sizeof *s);
  op->state = s;
  if (!s) return rt_oom(c, op, sizeof *s);
  s->f = fe_db_new(&g, rt_t(c, op, 0, NULL), rt_t(c, op, 1, NULL), rows, k + 6);
  return s->f ? 0 : rt_oom(c, op, (size_t)g.nmel * g.nfr * 4 + (size_t)N * 64);
}
#define FE_FRAMES 8
static void fe_task(void *a, int k, int w) {
  const op_job *j = a;
  fe_st *s = j->op->state;
  size_t wb = fe_db_work_bytes(s->f);
  float32x4_t *re = (float32x4_t *)((char *)s->work + (size_t)w * wb);
  const fe_db_cfg *g = fe_db_config(s->f);
  int nh = g->nfft / 2;
  int t0 = k * FE_FRAMES, t1 = rt_chunk_end(k, FE_FRAMES, g->nfr);
  fe_db_frames(s->f, rt_b(j->c, j->op, 0), t0, t1, re, re + nh, re + 2 * nh);
}
static int fe_db_krun(rt_ctx *c, rt_op *op) {
  fe_st *s = op->state;
  int T = rt_threads(c);
  if (T > 1) {
    /* one set of FFT work buffers per thread */
    if (s->nw < T && !(s->work = rt_worker_scratch(s->work, &s->nw, T, fe_db_work_bytes(s->f) * T))) return RT_E_OOM;
    op_job j = {c, op};
    rt_par(c, rt_nchunks(fe_db_config(s->f)->nfr, FE_FRAMES), fe_task, &j);
    fe_db_image(s->f, rt_b(c, op, 1));
    return 0;
  }
  fe_db_run(s->f, rt_b(c, op, 0), rt_b(c, op, 1));
  return 0;
}
static void fe_db_fini(rt_op *op) {
  fe_st *s = op->state;
  if (!s) return;
  fe_db_free(s->f);
  free(s->work);
  free(s);
}

/* -------- CONVK: b = [x (halo h, ld), out (halo opad, ldo), res | -1]; t = [w, b] (K = kh * kw * ld, order ky kx c);
 *          p = [N K Kp Np, H W Ho Wo kh kw s oy0 ox0 act ldo opad]: output (oy, ox) reads input rows oy*s + oy0 + ky,
 *          columns ox*s + ox0 + kx of the haloed buffer */
static int convk_init(rt_ctx *c, rt_op *op) {
  fpack *f = calloc(1, sizeof *f);
  op->state = f;
  if (!f) return rt_oom(c, op, sizeof *f);
  if (rt_need_args(c, op, 3, 2, 16) || rt_fpack_view(c, op, 0, 0, f)) return RT_E_PARAM;
  const int *p = op->p;
  int H = p[4], W = p[5], Ho = p[6], Wo = p[7], kh = p[8], kw = p[9], st = p[10], oy0 = p[11], ox0 = p[12], act = p[13];
  int ldo = p[14], opad = p[15];
  RT_REQ(c, op, H > 0 && W > 0 && Ho > 0 && Wo > 0 && kh >= 1 && kh <= 7 && kw >= 1 && kw <= 7 && st >= 1 && st <= 2);
  RT_REQ(c, op, act >= 0 && act <= 2 && opad >= 0 && opad <= 3 && oy0 >= 0 && ox0 >= 0);
  const rt_bufdesc *x = rt_bd(c, op, 0), *o = rt_bd(c, op, 1);
  if (!x || !o) return rt_bad(c, op, "buffer argument %d is absent", x ? 1 : 0);
  int h = (int)x->halo, ld = (int)x->ld, Wp = W + 2 * h;
  if (x->dtype != RT_F32 || x->H != (uint32_t)H || x->W != (uint32_t)W || ld % 4)
    return rt_bad(c, op, "buffer argument 0 (buffer #%d, input): expected fp32 %d x %d with ld a multiple of 4, found %s %u x %u ld %u",
                  op->b[0], H, W, rt_dtype_name(x->dtype), x->H, x->W, x->ld);
  RT_REQ(c, op, f->K == kh * kw * ld && f->Kp == f->K);
  RT_REQ(c, op, oy0 + (Ho - 1) * st + kh <= H + 2 * h && ox0 + (Wo - 1) * st + kw <= W + 2 * h);
  /* the 8-pixel tiles of the last output row read 7 strides past it, and the K loop 16 B ahead */
  size_t last = ((size_t)(oy0 + (Ho - 1) * st + kh - 1) * Wp + ox0 + (size_t)(((Wo + 7) & ~7) - 1) * st + kw) * ld + 4;
  if (rt_need(c, op, 0, RT_F32, last)) return RT_E_PARAM;
  if (o->dtype != RT_F32 || o->H != (uint32_t)Ho || o->W != (uint32_t)Wo || o->ld != (uint32_t)ldo || o->halo != (uint32_t)opad ||
      ldo < f->Np)
    return rt_bad(c, op, "buffer argument 1 (buffer #%d, output): expected fp32 %d x %d ld %d (>= %d) halo %d, found %s %u x %u ld %u halo %u",
                  op->b[1], Ho, Wo, ldo, f->Np, opad, rt_dtype_name(o->dtype), o->H, o->W, o->ld, o->halo);
  if (rt_need(c, op, 1, RT_F32, (size_t)(Ho + 2 * opad) * (Wo + 2 * opad) * ldo)) return RT_E_PARAM;
  const rt_bufdesc *r = rt_bd(c, op, 2);
  if (r) {
    if (r->dtype != RT_F32 || r->H != (uint32_t)Ho || r->W != (uint32_t)Wo || r->ld < (uint32_t)f->Np)
      return rt_bad(c, op, "buffer argument 2 (buffer #%d, residual): expected fp32 %d x %d with ld >= %d, found %s %u x %u ld %u",
                    op->b[2], Ho, Wo, f->Np, rt_dtype_name(r->dtype), r->H, r->W, r->ld);
    if (rt_need(c, op, 2, RT_F32, (size_t)(Ho + 2 * r->halo) * (Wo + 2 * r->halo) * r->ld)) return RT_E_PARAM;
  }
  return 0;
}
static void convk_rows(rt_ctx *c, rt_op *op, int y0, int y1) {
  const fpack *f = op->state;
  const int *p = op->p;
  int W = p[5], Wo = p[7], kh = p[8], kw = p[9], st = p[10], oy0 = p[11], ox0 = p[12], ldo = p[14], opad = p[15];
  const rt_bufdesc *x = rt_bd(c, op, 0), *r = rt_bd(c, op, 2);
  int h = (int)x->halo, ld = (int)x->ld, Wp = W + 2 * h;
  const float *xp = rt_b(c, op, 0), *res = rt_b(c, op, 2);
  float *out = rt_b(c, op, 1);
  for (int oy = y0; oy < y1; oy++) {
    const float *A = xp + ((size_t)(oy * st + oy0) * Wp + ox0) * ld;
    epi32 e = {.act = p[13], .out = out + (size_t)oy * (Wo + 2 * opad) * ldo, .ldo = ldo, .ow = Wo, .opad = opad};
    if (res) {
      int rh = (int)r->halo;
      e.res = res + ((size_t)(oy + rh) * (Wo + 2 * rh) + rh) * r->ld;
      e.ldr = (int)r->ld;
    }
    gemm_f32_seg(A, st * ld, Wo, kh, kw * ld, (long)Wp * ld, f, &e);
  }
}
static void convk_task(void *a, int k, int w) {
  const op_job *j = a;
  convk_rows(j->c, j->op, k, k + 1);
}
static int convk_run(rt_ctx *c, rt_op *op) {
  const int *p = op->p;
  if (p[15]) rt_zero_halo(rt_b(c, op, 1), p[6], p[7], p[14], p[15], 4);
  if (rt_threads(c) > 1) {
    op_job j = {c, op};
    rt_par(c, p[6], convk_task, &j);
  } else
    convk_rows(c, op, 0, p[6]);
  return 0;
}

/* -------- GEM: b = [x [P][ld], out [C]]; t = [consts {eps, p, q}]; p = [P C ld] */
/* log2 for x > 0 (normal): x = m 2^e, m in [sqrt(1/2), sqrt(2)); log2 m = 2 atanh(s) / ln 2, s = (m - 1)/(m + 1) */
static inline float32x4_t log2_atanh_f32x4(float32x4_t x) {
  int32x4_t xi = vreinterpretq_s32_f32(x);
  int32x4_t e = vshrq_n_s32(vsubq_s32(xi, vdupq_n_s32(0x3f3504f3)), 23);
  int32x4_t mi = vsubq_s32(xi, vshlq_n_s32(e, 23));
  float32x4_t m = vreinterpretq_f32_s32(mi);
  float32x4_t s = vdivq_f32(vsubq_f32(m, vdupq_n_f32(1.f)), vaddq_f32(m, vdupq_n_f32(1.f)));
  float32x4_t s2 = vmulq_f32(s, s);
  float32x4_t q = vfmaq_f32(vdupq_n_f32(1.f / 9.f), s2, vdupq_n_f32(1.f / 11.f));
  q = vfmaq_f32(vdupq_n_f32(1.f / 7.f), s2, q);
  q = vfmaq_f32(vdupq_n_f32(1.f / 5.f), s2, q);
  q = vfmaq_f32(vdupq_n_f32(1.f / 3.f), s2, q);
  q = vfmaq_f32(vdupq_n_f32(1.f), s2, q);
  float32x4_t ln = vmulq_f32(vmulq_n_f32(s, 2.f), q);
  return vfmaq_f32(vcvtq_f32_s32(e), ln, vdupq_n_f32(1.4426950408889634f));
}
static int gem_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 2, 1, 3)) return RT_E_PARAM;
  int P = op->p[0], C = op->p[1], ld = op->p[2];
  RT_REQ(c, op, P > 0 && C > 0 && C % 4 == 0 && ld >= C);
  if (rt_need_t(c, op, 0, 12, 0)) return RT_E_PARAM;
  const float *k = rt_t(c, op, 0, NULL);
  RT_REQ(c, op, k[0] > 0.f && k[1] > 0.f);
  return rt_need_nohalo(c, op, 0, RT_F32, (size_t)P * ld) || rt_need(c, op, 1, RT_F32, (size_t)C) ? RT_E_PARAM : 0;
}
static void gem_f32(const float *x, int P, int c0, int c1, int ld, const float *k, float *out) {
  const float32x4_t eps = vdupq_n_f32(k[0]);
  const float pp = k[1];
  for (int c = c0; c < c1; c += 4) {
    float32x4_t acc = vdupq_n_f32(0);
    for (int i = 0; i < P; i++) {
      float32x4_t v = vmaxq_f32(vld1q_f32(x + (size_t)i * ld + c), eps);
      acc = vaddq_f32(acc, exp2_f32x4(vmulq_n_f32(log2_atanh_f32x4(v), pp)));
    }
    float a[4];
    vst1q_f32(a, vdivq_f32(acc, vdupq_n_f32((float)P)));
    for (int j = 0; j < 4; j++) out[c + j] = powf(a[j], k[2]);
  }
}
static void gem_task(void *a, int k, int w) {
  const op_job *j = a;
  int C = j->op->p[1], c0 = k * CH_CHUNK, c1 = rt_chunk_end(k, CH_CHUNK, C);
  gem_f32(rt_b(j->c, j->op, 0), j->op->p[0], c0, c1, j->op->p[2], rt_t(j->c, j->op, 0, NULL), rt_b(j->c, j->op, 1));
}
static int gem_run(rt_ctx *c, rt_op *op) {
  if (rt_threads(c) > 1) {
    op_job j = {c, op};
    rt_par(c, rt_nchunks(op->p[1], CH_CHUNK), gem_task, &j);
    return 0;
  }
  gem_f32(rt_b(c, op, 0), op->p[0], 0, op->p[1], op->p[2], rt_t(c, op, 0, NULL), rt_b(c, op, 1));
  return 0;
}

/* -------- SED: b = [x [H][W][ld], out [NC], s [L8][Cp], pooled [L8][Cp], h [L8][hid_p], att [L8][NCp], cla [L8][NCp]];
 *          t = [dw [dk][C], db [C], fc w b, att w b, cla w b, consts {clip lo, clip hi, scale}];
 *          p = [fc N K Kp Np, att N K Kp Np, cla N K Kp Np, red H W C ld L pk dk has_clip] */
typedef struct { fpack fc, at, cl; } sed_st;
/* What SED and SED_S8 (rt_ops_s8n.c) check alike, after their three GEMM views: the reduction and pooling
 * parameters, how the fc / att / cla shapes (p[0..11], each [N K Kp Np]) chain, the depthwise tensors and the
 * consts tensor kt, the input and buffers 1..6; Cp = the channel stride of s and pooled */
int rt_sed_geom(rt_ctx *c, rt_op *op, int kt, int Cp) {
  const int *p = op->p;
  int fcN = p[0], fcK = p[1], fcNp = p[3], atN = p[4], atK = p[5], atNp = p[7], clN = p[8], clK = p[9], clNp = p[11];
  int red = p[12], H = p[13], W = p[14], C = p[15], ld = p[16], L = p[17], pk = p[18], dk = p[19], hc = p[20];
  int L8 = (L + 7) & ~7;
  RT_REQ(c, op, red == 2 || red == 3);
  RT_REQ(c, op, H > 0 && W > 0 && C > 0 && C % 4 == 0 && ld >= C && L == (red == 3 ? H : W));
  RT_REQ(c, op, pk >= 0 && pk <= 15 && (pk == 0 || pk % 2) && dk >= 0 && dk <= 15 && (dk == 0 || dk % 2) && (pk || dk));
  RT_REQ(c, op, hc == 0 || hc == 1);
  RT_REQ(c, op, L <= 64);
  RT_REQ(c, op, fcK == C && atK == fcN && clK == fcN && atN == clN);
  if (rt_need_t(c, op, 0, (size_t)(dk ? dk : 1) * C * 4, 0) || rt_need_t(c, op, 1, (size_t)C * 4, 0) ||
      rt_need_t(c, op, kt, 12, 0))
    return RT_E_PARAM;
  const rt_bufdesc *x = rt_bd(c, op, 0);
  if (!x || x->halo || x->dtype != RT_F32 || x->H != (uint32_t)H || x->W != (uint32_t)W)
    return rt_bad(c, op, "buffer argument 0: expected an unpadded fp32 activation %d x %d", H, W);
  return rt_need(c, op, 0, RT_F32, (size_t)H * W * ld) || rt_need(c, op, 1, RT_F32, (size_t)atN) ||
                 rt_need(c, op, 2, RT_F32, (size_t)L8 * Cp) || rt_need(c, op, 3, RT_F32, (size_t)L8 * Cp) ||
                 rt_need(c, op, 4, RT_F32, (size_t)L8 * fcNp) || rt_need(c, op, 5, RT_F32, (size_t)L8 * atNp) ||
                 rt_need(c, op, 6, RT_F32, (size_t)L8 * clNp)
             ? RT_E_PARAM
             : 0;
}
static int sed_init(rt_ctx *c, rt_op *op) {
  sed_st *s = calloc(1, sizeof *s);
  op->state = s;
  if (!s) return rt_oom(c, op, sizeof *s);
  if (rt_need_args(c, op, 7, 9, 21) || rt_fpack_view(c, op, 2, 0, &s->fc) || rt_fpack_view(c, op, 4, 4, &s->at) ||
      rt_fpack_view(c, op, 6, 8, &s->cl))
    return RT_E_PARAM;
  RT_REQ(c, op, s->fc.Kp % 8 == 0 && s->at.Kp == s->fc.Np && s->cl.Kp == s->fc.Np);
  return rt_sed_geom(c, op, 8, s->fc.Kp);
}
/* axis mean + pooling for channels [c0, c1) */
void rt_sed_pool(rt_ctx *c, rt_op *op, int Cp, int c0, int c1) {
  const int *p = op->p;
  int red = p[12], H = p[13], W = p[14], ld = p[16], L = p[17], pk = p[18], dk = p[19], C = p[15];
  int n = red == 3 ? W : H;
  const float *x = rt_b(c, op, 0), *dw = rt_t(c, op, 0, NULL), *db = rt_t(c, op, 1, NULL);
  float *s = rt_b(c, op, 2), *pl = rt_b(c, op, 3);
  for (int t = 0; t < L; t++)
    for (int ch = c0; ch < c1; ch++) {
      float a = 0.f;
      for (int u = 0; u < n; u++) a += red == 3 ? x[((size_t)t * W + u) * ld + ch] : x[((size_t)u * W + t) * ld + ch];
      s[(size_t)t * Cp + ch] = a / (float)n;
    }
  for (int t = 0; t < L; t++)
    for (int ch = c0; ch < c1; ch++) {
      float m = -INFINITY, cv = 0.f;
      for (int d = 0; d < pk; d++) {
        int u = t + d - pk / 2;
        if (u >= 0 && u < L) m = fmaxf(m, s[(size_t)u * Cp + ch]);
      }
      for (int d = 0; d < dk; d++) {
        int u = t + d - dk / 2;
        if (u >= 0 && u < L) cv += dw[(size_t)d * C + ch] * s[(size_t)u * Cp + ch];
      }
      if (dk) cv += db[ch];
      pl[(size_t)t * Cp + ch] = pk && dk ? m + cv : pk ? m : cv;
    }
}
/* attention softmax over time and the weighted class sum for classes [k0, k1) */
void rt_sed_out(rt_ctx *c, rt_op *op, int ldn, int ldc, int kt, int k0, int k1) {
  const int *p = op->p;
  int L = p[17], hc = p[20];
  const float *kk = rt_t(c, op, kt, NULL); /* {clip lo, clip hi, scale} */
  const float *att = rt_b(c, op, 5), *cla = rt_b(c, op, 6);
  float *out = rt_b(c, op, 1);
  for (int k = k0; k < k1; k++) {
    float a[64], mx = -INFINITY, z = 0.f, acc = 0.f;
    for (int t = 0; t < L; t++) {
      float v = att[(size_t)t * ldn + k];
      if (hc) v = v < kk[0] ? kk[0] : v > kk[1] ? kk[1] : v;
      a[t] = v; mx = fmaxf(mx, v);
    }
    for (int t = 0; t < L; t++) { a[t] = expf(a[t] - mx); z += a[t]; }
    for (int t = 0; t < L; t++) acc += a[t] / z * cla[(size_t)t * ldc + k];
    out[k] = acc * kk[2];
  }
}
static void sed_pool_task(void *a, int k, int w) {
  const op_job *j = a;
  int C = j->op->p[15], c0 = k * CH_CHUNK, c1 = rt_chunk_end(k, CH_CHUNK, C);
  rt_sed_pool(j->c, j->op, ((const sed_st *)j->op->state)->fc.Kp, c0, c1);
}
static void sed_out_task(void *a, int k, int w) {
  const op_job *j = a;
  const sed_st *st = j->op->state;
  int k0 = k * SED_CLS, k1 = rt_chunk_end(k, SED_CLS, st->at.N);
  rt_sed_out(j->c, j->op, st->at.Np, st->cl.Np, 8, k0, k1);
}
static int sed_run(rt_ctx *c, rt_op *op) {
  const sed_st *st = op->state;
  const int *p = op->p;
  int C = p[15], L = p[17], Cp = st->fc.Kp;
  float *s = rt_b(c, op, 2), *pl = rt_b(c, op, 3), *h = rt_b(c, op, 4);
  for (int t = 0; t < L; t++) { /* pad channels of the GEMM input */
    memset(s + (size_t)t * Cp + C, 0, sizeof(float) * (Cp - C));
    memset(pl + (size_t)t * Cp + C, 0, sizeof(float) * (Cp - C));
  }
  epi32 ef = {.act = ACT32_RELU, .out = h, .ldo = st->fc.Np};
  epi32 ea = {.act = ACT32_NONE, .out = rt_b(c, op, 5), .ldo = st->at.Np};
  epi32 ec = {.act = ACT32_NONE, .out = rt_b(c, op, 6), .ldo = st->cl.Np};
  if (rt_threads(c) > 1) {
    op_job j = {c, op};
    rt_par(c, rt_nchunks(C, CH_CHUNK), sed_pool_task, &j);
    rt_gemm_par(c, pl, Cp, L, &st->fc, &ef);
    rt_gemm_par(c, h, st->fc.Np, L, &st->at, &ea);
    rt_gemm_par(c, h, st->fc.Np, L, &st->cl, &ec);
    rt_par(c, rt_nchunks(st->at.N, SED_CLS), sed_out_task, &j);
    return 0;
  }
  rt_sed_pool(c, op, Cp, 0, C);
  gemm_f32(pl, Cp, L, &st->fc, &ef);
  gemm_f32(h, st->fc.Np, L, &st->at, &ea);
  gemm_f32(h, st->fc.Np, L, &st->cl, &ec);
  rt_sed_out(c, op, st->at.Np, st->cl.Np, 8, 0, st->at.N);
  return 0;
}

/* -------- VADD: b = [a, b, out]; p = [n] */
static int vadd_init(rt_ctx *c, rt_op *op) {
  if (rt_need_args(c, op, 3, 0, 1)) return RT_E_PARAM;
  RT_REQ(c, op, op->p[0] > 0);
  size_t n = (size_t)op->p[0];
  return rt_need(c, op, 0, RT_F32, n) || rt_need(c, op, 1, RT_F32, n) || rt_need(c, op, 2, RT_F32, n) ? RT_E_PARAM : 0;
}
static int vadd_run(rt_ctx *c, rt_op *op) {
  const float *a = rt_b(c, op, 0), *b = rt_b(c, op, 1);
  float *o = rt_b(c, op, 2);
  for (int i = 0; i < op->p[0]; i++) o[i] = a[i] + b[i];
  return 0;
}

/* -------- CONVW: b = [x (halo >= 1, ld C), out (halo opad, ldo), res | -1]; t = [U [36][Np/8][C][8], bias [Np + 8]];
 *          p = [C N Np H W act ldo opad S Nb] (output H x W) */
#define WINO_STRIP 8
#define WINO_NB 40  /* output channels per block with 2+ threads: four workers' V strips and M blocks share the L2 */
#define WINO_NB1 64 /* on one thread the L2 is one worker's: a larger block reads the V strip fewer times (K062) */
typedef struct { wino43_cfg g; float *scr, *bz; int nw; size_t scrf; } convw_st;
static int convw_init(rt_ctx *c, rt_op *op) {
  convw_st *s = calloc(1, sizeof *s);
  op->state = s;
  if (!s) return rt_oom(c, op, sizeof *s);
  if (rt_need_args(c, op, 3, 2, 10)) return RT_E_PARAM;
  const int *p = op->p;
  wino43_cfg g = {.C = p[0], .N = p[1], .Np = p[2], .H = p[3], .W = p[4], .Ho = p[3], .Wo = p[4], .act = p[5], .ldo = p[6],
                  .opad = p[7], .S = p[8], .Nb = p[9]};
  RT_REQ(c, op, g.C > 0 && g.C % 4 == 0 && g.N > 0 && g.Np == ((g.N + 7) & ~7) && g.H > 0 && g.W > 0);
  RT_REQ(c, op, g.act >= 0 && g.act <= 2 && g.opad >= 0 && g.opad <= 3 && g.ldo >= g.Np);
  RT_REQ(c, op, g.S >= 1 && g.S <= 256 && g.Nb >= 8 && g.Nb % 8 == 0);
  const rt_bufdesc *x = rt_bd(c, op, 0), *o = rt_bd(c, op, 1);
  if (!x || !o) return rt_bad(c, op, "buffer argument %d is absent", x ? 1 : 0);
  if (x->dtype != RT_F32 || x->H != (uint32_t)g.H || x->W != (uint32_t)g.W || x->ld != (uint32_t)g.C || x->halo < 1)
    return rt_bad(c, op, "buffer argument 0 (buffer #%d, input): expected fp32 %d x %d ld %d with a halo, found %s %u x %u ld %u halo %u",
                  op->b[0], g.H, g.W, g.C, rt_dtype_name(x->dtype), x->H, x->W, x->ld, x->halo);
  g.halo = (int)x->halo;
  if (rt_need(c, op, 0, RT_F32, (size_t)(g.H + 2 * g.halo) * (g.W + 2 * g.halo) * g.C)) return RT_E_PARAM;
  if (o->dtype != RT_F32 || o->H != (uint32_t)g.H || o->W != (uint32_t)g.W || o->ld != (uint32_t)g.ldo || o->halo != (uint32_t)g.opad)
    return rt_bad(c, op, "buffer argument 1 (buffer #%d, output): expected fp32 %d x %d ld %d halo %d", op->b[1], g.H, g.W, g.ldo, g.opad);
  if (rt_need(c, op, 1, RT_F32, (size_t)(g.H + 2 * g.opad) * (g.W + 2 * g.opad) * g.ldo)) return RT_E_PARAM;
  if (rt_need_t(c, op, 0, (size_t)36 * g.Np * g.C * 4, 1) || rt_need_t(c, op, 1, (size_t)g.Np * 4, 1)) return RT_E_PARAM;
  const rt_bufdesc *r = rt_bd(c, op, 2);
  if (r) {
    if (r->dtype != RT_F32 || r->H != (uint32_t)g.H || r->W != (uint32_t)g.W || r->ld < (uint32_t)g.Np)
      return rt_bad(c, op, "buffer argument 2 (buffer #%d, residual): expected fp32 %d x %d with ld >= %d", op->b[2], g.H, g.W, g.Np);
    if (rt_need(c, op, 2, RT_F32, (size_t)(g.H + 2 * r->halo) * (g.W + 2 * r->halo) * r->ld)) return RT_E_PARAM;
    g.rh = (int)r->halo; g.rld = (int)r->ld;
  }
  /* strip and channel block are the runtime's choice (the package's S, Nb are validated for compatibility but not
   * used): one 8-row GEMM tile of transform-domain rows per strip and 40 output channels per block keep each worker's
   * V strip, M block and weight block well inside its share of a 1 MB shared L2 at 4 threads (K060); on one thread
   * 64 channels per block measured best (K062), so the block is chosen per run from the thread count. Loop order
   * only: every output is computed as for any S, Nb. */
  g.S = WINO_STRIP;
  g.Nb = g.Np < WINO_NB1 ? g.Np : WINO_NB1;
  s->scrf = wino43_scratch_floats(&g); /* per-worker scratch for the larger block */
  s->g = g;
  s->bz = calloc((size_t)g.Np + 16, sizeof(float));
  return s->bz ? 0 : rt_oom(c, op, (size_t)g.Np * 4);
}
static void convw_task(void *a, int k, int w) {
  const op_job *j = a;
  convw_st *s = j->op->state;
  wino43_strip(&s->g, rt_b(j->c, j->op, 0), rt_t(j->c, j->op, 0, NULL), s->bz, rt_t(j->c, j->op, 1, NULL),
               rt_b(j->c, j->op, 1), k, s->scr + (size_t)w * s->scrf);
}
static int convw_run(rt_ctx *c, rt_op *op) {
  convw_st *s = op->state;
  int T = rt_threads(c);
  if (s->nw < T && !(s->scr = rt_worker_scratch(s->scr, &s->nw, T, s->scrf * 4 * T))) return RT_E_OOM;
  { int nb = T > 1 ? WINO_NB : WINO_NB1; s->g.Nb = s->g.Np < nb ? s->g.Np : nb; }
  s->g.res = rt_b(c, op, 2);
  if (s->g.opad) rt_zero_halo(rt_b(c, op, 1), s->g.Ho, s->g.Wo, s->g.ldo, s->g.opad, 4);
  int tiles = ((s->g.Ho + 3) / 4) * ((s->g.Wo + 3) / 4);
  op_job j = {c, op};
  rt_par(c, rt_nchunks(tiles, s->g.S), convw_task, &j);
  return 0;
}
static void convw_fini(rt_op *op) {
  convw_st *s = op->state;
  if (s) { free(s->scr); free(s->bz); }
  free(s);
}

const rt_kernel rt_k_f32x[RT_OP_N] = {
    [RT_OP_CONVW] = {convw_init, convw_run, convw_fini, "convw_f32", 0x2, 0x4},
    [RT_OP_FE_DBIMG] = {fe_db_init, fe_db_krun, fe_db_fini, "fe_dbimg", 0x2, 0},
    [RT_OP_CONVK] = {convk_init, convk_run, rt_free_state, "convk_f32", 0x2, 0x4},
    [RT_OP_GEM] = {gem_init, gem_run, NULL, "gem_f32", 0x2, 0},
    [RT_OP_SED] = {sed_init, sed_run, rt_free_state, "sed_f32", 0x7e, 0},
    [RT_OP_VADD] = {vadd_init, vadd_run, NULL, "vadd_f32", 0x4, 0},
};
