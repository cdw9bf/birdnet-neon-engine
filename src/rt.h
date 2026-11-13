// bnneon package runtime: .bnpk loader, buffer arena, op list executor and the kernel registry.
// Package format: see python/bnneon/compiler/pkgfmt.py (the writer) and docs/ADDING_A_MODEL.md.
#ifndef BNN_RT_H
#define BNN_RT_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RT_FORMAT_VERSION 1
#define RT_READER_VERSION 1
enum { RT_F32 = 0, RT_F16 = 1, RT_I8 = 2, RT_I32 = 3 };
enum { RT_HW_FP16 = 1, RT_HW_DOTPROD = 2 }; /* path requirements (HWCAP asimdhp, asimddp) */

typedef struct {
  char name[48];
  uint32_t dtype, ndim, dims[4];
  uint64_t offset, nbytes, reserved;
} rt_tensor; /* 96 bytes in the file */

typedef struct {
  uint32_t dtype, H, W, C, ld, halo;
  uint32_t slot;     /* memory slot: buffers with disjoint lifetimes share one (the compiler plans it) */
  uint32_t reserved;
  uint64_t elems;    /* allocated elements (incl. halo and channel padding); +64 zero bytes are added at allocation */
} rt_bufdesc;  /* 40 bytes in the file */

typedef struct {
  char name[24];
  uint32_t hwcap, n_buf, n_ops, ops_bytes, in_buf, out_buf, emb_buf, reserved;
  uint64_t buf_off, ops_off;
} rt_pathdesc; /* 72 bytes in the file */

/* 64-byte aligned allocations (free with free). aligned_alloc needs size % alignment == 0 (macOS enforces it), so
 * the size is rounded up to 64. Three meanings, chosen per call site:
 *   rt_al64(n)          n bytes, not initialized
 *   rt_zalloc64(n)      n bytes, zeroed
 *   rt_zalloc64_tail(n) n bytes plus at least 64 more, all zeroed: for buffers the K loops read ahead of */
static inline void *rt_al64(size_t n) { return aligned_alloc(64, (n + 63) & ~(size_t)63); }
static inline void *rt_zalloc64(size_t n) {
  void *p = rt_al64(n);
  if (p) memset(p, 0, n);
  return p;
}
static inline void *rt_zalloc64_tail(size_t n) {
  n = (n + 64 + 63) & ~(size_t)63;
  void *p = aligned_alloc(64, n);
  if (p) memset(p, 0, n);
  return p;
}

typedef struct rt_pkg rt_pkg;
typedef struct rt_ctx rt_ctx;
typedef struct rt_pool rt_pool;
/* intra-op thread pool (rt_pool.c): fn(arg, chunk, worker) for chunk 0..n-1, worker 0..size-1 (0 = the caller) */
typedef void (*rt_task_fn)(void *arg, int chunk, int worker);
rt_pool *rt_pool_new(int threads); /* NULL for threads < 2 */
void rt_pool_free(rt_pool *p);
int rt_pool_size(const rt_pool *p); /* 1 for NULL */
void rt_pool_run(rt_pool *p, int nchunks, rt_task_fn fn, void *arg); /* serial on the caller when p is NULL */

/* one decoded op: buffers, tensors, int params (pointers into the mapped file) */
typedef struct rt_op {
  uint16_t code, nb, nt, ni;
  const int32_t *b, *t, *p;
  void *state;     /* per-context op state (packed views, scratch), built by the op's init */
  const struct rt_kernel *k; /* the opcode's kernel, looked up once at load (NULL until then) */
} rt_op;

typedef struct rt_kernel {
  int (*init)(rt_ctx *c, rt_op *op);  /* 0 = ok; validates every buffer, tensor and parameter it uses and reports
                                         every failure through rt_bad / rt_oom / rt_need* (never a bare -1) */
  int (*run)(rt_ctx *c, rt_op *op);   /* 0 = ok, nonzero = window failed (caller falls back) */
  void (*fini)(rt_op *op);
  const char *name;
  uint32_t outmask;  /* buffer arguments the kernel writes (never the input window) */
  uint32_t optmask;  /* buffer arguments that may be -1 (absent) */
} rt_kernel;

struct rt_ctx {
  rt_pkg *pkg;
  const rt_pathdesc *path;
  const rt_bufdesc *bufd;
  void **buf;      /* buffer pointers (buf[in_buf] = the caller's window) */
  void *arena;
  rt_op *ops;
  int n_ops;
  int prof;
  double *t_op;
  int nprof;
  char *err;        /* the caller's error buffer while rt_ctx_new runs (kernel inits write their detail here) */
  size_t errlen;
  int err_kind;     /* 0 none, RT_E_PARAM or RT_E_OOM: set by rt_bad / rt_oom */
  char err_params[160]; /* the failing op's int parameters (appended after the hint when there is room) */
  const char *src;  /* per-op source names ("src_nodes/<path>" metadata, one line per op), NULL if absent */
  size_t src_len;
  int fail_op;      /* the op whose run reported a failed window (rt_run), -1 otherwise */
  const char *dump; /* BNN_DUMP directory (debugging), NULL otherwise */
  rt_pool *pool;    /* intra-op threads (rt_set_threads), NULL = this thread only */
  int no_madv;      /* BNN_NO_MADV was set when the context was made: rt_release_t keeps every page */
  size_t page;      /* the system page size (rt_release_t) */
};

rt_pkg *rt_pkg_open(const char *path, char *err, size_t errlen);
void rt_pkg_close(rt_pkg *p);
/* metadata: string / integer / float by key (returns -1 if absent) */
int rt_meta_str(const rt_pkg *p, const char *key, char *buf, size_t n);
int rt_meta_i64(const rt_pkg *p, const char *key, int64_t *v);
int rt_meta_f64(const rt_pkg *p, const char *key, double *v);
int rt_n_paths(const rt_pkg *p);
int rt_format_version(const rt_pkg *p);
const rt_pathdesc *rt_path(const rt_pkg *p, int i);
const rt_tensor *rt_tensor_at(const rt_pkg *p, int i, const void **data);
int rt_n_tensors(const rt_pkg *p);

/* kind (may be NULL): RT_E_PARAM for a package that violates a kernel's contract, RT_E_OOM for memory */
rt_ctx *rt_ctx_new(rt_pkg *p, int path_index, char *err, size_t errlen, int *kind);
int rt_run(rt_ctx *c, const float *pcm);
const float *rt_out(const rt_ctx *c);
const float *rt_emb(const rt_ctx *c);
void rt_ctx_free(rt_ctx *c);
/* threads for this context's ops (1 = none; outputs are bit-identical for every count): 0 ok, -1 out of memory */
int rt_set_threads(rt_ctx *c, int threads);
static inline int rt_threads(const rt_ctx *c) { return rt_pool_size(c->pool); }
static inline void rt_par(const rt_ctx *c, int nchunks, rt_task_fn fn, void *arg) { rt_pool_run(c->pool, nchunks, fn, arg); }

/* source name of op i (the ONNX node the compiler lowered it from), "" if the package carries none */
int rt_op_src(const rt_ctx *c, int i, char *buf, size_t n);
/* kernel name of op i ("pw_f16", ...), "?" outside the op list */
const char *rt_op_name(const rt_ctx *c, int i);
const char *rt_dtype_name(uint32_t dt);

/* helpers for kernels */
const void *rt_t(const rt_ctx *c, const rt_op *op, int i, size_t *nbytes);
const rt_bufdesc *rt_bd(const rt_ctx *c, const rt_op *op, int i);
/* for an init that repacked tensor argument i and never reads it again: drop its pages from the resident set */
void rt_release_t(const rt_ctx *c, const rt_op *op, int i);
/* error reporting from kernel inits. Every check records WHAT was expected and WHAT was found; the loader prefixes
 * the op index, kernel name and source node and appends a hint. The first message recorded wins. */
enum { RT_E_PARAM = -1, RT_E_OOM = -2 };
int rt_bad(const rt_ctx *c, const rt_op *op, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
int rt_oom(const rt_ctx *c, const rt_op *op, size_t bytes);
/* an op's per-worker scratch, sized for the context's T threads: when *nw < T the old block is freed and a new one
 * of `bytes` (64-byte aligned, not initialized) is returned with *nw = T; NULL (and *nw = 0) when out of memory.
 * Call it only when *nw < T. */
void *rt_worker_scratch(void *old, int *nw, int T, size_t bytes);
/* 0 if buffer argument i exists with dtype dt and >= elems elements (else records the mismatch, returns RT_E_PARAM) */
int rt_need(const rt_ctx *c, const rt_op *op, int i, int dt, size_t elems);
/* rt_need for a zero-haloed activation [H + 2h][W + 2h][ld]: also halo == h and channel stride == ld */
int rt_need_haloed(const rt_ctx *c, const rt_op *op, int i, int dt, int H, int W, int ld, int h);
/* rt_need for unpadded rows, `rows` x channel stride ld: also no halo and channel stride == ld */
int rt_need_rows(const rt_ctx *c, const rt_op *op, int i, int dt, size_t rows, int ld);
/* rt_need for an unpadded input whose channel stride may exceed its channels (elems = rows x stride): also no halo */
int rt_need_nohalo(const rt_ctx *c, const rt_op *op, int i, int dt, size_t elems);
/* 0 if tensor argument i exists with exactly nbytes (atleast = 0) or >= nbytes (atleast = 1) bytes */
int rt_need_t(const rt_ctx *c, const rt_op *op, int i, size_t nbytes, int atleast);
/* 0 if the op record has exactly nb buffer, nt tensor and ni int arguments (-1 = any) */
int rt_need_args(const rt_ctx *c, const rt_op *op, int nb, int nt, int ni);
/* a parameter / geometry condition: records the condition text and the op's int parameters when it fails */
#define RT_REQ(c, op, cond) \
  do { if (!(cond)) return rt_bad(c, op, "parameter check failed: %s", #cond); } while (0)
static inline void *rt_b(const rt_ctx *c, const rt_op *op, int i) { return op->b[i] < 0 ? NULL : c->buf[op->b[i]]; }

/* registry (rt_pkg.c): the kernel of an opcode (base | variant << 8), NULL if this library has none */
const rt_kernel *rt_kernel_for(int code);
/* zeroes the halo of a padded activation [H + 2h][W + 2h][C] of esz-byte elements; the interior is not touched */
void rt_zero_halo(void *xp, int H, int W, int C, int h, size_t esz);

/* opcodes (append only; a reader refuses unknown codes) */
enum {
  RT_OP_FE_STFT2 = 1,   /* v2.4 front end: pcm -> img f32 [96][511][2] */
  RT_OP_C0POOL = 2,     /* v2.4 c0 4x8 s2 + ReLU + max|avg pool2: img -> [48][128][48] */
  RT_OP_PW = 3,         /* 1x1 conv / GEMM with epilogue (act, residual, padded output) */
  RT_OP_DW3 = 4,        /* depthwise 3x3 (pad 1 halo, TF-same offsets) + SiLU + channel sums */
  RT_OP_SE = 5,         /* squeeze-excitation (mean from channel sums), in place */
  RT_OP_POSTPC = 6,     /* v2.4 post BN + ReLU + pc 3x3 valid + ReLU + mean (+ non-finite guard) */
  RT_OP_DENSE = 7,      /* dense head over a 1024 feature vector */
  RT_OP_FE_LOGMEL = 8,  /* framed log-mel front end (Perch) */
  RT_OP_STEM = 9,       /* dense kxk conv via im2col on a 1-channel fp32 image (Perch stem), fp32 math */
  RT_OP_DWK = 10,       /* depthwise kxk (k 3|5|7, stride 1|2, asymmetric pads) + act + channel sums */
  RT_OP_PAD = 11,       /* copy an activation into a zero-haloed buffer */
  RT_OP_MEAN = 12,      /* spatial mean of an activation -> f32 vector */
  RT_OP_PROTO = 13,     /* ProtoPNet head: L2-normalise cells, prototype GEMM, max over cells, class weights */
  RT_OP_PROTO_F32H = 14, /* the same in fp32 math with fp16-stored prototypes (half the weight memory) */
  RT_OP_GEMV = 15,      /* dense layer on a vector: any K, N, act none|relu (fp32 weights; variant 1: fp16 weights) */
  RT_OP_FE_DBIMG = 16,  /* dB-image front end: STFT power, mel, dB, top-db, row selection, min/max, resize, channels */
  RT_OP_CONVK = 17,     /* dense k x k conv (group 1, stride 1|2, asymmetric pads) on a zero-haloed input, no im2col */
  RT_OP_GEM = 18,       /* GeM pooling: (mean over pixels of max(x, eps)^p)^q -> vector */
  RT_OP_SED = 19,       /* SED attention head: axis mean, maxpool + depthwise conv1d, fc + ReLU, attention softmax */
  RT_OP_VADD = 20,      /* out = a + b over n floats */
  RT_OP_CONVW = 21,     /* dense 3x3 / stride 1 / pad 1 conv by Winograd F(4x4, 3x3), fp32 */
  RT_OP_PW_S8 = 22,     /* 1x1 conv / GEMM, int8 weights + per-row dynamic int8 activations, no dotprod (SMULL/SADALP) */
  RT_OP_GEMV_S8 = 23,   /* dense layer on a vector, int8 (as PW_S8) */
  RT_OP_SED_S8 = 24,    /* SED attention head with int8 fc / attention / class GEMMs */
  RT_OP_SE_GATE = 25,   /* squeeze-excitation gate vector only (mean, fc + swish, fc, sigmoid); the next 1x1 conv applies it */
  /* dotprod + fp16 family (rt_ops_dot.c, variant 2): the mixed int8 SDOT / fp16 plan of dense-conv models */
  RT_OP_STEM3 = 26,     /* 3x3 stride-2 conv on a 3-channel fp32 image (TF-same), fp32 math, SiLU -> zero-haloed fp16 */
  RT_OP_CONV3C24 = 27,  /* 3x3 stride-1 conv, 24 -> 24 channels, fp16 Winograd F(2,3), SiLU + residual (zero-haloed fp16) */
  RT_OP_UNPAD = 28,     /* zero-haloed fp16 activation -> unpadded fp32 rows */
  RT_OP_WINO_S8 = 29,   /* dense 3x3 / stride 1 / pad 1 conv, int8 Winograd F(2,3) (per-tile dynamic), SiLU -> fp16 rows */
  RT_OP_QCONV3 = 30,    /* dense 3x3 conv (TF-same, stride 1|2): per-tensor int8 input, im2col SDOT GEMM, SiLU -> fp16 rows */
  RT_OP_QPW = 31,       /* 1x1 conv / GEMM: per-row dynamic int8 input (fp32 or gated fp16 rows), SDOT, act, residual */
  RT_OP_SE_GATE16 = 32, /* squeeze-excitation gate vector in fp16 from the channel sums (fp16-stored weights) */
  RT_OP_GEM_A = 33,     /* GeM pooling with the vector exp2 / log2 power: rows -> vector */
  RT_OP_HEADS_S8 = 34,  /* dense head + two SED attention heads on one feature map, int8 SDOT GEMMs, summed logits */
  RT_OP_PROTO_S8 = 35,  /* ProtoPNet head, int8 prototypes + per-cell int8 activations, no dotprod (SMULL/SADALP) */
  RT_OP_N
};
enum { RT_ACT_NONE = 0, RT_ACT_SILU = 1, RT_ACT_RELU = 2 };
#endif
