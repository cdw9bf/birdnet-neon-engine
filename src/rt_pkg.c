// .bnpk package loader (mmap, bounds-checked tables, metadata) and the executor (buffer arena, op list).
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <errno.h>
#include <stdarg.h>

#include "rt_ops.h"

struct rt_pkg {
  void *map;
  size_t size;
  uint32_t n_meta, n_tensors, n_paths;
  int version;
  const uint8_t *meta;      /* first record */
  size_t meta_bytes;
  rt_tensor *tensors;       /* aligned copy */
  rt_pathdesc *paths;       /* aligned copy */
};

#define FAIL(...) do { snprintf(err, errlen, __VA_ARGS__); goto fail; } while (0)

rt_pkg *rt_pkg_open(const char *path, char *err, size_t errlen) {
  rt_pkg *p = calloc(1, sizeof *p);
  if (!p) { snprintf(err, errlen, "out of memory"); return NULL; }
  int fd = open(path, O_RDONLY);
  if (fd < 0) { snprintf(err, errlen, "cannot open %s (%s)", path, strerror(errno)); free(p); return NULL; }
  struct stat sb;
  int no_stat = fstat(fd, &sb) != 0;
  if (no_stat || sb.st_size < 64) {
    close(fd);
    snprintf(err, errlen, "%s: %lld bytes, smaller than a package header (64); hint: incomplete download or the wrong file",
             path, no_stat ? -1LL : (long long)sb.st_size);
    free(p);
    return NULL;
  }
  p->size = (size_t)sb.st_size;
  p->map = mmap(NULL, p->size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p->map == MAP_FAILED) { p->map = NULL; snprintf(err, errlen, "%s: mmap of %zu bytes failed (%s); hint: memory or a filesystem that cannot map files", path, p->size, strerror(errno)); free(p); return NULL; }
  const uint8_t *m = p->map;
  if (memcmp(m, "BNNPKG01", 8))
    FAIL("%s: not a .bnpk package (magic '%.8s', expected BNNPKG01); hint: the file is not a compiled model package: "
         "compile the model with compiler/bnc.py, or check the download (sha256)", path, m);
  uint32_t bom;
  uint16_t ver[2];
  memcpy(&bom, m + 8, 4);
  memcpy(ver, m + 12, 4);
  if (bom != 0x01020304u) FAIL("%s: byte-order mark 0x%08x, expected 0x01020304 (little endian); hint: corrupt file", path, bom);
  p->version = ver[0];
  if (ver[1] > RT_READER_VERSION)
    FAIL("%s: the package needs reader version %u, this library reads format version %u; hint: update libbnneon, or "
         "recompile the model with the compiler that matches this library", path, ver[1], RT_READER_VERSION);
  uint32_t cnt[4];
  uint64_t off[4];
  memcpy(cnt, m + 16, 16);
  memcpy(off, m + 32, 32);
  p->n_meta = cnt[0]; p->n_tensors = cnt[1]; p->n_paths = cnt[2];
  if (off[3] != p->size)
    FAIL("%s: package truncated (size %zu, header says %llu); hint: incomplete download or copy; compare the sha256", path,
         p->size, (unsigned long long)off[3]);
  if (p->n_tensors > 65536 || p->n_paths < 1 || p->n_paths > 8 || p->n_meta > 4096)
    FAIL("%s: corrupt header (%u metadata records, %u tensors, %u paths; limits 4096 / 65536 / 1..8); hint: corrupt file or "
         "not a package", path, p->n_meta, p->n_tensors, p->n_paths);
  if (off[0] > p->size || off[1] > p->size || off[2] > p->size || off[1] < off[0] ||
      off[1] + (uint64_t)p->n_tensors * sizeof(rt_tensor) > p->size || off[2] + (uint64_t)p->n_paths * sizeof(rt_pathdesc) > p->size)
    FAIL("%s: table offsets outside the file (meta %llu, tensors %llu, paths %llu, size %zu); hint: corrupt file", path,
         (unsigned long long)off[0], (unsigned long long)off[1], (unsigned long long)off[2], p->size);
  p->meta = m + off[0];
  p->meta_bytes = off[1] - off[0];
  p->tensors = malloc(sizeof(rt_tensor) * (p->n_tensors ? p->n_tensors : 1));
  p->paths = malloc(sizeof(rt_pathdesc) * p->n_paths);
  if (!p->tensors || !p->paths) FAIL("out of memory");
  memcpy(p->tensors, m + off[1], sizeof(rt_tensor) * p->n_tensors);
  memcpy(p->paths, m + off[2], sizeof(rt_pathdesc) * p->n_paths);
  for (uint32_t i = 0; i < p->n_tensors; i++) {
    const rt_tensor *t = &p->tensors[i];
    /* every blob is followed by >= 64 bytes of the file (zero padding): the K loops read ahead */
    if (t->offset % 64 || t->offset > p->size || t->nbytes > p->size - t->offset || t->offset + t->nbytes + 64 > p->size)
      FAIL("%s: tensor %u (%.48s) lies outside the file (offset %llu, %llu bytes, file %zu); hint: truncated or corrupt file",
           path, i, t->name, (unsigned long long)t->offset, (unsigned long long)t->nbytes, p->size);
  }
  for (uint32_t i = 0; i < p->n_paths; i++) {
    const rt_pathdesc *d = &p->paths[i];
    if (d->n_buf == 0 || d->n_buf > 4096 || d->n_ops == 0 || d->n_ops > 4096 || d->buf_off % 8 || d->ops_off % 8 ||
        d->buf_off + (uint64_t)d->n_buf * sizeof(rt_bufdesc) > p->size || d->ops_off + d->ops_bytes > p->size ||
        d->in_buf >= d->n_buf || d->out_buf >= d->n_buf || (d->emb_buf != 0xffffffffu && d->emb_buf >= d->n_buf))
      FAIL("%s: path %u (%.24s) has corrupt tables (%u buffers, %u ops, in %u out %u emb %u); hint: corrupt file", path, i,
           d->name, d->n_buf, d->n_ops, d->in_buf, d->out_buf, d->emb_buf);
  }
  return p;
fail:
  rt_pkg_close(p);
  return NULL;
}

void rt_pkg_close(rt_pkg *p) {
  if (!p) return;
  if (p->map) munmap(p->map, p->size);
  free(p->tensors);
  free(p->paths);
  free(p);
}

/* metadata records: u16 key_len, u16 type (0 str, 1 i64, 2 f64, 3 bytes), u32 val_len, key, value, pad to 8 */
static const uint8_t *meta_find(const rt_pkg *p, const char *key, uint16_t *type, uint32_t *vlen) {
  const uint8_t *q = p->meta, *end = p->meta + p->meta_bytes;
  size_t kl = strlen(key);
  for (uint32_t i = 0; i < p->n_meta && q + 8 <= end; i++) {
    uint16_t h[2];
    uint32_t vl;
    memcpy(h, q, 4);
    memcpy(&vl, q + 4, 4);
    size_t rec = (8 + (size_t)h[0] + vl + 7) & ~(size_t)7;
    if ((size_t)(end - q) < rec) return NULL;
    if (h[0] == kl && !memcmp(q + 8, key, kl)) { *type = h[1]; *vlen = vl; return q + 8 + h[0]; }
    q += rec;
  }
  return NULL;
}

int rt_meta_str(const rt_pkg *p, const char *key, char *buf, size_t n) {
  uint16_t ty;
  uint32_t vl;
  const uint8_t *v = meta_find(p, key, &ty, &vl);
  if (!v) return -1;
  char tmp[64];
  if (ty == 1 || ty == 2) { /* numbers as decimal text */
    if (ty == 1) { int64_t x; memcpy(&x, v, 8); snprintf(tmp, sizeof tmp, "%lld", (long long)x); }
    else { double x; memcpy(&x, v, 8); snprintf(tmp, sizeof tmp, "%.17g", x); }
    v = (const uint8_t *)tmp; vl = (uint32_t)strlen(tmp);
  }
  if (buf && n) {
    size_t k = vl < n - 1 ? vl : n - 1;
    memcpy(buf, v, k);
    buf[k] = 0;
  }
  return (int)vl;
}
/* an 8-byte number of the given record type (1 i64, 2 f64) */
static int meta_num(const rt_pkg *p, const char *key, uint16_t type, void *x) {
  uint16_t ty;
  uint32_t vl;
  const uint8_t *v = meta_find(p, key, &ty, &vl);
  if (!v || ty != type || vl != 8) return -1;
  memcpy(x, v, 8);
  return 0;
}
int rt_meta_i64(const rt_pkg *p, const char *key, int64_t *x) { return meta_num(p, key, 1, x); }
int rt_meta_f64(const rt_pkg *p, const char *key, double *x) { return meta_num(p, key, 2, x); }
int rt_n_paths(const rt_pkg *p) { return (int)p->n_paths; }
int rt_format_version(const rt_pkg *p) { return p->version; }
const rt_pathdesc *rt_path(const rt_pkg *p, int i) { return i >= 0 && i < (int)p->n_paths ? &p->paths[i] : NULL; }
int rt_n_tensors(const rt_pkg *p) { return (int)p->n_tensors; }
const rt_tensor *rt_tensor_at(const rt_pkg *p, int i, const void **data) {
  if (i < 0 || i >= (int)p->n_tensors) return NULL;
  if (data) *data = (const uint8_t *)p->map + p->tensors[i].offset;
  return &p->tensors[i];
}

/* ------------------------------------------------------------------ executor */
static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

const void *rt_t(const rt_ctx *c, const rt_op *op, int i, size_t *nbytes) {
  if (i >= op->nt || op->t[i] < 0) return NULL;
  const void *d;
  const rt_tensor *t = rt_tensor_at(c->pkg, op->t[i], &d);
  if (!t) return NULL;
  if (nbytes) *nbytes = t->nbytes;
  return d;
}
/* An op that repacked tensor argument i at init no longer reads it: release its whole pages from the mapping. They
 * are clean and file-backed, so a later read (another context's init) faults them back in; no result can change.
 * This keeps the resident size at the packed weights instead of packed + file. BNN_NO_MADV keeps every page. */
void rt_release_t(const rt_ctx *c, const rt_op *op, int i) {
  size_t n = 0;
  const void *d = rt_t(c, op, i, &n);
  if (!d || c->no_madv) return;
  uintptr_t ps = (uintptr_t)c->page, lo = (uintptr_t)c->pkg->map, hi = lo + c->pkg->size;
  uintptr_t a = ((uintptr_t)d + ps - 1) & ~(ps - 1), b = ((uintptr_t)d + n) & ~(ps - 1);
  if (a >= lo && b <= hi && b > a) madvise((void *)a, b - a, MADV_DONTNEED);
}
void rt_zero_halo(void *xp, int H, int W, int C, int h, size_t esz) {
  char *x = xp;
  size_t rowl = (size_t)(W + 2 * h) * C * esz, side = (size_t)C * h * esz;
  memset(x, 0, rowl * h);
  memset(x + (size_t)(H + h) * rowl, 0, rowl * h);
  for (int y = h; y < H + h; y++) {
    memset(x + (size_t)y * rowl, 0, side);
    memset(x + (size_t)y * rowl + (size_t)(W + h) * C * esz, 0, side);
  }
}
void rt_free_state(rt_op *op) { free(op->state); }
void *rt_worker_scratch(void *old, int *nw, int T, size_t bytes) {
  free(old);
  void *p = rt_al64(bytes);
  *nw = p ? T : 0;
  return p;
}
const rt_bufdesc *rt_bd(const rt_ctx *c, const rt_op *op, int i) {
  return i >= op->nb || op->b[i] < 0 ? NULL : &c->bufd[op->b[i]];
}

const char *rt_dtype_name(uint32_t dt) {
  static const char *nm[] = {"f32", "f16", "i8", "i32"};
  return dt < 4 ? nm[dt] : "?";
}

/* ---- error reporting: the first message wins; the loader adds the op prefix and the hint */
static void err_set(const rt_ctx *c, int kind, const char *msg) {
  rt_ctx *w = (rt_ctx *)c; /* the error fields are the only ones written through a const context */
  if (!w->err || w->err_kind) return;
  w->err_kind = kind;
  snprintf(w->err, w->errlen, "%s", msg);
}
static void int_params(const rt_op *op, char *buf, size_t n) {
  size_t k = 0;
  buf[0] = 0;
  for (int i = 0; i < op->ni && k + 16 < n; i++) k += (size_t)snprintf(buf + k, n - k, "%s%d", i ? " " : "", op->p[i]);
  if (op->ni && k + 16 >= n) snprintf(buf + k, n - k, " ...");
}
int rt_bad(const rt_ctx *c, const rt_op *op, const char *fmt, ...) {
  char msg[320];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  if (c->err && !c->err_kind) int_params(op, ((rt_ctx *)c)->err_params, sizeof c->err_params);
  err_set(c, RT_E_PARAM, msg);
  return RT_E_PARAM;
}
int rt_oom(const rt_ctx *c, const rt_op *op, size_t bytes) {
  char msg[128];
  snprintf(msg, sizeof msg, "out of memory allocating %zu bytes of kernel scratch", bytes);
  err_set(c, RT_E_OOM, msg);
  return RT_E_OOM;
}
int rt_need(const rt_ctx *c, const rt_op *op, int i, int dt, size_t elems) {
  const rt_bufdesc *d = rt_bd(c, op, i);
  if (!d)
    return rt_bad(c, op, "buffer argument %d is absent (the op record has %d buffer arguments)", i, op->nb);
  if ((int)d->dtype == dt && d->elems >= elems) return 0;
  return rt_bad(c, op, "buffer argument %d (buffer #%d): expected dtype %s with >= %zu elements, found dtype %s "
                "[%ux%ux%u ld %u halo %u] with %llu elements", i, op->b[i], rt_dtype_name((uint32_t)dt), elems,
                rt_dtype_name(d->dtype), d->H, d->W, d->C, d->ld, d->halo, (unsigned long long)d->elems);
}
int rt_need_haloed(const rt_ctx *c, const rt_op *op, int i, int dt, int H, int W, int ld, int h) {
  /* summed in size_t, so that an absurd H or W in the op record does not overflow an int here (the product can
   * still wrap for sizes near 2^64 / ld; the callers' own parameter checks bound H and W) */
  if (rt_need(c, op, i, dt, ((size_t)H + 2 * (size_t)h) * ((size_t)W + 2 * (size_t)h) * (size_t)ld)) return RT_E_PARAM;
  const rt_bufdesc *d = rt_bd(c, op, i);
  if ((int)d->halo != h || (int)d->ld != ld)
    return rt_bad(c, op, "buffer argument %d (buffer #%d): expected halo %d and channel stride %d, found halo %u and stride %u",
                  i, op->b[i], h, ld, d->halo, d->ld);
  return 0;
}
int rt_need_rows(const rt_ctx *c, const rt_op *op, int i, int dt, size_t rows, int ld) {
  if (rt_need(c, op, i, dt, rows * (size_t)ld)) return RT_E_PARAM;
  const rt_bufdesc *d = rt_bd(c, op, i);
  if (d->halo || (int)d->ld != ld)
    return rt_bad(c, op, "buffer argument %d (buffer #%d): expected unpadded rows of channel stride %d, found halo %u and stride %u",
                  i, op->b[i], ld, d->halo, d->ld);
  return 0;
}
int rt_need_nohalo(const rt_ctx *c, const rt_op *op, int i, int dt, size_t elems) {
  if (rt_need(c, op, i, dt, elems)) return RT_E_PARAM;
  const rt_bufdesc *d = rt_bd(c, op, i);
  if (d->halo) return rt_bad(c, op, "buffer argument %d (buffer #%d, input): expected no halo, found halo %u", i, op->b[i], d->halo);
  return 0;
}
int rt_need_t(const rt_ctx *c, const rt_op *op, int i, size_t nbytes, int atleast) {
  size_t nb = 0;
  if (!rt_t(c, op, i, &nb))
    return rt_bad(c, op, "tensor argument %d is absent (the op record has %d tensor arguments)", i, op->nt);
  if (atleast ? nb >= nbytes : nb == nbytes) return 0;
  const rt_tensor *t = rt_tensor_at(c->pkg, op->t[i], NULL);
  return rt_bad(c, op, "tensor argument %d (tensor #%d '%.48s' %s [%u %u %u %u]): expected %s%zu bytes, found %llu bytes",
                i, op->t[i], t->name, rt_dtype_name(t->dtype), t->dims[0], t->dims[1], t->dims[2], t->dims[3],
                atleast ? ">= " : "", nbytes, (unsigned long long)t->nbytes);
}
int rt_need_args(const rt_ctx *c, const rt_op *op, int nb, int nt, int ni) {
  if ((nb < 0 || op->nb == nb) && (nt < 0 || op->nt == nt) && (ni < 0 || op->ni == ni)) return 0;
  return rt_bad(c, op, "op record shape: expected %d buffer, %d tensor and %d int arguments, found %d, %d and %d",
                nb, nt, ni, op->nb, op->nt, op->ni);
}

int rt_op_src(const rt_ctx *c, int i, char *buf, size_t n) {
  if (n) buf[0] = 0;
  if (!c->src) return -1;
  const char *p = c->src, *end = c->src + c->src_len;
  for (int k = 0; k < i && p < end; k++) {
    const char *nl = memchr(p, '\n', (size_t)(end - p));
    if (!nl) return -1;
    p = nl + 1;
  }
  if (p >= end) return -1;
  const char *nl = memchr(p, '\n', (size_t)(end - p));
  size_t l = (size_t)((nl ? nl : end) - p);
  if (n) { if (l > n - 1) l = n - 1; memcpy(buf, p, l); buf[l] = 0; }
  return 0;
}

const char *rt_op_name(const rt_ctx *c, int i) { return i >= 0 && i < c->n_ops && c->ops[i].k ? c->ops[i].k->name : "?"; }

static size_t dsize(uint32_t dt) { return dt == RT_F16 ? 2 : dt == RT_I8 ? 1 : 4; }
/* The registry: every kernel table with the variant it serves (opcode = base | variant << 8). The first table of
 * the variant that has a run for the base opcode is the kernel's; the tables of a variant do not overlap. Only the
 * tables are read here: no code of the A76 objects runs for a lookup. */
static const struct { int var; const rt_kernel *tab; } TABLES[] = {
    {0, rt_k_f32}, {0, rt_k_f32x}, {0, rt_k_s8n}, /* fp32 arithmetic (and int8 GEMMs without dotprod): any AArch64 */
    {1, rt_k_f16},                                /* fp16 arithmetic */
    {2, rt_k_f16_s8}, {2, rt_k_dot},              /* dotprod + fp16 */
};
const rt_kernel *rt_kernel_for(int code) {
  int base = code & 0xff, var = code >> 8;
  if (base >= RT_OP_N) return NULL;
  for (size_t i = 0; i < sizeof TABLES / sizeof TABLES[0]; i++)
    if (TABLES[i].var == var && TABLES[i].tab[base].run) return &TABLES[i].tab[base];
  return NULL;
}

/* "op 12 pw_f16 (ONNX node 'model/block_2/conv/Conv' +4)"; long TF names are cut to keep the hint inside the host's
 * 512-byte buffer */
static const char *op_prefix(const rt_ctx *c, int i, const char *kname, char *buf, size_t n) {
  char src[160];
  if (rt_op_src(c, i, src, sizeof src) == 0 && src[0]) {
    if (strlen(src) > 96) strcpy(src + 93, "...'");
    snprintf(buf, n, "op %d %s (ONNX node %s)", i, kname ? kname : "?", src);
  } else
    snprintf(buf, n, "op %d %s (no source node names in the package: compile with bnc 2 or newer)", i, kname ? kname : "?");
  return buf;
}
#define HINT_PKG "hint: package/library mismatch or a lowering bug: recompile with this checkout's compiler/bnc.py; if it " \
                 "reproduces, report this message"
#define KIND(k) do { if (kind) *kind = (k); } while (0)
#define PFX(kname) op_prefix(c, (int)i, kname, pfx, sizeof pfx) /* op i's prefix, written to pfx */

rt_ctx *rt_ctx_new(rt_pkg *p, int pi, char *err, size_t errlen, int *kind) {
  KIND(RT_E_PARAM);
  const rt_pathdesc *d = rt_path(p, pi);
  if (!d) { snprintf(err, errlen, "no path %d in the package (%d paths)", pi, rt_n_paths(p)); return NULL; }
  rt_ctx *c = calloc(1, sizeof *c);
  if (!c) { KIND(RT_E_OOM); snprintf(err, errlen, "out of memory (%zu bytes for the context)", sizeof *c); return NULL; }
  c->pkg = p;
  c->path = d;
  c->fail_op = -1;
  c->err = err;
  c->errlen = errlen;
  c->no_madv = getenv("BNN_NO_MADV") != NULL;
  c->page = (size_t)sysconf(_SC_PAGESIZE);
  { /* per-op source node names: optional metadata "src_nodes/<path>" (one line per op) */
    char key[64];
    uint16_t ty;
    uint32_t vl;
    snprintf(key, sizeof key, "src_nodes/%.24s", d->name);
    const uint8_t *v = meta_find(p, key, &ty, &vl);
    if (v && (ty == 0 || ty == 3)) { c->src = (const char *)v; c->src_len = vl; }
  }
  const uint8_t *m = p->map;
  char pfx[256];
  size_t *slot_sz = NULL, total = 0; /* slot sizes and pointers: freed on every way out */
  void **sp = NULL;
  rt_bufdesc *bd = malloc(sizeof(rt_bufdesc) * d->n_buf);
  c->bufd = bd;
  c->buf = calloc(d->n_buf, sizeof(void *));
  if (!bd || !c->buf) { KIND(RT_E_OOM); FAIL("out of memory (%zu bytes for %u buffer descriptors)", sizeof(rt_bufdesc) * d->n_buf, d->n_buf); }
  memcpy(bd, m + d->buf_off, sizeof(rt_bufdesc) * d->n_buf);
  /* slots: one zeroed arena; every slot 64-byte aligned, sized for its largest buffer + >= 64 zero bytes (K-loop
   * read-ahead). The input window's slot is the caller's memory. */
  slot_sz = calloc(d->n_buf, sizeof(size_t));
  if (!slot_sz) { KIND(RT_E_OOM); FAIL("out of memory (slot table)"); }
  for (uint32_t i = 0; i < d->n_buf; i++) {
    if (bd[i].dtype > RT_I32 || bd[i].elems > ((size_t)1 << 30) || bd[i].slot >= d->n_buf)
      FAIL("path %.24s buffer #%u: bad descriptor (dtype %u, %llu elements, slot %u of %u); " HINT_PKG, d->name, i,
           bd[i].dtype, (unsigned long long)bd[i].elems, bd[i].slot, d->n_buf);
    if ((i == d->in_buf) != (bd[i].slot == bd[d->in_buf].slot))
      FAIL("path %.24s buffer #%u shares the input window's memory slot %u; " HINT_PKG, d->name, i, bd[i].slot);
    size_t b = ((size_t)bd[i].elems * dsize(bd[i].dtype) + 64 + 63) & ~(size_t)63;
    if (b > slot_sz[bd[i].slot]) slot_sz[bd[i].slot] = b;
  }
  for (uint32_t s = 0; s < d->n_buf; s++) {
    if (s == bd[d->in_buf].slot) continue;
    total += slot_sz[s];
    if (total > ((size_t)1 << 30))
      FAIL("path %.24s needs more than 1 GB of activation buffers; hint: the package is corrupt or the model is far "
           "larger than this runtime is built for", d->name);
  }
  c->arena = aligned_alloc(64, total ? total : 64);
  if (!c->arena) {
    KIND(RT_E_OOM);
    FAIL("out of memory: %zu bytes (%.1f MB) for the activation buffers of path %.24s; hint: free memory or use a "
         "smaller model", total, total / 1e6, d->name);
  }
  memset(c->arena, 0, total ? total : 64);
  sp = calloc(d->n_buf, sizeof(void *));
  if (!sp) { KIND(RT_E_OOM); FAIL("out of memory (slot pointers)"); }
  size_t o = 0;
  for (uint32_t s = 0; s < d->n_buf; s++) {
    if (s == bd[d->in_buf].slot || !slot_sz[s]) continue;
    sp[s] = (uint8_t *)c->arena + o;
    o += slot_sz[s];
  }
  for (uint32_t i = 0; i < d->n_buf; i++) {
    if (i != d->in_buf) c->buf[i] = sp[bd[i].slot];
    /* the outputs are read after the op list: their slots are theirs alone */
    if (i != d->out_buf && (bd[i].slot == bd[d->out_buf].slot ||
                            (d->emb_buf != 0xffffffffu && i != d->emb_buf && bd[i].slot == bd[d->emb_buf].slot)))
      FAIL("path %.24s buffer #%u shares memory slot %u with the output buffer (#%u / #%u); " HINT_PKG, d->name, i,
           bd[i].slot, d->out_buf, d->emb_buf);
  }
  free(sp);
  free(slot_sz);
  sp = NULL;
  slot_sz = NULL;
  c->ops = calloc(d->n_ops, sizeof(rt_op));
  if (!c->ops) { KIND(RT_E_OOM); FAIL("out of memory (%u op records)", d->n_ops); }
  const uint8_t *q = m + d->ops_off, *end = q + d->ops_bytes;
  for (uint32_t i = 0; i < d->n_ops; i++) {
    uint16_t h[6];
    if (q + 12 > end) FAIL("op %u: truncated op list (%u ops declared, %u bytes); " HINT_PKG, i, d->n_ops, d->ops_bytes);
    memcpy(h, q, 12);
    size_t need = 12 + 4 * ((size_t)h[2] + h[3] + h[4]);
    if (h[1] < need || q + h[1] > end || ((uintptr_t)(q - m) % 4))
      FAIL("op %u: bad op record (%u bytes for %u buffer, %u tensor, %u int arguments); " HINT_PKG, i, h[1], h[2], h[3], h[4]);
    rt_op *op = &c->ops[i];
    op->code = h[0]; op->nb = h[2]; op->nt = h[3]; op->ni = h[4];
    op->b = (const int32_t *)(q + 12);
    op->t = op->b + op->nb;
    op->p = op->t + op->nt;
    for (int k = 0; k < op->nb; k++)
      if (op->b[k] >= (int32_t)d->n_buf || op->b[k] < -1)
        FAIL("%s: buffer argument %d names buffer #%d but the path has %u buffers; " HINT_PKG,
             PFX(NULL), k, op->b[k], d->n_buf);
    for (int k = 0; k < op->nt; k++)
      if (op->t[k] >= (int32_t)p->n_tensors)
        FAIL("%s: tensor argument %d names tensor #%d but the package has %u tensors; " HINT_PKG,
             PFX(NULL), k, op->t[k], p->n_tensors);
    int var = op->code >> 8;
    if (var && !(d->hwcap & RT_HW_FP16))
      FAIL("%s: fp16 kernel (opcode 0x%x) in path %.24s, which does not declare the fp16 requirement; " HINT_PKG,
           PFX(NULL), op->code, d->name);
    if (var == 2 && !(d->hwcap & RT_HW_DOTPROD))
      FAIL("%s: dotprod kernel (opcode 0x%x) in path %.24s, which does not declare the dotprod requirement; " HINT_PKG,
           PFX(NULL), op->code, d->name);
    const rt_kernel *k = op->k = rt_kernel_for(op->code);
    if (!k) {
      char comp[48];
      if (rt_meta_str(p, "compiler", comp, sizeof comp) < 0) snprintf(comp, sizeof comp, "unknown");
      FAIL("%s: unknown opcode 0x%x (base %d variant %d); this library knows opcodes 1..%d (reader version %d), the "
           "package was written by compiler '%s'; hint: update libbnneon, or recompile the model with the compiler that "
           "matches this library", PFX(NULL), op->code, op->code & 0xff, var,
           RT_OP_N - 1, RT_READER_VERSION, comp);
    }
    PFX(k->name);
    for (int a = 0; a < op->nb; a++) {
      if (op->b[a] < 0 && !(a < 32 && (k->optmask >> a & 1)))
        FAIL("%s: buffer argument %d is required by the kernel but absent (-1) in the op record; " HINT_PKG, pfx, a);
      if (op->b[a] == (int32_t)d->in_buf && a < 32 && (k->outmask >> a & 1))
        FAIL("%s: buffer argument %d is the input window (#%u) but the kernel writes it; " HINT_PKG, pfx, a, d->in_buf);
    }
    for (int a = 0; a < op->nb; a++) /* a written buffer may not share memory with another argument of the op */
      for (int b2 = 0; b2 < op->nb; b2++)
        if (a != b2 && op->b[a] >= 0 && op->b[b2] >= 0 && a < 32 && (k->outmask >> a & 1) &&
            (op->b[a] == op->b[b2] || bd[op->b[a]].slot == bd[op->b[b2]].slot))
          FAIL("%s: buffer arguments %d (#%d, written) and %d (#%d) share memory slot %u; " HINT_PKG, pfx, a, op->b[a],
               b2, op->b[b2], bd[op->b[a]].slot);
    c->n_ops = i + 1; /* fini runs for this op even if its init fails */
    c->err_kind = 0;
    c->err_params[0] = 0;
    if (k->init && k->init(c, op)) {
      char detail[400];
      snprintf(detail, sizeof detail, "%s", c->err_kind ? err : "init failed without a message (kernel bug)");
      if (c->err_kind == RT_E_OOM) {
        KIND(RT_E_OOM);
        snprintf(err, errlen, "%s: %s; hint: this path already holds %.1f MB of activation buffers; free memory or use "
                 "a smaller model", pfx, detail, total / 1e6);
      } else {
        int n = snprintf(err, errlen, "%s: %s; " HINT_PKG, pfx, detail);
        if (n > 0 && (size_t)n + 24 < errlen && c->err_params[0]) /* the op record, when it still fits */
          snprintf(err + n, errlen - (size_t)n, " (int params: [%s])", c->err_params);
      }
      goto fail;
    }
    q += h[1];
  }
  c->err = NULL;
  const char *pe = getenv("BNN_PROF");
  c->prof = pe ? atoi(pe) : 0;
  c->dump = getenv("BNN_DUMP");
  if (c->prof) c->t_op = calloc(d->n_ops, sizeof(double));
  return c;
fail:
  free(sp);
  free(slot_sz);
  c->err = NULL;
  rt_ctx_free(c);
  return NULL;
}
#undef PFX

/* BNN_DUMP=<dir> (debugging): every buffer an op writes, after the op, as raw <dir>/op<i>_b<arg>.bin (first window) */
static void dump_op(rt_ctx *c, int i, const rt_kernel *k) {
  if (c->nprof) return;
  const rt_op *op = &c->ops[i];
  for (int a = 0; a < op->nb && a < 32; a++) {
    if (!(k->outmask >> a & 1) || op->b[a] < 0) continue;
    const rt_bufdesc *d = &c->bufd[op->b[a]];
    char fn[512];
    snprintf(fn, sizeof fn, "%s/op%03d_b%d.bin", c->dump, i, a);
    FILE *f = fopen(fn, "wb");
    if (!f) continue;
    fwrite(c->buf[op->b[a]], dsize(d->dtype), (size_t)d->elems, f);
    fclose(f);
  }
}

int rt_run(rt_ctx *c, const float *pcm) {
  c->buf[c->path->in_buf] = (void *)pcm;
  c->fail_op = -1;
  double t0 = c->prof ? now_s() : 0;
  for (int i = 0; i < c->n_ops; i++) {
    rt_op *op = &c->ops[i];
    const rt_kernel *k = op->k;
    int r = k->run(c, op);
    if (c->prof) { double t = now_s(); c->t_op[i] += t - t0; t0 = t; }
    if (c->dump) dump_op(c, i, k);
    if (r) { c->fail_op = i; return r; }
  }
  c->nprof++;
  return 0;
}

const float *rt_out(const rt_ctx *c) { return c->buf[c->path->out_buf]; }
const float *rt_emb(const rt_ctx *c) { return c->path->emb_buf == 0xffffffffu ? NULL : c->buf[c->path->emb_buf]; }

int rt_set_threads(rt_ctx *c, int threads) {
  if (threads < 1) threads = 1;
  if (threads > 64) threads = 64;
  if (rt_pool_size(c->pool) == threads) return 0;
  rt_pool_free(c->pool);
  c->pool = NULL;
  if (threads == 1) return 0;
  c->pool = rt_pool_new(threads);
  return c->pool ? 0 : -1;
}

/* BNN_PROF=3: multiply-accumulates of an op, estimated as (output pixels) x (elements of its largest weight tensor);
 * ops that apply their weights once per window (vector ops) count one pixel, the prototype head its input cells.
 * Winograd convolutions are counted as the direct convolution they replace (CONVW's stored weights are already
 * transformed, 4x the direct count), so their GMAC/s is an equivalent rate, not their multiplier rate. */
static double prof_macs(const rt_ctx *c, const rt_op *op, const rt_kernel *k) {
  double w = 0, px = 1;
  for (int i = 0; i < op->nt; i++) {
    if (op->t[i] < 0) continue;
    const rt_tensor *t = rt_tensor_at(c->pkg, op->t[i], NULL);
    double e = t ? (double)t->nbytes / (double)dsize(t->dtype) : 0;
    if (e > w) w = e;
  }
  int code = op->code & 0xff, a = -1;
  if (code == RT_OP_HEADS_S8 && op->ni >= 5) { /* p = [H W C hid nc]: the dense head + per time step fc, attention, class */
    const double H = op->p[0], W = op->p[1], C = op->p[2], hid = op->p[3], nc = op->p[4];
    return nc * C + (H + W) * (hid * C + 2 * nc * hid);
  }
  if (code == RT_OP_PROTO || code == RT_OP_PROTO_F32H || code == RT_OP_PROTO_S8) a = 0; /* cells of the input */
  else if (code == RT_OP_PW || code == RT_OP_DW3 || code == RT_OP_DWK || code == RT_OP_STEM || code == RT_OP_CONVK ||
           code == RT_OP_CONVW || code == RT_OP_PW_S8 || code == RT_OP_C0POOL || code == RT_OP_POSTPC ||
           code == RT_OP_STEM3 || code == RT_OP_CONV3C24 || code == RT_OP_WINO_S8 || code == RT_OP_QCONV3 ||
           code == RT_OP_QPW)
    for (int i = 0; i < op->nb && i < 32; i++)
      if ((k->outmask >> i & 1) && op->b[i] >= 0) { a = i; break; }
  if (a >= 0 && a < op->nb && op->b[a] >= 0) px = (double)c->bufd[op->b[a]].H * c->bufd[op->b[a]].W;
  return w * (px > 0 ? px : 1);
}

void rt_ctx_free(rt_ctx *c) {
  if (!c) return;
  rt_pool_free(c->pool);
  c->pool = NULL;
  if (c->prof && c->nprof && c->t_op) {
    double tot = 0;
    fprintf(stderr, "profile over %d windows (ms/window), path %s%s:\n", c->nprof, c->path->name,
            c->prof > 2 ? "; MACs are estimates of the direct convolution (Winograd ops do fewer multiplies)" : "");
    for (int i = 0; i < c->n_ops; i++) {
      const rt_op *op = &c->ops[i];
      const rt_kernel *k = op->k;
      tot += c->t_op[i];
      if (c->prof > 1) {
        char src[120];
        rt_op_src(c, i, src, sizeof src);
        if (c->prof > 2) { /* estimated multiply-accumulates and the rate reached (the profile's audit columns) */
          double mm = prof_macs(c, op, k) * 1e-6, ms = 1e3 * c->t_op[i] / c->nprof;
          fprintf(stderr, "  %3d %-14s %8.3f  %9.2f MMAC %6.1f GMAC/s  %s\n", i, k->name, ms, mm, ms > 0 ? mm / ms : 0, src);
        } else
          fprintf(stderr, "  %3d %-14s %8.3f  %s\n", i, k->name, 1e3 * c->t_op[i] / c->nprof, src);
      }
    }
    /* per kernel totals */
    for (int code = 0; code < 0x300; code++) {
      double s = 0, mm = 0;
      int n = 0;
      const char *nm = NULL;
      for (int i = 0; i < c->n_ops; i++)
        if (c->ops[i].code == code) {
          const rt_kernel *k = c->ops[i].k;
          s += c->t_op[i]; n++;
          nm = k->name;
          if (c->prof > 2) mm += prof_macs(c, &c->ops[i], k) * 1e-6;
        }
      if (n && c->prof > 2)
        fprintf(stderr, "  %-14s x%-3d %8.3f  %9.2f MMAC %6.1f GMAC/s\n", nm, n, 1e3 * s / c->nprof, mm, s > 0 ? mm * c->nprof / (1e3 * s) : 0);
      else if (n) fprintf(stderr, "  %-14s x%-3d %8.3f\n", nm, n, 1e3 * s / c->nprof);
    }
    fprintf(stderr, "  %-14s      %8.3f\n", "total", 1e3 * tot / c->nprof);
  }
  for (int i = 0; i < c->n_ops; i++) {
    rt_op *op = &c->ops[i];
    if (op->k && op->k->fini) op->k->fini(op);
  }
  free(c->ops);
  free(c->t_op);
  free(c->arena);
  free(c->buf);
  free((void *)c->bufd);
  free(c);
}
