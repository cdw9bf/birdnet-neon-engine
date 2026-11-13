// .bnp2 weight file loader (Perch v2; mmap, zero-copy fp32 views, bounds-checked)
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "perch_internal.h"

double pc_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static const float *pc_tensor(const pc_model *m, const char *name, long expect_numel) {
  for (int i = 0; i < m->n_tensors; i++) {
    const pc_tensor_entry *e = &m->dir[i];
    if (strncmp(e->name, name, 48) == 0) {
      if (e->dtype != 0) return NULL;
      if (expect_numel >= 0 && e->nbytes != (uint64_t)expect_numel * 4) return NULL;
      return (const float *)((const char *)m->map + e->offset);
    }
  }
  return NULL;
}

void pc_model_release(const pc_model *m, const void *p, size_t nbytes) {
#if defined(__linux__)
  long pg = sysconf(_SC_PAGESIZE);
  uintptr_t a = ((uintptr_t)p + pg - 1) & ~(uintptr_t)(pg - 1), e = ((uintptr_t)p + nbytes) & ~(uintptr_t)(pg - 1);
  if (e > a && a >= (uintptr_t)m->map && e <= (uintptr_t)m->map + m->map_size) madvise((void *)a, e - a, MADV_DONTNEED);
#else
  (void)m; (void)p; (void)nbytes;
#endif
}

int pc_model_load(pc_model *m, const char *path, char *err, size_t errlen) {
  memset(m, 0, sizeof *m);
  int fd = open(path, O_RDONLY);
  if (fd < 0) { snprintf(err, errlen, "cannot open %s", path); return -1; }
  struct stat sb;
  if (fstat(fd, &sb) != 0 || sb.st_size < 128) { close(fd); snprintf(err, errlen, "cannot stat %s or file too small", path); return -1; }
  m->map_size = sb.st_size;
  m->map = mmap(NULL, m->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (m->map == MAP_FAILED) { snprintf(err, errlen, "mmap failed"); m->map = NULL; return -1; }
  const char *p = m->map;
  if (memcmp(p, "BNP2ENG1", 8) != 0) { snprintf(err, errlen, "bad magic (not a .bnp2 file)"); goto fail; }
  uint32_t h[5];
  memcpy(h, p + 8, sizeof h);
  if (h[0] != 1) { snprintf(err, errlen, "unsupported .bnp2 version %u", h[0]); goto fail; }
  m->n_tensors = h[1]; m->n_classes = h[2]; m->n_blocks = h[3];
  if (h[1] == 0 || h[1] > 4096 || h[3] == 0 || h[3] > PC_MAX_BLOCKS || h[2] == 0 || h[2] > 100000 || h[4] > (1u << 20) ||
      h[4] % 8) {
    snprintf(err, errlen, "corrupt header"); goto fail;
  }
  size_t off = 8 + 20 + h[4];
  if (off + 32 + (size_t)m->n_blocks * 32 > m->map_size) { snprintf(err, errlen, "truncated header"); goto fail; }
  memcpy(m->labels_sha, p + off, 32);
  off += 32;
  int32_t bt[PC_MAX_BLOCKS * 8];
  memcpy(bt, p + off, (size_t)m->n_blocks * 32);
  off += (size_t)m->n_blocks * 32;
  if (off + (size_t)m->n_tensors * sizeof(pc_tensor_entry) > m->map_size) { snprintf(err, errlen, "truncated tensor directory"); goto fail; }
  pc_tensor_entry *dir = malloc(sizeof(pc_tensor_entry) * m->n_tensors);
  if (!dir) { snprintf(err, errlen, "out of memory"); goto fail; }
  memcpy(dir, p + off, sizeof(pc_tensor_entry) * m->n_tensors);
  m->dir = dir;
  for (int i = 0; i < m->n_tensors; i++) {
    const pc_tensor_entry *e = &m->dir[i];
    if (e->offset % 64 || e->offset > m->map_size || e->nbytes > m->map_size - e->offset) {
      snprintf(err, errlen, "tensor %d out of file bounds (truncated/corrupt file)", i); goto fail;
    }
  }
#define T(dst, nm, n)                                                            \
  do {                                                                           \
    dst = pc_tensor(m, nm, n);                                                   \
    if (!dst) { snprintf(err, errlen, "missing/bad tensor %s", nm); goto fail; } \
  } while (0)
  const float *fc;
  T(fc, "fe.consts", 8);
  /* the engine is built for this front end; anything else is a different model */
  if (fc[0] != PC_FRAME || fc[1] != PC_HOP || fc[2] != PC_PAD || fc[3] != PC_NFFT || !(fc[4] > 0) || !(fc[7] >= 1) ||
      fc[7] > 16) {
    snprintf(err, errlen, "unsupported front end / head constants"); goto fail;
  }
  m->log_floor = fc[4]; m->log_scale = fc[5]; m->norm_eps = fc[6]; m->ppc = (int)fc[7];
  m->n_proto = m->n_classes * m->ppc;
  T(m->win, "fe.window", PC_FRAME);
  T(m->mel, "fe.mel", PC_NMEL * PC_BINS);
  m->stem_c = 40;
  T(m->stem_w, "stem.w", 40 * 9); T(m->stem_b, "stem.b", 40);
  int H = (PC_NFR - 3) / 2 + 1, W = (PC_NMEL - 3) / 2 + 1, C = 40;
  char nm[64];
  for (int i = 0; i < m->n_blocks; i++) {
    pc_block *b = &m->blocks[i];
    const int32_t *q = bt + 8 * i;
    b->k = q[0]; b->stride = q[1]; b->cin = q[2]; b->has_ex = q[3] != 0; b->cexp = q[3] ? q[3] : q[2];
    b->cout = q[4]; b->se_rd = q[5]; b->skip = q[6];
    b->pt = q[7] & 255; b->pl = (q[7] >> 8) & 255; b->pb = (q[7] >> 16) & 255; b->pr = (q[7] >> 24) & 255;
    int hk = (b->k - 1) / 2;
    /* kernel limits: k 3|5, stride 1|2, pads <= (k-1)/2 (symmetric halo of (k-1)/2), channels % 8, SE rd <= 512 */
    if (b->cin != C || (b->k != 3 && b->k != 5) || (b->stride != 1 && b->stride != 2) || b->pt > hk || b->pl > hk ||
        b->pb > hk || b->pr > hk || b->cexp < 8 || b->cexp > 4096 || b->cexp % 8 || b->cout < 8 || b->cout > 4096 ||
        b->cout % 8 || b->se_rd < 1 || b->se_rd > 512 || (b->skip && (b->stride != 1 || b->cin != b->cout))) {
      snprintf(err, errlen, "block %d: unsupported shape", i); goto fail;
    }
    b->hin = H; b->win = W;
    b->hout = (H + b->pt + b->pb - b->k) / b->stride + 1;
    b->wout = (W + b->pl + b->pr - b->k) / b->stride + 1;
#define TB(fld, suf, n) do { snprintf(nm, sizeof nm, "m%d.%s", i, suf); T(b->fld, nm, n); } while (0)
    if (b->has_ex) { TB(ex_w, "ex.w", (long)b->cexp * b->cin); TB(ex_b, "ex.b", b->cexp); }
    TB(dw_w, "dw.w", (long)b->k * b->k * b->cexp); TB(dw_b, "dw.b", b->cexp);
    TB(se1_w, "se1.w", (long)b->se_rd * b->cexp); TB(se1_b, "se1.b", b->se_rd);
    TB(se2_w, "se2.w", (long)b->cexp * b->se_rd); TB(se2_b, "se2.b", b->cexp);
    TB(pj_w, "pj.w", (long)b->cout * b->cexp); TB(pj_b, "pj.b", b->cout);
#undef TB
    H = b->hout; W = b->wout; C = b->cout;
  }
  if (H * W != PC_CELLS || C != 384) { snprintf(err, errlen, "unexpected body output %dx%dx%d", H, W, C); goto fail; }
  T(m->top_w, "top.w", PC_EMB * 384); T(m->top_b, "top.b", PC_EMB);
  T(m->proto, "proto", (long)m->n_proto * PC_EMB);
  T(m->cls_w, "cls.w", (long)m->n_classes * m->ppc); T(m->cls_b, "cls.b", m->n_classes);
#undef T
  for (int j = 0; j < PC_NMEL; j++) {
    const float *f = m->mel + (size_t)j * PC_BINS;
    int l = 0, hh = 0;
    for (int k = 0; k < PC_BINS; k++)
      if (f[k] != 0.f) { if (hh == 0) l = k; hh = k + 1; }
    m->mel_lo[j] = l; m->mel_hi[j] = hh;
  }
  return 0;
fail:
  free((void *)m->dir);
  m->dir = NULL;
  munmap(m->map, m->map_size);
  m->map = NULL;
  return -1;
}

void pc_model_unload(pc_model *m) {
  free((void *)m->dir);
  m->dir = NULL;
  if (m->map) munmap(m->map, m->map_size);
  m->map = NULL;
}
