// .bn24 weight file loader (mmap, zero-copy fp32 views, bounds-checked)
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "v24_internal.h"

double b24_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

const float *b24_tensor(const b24_model *m, const char *name, int expect_numel) {
  for (int i = 0; i < m->n_tensors; i++) {
    const b24_tensor_entry *e = &m->dir[i];
    if (strncmp(e->name, name, 48) == 0) {
      if (e->dtype != 0) return NULL;
      if (expect_numel >= 0 && e->nbytes != (uint64_t)expect_numel * 4) return NULL;
      return (const float *)((const char *)m->map + e->offset);
    }
  }
  return NULL;
}

static void nz_range(const float *mel, int bins, int *lo, int *hi) {
  for (int j = 0; j < NMEL; j++) {
    const float *f = mel + (size_t)j * bins;
    int l = 0, h = 0;
    for (int k = 0; k < bins; k++)
      if (f[k] != 0.f) { if (h == 0) l = k; h = k + 1; }
    lo[j] = l; hi[j] = h;
  }
}

int b24_model_load(b24_model *m, const char *path, char *err, size_t errlen) {
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
  if (memcmp(p, "BN24ENG1", 8) != 0) { snprintf(err, errlen, "bad magic (not a .bn24 file)"); goto fail; }
  uint32_t h[5];
  memcpy(h, p + 8, sizeof h);
  if (h[0] != 1) { snprintf(err, errlen, "unsupported .bn24 version %u", h[0]); goto fail; }
  m->n_tensors = h[1]; m->n_classes = h[2]; m->n_blocks = h[3];
  if (h[1] == 0 || h[1] > 4096 || h[3] == 0 || h[3] > MAX_BLOCKS || h[2] == 0 || h[2] > 100000 || h[4] > (1u << 20) ||
      h[4] % 8) {
    snprintf(err, errlen, "corrupt header"); goto fail;
  }
  size_t off = 8 + 20 + h[4];
  if (off + 32 + (size_t)m->n_blocks * 32 > m->map_size) { snprintf(err, errlen, "truncated header"); goto fail; }
  memcpy(m->labels_sha, p + off, 32);
  off += 32;
  const int32_t *bt = (const int32_t *)(p + off);
  off += (size_t)m->n_blocks * 32;
  if (off + (size_t)m->n_tensors * sizeof(b24_tensor_entry) > m->map_size) { snprintf(err, errlen, "truncated tensor directory"); goto fail; }
  /* the directory is only 4-byte aligned in the file: copy it (it holds uint64 fields) */
  b24_tensor_entry *dir = malloc(sizeof(b24_tensor_entry) * m->n_tensors);
  if (!dir) { snprintf(err, errlen, "out of memory"); goto fail; }
  memcpy(dir, p + off, sizeof(b24_tensor_entry) * m->n_tensors);
  m->dir = dir;
  for (int i = 0; i < m->n_tensors; i++) {
    const b24_tensor_entry *e = &m->dir[i];
    if (e->offset % 64 || e->offset > m->map_size || e->nbytes > m->map_size - e->offset) {
      snprintf(err, errlen, "tensor %d out of file bounds (truncated/corrupt file)", i); goto fail;
    }
  }
#define T(dst, nm, n)                                                            \
  do {                                                                           \
    dst = b24_tensor(m, nm, n);                                                   \
    if (!dst) { snprintf(err, errlen, "missing/bad tensor %s", nm); goto fail; } \
  } while (0)
  T(m->win1, "fe1.window", N1); T(m->dft1, "fe1.dft", N1 * B1); T(m->mel1, "fe1.mel", NMEL * B1); T(m->pow1, "fe1.pow", 1);
  T(m->win2, "fe2.window", N2); T(m->dft2, "fe2.dft", N2 * B2); T(m->mel2, "fe2.mel", NMEL * B2); T(m->pow2, "fe2.pow", 1);
  T(m->fe_bn_s, "fe.bn_s", 2); T(m->fe_bn_b, "fe.bn_b", 2);
  T(m->c0_w, "c0.w", 24 * 4 * 8 * 2); T(m->c0_b, "c0.b", 24);
  T(m->p0_w, "p0.w", 24 * 48); T(m->p0_b, "p0.b", 24);
  int H = 48, W = 128, C = 24;
  int stage = 1, idx = 0;
  char nm[64];
  for (int i = 0; i < m->n_blocks; i++) {
    b24_block *b = &m->blocks[i];
    const int32_t *q = bt + 8 * i;
    b->stride = q[0]; b->cin = q[1]; b->cmid = q[2]; b->cout = q[3]; b->se_rd = q[4]; b->skip = q[5];
    /* limits of the kernels: dw3x3_f16 / SE need cmid % 8 == 0 and cmid <= 2048; SE rd <= 512 */
    if (b->cin != C || (b->stride != 1 && b->stride != 2) || b->cmid < 8 || b->cmid > 2048 || b->cmid % 8 ||
        b->cout < 1 || b->cout > 2048 || b->se_rd < 0 || b->se_rd > 512 ||
        (b->skip && (b->stride != 1 || b->cin != b->cout))) {
      snprintf(err, errlen, "block %d: inconsistent shape", i); goto fail;
    }
    if (i > 0 && b->stride == 2) { stage++; idx = 0; }
    idx++;
    snprintf(b->tag, sizeof b->tag, "b%d.%d", stage, idx);
    b->hin = H; b->win = W;
    b->hout = (H + 2 - 3) / b->stride + 1;
    b->wout = (W + 2 - 3) / b->stride + 1;
#define TB(fld, suf, n) do { snprintf(nm, sizeof nm, "%s.%s", b->tag, suf); T(b->fld, nm, n); } while (0)
    TB(pw_w, "pw.w", b->cmid * b->cin); TB(pw_b, "pw.b", b->cmid);
    TB(dw_w, "dw.w", 9 * b->cmid); TB(dw_b, "dw.b", b->cmid);
    if (b->se_rd) {
      TB(se1_w, "se1.w", b->se_rd * b->cmid); TB(se1_b, "se1.b", b->se_rd);
      TB(se2_w, "se2.w", b->cmid * b->se_rd); TB(se2_b, "se2.b", b->cmid);
    }
    TB(pwl_w, "pwl.w", b->cout * b->cmid); TB(pwl_b, "pwl.b", b->cout);
#undef TB
    H = b->hout; W = b->wout; C = b->cout;
  }
  if (H != 3 || W != 8 || C != 192) { snprintf(err, errlen, "unexpected body output %dx%dx%d", H, W, C); goto fail; }
  T(m->post_s, "post.bn_s", 192); T(m->post_b, "post.bn_b", 192);
  T(m->pc_w, "pc.w", FEAT_C * 9 * 192); T(m->pc_b, "pc.b", FEAT_C);
  T(m->head_w, "head.w", m->n_classes * FEAT_C); T(m->head_b, "head.b", m->n_classes);
#undef T
  nz_range(m->mel1, B1, m->mel1_lo, m->mel1_hi);
  nz_range(m->mel2, B2, m->mel2_lo, m->mel2_hi);
  return 0;
fail:
  free((void *)m->dir);
  m->dir = NULL;
  munmap(m->map, m->map_size);
  m->map = NULL;
  return -1;
}

void b24_model_unload(b24_model *m) {
  free((void *)m->dir);
  m->dir = NULL;
  if (m->map) munmap(m->map, m->map_size);
  m->map = NULL;
}
