#ifndef BN24_INTERNAL_H
#define BN24_INTERNAL_H
#include <stddef.h>
#include <stdint.h>

#define NSAMP 144000
#define NFR 511     /* frames per spectrogram */
#define NMEL 96
#define N1 2048     /* fe1: n_fft 2048, hop 278 */
#define HOP1 278
#define N2 1024     /* fe2: n_fft 1024, hop 280 */
#define HOP2 280
#define B1 1025
#define B2 513
#define FEAT_C 1024
#define MAX_BLOCKS 32

typedef struct {
  char name[48];
  uint32_t dtype, ndim, dims[4];
  uint64_t offset, nbytes;
} b24_tensor_entry;

typedef struct {
  int stride, cin, cmid, cout, se_rd, skip;
  char tag[16];
  int hin, win, hout, wout;
  const float *pw_w, *pw_b;   /* [cmid][cin] */
  const float *dw_w, *dw_b;   /* [3][3][cmid] */
  const float *se1_w, *se1_b; /* [rd][cmid] */
  const float *se2_w, *se2_b; /* [cmid][rd] */
  const float *pwl_w, *pwl_b; /* [cout][cmid] */
} b24_block;

typedef struct {
  void *map;
  size_t map_size;
  int n_tensors, n_classes, n_blocks;
  const b24_tensor_entry *dir;
  unsigned char labels_sha[32];
  b24_block blocks[MAX_BLOCKS];
  const float *win1, *win2, *dft1, *dft2, *mel1, *mel2, *pow1, *pow2, *fe_bn_s, *fe_bn_b;
  int mel1_lo[NMEL], mel1_hi[NMEL], mel2_lo[NMEL], mel2_hi[NMEL]; /* nonzero bin ranges */
  const float *c0_w, *c0_b, *p0_w, *p0_b, *post_s, *post_b, *pc_w, *pc_b, *head_w, *head_b;
} b24_model;

int b24_model_load(b24_model *m, const char *path, char *err, size_t errlen);
void b24_model_unload(b24_model *m);
const float *b24_tensor(const b24_model *m, const char *name, int expect_numel);

/* reference (fp32 oracle, optional fake quantisation via BN24_Q) */
typedef struct b24_ref b24_ref;
b24_ref *b24_ref_new(const b24_model *m);
void b24_ref_free(b24_ref *r);
int b24_ref_infer(b24_ref *r, const float *pcm, float *logits, float *emb);
/* front end only: img [96][511][2] (NHWC, after the spectrogram BN) */
void b24_ref_frontend(const b24_model *m, const float *pcm, float *img, int use_fft);

/* fast engine */
typedef struct b24_fast b24_fast;
b24_fast *b24_fast_new(const b24_model *m, int mode);
void b24_fast_free(b24_fast *f);
int b24_fast_infer(b24_fast *f, const float *pcm, float *logits, float *emb);
const char *b24_fast_precision(const b24_fast *f);

/* portable fp32 NEON engine (v24_f32.c; plain ARMv8-A, the Cortex-A72 / Pi 4 path) */
typedef struct b24_f32 b24_f32;
b24_f32 *b24_f32_new(const b24_model *m);
void b24_f32_free(b24_f32 *f);
int b24_f32_infer(b24_f32 *f, const float *pcm, float *logits, float *emb);
const char *b24_f32_precision(const b24_f32 *f);
/* CPU features (bnneon.c, HWCAP): fp16 arithmetic */
int bn24_cpu_fp16(void);

double b24_now(void);
#endif
