// Perch v2 engine internals (weights file .bnp2, oracle, fast paths)
#ifndef PERCH_INTERNAL_H
#define PERCH_INTERNAL_H
#include <stddef.h>
#include <stdint.h>

#define PC_SAMPLES 160000
#define PC_FRAME 640
#define PC_HOP 320
#define PC_PAD 160
#define PC_NFFT 1024
#define PC_BINS 513
#define PC_NFR 500
#define PC_NMEL 128
#define PC_EMB 1536
#define PC_CELLS 64 /* 16 x 4 spatial cells of the final feature map */
#define PC_MAX_BLOCKS 32

typedef struct {
  char name[48];
  uint32_t dtype, ndim, dims[4];
  uint64_t offset, nbytes;
} pc_tensor_entry;

typedef struct {
  int k, stride, cin, cexp, cout, se_rd, skip; /* cexp = width of the depthwise stage (cin if no expansion conv) */
  int has_ex;
  int pt, pl, pb, pr;                          /* depthwise pads (TF SAME, possibly asymmetric) */
  int hin, win, hout, wout;
  const float *ex_w, *ex_b;   /* [cexp][cin] */
  const float *dw_w, *dw_b;   /* [k][k][cexp] */
  const float *se1_w, *se1_b; /* [rd][cexp] */
  const float *se2_w, *se2_b; /* [cexp][rd] */
  const float *pj_w, *pj_b;   /* [cout][cexp] */
} pc_block;

typedef struct {
  void *map;
  size_t map_size;
  int n_tensors, n_classes, n_blocks, n_proto, ppc;
  const pc_tensor_entry *dir;
  unsigned char labels_sha[32];
  pc_block blocks[PC_MAX_BLOCKS];
  const float *win, *mel; /* [640], [128][513] */
  int mel_lo[PC_NMEL], mel_hi[PC_NMEL];
  float log_floor, log_scale, norm_eps;
  const float *stem_w, *stem_b; /* OHWI [40][3][3][1] */
  int stem_c;
  const float *top_w, *top_b;   /* [1536][384] */
  const float *proto;           /* [n_proto][1536] */
  const float *cls_w, *cls_b;   /* [n_classes][ppc], [n_classes] */
} pc_model;

int pc_model_load(pc_model *m, const char *path, char *err, size_t errlen);
void pc_model_unload(pc_model *m);
/* drop the page-cache pages of a tensor's mmap range once the engine holds its own packed copy (RSS) */
void pc_model_release(const pc_model *m, const void *p, size_t nbytes);

/* fp32 oracle (perch_ref.c; fake quantisation via env PERCH_Q) */
typedef struct pc_ref pc_ref;
pc_ref *pc_ref_new(const pc_model *m);
void pc_ref_free(pc_ref *r);
int pc_ref_infer(pc_ref *r, const float *pcm, float *logits, float *emb);
/* front end only (exact double FFT): spec [500][128] */
void pc_ref_frontend(const pc_model *m, const float *pcm, float *spec);

double pc_now(void);
#endif
