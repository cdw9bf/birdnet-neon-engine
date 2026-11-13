/* dB-image front end (fe_dbimg.c): pcm -> zero-haloed NHWC image; see RT_OP_FE_DBIMG in rt.h */
#ifndef FE_DBIMG_H
#define FE_DBIMG_H
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
  int nsamp, nfft, hop, pad, padmode; /* padmode 0 none, 1 reflect, 2 zero */
  int nfr, nmel, R;                   /* frames, mel bands, selected rows */
  int has_top_db, norm;               /* norm 0 none, 1 (x - min) / (max - min + eps), 2 (max - x) / (...) */
  int post[3];                        /* after logf: 0 none, 1 mul, 2 div, 3 add, 4 sub */
  int Ho, Wo, C, chan[2], ld, halo;
  float floor, top_db, eps, postv[3];
} fe_db_cfg;

typedef struct fe_db fe_db;
/* mel [nmel][nfft/2 + 1] (dense; stored sparse), rows [R], chv [2][C] */
fe_db *fe_db_new(const fe_db_cfg *g, const float *window, const float *mel, const int32_t *rows, const float *chv);
void fe_db_free(fe_db *f);
void fe_db_run(fe_db *f, const float *pcm, float *img);
/* the two stages, for a frame split over threads: frames [t0, t1) (t0 % 4 == 0) with caller work buffers of
 * fe_db_work_bytes (re [N/2], im [N/2], pw [N/2 + 1] vectors), then the image */
void fe_db_frames(fe_db *f, const float *pcm, int t0, int t1, float32x4_t *re, float32x4_t *im, float32x4_t *pw);
void fe_db_image(fe_db *f, float *img);
size_t fe_db_work_bytes(const fe_db *f);
const fe_db_cfg *fe_db_config(const fe_db *f);
#endif
