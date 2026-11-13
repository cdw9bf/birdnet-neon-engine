#ifndef V24_FE_H
#define V24_FE_H
#include "v24_internal.h"
typedef struct b24_fe b24_fe;
b24_fe *b24_fe_new(const b24_model *m);
void b24_fe_free(b24_fe *f);
/* pcm [144000] -> img [96][511][2] fp32 */
void b24_fe_run(b24_fe *f, const float *pcm, float *img);
#endif
