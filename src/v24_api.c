// C API (src/bn24.h)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bn24.h"
#include "v24_internal.h"

struct bn24_ctx {
  b24_model m;
  b24_ref *ref;
  b24_fast *fast; /* fp16 path (ARMv8.2 fp16 arithmetic) */
  b24_f32 *f32;   /* portable fp32 path */
};

static _Thread_local char g_err[256];

const char *bn24_last_error(void) { return g_err; }

bn24_ctx *bn24_init(const char *weights_path, int mode) {
  bn24_ctx *c = calloc(1, sizeof *c);
  if (!c) { snprintf(g_err, sizeof g_err, "out of memory"); return NULL; }
  if (b24_model_load(&c->m, weights_path, g_err, sizeof g_err) != 0) { free(c); return NULL; }
  /* mode 0: the fp16 path where the CPU has fp16 arithmetic, else the portable fp32 path; 1: reference;
   * 2: fp32 path; 3: fp16 path (needs fp16 arithmetic); modes the CPU cannot run are refused here (never SIGILL) */
  int fp16 = bn24_cpu_fp16();
  g_err[0] = 0;
  if (mode == 1) c->ref = b24_ref_new(&c->m);
  else if (mode == 2 || (mode == 0 && !fp16)) c->f32 = b24_f32_new(&c->m);
  else if (mode == 0 || mode == 3) {
    if (!fp16) snprintf(g_err, sizeof g_err, "mode %d needs fp16 arithmetic, which this CPU lacks", mode);
    else c->fast = b24_fast_new(&c->m, mode);
  } else snprintf(g_err, sizeof g_err, "unknown mode %d (0 default, 1 reference, 2 fp32, 3 fp16)", mode);
  if (!c->ref && !c->fast && !c->f32) {
    if (!g_err[0]) snprintf(g_err, sizeof g_err, "engine init failed (mode %d)", mode);
    b24_model_unload(&c->m);
    free(c);
    return NULL;
  }
  return c;
}

int bn24_infer(bn24_ctx *c, const float *pcm, float *logits, float *emb) {
  if (!c || !pcm || !logits) return -1;
  if (c->ref) return b24_ref_infer(c->ref, pcm, logits, emb);
  if (c->f32) return b24_f32_infer(c->f32, pcm, logits, emb);
  return b24_fast_infer(c->fast, pcm, logits, emb);
}

int bn24_num_classes(const bn24_ctx *c) { return c ? c->m.n_classes : 0; }
const unsigned char *bn24_labels_sha256(const bn24_ctx *c) { return c ? c->m.labels_sha : NULL; }
const char *bn24_precision(const bn24_ctx *c) {
  return !c ? "" : c->ref ? "fp32 (reference)" : c->f32 ? b24_f32_precision(c->f32) : b24_fast_precision(c->fast);
}

void bn24_free(bn24_ctx *c) {
  if (!c) return;
  b24_ref_free(c->ref);
  b24_fast_free(c->fast);
  b24_f32_free(c->f32);
  b24_model_unload(&c->m);
  free(c);
}
