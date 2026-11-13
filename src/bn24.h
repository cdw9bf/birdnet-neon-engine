/* bn24: standalone BirdNET v2.4 inference engine (front end + CNN body + 6522-class head), C API.
 * Weights: a .bn24 file exported by scripts/export_weights.py from BirdNET_v2.4_fp32.onnx
 * (CC BY-NC-SA 4.0, derived from BirdNET v2.4).
 * Input: 144000 float samples (3 s at 48 kHz, same scale as the ONNX "input"). Output: n_classes LOGITS (the same
 * quantity as the ONNX "output"; BirdNET-Go applies the sigmoid with its sensitivity).
 * Thread safety: one bn24_ctx per thread; contexts share nothing mutable. */
#ifndef BN24_H
#define BN24_H
#ifdef __cplusplus
extern "C" {
#endif

#define BN24_SAMPLES 144000

typedef struct bn24_ctx bn24_ctx;

/* mode: 0 = default (fastest validated path for this CPU), 1 = fp32 reference (slow oracle), 2 = fp32 fast path
 * (portable NEON fp32; the Cortex-A72 / Pi 4 path), 3 = fp16 path (needs ARMv8.2 fp16). Returns NULL on error. */
bn24_ctx *bn24_init(const char *weights_path, int mode);
/* Runs one 3 s window. logits must hold bn24_num_classes() floats; emb (optional, may be NULL) 1024 floats.
 * Returns 0 on success. */
int bn24_infer(bn24_ctx *ctx, const float *pcm, float *logits, float *emb);
int bn24_num_classes(const bn24_ctx *ctx);
/* sha256 of the label file the weights were exported with (32 bytes) */
const unsigned char *bn24_labels_sha256(const bn24_ctx *ctx);
/* human-readable description of the active path, e.g. "int8/fp16 (dotprod)" */
const char *bn24_precision(const bn24_ctx *ctx);
const char *bn24_last_error(void);
void bn24_free(bn24_ctx *ctx);

#ifdef __cplusplus
}
#endif
#endif
