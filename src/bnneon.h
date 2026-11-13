/* bnneon API v2: one NEON inference library; the weights file selects the model.
 *   .bnpk (magic BNNPKG01): compiled model package (any model: Perch v2, BattyBirdNET, BirdNET v2.4, ...); metadata
 *                           inside the package describes it completely (format_version >= 1)
 *   .bn24 (legacy, format_version 0): BirdNET v2.4 for the hand-written v2.4 engine
 *   .bnv3 (legacy BirdNET v3.0): refused by bnn_probe / bnn_init with BNN_ERR_PACKAGE and a message naming the BirdNET
 *         v3.0 packages to use instead (its hand-written engine was retired; libraries before that ran it)
 * Input convention: raw float PCM in [-1, 1) at info.sample_rate (the engine does no resampling; a model that wants
 * normalized input normalizes itself). Output: raw logits (output_kind LOGITS) and `post` says what the caller
 * applies (legacy .bn24: post SIGMOID). Contexts share nothing, so contexts of different models may run concurrently
 * on different threads.
 * Licenses: a package carries the license and attribution of the model it was compiled from (bnn_probe_meta "license"
 * / "attribution"); legacy .bn24 (v2.4) CC BY-NC-SA 4.0.
 * Thread safety: one context per thread. All strings returned are valid until bnn_free (or are static). */
#ifndef BNNEON_H
#define BNNEON_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define BNN_API_VERSION 2

/* family codes (informational: a generic host should only need the metadata below) */
enum { BNN_MODEL_V24 = 24, BNN_MODEL_V30 = 30, BNN_MODEL_PERCH_V2 = 120, BNN_MODEL_BAT_V24 = 224 };
/* output_kind: what bnn_infer writes. SIGMOID was the retired .bnv3 engine's (confidences); this library writes
 * LOGITS for every file it loads, and the value stays defined for hosts that also run older libraries */
enum { BNN_OUT_LOGITS = 0, BNN_OUT_SIGMOID = 1 };
enum { BNN_POST_NONE = 0, BNN_POST_SIGMOID = 1, BNN_POST_SOFTMAX = 2 }; /* post: what the caller applies to logits */
/* modes: 0 = fastest validated path in the file for this CPU; 1 = fp32 reference (slow oracle; legacy files only,
 * refused for .bnpk); 2 = the portable "neon-fp32" path; 3 = the "neon-fp16" path (fp16 arithmetic, + dotprod if the
 * path says so). A mode the file or CPU cannot run is refused at init (never SIGILL). */
/* 4 = the "neon-int8" path (int8 GEMMs without dotprod: any AArch64; packages only) */
enum { BNN_MODE_DEFAULT = 0, BNN_MODE_REFERENCE = 1, BNN_MODE_FP32 = 2, BNN_MODE_FP16 = 3, BNN_MODE_INT8 = 4 };

typedef struct {
  uint32_t struct_size;  /* in: the caller's sizeof(bnn_info); out: the number of bytes the library filled
                            (min of both); fields beyond it are untouched. Fields are only ever appended. */
  int model;             /* family code BNN_MODEL_* */
  int format_version;    /* 0 = legacy .bn24, >= 1 = .bnpk */
  int sample_rate;       /* Hz of the input samples */
  int n_samples;         /* samples per window */
  int hop_samples;       /* the model's default hop between windows (0 = caller decides) */
  int n_classes;         /* length of the output vector */
  int output_kind;       /* BNN_OUT_*: raw logits (sigmoid only from older libraries' legacy v3 engine) */
  int post;              /* BNN_POST_*: what the caller applies to obtain confidences */
  float logit_scale;     /* multiply logits by this before post (1.0 unless the model says otherwise) */
  int emb_dim;           /* embedding length written by bnn_infer when emb != NULL (0 = none) */
  int has_labels_sha;
  unsigned char labels_sha256[32]; /* sha256 of the label file the package was compiled with (file bytes as shipped) */
  char model_id[64];     /* e.g. "perch_v2", "birdnet_v2.4", "battybirdnet_eu_256khz" */
  char path[32];         /* CPU path: after init the chosen one ("neon-fp16", "neon-fp32"); after probe the
                            best one in the file this CPU can run, "" if none (legacy files: same names) */
  const char *precision; /* human-readable, e.g. "fp16 body, int8 head" (static string; "" after probe) */
} bnn_info;

typedef struct bnn_ctx bnn_ctx;

int bnn_api_version(void);
/* Legacy files only: 1 if this CPU can run the legacy engine of the given family in mode 0 (BNN_MODEL_V24, the .bn24
 * engine: any AArch64). BNN_MODEL_V30 answers as while the .bnv3 engine existed (1 with fp16 arithmetic + dotprod),
 * since hosts ask before loading any file of the family. It does not answer for packages: a family's packages exist
 * for several CPU classes (BirdNET v3.0 has neon-fp16, neon-fp32 and neon-int8 packages), and whether this CPU runs a
 * given .bnpk is bnn_probe's info.path ("" = none of its paths). The Perch v2 and BattyBirdNET codes return 1 as they
 * always have; any other code returns 0. */
int bnn_cpu_supported(int model);
/* reads the header/metadata only: fills info or returns -1 (bnn_last_error) */
int bnn_probe(const char *weights_path, bnn_info *info);
/* metadata string by key ("license", "attribution", "model_id", "source_sha256", "labels_file", "notes", ...;
 * numbers are returned as decimal text; "labels" is the label file itself and "labels_sha256" its 32 raw sha256
 * bytes (also in bnn_info)): copies at most n-1 bytes + NUL into buf; returns the full length, or -1 if
 * the key is absent. Legacy files answer "license" and "model_id". "licence", the key's spelling in packages from
 * compilers before bnc 3, is the same key: either spelling reads the license of old and new packages. */
int bnn_probe_meta(const char *weights_path, const char *key, char *buf, size_t n);
int bnn_meta(const bnn_ctx *ctx, const char *key, char *buf, size_t n);
bnn_ctx *bnn_init(const char *weights_path, int mode);
int bnn_get_info(const bnn_ctx *ctx, bnn_info *info);
/* one window of info.n_samples floats -> info.n_classes outputs (+ optional embedding). 0 on success; nonzero = the
 * window could not be scored (e.g. non-finite features on the fp16 path): the caller should fall back. */
int bnn_infer(bnn_ctx *ctx, const float *pcm, float *out, float *emb);
/* 0 if labels_path is byte-for-byte the label file the package was compiled with (its sha256 is in bnn_info);
 * -1 with bnn_last_error naming both hashes and line counts otherwise. The package also ships the label file itself
 * (bnn_probe_meta "labels"), which is always the right one. */
int bnn_verify_labels(const char *weights_path, const char *labels_path);
/* Every failure sets a one-line message of the form "<what failed>: <expected> vs <found>; hint: <what to do>", which
 * the host should surface unchanged (up to 511 bytes). Load/validation failures name the op index, the kernel, the
 * ONNX node it came from (when the package records source names), the buffer or tensor argument and the values. */
const char *bnn_last_error(void);
/* the kind of the last error, for hosts that act on it (the text stays the thing to show the user) */
enum { BNN_ERR_NONE = 0, BNN_ERR_PACKAGE = 1 /* bad, corrupt or incompatible file, or a kernel-contract violation */,
       BNN_ERR_OOM = 2 /* out of memory */, BNN_ERR_CPU = 3 /* no path or mode this CPU can run */,
       BNN_ERR_WINDOW = 4 /* one window could not be scored */, BNN_ERR_LABELS = 5, BNN_ERR_USAGE = 6 };
int bnn_last_error_kind(void);
/* intra-op threads for a compiled package's context (1 = the calling thread only, the default; at most 64). The
 * outputs are bit-identical for every thread count. Legacy .bn24 contexts are single-threaded: any count but 1
 * is refused (-1, BNN_ERR_USAGE). Not thread-safe against a concurrent bnn_infer on the same context. 0 on success.
 * Added after API version 2 shipped: a host can check for the symbol (dlsym) to support older libraries. */
int bnn_set_threads(bnn_ctx *ctx, int threads);
void bnn_free(bnn_ctx *ctx);

#ifdef __cplusplus
}
#endif
#endif
