// bnneon: one API over the compiled-package runtime (.bnpk, rt_*.c) and the legacy BirdNET v2.4 engine (bn24.h),
// selected by weights-file magic. A legacy BirdNET v3.0 .bnv3 is recognized and refused with a message (its
// hand-written engine was retired for the BirdNET v3.0 packages).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1 << 10)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif

#include "bn24.h"
#include "bnneon.h"
#include "rt.h"
#include "sha256.h"

struct bnn_ctx {
  bn24_ctx *v24;
  rt_pkg *pkg;
  rt_ctx *rt;
  bnn_info info;
};

static __thread char g_err[512];
static __thread int g_kind;
const char *bnn_last_error(void) { return g_err; }
int bnn_last_error_kind(void) { return g_kind; }
#define ERR(kind, ...) (g_kind = (kind), snprintf(g_err, sizeof g_err, __VA_ARGS__))
static const char *base(const char *path) { const char *s = strrchr(path, '/'); return s ? s + 1 : path; }
int bnn_api_version(void) { return BNN_API_VERSION; }

static unsigned cpu_caps(void) { /* RT_HW_* bits this CPU has */
#if defined(__linux__)
  unsigned long h = getauxval(AT_HWCAP);
  return ((h & HWCAP_ASIMDHP) ? RT_HW_FP16 : 0) | ((h & HWCAP_ASIMDDP) ? RT_HW_DOTPROD : 0);
#else
  return RT_HW_FP16 | RT_HW_DOTPROD; /* Apple silicon */
#endif
}
static int cpu_dotprod_fp16(void) { return (cpu_caps() & (RT_HW_FP16 | RT_HW_DOTPROD)) == (RT_HW_FP16 | RT_HW_DOTPROD); }
int bn24_cpu_fp16(void) { return (cpu_caps() & RT_HW_FP16) != 0; }
/* the legacy engine's answer per family (bnneon.h): the .bn24 engine has an fp32 path for any AArch64. BirdNET v3.0
 * answers as it did while its .bnv3 engine (Cortex-A76 code) existed, because hosts ask before they load any file of
 * a family and refuse on 0 (BirdNET-Go's loadNeonLibrary). (The answer for a package is its paths: pick_path.) */
int bnn_cpu_supported(int model) {
  if (model == BNN_MODEL_V30) return cpu_dotprod_fp16();
  return model == BNN_MODEL_V24 || model == BNN_MODEL_PERCH_V2 || model == BNN_MODEL_BAT_V24;
}

/* the legacy header of the open file f: magic, class count and (.bn24) the label hash */
static int read_magic(FILE *f, const char *path, char magic[8], unsigned char sha[32], int *nclass) {
  unsigned char h[28] = {0};
  int ok = fseek(f, 0, SEEK_SET) == 0 && fread(h, 1, sizeof h, f) == sizeof h;
  memcpy(magic, h, 8);
  uint32_t v[5];
  memcpy(v, h + 8, sizeof v);
  *nclass = (int)v[2];
  if (ok && !memcmp(magic, "BN24ENG1", 8)) {
    ok = v[4] < (1u << 20) && fseek(f, 28 + (long)v[4], SEEK_SET) == 0 && fread(sha, 1, 32, f) == 32;
  }
  if (!ok) { ERR(BNN_ERR_PACKAGE, "%s: truncated header", path); return -1; }
  return 0;
}

/* the package paths a mode can name; tag: the part of a path name that tells its precision (first match wins, the
 * last row matches every name) */
static const struct { int mode; const char *name, *tag, *precision; } PATHS[] = {
    {BNN_MODE_FP16, "neon-fp16", "fp16", "fp16"},
    {BNN_MODE_INT8, "neon-int8", "int8", "int8 GEMMs, fp32 elsewhere"},
    {BNN_MODE_FP32, "neon-fp32", "", "fp32"},
};
enum { N_PATHS = sizeof PATHS / sizeof PATHS[0] };

/* the best path of a package this CPU can run (paths are stored fastest first); want: "" = any */
static int pick_path(const rt_pkg *p, const char *want) {
  unsigned caps = cpu_caps();
  for (int i = 0; i < rt_n_paths(p); i++) {
    const rt_pathdesc *d = rt_path(p, i);
    if ((d->hwcap & caps) != d->hwcap) continue;
    if (want[0] && strncmp(d->name, want, sizeof d->name)) continue;
    return i;
  }
  return -1;
}
static const char *hw_words(unsigned hw, char *buf, size_t n) {
  snprintf(buf, n, "%s%s%s", (hw & RT_HW_FP16) ? "fp16 arithmetic" : "", (hw & RT_HW_FP16) && (hw & RT_HW_DOTPROD) ? " + " : "",
           (hw & RT_HW_DOTPROD) ? "dotprod" : "");
  if (!buf[0]) snprintf(buf, n, "any AArch64");
  return buf;
}
/* why pick_path failed: every path of the package, what it needs, what this CPU has, and what to do */
static void no_path_error(const char *path, const rt_pkg *p, const char *want) {
  unsigned caps = cpu_caps();
  char have[64], list[256] = "", id[64] = "";
  hw_words(caps, have, sizeof have);
  rt_meta_str(p, "model_id", id, sizeof id);
  size_t k = 0;
  int present = 0;
  for (int i = 0; i < rt_n_paths(p) && k + 64 < sizeof list; i++) {
    const rt_pathdesc *d = rt_path(p, i);
    char need[64];
    if (want[0] && !strncmp(d->name, want, sizeof d->name)) present = 1;
    k += (size_t)snprintf(list + k, sizeof list - k, "%s%.24s (needs %s)", i ? ", " : "", d->name, hw_words(d->hwcap, need, sizeof need));
  }
  g_kind = BNN_ERR_CPU;
  if (want[0] && !present)
    snprintf(g_err, sizeof g_err, "%s: path %s is not in this package (it has: %s); hint: use mode 0 to let the library "
             "pick, or a package of this model compiled for %s, if the model has that path (it would be %s-%s.bnpk; "
             "`bnc.py compile --paths %s` says whether it does)", path, want, list, want, id[0] ? id : "<model_id>", want, want);
  else
    snprintf(g_err, sizeof g_err, "%s: this CPU (%s) can run none of the package's paths%s%s: %s; hint: use the neon-fp32 "
             "package of this model (%s-neon-fp32.bnpk, any AArch64) or compile one with --paths neon-fp32",
             path, have, want[0] ? " named " : "", want, list, id[0] ? id : "<model_id>");
}

static void pkg_info(const rt_pkg *p, int pi, bnn_info *info) {
  int64_t v;
  double x;
#define I(key, fld) do { if (!rt_meta_i64(p, key, &v)) info->fld = (int)v; } while (0)
  I("family", model); I("sample_rate", sample_rate); I("n_samples", n_samples); I("hop_samples", hop_samples);
  I("n_classes", n_classes); I("post", post); I("emb_dim", emb_dim);
#undef I
  info->format_version = rt_format_version(p);
  info->output_kind = BNN_OUT_LOGITS;
  info->logit_scale = rt_meta_f64(p, "logit_scale", &x) ? 1.f : (float)x;
  char sha[40];
  if (rt_meta_str(p, "labels_sha256", sha, sizeof sha) == 32) { memcpy(info->labels_sha256, sha, 32); info->has_labels_sha = 1; }
  rt_meta_str(p, "model_id", info->model_id, sizeof info->model_id);
  if (pi >= 0) snprintf(info->path, sizeof info->path, "%.24s", rt_path(p, pi)->name);
}

static void copy_info(bnn_info *dst, const bnn_info *src) {
  uint32_t n = dst->struct_size;
  if (n > sizeof *src) n = sizeof *src;
  bnn_info tmp = *src;
  tmp.struct_size = n;
  memcpy(dst, &tmp, n);
}

/* keep (may be NULL): for a package, receives the open package instead of closing it (the caller closes it) */
static int probe_full(const char *path, bnn_info *info, rt_pkg **keep) {
  char magic[8];
  memset(info, 0, sizeof *info);
  info->struct_size = sizeof *info;
  info->logit_scale = 1.f;
  info->precision = "";
  FILE *f = fopen(path, "rb");
  if (!f) { ERR(BNN_ERR_PACKAGE, "cannot open %s; hint: check the path and permissions", path); return -1; }
  int ok = fread(magic, 1, 8, f) == 8;
  if (!ok) { fclose(f); ERR(BNN_ERR_PACKAGE, "%s: shorter than 8 bytes; hint: incomplete download or the wrong file", path); return -1; }
  if (!memcmp(magic, "BNNPKG01", 8)) {
    fclose(f);
    g_kind = BNN_ERR_PACKAGE;
    rt_pkg *p = rt_pkg_open(path, g_err, sizeof g_err);
    if (!p) return -1;
    pkg_info(p, pick_path(p, ""), info);
    if (keep) *keep = p;
    else rt_pkg_close(p);
    return 0;
  }
  int nc = 0;
  ok = read_magic(f, path, magic, info->labels_sha256, &nc) == 0;
  fclose(f);
  if (!ok) return -1;
  info->n_classes = nc;
  if (!memcmp(magic, "BN24ENG1", 8)) {
    info->model = BNN_MODEL_V24; info->sample_rate = 48000; info->n_samples = BN24_SAMPLES;
    info->output_kind = BNN_OUT_LOGITS; info->post = BNN_POST_SIGMOID; info->emb_dim = 1024; info->has_labels_sha = 1;
    snprintf(info->model_id, sizeof info->model_id, "birdnet_v2.4");
    snprintf(info->path, sizeof info->path, "%s", bn24_cpu_fp16() ? "neon-fp16" : "neon-fp32");
  } else if (!memcmp(magic, "BNV3ENG1", 8)) {
    ERR(BNN_ERR_PACKAGE, "%s: a legacy BirdNET v3.0 .bnv3 file, whose hand-written engine this library no longer has; "
        "hint: use the BirdNET v3.0 package compiled from the same ONNX model (birdnet_v3.0[_<region>]-neon-fp16.bnpk on "
        "a Cortex-A76 class CPU, -neon-int8 or -neon-fp32 on any AArch64; scripts/compile_models.sh v3, guide section "
        "2.1), or a libbnneon built before the engine's removal", path);
    return -1;
  } else {
    ERR(BNN_ERR_PACKAGE, "%s: not a bnneon weights file (magic '%.8s'; expected BNNPKG01 for a compiled package, "
             "BN24ENG1 for a legacy BirdNET v2.4 file); hint: compile the model with compiler/bnc.py, or check the "
             "download (sha256)", path, magic);
    return -1;
  }
  return 0;
}

int bnn_probe(const char *path, bnn_info *info) {
  if (!info || info->struct_size < 8) { ERR(BNN_ERR_USAGE, "bnn_probe: bnn_info.struct_size not set"); return -1; }
  bnn_info full;
  if (probe_full(path, &full, NULL)) return -1;
  copy_info(info, &full);
  return 0;
}

/* "license" was spelled "licence" by compilers before bnc 3: either key finds the license of any file */
static int is_license_key(const char *key) { return !strcmp(key, "license") || !strcmp(key, "licence"); }

/* the two keys a legacy (.bn24) file answers */
static int legacy_meta(const bnn_info *info, const char *key, char *buf, size_t n) {
  const char *v = is_license_key(key) ? "CC BY-NC-SA 4.0 (BirdNET v2.4)" : !strcmp(key, "model_id") ? info->model_id : NULL;
  if (!v) return -1;
  if (buf && n) snprintf(buf, n, "%s", v);
  return (int)strlen(v);
}

/* a package's metadata string; the license under either spelling of its key */
static int pkg_meta(const rt_pkg *p, const char *key, char *buf, size_t n) {
  int r = rt_meta_str(p, key, buf, n);
  if (r < 0 && is_license_key(key)) r = rt_meta_str(p, strcmp(key, "licence") ? "licence" : "license", buf, n);
  return r;
}

int bnn_probe_meta(const char *path, const char *key, char *buf, size_t n) {
  bnn_info info;
  rt_pkg *p = NULL;
  if (probe_full(path, &info, &p)) return -1;
  int r = info.format_version >= 1 ? pkg_meta(p, key, buf, n) : legacy_meta(&info, key, buf, n);
  rt_pkg_close(p);
  return r;
}

int bnn_meta(const bnn_ctx *c, const char *key, char *buf, size_t n) {
  if (!c) return -1;
  if (c->pkg) return pkg_meta(c->pkg, key, buf, n);
  return legacy_meta(&c->info, key, buf, n);
}

/* c->pkg is the open package (bnn_init's probe kept it) */
static bnn_ctx *init_pkg(const char *path, int mode, bnn_ctx *c) {
  g_kind = BNN_ERR_PACKAGE;
  const char *want = "";
  for (int i = 0; i < N_PATHS; i++)
    if (PATHS[i].mode == mode) want = PATHS[i].name;
  if (mode == BNN_MODE_REFERENCE || mode < 0 || mode > BNN_MODE_INT8) {
    ERR(BNN_ERR_USAGE, "%s: mode %d is not available for compiled packages; hint: use mode 0 (best path for "
             "this CPU), 2 (the neon-fp32 path), 3 (the neon-fp16 path) or 4 (the neon-int8 path); mode 1 (fp32 reference) "
             "exists for the legacy .bn24 files only", path, mode);
    return NULL;
  }
  int pi = pick_path(c->pkg, want);
  if (pi < 0) { no_path_error(path, c->pkg, want); return NULL; }
  char rerr[sizeof g_err];
  int kind = RT_E_PARAM;
  if (!(c->rt = rt_ctx_new(c->pkg, pi, rerr, sizeof rerr, &kind))) {
    ERR(kind == RT_E_OOM ? BNN_ERR_OOM : BNN_ERR_PACKAGE, "%s path %.24s: %s", base(path), rt_path(c->pkg, pi)->name, rerr);
    return NULL;
  }
  pkg_info(c->pkg, pi, &c->info);
  /* the metadata must agree with the buffers bnn_infer reads and the window the caller passes */
  const rt_pathdesc *d = c->rt->path;
  const rt_bufdesc *ib = &c->rt->bufd[d->in_buf], *ob = &c->rt->bufd[d->out_buf];
  const rt_bufdesc *eb = d->emb_buf == 0xffffffffu ? NULL : &c->rt->bufd[d->emb_buf];
#define META_HINT "; hint: the package is inconsistent: recompile it (compiler/bnc.py) and check its 'compiler' metadata"
  g_kind = BNN_ERR_PACKAGE;
  if (c->info.n_samples <= 0 || ib->dtype != RT_F32 || ib->elems != (uint64_t)c->info.n_samples) {
    snprintf(g_err, sizeof g_err, "%s: metadata n_samples = %d but the input buffer is %s with %llu elements" META_HINT, path,
             c->info.n_samples, rt_dtype_name(ib->dtype), (unsigned long long)ib->elems);
    return NULL;
  }
  if (c->info.n_classes <= 0 || ob->dtype != RT_F32 || ob->elems < (uint64_t)c->info.n_classes) {
    snprintf(g_err, sizeof g_err, "%s: metadata n_classes = %d but the output buffer is %s with %llu elements" META_HINT, path,
             c->info.n_classes, rt_dtype_name(ob->dtype), (unsigned long long)ob->elems);
    return NULL;
  }
  if (c->info.emb_dim < 0 || (c->info.emb_dim && (!eb || eb->dtype != RT_F32 || eb->elems < (uint64_t)c->info.emb_dim))) {
    snprintf(g_err, sizeof g_err, "%s: metadata emb_dim = %d but the path's embedding buffer is %s" META_HINT, path,
             c->info.emb_dim, !eb ? "absent" : rt_dtype_name(eb->dtype));
    return NULL;
  }
  c->info.precision = "int8/fp16"; /* a dotprod path */
  if (!(d->hwcap & RT_HW_DOTPROD))
    for (int i = 0; i < N_PATHS; i++)
      if (strstr(d->name, PATHS[i].tag)) { c->info.precision = PATHS[i].precision; break; }
  return c;
}

bnn_ctx *bnn_init(const char *path, int mode) {
  bnn_info pi;
  rt_pkg *pkg = NULL; /* a package stays open from the probe to the context: one open, one mapping */
  if (probe_full(path, &pi, &pkg)) return NULL;
  if (pi.format_version < 1) { rt_pkg_close(pkg); pkg = NULL; } /* not a format this reader runs as a package */
  bnn_ctx *c = calloc(1, sizeof *c);
  if (!c) { rt_pkg_close(pkg); ERR(BNN_ERR_OOM, "out of memory (%zu bytes for the context)", sizeof *c); return NULL; }
  c->info = pi;
  c->pkg = pkg;
  if (pi.format_version >= 1) {
    if (!init_pkg(path, mode, c)) { bnn_free(c); return NULL; }
  } else {
    c->v24 = bn24_init(path, mode);
    if (!c->v24) { ERR(BNN_ERR_PACKAGE, "%s", bn24_last_error()); free(c); return NULL; }
    c->info.n_classes = bn24_num_classes(c->v24);
    c->info.precision = bn24_precision(c->v24);
    snprintf(c->info.path, sizeof c->info.path, "%s",
             mode == 1 ? "reference" : (mode == 2 || !bn24_cpu_fp16()) ? "neon-fp32" : "neon-fp16");
  }
  return c;
}

int bnn_get_info(const bnn_ctx *c, bnn_info *info) {
  if (!c || !info || info->struct_size < 8) { ERR(BNN_ERR_USAGE, "bnn_get_info: no context or struct_size"); return -1; }
  copy_info(info, &c->info);
  return 0;
}

int bnn_infer(bnn_ctx *c, const float *pcm, float *out, float *emb) {
  if (!c || !pcm || !out) { ERR(BNN_ERR_USAGE, "bnn_infer: NULL context, pcm or out"); return -1; }
  if (c->v24) return bn24_infer(c->v24, pcm, out, emb);
  int r = rt_run(c->rt, pcm);
  if (r) {
    char src[160];
    int i = c->rt->fail_op;
    const char *kn = rt_op_name(c->rt, i);
    if (i < 0 || rt_op_src(c->rt, i, src, sizeof src)) src[0] = 0;
    int f16 = strstr(c->info.path, "fp16") != NULL;
    ERR(BNN_ERR_WINDOW, "window not scored: op %d %s%s%s%s reported a failed window (code %d: non-finite values on the %s "
        "path); hint: %s", i, kn, src[0] ? " (ONNX node " : "", src, src[0] ? ")" : "", r, c->info.path,
        f16 ? "skip this window or score it with the neon-fp32 path (mode 2)"
            : "skip this window (the input itself is probably non-finite or extreme)");
    return r;
  }
  memcpy(out, rt_out(c->rt), sizeof(float) * c->info.n_classes);
  const float *e = rt_emb(c->rt);
  if (emb && e) memcpy(emb, e, sizeof(float) * c->info.emb_dim);
  return 0;
}

int bnn_verify_labels(const char *weights_path, const char *labels_path) {
  bnn_info info;
  if (probe_full(weights_path, &info, NULL)) return -1;
  g_kind = BNN_ERR_LABELS;
  if (!info.has_labels_sha) {
    snprintf(g_err, sizeof g_err, "%s carries no label hash (legacy file or package compiled without --labels)", weights_path);
    return -1;
  }
  FILE *f = fopen(labels_path, "rb");
  if (!f) { snprintf(g_err, sizeof g_err, "cannot open label file %s", labels_path); return -1; }
  unsigned char buf[65536];
  size_t n, total = 0;
  int lines = 0;
  /* sha256 of the file bytes as shipped (no newline normalisation), the same as the compiler's */
  sha256_ctx h;
  sha256_init(&h);
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) {
    sha256_update(&h, buf, n);
    for (size_t i = 0; i < n; i++) lines += buf[i] == '\n';
    total += n;
  }
  fclose(f);
  unsigned char sum[32];
  sha256_final(&h, sum);
  if (!memcmp(sum, info.labels_sha256, 32)) return 0;
  char a[17], b[17];
  for (int i = 0; i < 8; i++) { snprintf(a + 2 * i, 3, "%02x", sum[i]); snprintf(b + 2 * i, 3, "%02x", info.labels_sha256[i]); }
  snprintf(g_err, sizeof g_err, "label file %s (sha256 %s..., %d lines, %zu bytes) is not the label file %s was compiled with "
           "(sha256 %s..., %d classes); hint: use the label file inside the package (bnn_probe_meta \"labels\", or `cli <pkg> "
           "--labels`) or recompile the package with --labels %s", labels_path, a, lines, total, weights_path, b, info.n_classes, labels_path);
  return -1;
}

int bnn_set_threads(bnn_ctx *c, int threads) {
  if (!c) { ERR(BNN_ERR_USAGE, "bnn_set_threads: NULL context"); return -1; }
  if (threads < 1 || threads > 64) {
    ERR(BNN_ERR_USAGE, "bnn_set_threads: %d threads; hint: pass 1..64 (a host's 0 = all CPUs must be resolved by the host)", threads);
    return -1;
  }
  if (!c->rt) {
    if (threads == 1) return 0;
    ERR(BNN_ERR_USAGE, "bnn_set_threads: %d threads for the legacy %s engine, which is single-threaded; hint: use a compiled "
        ".bnpk package for threads, or 1", threads, c->info.model_id);
    return -1;
  }
  if (rt_set_threads(c->rt, threads)) { ERR(BNN_ERR_OOM, "bnn_set_threads: cannot start %d threads", threads); return -1; }
  return 0;
}

void bnn_free(bnn_ctx *c) {
  if (!c) return;
  if (c->v24) bn24_free(c->v24);
  rt_ctx_free(c->rt);
  rt_pkg_close(c->pkg);
  free(c);
}
