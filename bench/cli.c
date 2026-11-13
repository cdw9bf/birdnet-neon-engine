// bnneon CLI (any model, .bnpk or legacy): run windows through the engine and time it.
//   cli <weights> <mode> <windows.f32> <out.f32> [reps] [warmup]               windows = n x n_samples floats
//   cli <weights> <mode> --stream <stream.f32> <offsets.i32> <out.f32>           gate scoring of a whole stream
//   cli <weights> <mode> --pipe                                                   same over stdin/stdout (see below)
//   cli <weights> <mode> --pipe-emb                                               same, writing the embeddings
//   cli <weights> --info                                                          metadata, paths, chosen path
//   cli <weights> --verify-labels <labels.txt>                                    0 if the label file matches
//   cli <weights> --labels                                                        print the embedded label file
// BNN_THREADS=N: intra-op threads for a package (bnn_set_threads; outputs identical for every N).
// Prints per-window median/min ms and VmRSS on Linux. Every failure prints bnn_last_error, which names
// the op, kernel, source node, the expected and found values and a hint.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include "../src/bnneon.h"
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void *slurp(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb");
  if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END);
  *n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *b = malloc(*n ? *n : 1);
  if (fread(b, 1, *n, f) != *n) { perror("read"); exit(1); }
  fclose(f);
  return b;
}

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

static void rss(void) {
  FILE *f = fopen("/proc/self/status", "r");
  if (!f) return;
  char l[256];
  while (fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS", 5) || !strncmp(l, "VmHWM", 5)) fputs(l, stderr);
  fclose(f);
}

static int info_cmd(const char *w) {
  bnn_info inf = {.struct_size = sizeof inf};
  if (bnn_probe(w, &inf)) { fprintf(stderr, "probe: %s\n", bnn_last_error()); return 1; }
  static const char *posts[] = {"none", "sigmoid", "softmax"};
  printf("model_id %s family %d format_version %d\nsample_rate %d n_samples %d hop %d n_classes %d emb_dim %d post %d (%s) "
         "logit_scale %g\nbest path for this CPU: %s\n", inf.model_id, inf.model, inf.format_version, inf.sample_rate,
         inf.n_samples, inf.hop_samples, inf.n_classes, inf.emb_dim, inf.post,
         inf.post >= 0 && inf.post <= 2 ? posts[inf.post] : "?", inf.logit_scale, inf.path[0] ? inf.path : "(none)");
  if (inf.has_labels_sha) {
    printf("labels_sha256 ");
    for (int i = 0; i < 32; i++) printf("%02x", inf.labels_sha256[i]);
    printf("\n");
  }
  static const char *keys[] = {"compiler", "license", "attribution", "source_sha256", "labels_file", "notes", "input", NULL};
  char buf[8192];
  for (int i = 0; keys[i]; i++) {
    int r = bnn_probe_meta(w, keys[i], buf, sizeof buf);
    if (r >= 0) printf("%s: %s%s\n", keys[i], buf, r >= (int)sizeof buf ? " ...(truncated)" : "");
  }
  /* which modes initialize (1 = the fp32 reference: legacy files only). A package's mode 0 is the explicit mode that
     names its best path, so that one initialization answers both. */
  static const char *mode_path[] = {"", "", "neon-fp32", "neon-fp16", "neon-int8"};
  char res[5][512];
  for (int m = 4; m >= 0; m--) {
    int same = 0;
    for (int k = 2; k <= 4 && m == 0 && inf.format_version >= 1; k++)
      if (!strcmp(inf.path, mode_path[k])) same = k;
    if (m == 1) continue;
    if (same) { memcpy(res[0], res[same], sizeof res[0]); continue; }
    bnn_ctx *c = bnn_init(w, m);
    snprintf(res[m], sizeof res[m], "%s", c ? "ok" : bnn_last_error());
    if (c) bnn_free(c);
  }
  for (int m = 0; m <= 4; m++)
    if (m != 1) printf("mode %d: %s\n", m, res[m]);
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 3 && !strcmp(argv[2], "--info")) return info_cmd(argv[1]);
  if (argc == 3 && !strcmp(argv[2], "--labels")) { /* the label file compiled into the package, byte for byte */
    int n = bnn_probe_meta(argv[1], "labels", NULL, 0);
    if (n < 0) { fprintf(stderr, "labels: %s carries no label file (%s)\n", argv[1], bnn_last_error()); return 1; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fprintf(stderr, "labels: out of memory (%d bytes)\n", n); return 1; }
    bnn_probe_meta(argv[1], "labels", buf, (size_t)n + 1);
    fwrite(buf, 1, (size_t)n, stdout);
    free(buf);
    return 0;
  }
  if (argc == 4 && !strcmp(argv[2], "--verify-labels")) {
    if (bnn_verify_labels(argv[1], argv[3])) { fprintf(stderr, "labels: %s\n", bnn_last_error()); return 1; }
    printf("labels match\n");
    return 0;
  }
  int pipe = argc == 4 && (!strcmp(argv[3], "--pipe") || !strcmp(argv[3], "--pipe-emb"));
  if (argc < 5 && !pipe) {
    fprintf(stderr,
            "usage: cli <weights> <mode> <windows.f32> <out.f32> [reps] [warmup]   score windows (n x n_samples float32),\n"
            "                                                                     write n x n_classes float32 logits, print ms/window\n"
            "       cli <weights> <mode> --stream <stream.f32> <offsets.i32> <out.f32>   score a long stream at the given sample offsets\n"
            "       cli <weights> <mode> --pipe | --pipe-emb                            the same over stdin/stdout (logits | embeddings)\n"
            "       cli <weights> --info                                                metadata, paths, which modes initialize\n"
            "       cli <weights> --verify-labels <labels.txt>                          0 if the label file is the one compiled in\n"
            "       cli <weights> --labels                                              print the label file compiled in (stdout)\n"
            "weights: a .bnpk package (or a legacy .bn24 file); mode: 0 = the best path this CPU can run, 2 = the neon-fp32\n"
            "path, 3 = the neon-fp16 path, 4 = the neon-int8 path (1 = fp32 reference, legacy files only). Every failure prints\n"
            "bnn_last_error. Environment: BNN_THREADS=N runs a package's ops on N threads (outputs identical for every N).\n");
    return 2;
  }
  bnn_ctx *c = bnn_init(argv[1], atoi(argv[2]));
  if (c && getenv("BNN_THREADS") && bnn_set_threads(c, atoi(getenv("BNN_THREADS")))) {
    fprintf(stderr, "threads: %s\n", bnn_last_error());
    return 1;
  }
  if (!c) { fprintf(stderr, "init: %s\n", bnn_last_error()); return 1; }
  bnn_info inf = {.struct_size = sizeof inf};
  bnn_get_info(c, &inf);
  int nc = inf.n_classes, NS = inf.n_samples;
  if (pipe) {
    int emb_out = !strcmp(argv[3], "--pipe-emb"); /* write the embedding (info.emb_dim floats) instead of the logits */
    float *ev = malloc(sizeof(float) * (inf.emb_dim ? inf.emb_dim : 1));
    if (emb_out && !inf.emb_dim) { fprintf(stderr, "pipe: this model has no embedding output\n"); return 1; }
    /* stdin: int32 n, int32 offsets[n] (ascending, in samples), then the float32 stream; stdout: n x classes float32
       logits. Used by the gates' scorers (scripts/pipe_score.py, the bridge modelgate extern runs, and gate_v3_pkg.py)
       through docker exec -i (no files on a shared mount). */
    int32_t n;
    if (fread(&n, 4, 1, stdin) != 1 || n <= 0) { fprintf(stderr, "pipe: bad header\n"); return 1; }
    int32_t *off = malloc(sizeof(int32_t) * n);
    if (fread(off, 4, n, stdin) != (size_t)n) { fprintf(stderr, "pipe: short offsets\n"); return 1; }
    size_t cap = (size_t)NS * 4, have = 0, base = 0; /* buffer holds samples [base, base + have) */
    float *buf = malloc(sizeof(float) * cap), *lg = malloc(sizeof(float) * nc);
    for (int32_t i = 0; i < n; i++) {
      size_t o = (size_t)off[i];
      if (o < base) { fprintf(stderr, "pipe: offsets not ascending\n"); return 1; }
      if (o > base) { /* drop samples before o */
        size_t d = o - base;
        if (d >= have) { /* skip ahead in the input */
          size_t skip = d - have;
          while (skip) { size_t k = skip < cap ? skip : cap; if (fread(buf, 4, k, stdin) != k) { fprintf(stderr, "pipe: eof\n"); return 1; } skip -= k; }
          have = 0;
        } else { memmove(buf, buf + d, sizeof(float) * (have - d)); have -= d; }
        base = o;
      }
      while (have < (size_t)NS) {
        size_t k = fread(buf + have, 4, cap - have, stdin);
        if (k == 0) { fprintf(stderr, "pipe: eof at window %d\n", i); return 1; }
        have += k;
      }
      if (bnn_infer(c, buf, lg, emb_out ? ev : NULL)) { fprintf(stderr, "infer failed: %s\n", bnn_last_error()); return 1; }
      size_t nw = emb_out ? (size_t)inf.emb_dim : (size_t)nc;
      if (fwrite(emb_out ? ev : lg, sizeof(float), nw, stdout) != nw) { fprintf(stderr, "pipe: write failed\n"); return 1; }
    }
    fflush(stdout);
    bnn_free(c);
    return 0;
  }
  if (!strcmp(argv[3], "--stream")) {
    size_t ns, no;
    /* the stream can be hours long and may sit on a shared (virtiofs) mount where mmap is unreliable while the host
       writes other files: read each window with pread */
    int fd = open(argv[4], O_RDONLY);
    struct stat sb;
    if (fd < 0 || fstat(fd, &sb)) { perror(argv[4]); return 1; }
    ns = (size_t)sb.st_size;
    float *x = malloc(sizeof(float) * NS);
    int32_t *off = slurp(argv[5], &no);
    size_t n = no / 4;
    FILE *o = fopen(argv[6], "wb");
    if (!o) { perror(argv[6]); return 1; }
    float *lg = malloc(sizeof(float) * nc);
    for (size_t i = 0; i < n; i++) {
      if ((size_t)off[i] + NS > ns / 4) { fprintf(stderr, "offset out of range\n"); return 1; }
      if (pread(fd, x, sizeof(float) * NS, (off_t)off[i] * 4) != (ssize_t)(sizeof(float) * NS)) { perror("pread"); return 1; }
      if (bnn_infer(c, x, lg, NULL)) { fprintf(stderr, "infer failed: %s\n", bnn_last_error()); return 1; }
      fwrite(lg, sizeof(float), nc, o);
    }
    fclose(o);
    bnn_free(c);
    return 0;
  }
  size_t nb;
  float *x = slurp(argv[3], &nb);
  int nw = (int)(nb / (sizeof(float) * NS));
  int reps = argc > 5 ? atoi(argv[5]) : 1, warm = argc > 6 ? atoi(argv[6]) : 0;
  float *lg = malloc(sizeof(float) * nc * nw);
  double *t = malloc(sizeof(double) * nw * reps);
  int nt = 0;
  for (int w = 0; w < warm; w++) bnn_infer(c, x + (size_t)(w % nw) * NS, lg, NULL);
  for (int r = 0; r < reps; r++)
    for (int i = 0; i < nw; i++) {
      double t0 = now_s();
      if (bnn_infer(c, x + (size_t)i * NS, lg + (size_t)i * nc, NULL)) { fprintf(stderr, "infer failed: %s\n", bnn_last_error()); return 1; }
      t[nt++] = (now_s() - t0) * 1e3;
    }
  qsort(t, nt, sizeof(double), cmpd);
  fprintf(stderr, "%s: %d windows x %d reps: median %.2f ms/window, min %.2f ms/window\n", inf.precision, nw, reps, t[nt / 2], t[0]);
  rss();
  FILE *o = fopen(argv[4], "wb");
  fwrite(lg, sizeof(float), (size_t)nc * nw, o);
  fclose(o);
  bnn_free(c);
  return 0;
}
