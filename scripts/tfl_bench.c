// TFLite C-API benchmark and gate scorer, set up the way BirdNET-Go's TFLite route sets up its interpreter
// (internal/inference/tflite/classifier.go): TfLiteXNNPackDelegateOptionsDefault() with only num_threads changed
// (default delegate flags), the delegate added to the interpreter options, interpreter threads 1. BirdNET-Go passes
// max(1, threads-1) as the delegate thread count; this harness takes the delegate thread count directly.
// Link against libtensorflowlite_c 2.17.1 (github.com/tphakala/tflite_c, the library BirdNET-Go's image ships).
//
// bench: tfl_bench bench MODEL WIN.f32 NWIN RUNS WARM THREADS [--out NAME] [--head HEAD.tflite --emb NAME]
//   WIN.f32 = NWIN windows of the model's input length (raw float32). Run i uses window i % NWIN (as pi_prof.py).
//   --head: two-stage pipeline (BattyBirdNET): output NAME of MODEL (the embedding) feeds HEAD's input; the timed
//   unit is backbone + head. Prints one RESULT line: median / min / p90 ms, peak RSS (VmHWM) in MB.
// score: tfl_bench score MODEL THREADS [--out NAME] (--pipe | IN.f32 OFF.i32 OUT.f32)
//   IN = the whole stream (float32), OFF = int32 window offsets in samples, OUT = n x D float32 of the selected output
//   (modelgate.py extern file contract). --pipe: int32 n, int32 offsets[n], then the float32 stream
//   on stdin until EOF; n x D float32 on stdout (the scripts/pipe_score.py contract).
// --flags HEX: set the XNNPACK delegate flags instead of the default (0x3 = QS8|QU8 in 2.17.1); BirdNET-Go cannot do
//   this today (go-tflite's DelegateOptions has NumThreads only). 0x0 = fp32 delegate only (int8 ops on TFLite kernels).
// Output selection: --out NAME picks the output tensor whose name starts with NAME; else an output named
// "predictions*", else output 0.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static long peak_rss_mb(void) {
  FILE *f = fopen("/proc/self/status", "r"); char line[256]; long kb = -1;
  if (!f) return -1;
  while (fgets(line, sizeof line, f)) if (sscanf(line, "VmHWM: %ld kB", &kb) == 1) break;
  fclose(f); return kb < 0 ? -1 : kb / 1024;
}
static void die(const char *m) { fprintf(stderr, "tfl_bench: %s\n", m); exit(1); }

typedef struct { TfLiteModel *m; TfLiteInterpreterOptions *o; TfLiteDelegate *d; TfLiteInterpreter *it;
                 TfLiteTensor *in; const TfLiteTensor *out; size_t inlen, outlen; } Net;

static size_t numel(const TfLiteTensor *t) { size_t n = 1; for (int d = 0; d < TfLiteTensorNumDims(t); d++) n *= TfLiteTensorDim(t, d); return n; }

static long xnn_flags = -1;  // -1 = the library default
static void net_open(Net *n, const char *path, int threads, const char *outname, int verbose) {
  n->m = TfLiteModelCreateFromFile(path); if (!n->m) die("model load failed");
  n->o = TfLiteInterpreterOptionsCreate();
  TfLiteXNNPackDelegateOptions xo = TfLiteXNNPackDelegateOptionsDefault();
  xo.num_threads = threads;
  if (xnn_flags >= 0) xo.flags = (uint32_t)xnn_flags;
  n->d = TfLiteXNNPackDelegateCreate(&xo);
  if (n->d) { TfLiteInterpreterOptionsAddDelegate(n->o, n->d); TfLiteInterpreterOptionsSetNumThreads(n->o, 1); }
  else { fprintf(stderr, "tfl_bench: XNNPACK delegate create failed, CPU kernels\n"); TfLiteInterpreterOptionsSetNumThreads(n->o, threads); }
  if (verbose) fprintf(stderr, "tflite %s, xnnpack flags 0x%x, delegate threads %d, interpreter threads %d: %s\n",
                       TfLiteVersion(), xo.flags, threads, n->d ? 1 : threads, path);
  n->it = TfLiteInterpreterCreate(n->m, n->o); if (!n->it) die("interpreter create failed");
  if (TfLiteInterpreterAllocateTensors(n->it) != kTfLiteOk) die("allocate failed");
  n->in = TfLiteInterpreterGetInputTensor(n->it, 0);
  int no = TfLiteInterpreterGetOutputTensorCount(n->it); n->out = NULL;
  for (int k = 0; k < no && !n->out; k++) {
    const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(n->it, k); const char *nm = TfLiteTensorName(t);
    if (outname && strncmp(nm, outname, strlen(outname)) == 0) n->out = t;
  }
  for (int k = 0; k < no && !n->out && !outname; k++) {
    const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(n->it, k);
    if (strncmp(TfLiteTensorName(t), "predictions", 11) == 0) n->out = t;
  }
  if (!n->out && outname) die("no output with that name");
  if (!n->out) n->out = TfLiteInterpreterGetOutputTensor(n->it, 0);
  n->inlen = numel(n->in); n->outlen = numel(n->out);
  if (TfLiteTensorType(n->in) != kTfLiteFloat32 || TfLiteTensorType(n->out) != kTfLiteFloat32) die("expect float32 io");
  if (verbose) fprintf(stderr, "input %s len %zu, output %s len %zu (%d outputs)\n", TfLiteTensorName(n->in), n->inlen,
                       TfLiteTensorName(n->out), n->outlen, no);
}
static void net_close(Net *n) { TfLiteInterpreterDelete(n->it); TfLiteInterpreterOptionsDelete(n->o); if (n->d) TfLiteXNNPackDelegateDelete(n->d); TfLiteModelDelete(n->m); }
static void net_run(Net *n, const float *x) {
  if (TfLiteTensorCopyFromBuffer(n->in, x, n->inlen * sizeof(float)) != kTfLiteOk) die("copy in failed");
  if (TfLiteInterpreterInvoke(n->it) != kTfLiteOk) die("invoke failed");
}

static const char *opt(int argc, char **argv, const char *k) { for (int i = 0; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return NULL; }
static int has(int argc, char **argv, const char *k) { for (int i = 0; i < argc; i++) if (!strcmp(argv[i], k)) return 1; return 0; }

static int bench(int argc, char **argv) {
  if (argc < 8) die("bench MODEL WIN.f32 NWIN RUNS WARM THREADS [--out NAME] [--head HEAD --emb NAME]");
  const char *mp = argv[2], *wp = argv[3]; int nwin = atoi(argv[4]), runs = atoi(argv[5]), warm = atoi(argv[6]), thr = atoi(argv[7]);
  const char *outname = opt(argc, argv, "--out"), *head = opt(argc, argv, "--head"), *emb = opt(argc, argv, "--emb");
  Net a, h; net_open(&a, mp, thr, head ? emb : outname, 1);
  if (head) { net_open(&h, head, thr, NULL, 1); if (h.inlen != a.outlen) die("head input != embedding width"); }
  float *X = malloc((size_t)nwin * a.inlen * sizeof(float)); FILE *f = fopen(wp, "rb"); if (!f) die("windows open failed");
  if (fread(X, sizeof(float), (size_t)nwin * a.inlen, f) != (size_t)nwin * a.inlen) die("short windows file");
  fclose(f);
  double *ts = malloc(runs * sizeof(double)), t0;
  for (int i = 0; i < warm + runs; i++) {
    t0 = now_ms(); net_run(&a, X + (size_t)(i % nwin) * a.inlen);
    if (head) net_run(&h, TfLiteTensorData(a.out));
    if (i >= warm) ts[i - warm] = now_ms() - t0;
  }
  qsort(ts, runs, sizeof(double), cmpd);
  const Net *o = head ? &h : &a; const float *y = TfLiteTensorData(o->out);
  printf("RESULT model=%s%s%s threads=%d runs=%d warm=%d nwin=%d median=%.2f min=%.2f p90=%.2f peakRSS=%ldMB out0=%.5f out1=%.5f\n",
         mp, head ? "+" : "", head ? head : "", thr, runs, warm, nwin, ts[runs / 2], ts[0], ts[(runs * 9) / 10], peak_rss_mb(), y[0], y[1]);
  if (head) net_close(&h);
  net_close(&a); free(X); free(ts);
  return 0;
}

static int score(int argc, char **argv) {
  if (argc < 4) die("score MODEL THREADS [--out NAME] (--pipe | IN OFF OUT)");
  Net a; net_open(&a, argv[2], atoi(argv[3]), opt(argc, argv, "--out"), 1);
  int pipe = has(argc, argv, "--pipe"); int32_t n = 0; int32_t *off = NULL; float *x = NULL; size_t ns = 0, cap = 0;
  FILE *fs, *fo; const char *pos[3]; int np = 0;
  for (int i = 4; i < argc; i++) { if (!strcmp(argv[i], "--out")) { i++; continue; } if (!strcmp(argv[i], "--pipe")) continue; if (np < 3) pos[np++] = argv[i]; }
  if (pipe) {
    fs = stdin; fo = stdout;
    if (fread(&n, 4, 1, fs) != 1) die("pipe: no count");
    off = malloc((size_t)n * 4); if (fread(off, 4, n, fs) != (size_t)n) die("pipe: short offsets");
  } else {
    if (np != 3) die("score: need IN OFF OUT");
    FILE *fo2 = fopen(pos[1], "rb"); if (!fo2) die("offsets open failed");
    fseek(fo2, 0, SEEK_END); n = (int32_t)(ftell(fo2) / 4); rewind(fo2);
    off = malloc((size_t)n * 4); if (fread(off, 4, n, fo2) != (size_t)n) die("short offsets"); fclose(fo2);
    fs = fopen(pos[0], "rb"); fo = fopen(pos[2], "wb"); if (!fs || !fo) die("io open failed");
  }
  for (;;) {  // read the whole stream
    if (ns + (1 << 20) > cap) { cap = cap ? cap * 2 : (1 << 24); x = realloc(x, cap * sizeof(float)); if (!x) die("oom"); }
    size_t r = fread(x + ns, sizeof(float), 1 << 20, fs); ns += r; if (r == 0) break;
  }
  float *buf = calloc(a.inlen, sizeof(float));
  for (int32_t i = 0; i < n; i++) {
    size_t o = (size_t)off[i];
    if (o + a.inlen <= ns) net_run(&a, x + o);
    else { memset(buf, 0, a.inlen * sizeof(float)); if (o < ns) memcpy(buf, x + o, (ns - o) * sizeof(float)); net_run(&a, buf); }  // zero-pad a short tail
    if (fwrite(TfLiteTensorData(a.out), sizeof(float), a.outlen, fo) != a.outlen) die("write failed");
  }
  fflush(fo); if (!pipe) { fclose(fo); fclose(fs); }
  fprintf(stderr, "scored %d windows of %zu samples from %zu samples\n", n, a.inlen, ns);
  net_close(&a); free(x); free(off); free(buf);
  return 0;
}

int main(int argc, char **argv) {
  const char *fl = opt(argc, argv, "--flags"); if (fl) xnn_flags = strtol(fl, NULL, 16);
  if (argc > 1 && !strcmp(argv[1], "bench")) return bench(argc, argv);
  if (argc > 1 && !strcmp(argv[1], "score")) return score(argc, argv);
  fprintf(stderr, "usage: tfl_bench bench|score ... (see the header)\n"); return 2;
}
