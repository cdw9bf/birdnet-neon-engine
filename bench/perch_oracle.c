// Perch v2 fp32 oracle CLI (development tool, not part of libbnneon): runs the readable C reference (src/perch_ref.c) on a
// .bnp2 file exported by scripts/export_perch.py, optionally with fake quantization (env PERCH_Q, PERCH_FLUSH).
//   perch_oracle <weights.bnp2> <windows.f32> <logits.f32> [emb.f32]
#include <stdio.h>
#include <stdlib.h>

#include "../src/perch_internal.h"

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: perch_oracle <weights.bnp2> <windows.f32> <logits.f32> [emb.f32]\n"); return 2; }
  pc_model m;
  char err[256];
  if (pc_model_load(&m, argv[1], err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
  pc_ref *r = pc_ref_new(&m);
  FILE *fi = fopen(argv[2], "rb"), *fo = fopen(argv[3], "wb"), *fe = argc > 4 ? fopen(argv[4], "wb") : NULL;
  if (!fi || !fo) { perror("open"); return 1; }
  float *x = malloc(sizeof(float) * PC_SAMPLES), *lg = malloc(sizeof(float) * m.n_classes), emb[PC_EMB];
  int n = 0;
  double t0 = pc_now();
  while (fread(x, sizeof(float), PC_SAMPLES, fi) == PC_SAMPLES) {
    pc_ref_infer(r, x, lg, emb);
    fwrite(lg, sizeof(float), m.n_classes, fo);
    if (fe) fwrite(emb, sizeof(float), PC_EMB, fe);
    n++;
  }
  fprintf(stderr, "oracle: %d windows, %.1f ms/window\n", n, 1e3 * (pc_now() - t0) / (n ? n : 1));
  fclose(fi); fclose(fo);
  if (fe) fclose(fe);
  pc_ref_free(r);
  pc_model_unload(&m);
  return 0;
}
