// front-end timing: bench_fe <weights.bn24> <windows.f32> [reps]  (build with -DFE_PROF)
#include <stdio.h>
#include <stdlib.h>
#include "../src/v24_fe.h"
extern double fe_prof[4];
int main(int argc, char **argv) {
  b24_model m;
  char err[256];
  if (b24_model_load(&m, argv[1], err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
  FILE *f = fopen(argv[2], "rb");
  float *x = malloc(sizeof(float) * 144000 * 8);
  if (fread(x, sizeof(float), 144000 * 8, f) != 144000 * 8) return 1;
  int reps = argc > 3 ? atoi(argv[3]) : 50;
  b24_fe *fe = b24_fe_new(&m);
  float *img = malloc(sizeof(float) * 96 * 511 * 2);
  double t0 = b24_now();
  for (int r = 0; r < reps; r++) b24_fe_run(fe, x + (size_t)(r % 8) * 144000, img);
  double t = (b24_now() - t0) / reps;
  printf("front end %.3f ms/window: load %.3f fft %.3f split %.3f mel+pow %.3f\n", t * 1e3, fe_prof[0] * 1e3 / reps,
         fe_prof[1] * 1e3 / reps, fe_prof[2] * 1e3 / reps, fe_prof[3] * 1e3 / reps);
  return 0;
}
