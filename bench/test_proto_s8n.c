// Test: the int8 prototype head without dotprod (proto_s8n.c) gives the SDOT head's column maxima (pc_proto_s8.c) bit
// for bit on the same int8 cells and prototypes, with the n=4 panel flags set by the compiler's rule (every int16
// lane's 4 weights per 32-byte step have sum |w| <= N4_LIMIT = 258), for any panel split. The cases include the ones
// a wrong rule gets wrong: cells of +-127 sign-aligned with a lane whose weights sum to exactly 258 (must be exact
// with the flag) and to 259-400 (must not be flagged), cells that the quantizer would turn into -128, K that needs
// padding (C = 1528) or switches the n=4 loop off (C = 1544, also with every flag forced on) and N that is not a
// multiple of 4. The loader's flag check (pc_proto_n4_check) must agree with the rule in every case.
// Needs a CPU with dotprod for the reference kernel (Cortex-A76, Apple silicon):
//   cc -O2 -march=armv8.2-a+dotprod+fp16 -o test_proto_s8n bench/test_proto_s8n.c src/proto_s8n.c src/pc_proto_s8.c -lm
// -DN4_LIMIT=330 builds the mutant (a rule that is too loose): the test must then FAIL.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/pc_kern.h"

#ifndef N4_LIMIT
#define N4_LIMIT 258
#endif
enum { P = 64, KMAX = 1552, NMAX = 1440 };

static int8_t cells[P][KMAX], protos[NMAX][KMAX];
static float sx[P], sw[NMAX + 12];
static int failures;

static int8_t rnd8(int amp) { return (int8_t)(rand() % (2 * amp + 1) - amp); }

/* the compiler's rule (compiler/lower.py n4_panel_flags) for prototypes rows [N][K] zero-padded to Np x Kp */
static int panel_flags(int N, int K, int Kp, uint8_t *n4) {
  int Np = (N + 3) & ~3, safe = 0;
  for (int pn = 0; pn < Np / 4; pn++) {
    int worst = 0;
    for (int j = 4 * pn; j < 4 * pn + 4 && j < N; j++)
      for (int g = 0; g + 32 <= Kp; g += 32)
        for (int l = 0; l < 8; l++) {
          int s = 0;
          for (int c = 0; c < 4; c++) s += g + 8 * c + l < K ? abs(protos[j][g + 8 * c + l]) : 0;
          if (s > worst) worst = s;
        }
    n4[pn] = Kp % 32 == 0 && worst <= N4_LIMIT;
    safe += n4[pn];
  }
  return safe;
}

/* both kernels on cells [P][C] and prototypes [N][C]; every panel split, flags on and off. force: every flag set to 1
 * (only valid where the kernel must ignore the flags: Kp % 32 != 0). The loader's flag check must accept the rule's
 * flags and, when a panel breaks the rule, refuse all-ones flags. */
static void compare_flags(const char *what, int C, int N, int force) {
  const int Kp = (C + 15) & ~15, Np = (N + 3) & ~3, N12 = (N + 11) / 12 * 12;
  int8_t *qa_dot = calloc((size_t)P * C, 1), *qw_dot = calloc((size_t)N12 * C, 1);
  int8_t *qa = calloc((size_t)P * Kp, 1), *qw = calloc((size_t)Np * Kp, 1);
  float *ref = calloc(N12, sizeof(float)), *out = calloc(Np, sizeof(float));
  uint8_t *n4 = calloc(Np / 4, 1);
  for (int p = 0; p < P; p++)
    for (int k = 0; k < C; k++) {
      qa_dot[((size_t)(p / 8) * (C / 4) + k / 4) * 32 + (p % 8) * 4 + k % 4] = cells[p][k];
      qa[(size_t)p * Kp + k] = cells[p][k];
    }
  for (int j = 0; j < N; j++)
    for (int k = 0; k < C; k++) {
      qw_dot[((size_t)(j / 12) * (C / 4) + k / 4) * 48 + (j % 12) * 4 + k % 4] = protos[j][k];
      qw[((size_t)(j / 4) * (Kp / 16) + k / 16) * 64 + (j % 4) * 16 + k % 16] = protos[j][k];
    }
  int safe = panel_flags(N, C, Kp, n4);
  int check_ok = pc_proto_n4_check(qw, Kp, n4, Np / 4) == -1;
  if (Kp % 32 == 0 && safe < Np / 4) {
    uint8_t *ones = malloc(Np / 4);
    memset(ones, 1, Np / 4);
    int first = pc_proto_n4_check(qw, Kp, ones, Np / 4);
    check_ok = check_ok && first >= 0 && !n4[first];
    free(ones);
  }
  if (force) memset(n4, 1, Np / 4);
  pc_proto_colmax_s8(qa_dot, sx, P, C, qw_dot, sw, N12, ref);
  static const int split[] = {1 << 30, 64, 7, 1};
  int bad = 0;
  for (unsigned s = 0; s < sizeof split / sizeof split[0]; s++)
    for (int use = 0; use < 2; use++) {
      memset(out, 0, sizeof(float) * Np);
      for (int p0 = 0; p0 < Np / 4;) {
        int p1 = Np / 4 - p0 > split[s] ? p0 + split[s] : Np / 4;
        pc_proto_colmax_s8n(qa, Kp, sx, P, qw, sw, use ? n4 : NULL, p0, p1, out);
        p0 = p1;
      }
      bad += memcmp(out, ref, sizeof(float) * N) != 0;
    }
  printf("%-58s C %4d N %4d, %3d of %3d panels flagged%s: %s%s\n", what, C, N, safe, Np / 4, force ? " (all forced on)" : "",
         bad ? "DIFFERENT" : "identical to the SDOT head", check_ok ? "" : "; LOADER CHECK WRONG");
  failures += bad != 0 || !check_ok;
  free(qa_dot); free(qw_dot); free(qa); free(qw); free(ref); free(out); free(n4);
}

static void compare(const char *what, int C, int N) { compare_flags(what, C, N, 0); }

static void random_scales(int N) {
  for (int p = 0; p < P; p++) sx[p] = 1e-4f * (float)(1 + rand() % 1000);
  for (int j = 0; j < N + 12; j++) sw[j] = 1e-3f * (float)(1 + rand() % 1000);
}

static void random_case(const char *what, int C, int N) {
  for (int p = 0; p < P; p++) {
    for (int k = 0; k < C; k++) cells[p][k] = rnd8(127);
    cells[p][rand() % C] = 127;
  }
  for (int j = 0; j < N; j++) {
    int c = (j / 4) % 3, amp = c == 0 ? 127 : c == 1 ? 70 : 20; /* per panel: large, borderline and n=4-safe weights */
    for (int k = 0; k < C; k++) protos[j][k] = rnd8(amp);
  }
  random_scales(N);
  compare(what, C, N);
}

/* every prototype is zero except one lane of one 32-byte step, whose 4 weights sum to `sum` (mixed signs); the cells
 * are +-127 with the weights' signs on that lane (and the opposite on odd cells), so the lane's partial sum reaches
 * +-127 * sum: exact up to 258, an int16 wrap beyond if the panel were flagged */
static void aligned_case(const char *what, int C, int N, int sum_lo, int sum_hi) {
  memset(protos, 0, sizeof protos);
  for (int j = 0; j < N; j++) {
    int sum = sum_lo + j % (sum_hi - sum_lo + 1), step = (j * 7) % (C / 32), lane = j % 8;
    int w[4] = {sum >= 127 ? 127 : sum, 0, 0, 0};
    w[1] = sum - w[0] >= 127 ? 127 : sum - w[0];
    w[2] = sum - w[0] - w[1] >= 127 ? 127 : sum - w[0] - w[1];
    w[3] = sum - w[0] - w[1] - w[2];
    for (int c = 0; c < 4; c++) protos[j][32 * step + 8 * c + lane] = (int8_t)((j + c) % 2 ? -w[c] : w[c]);
  }
  for (int k = 0; k < C; k++) {
    int sign = 1; /* the sign of the first prototype with a weight at k: that prototype is fully aligned */
    for (int j = 0; j < N; j++)
      if (protos[j][k]) { sign = protos[j][k] > 0 ? 1 : -1; break; }
    for (int p = 0; p < P; p++) cells[p][k] = (int8_t)((p % 2 ? -sign : sign) * 127);
  }
  random_scales(N);
  compare(what, C, N);
}

/* pc_quant_cells_f32: bytes in [-127, 127] even for a cell whose scale overflows, zero padding beyond C, and the
 * fp16 quantizer's bytes and scales on fp16-representable cells */
static void quantizer_case(void) {
  enum { C = 1528, Kp = 1536 };
  static float x[P][C];
  static f16 x16[P][C];
  static int8_t qa[P * Kp], qd[P * C];
  static float s32[P], s16[P];
  for (int p = 0; p < P; p++)
    for (int k = 0; k < C; k++) {
      x16[p][k] = (f16)((float)(rand() % 2001 - 1000) / 512.f);
      x[p][k] = (float)x16[p][k];
    }
  for (int k = 0; k < C; k++) x[1][k] = k % 2 ? -1e-38f : 1e-38f; /* 127 / max|x| is +inf: negatives would be -128 */
  for (int k = 0; k < C; k++) x[2][k] = 0.f;
  memset(qa, 0x55, sizeof qa);
  pc_quant_cells_f32(&x[0][0], P, C, C, 1e-5f, qa, Kp, s32);
  for (int k = 0; k < C; k++) { x16[1][k] = 0; x16[2][k] = 0; }
  pc_quant_cells_f16(&x16[0][0], P, C, C, 1e-5f, qd, s16);
  int min = 127, pad = 0, same = 1;
  for (int p = 0; p < P; p++) {
    for (int k = 0; k < Kp; k++) {
      int v = qa[(size_t)p * Kp + k];
      if (k >= C) pad |= v != 0;
      else if (v < min) min = v;
      if (k < C && p != 1 && v != qd[((size_t)(p / 8) * (C / 4) + k / 4) * 32 + (p % 8) * 4 + k % 4]) same = 0;
    }
    if (p != 1 && memcmp(&s32[p], &s16[p], sizeof(float))) same = 0;
  }
  int ok = min >= -127 && !pad && same && isfinite(s32[1]);
  printf("%-58s smallest byte %d, padding %s, fp16 quantizer %s: %s\n", "quantizer: tiny cell, zero cell, K padding", min,
         pad ? "NOT zero" : "zero", same ? "equal" : "DIFFERENT", ok ? "ok" : "WRONG");
  failures += !ok;
  /* the quantized tiny cell through both kernels (its bytes are +-127: the worst case for a flagged panel) */
  for (int p = 0; p < P; p++)
    for (int k = 0; k < C; k++) cells[p][k] = qa[(size_t)(p % 2 ? 1 : p) * Kp + k];
}

int main(void) {
  srand(7);
  random_case("random cells, weights large / borderline / small per panel", 1536, 1440);
  random_case("K padded to 16 (C = 1528)", 1528, 1440);
  random_case("Kp not a multiple of 32 (C = 1544): n=4 loop off", 1544, 1440);
  compare_flags("the same with every flag forced on (must be ignored)", 1544, 1440, 1);
  random_case("N not a multiple of 4", 1536, 1438);
  aligned_case("sign-aligned +-127 cells, lane sums exactly 258", 1536, 1440, 258, 258);
  aligned_case("sign-aligned +-127 cells, lane sums 200-400", 1536, 1440, 200, 400);
  aligned_case("sign-aligned +-127 cells, lane sums 255-262, C = 1528", 1528, 1438, 255, 262);
  quantizer_case();
  for (int j = 0; j < NMAX; j++)
    for (int k = 0; k < 1528; k++) protos[j][k] = rnd8(j % 8 < 4 ? 60 : 127);
  random_scales(NMAX);
  compare("cells from the quantizer (one of denormal size), K padded", 1528, 1440);
  printf("%s\n", failures ? "FAIL" : "PASS");
  return failures != 0;
}
