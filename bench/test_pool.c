// Stress test of the intra-op thread pool (src/rt_pool.c): pools are created, run and freed in a loop with changing
// thread counts; every chunk of every job must run exactly once, and no chunk may run after its job has returned
// (the job argument lives on the caller's stack). Run it plain and under -fsanitize=thread and -fsanitize=address:
//   make OUT=build build/test_pool && build/test_pool [iterations]
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/rt.h"

#define MAXC 64
typedef struct {
  atomic_int runs[MAXC];
  atomic_int open; /* 1 while rt_pool_run of this job has not returned */
  int n;
} jobarg;

static atomic_int g_late, g_range;

static void task(void *arg, int k, int worker) {
  jobarg *a = arg;
  (void)worker;
  if (!atomic_load(&a->open)) atomic_fetch_add(&g_late, 1);
  if (k < 0 || k >= a->n) { atomic_fetch_add(&g_range, 1); return; }
  atomic_fetch_add(&a->runs[k], 1);
  volatile unsigned x = 0;
  for (int i = 0; i < 200 * (k % 5); i++) x += (unsigned)i; /* uneven chunk lengths */
}

/* runs one job of n chunks and checks it; the argument dies when this returns */
static int run_job(rt_pool *p, int n) {
  jobarg a;
  memset(&a, 0, sizeof a);
  a.n = n;
  atomic_store(&a.open, 1);
  rt_pool_run(p, n, task, &a);
  atomic_store(&a.open, 0);
  int bad = 0;
  for (int k = 0; k < n; k++) bad += atomic_load(&a.runs[k]) != 1;
  return bad;
}

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 3000, bad = 0;
  unsigned seed = 12345;
  for (int it = 0; it < iters; it++) {
    seed = seed * 1103515245u + 12345u;
    int threads = 2 + (int)((seed >> 16) % 5), jobs = 1 + (int)((seed >> 8) % 4);
    rt_pool *p = rt_pool_new(threads);
    if (!p) { fprintf(stderr, "rt_pool_new(%d) failed\n", threads); return 2; }
    for (int j = 0; j < jobs; j++) {
      seed = seed * 1103515245u + 12345u;
      bad += run_job(p, 1 + (int)((seed >> 12) % MAXC));
    }
    rt_pool_free(p); /* straight after the last job: workers may still be inside their claim loop */
  }
  printf("%d pools: %d chunks not run exactly once, %d chunks run after their job returned, %d chunk indices out of range\n",
         iters, bad, atomic_load(&g_late), atomic_load(&g_range));
  return bad || atomic_load(&g_late) || atomic_load(&g_range);
}
