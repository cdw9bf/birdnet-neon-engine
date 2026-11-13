// Intra-op thread pool for one runtime context: threads - 1 workers plus the calling thread. A job is a function over
// chunks 0..n-1 taken from an atomic counter (dynamic: a loaded core takes fewer chunks). Kernels split only over
// independent output elements, so which thread computes a chunk never changes a result: any thread count is
// bit-identical to one. Workers spin briefly on the job generation, then sleep on a condition variable (ops inside
// a window run back to back; windows are seconds apart and the host shares the cores).
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <time.h>

#include "rt.h"

/* A job is published as (sequence << 32 | next chunk) in one 64-bit word; its descriptor lives in slot seq & 1. A
 * worker claims a chunk by a CAS on the word, so a claim can only succeed for the job whose descriptor it read (a
 * stale read fails the CAS and is retried), and the caller returns only when every chunk of its job is done. */
/* fn, arg and n are atomics only because a worker may read a slot while the caller rewrites it for a later job (it
 * then drops what it read: its CAS on the word fails). Relaxed loads and stores: plain moves on AArch64. */
typedef struct {
  _Atomic(rt_task_fn) fn;
  _Atomic(void *) arg;
  atomic_int n;
  atomic_int done;
} job;

struct rt_pool {
  int n;                      /* threads including the caller */
  pthread_t *th;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  _Atomic unsigned long long word;
  job jobs[2];
  atomic_int sleeping;
  atomic_int quit;
};

typedef struct { rt_pool *p; int id; } worker_arg;

static double mono_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

/* claims and runs chunks of the current job until none is left */
static void work(rt_pool *p, int id) {
  unsigned long long v = atomic_load_explicit(&p->word, memory_order_acquire);
  for (;;) {
    if (atomic_load_explicit(&p->quit, memory_order_acquire)) return; /* shutting down: no job is open */
    job *j = &p->jobs[(v >> 32) & 1];
    int n = atomic_load_explicit(&j->n, memory_order_relaxed);
    rt_task_fn fn = atomic_load_explicit(&j->fn, memory_order_relaxed);
    void *arg = atomic_load_explicit(&j->arg, memory_order_relaxed);
    if ((int)(v & 0xffffffffu) >= n) return;
    if (!atomic_compare_exchange_weak_explicit(&p->word, &v, v + 1, memory_order_acq_rel, memory_order_acquire)) continue;
    fn(arg, (int)(v & 0xffffffffu), id);
    atomic_fetch_add_explicit(&j->done, 1, memory_order_release);
    v = atomic_load_explicit(&p->word, memory_order_acquire);
  }
}

#define SPIN_US 60.0
static void *worker(void *a) {
  rt_pool *p = ((worker_arg *)a)->p;
  int id = ((worker_arg *)a)->id;
  free(a);
  unsigned seen = (unsigned)(atomic_load(&p->word) >> 32);
  for (;;) {
    double t0 = mono_us();
    unsigned g;
    int spins = 0;
    while ((g = (unsigned)(atomic_load_explicit(&p->word, memory_order_acquire) >> 32)) == seen && !atomic_load(&p->quit)) {
      if (++spins % 64 == 0 && mono_us() - t0 > SPIN_US) {
        pthread_mutex_lock(&p->mu);
        atomic_fetch_add(&p->sleeping, 1);
        while ((g = (unsigned)(atomic_load(&p->word) >> 32)) == seen && !atomic_load(&p->quit)) pthread_cond_wait(&p->cv, &p->mu);
        atomic_fetch_sub(&p->sleeping, 1);
        pthread_mutex_unlock(&p->mu);
        break;
      }
#if defined(__aarch64__)
      __asm__ volatile("yield");
#endif
    }
    if (atomic_load(&p->quit)) return NULL;
    seen = g;
    work(p, id);
  }
}

rt_pool *rt_pool_new(int n) {
  if (n < 2) return NULL;
  rt_pool *p = calloc(1, sizeof *p);
  if (!p) return NULL;
  p->n = n;
  p->th = calloc((size_t)n, sizeof(pthread_t));
  pthread_mutex_init(&p->mu, NULL);
  pthread_cond_init(&p->cv, NULL);
  atomic_init(&p->word, 0ull);
  atomic_init(&p->jobs[0].done, 0);
  atomic_init(&p->jobs[1].done, 0);
  atomic_init(&p->sleeping, 0);
  if (!p->th) { free(p); return NULL; }
  for (int i = 1; i < n; i++) {
    worker_arg *a = malloc(sizeof *a);
    if (!a) { p->n = i; break; }
    a->p = p; a->id = i;
    if (pthread_create(&p->th[i], NULL, worker, a)) { free(a); p->n = i; break; }
  }
  if (p->n < 2) { rt_pool_free(p); return NULL; }
  return p;
}

void rt_pool_free(rt_pool *p) {
  if (!p) return;
  pthread_mutex_lock(&p->mu);
  /* Wake the workers with a new sequence whose chunk index is past every job size. Adding 1 << 32 alone would keep
   * the old index and point the sequence at the other slot, which still holds the previous job's descriptor: a worker
   * inside work() could then claim one of its chunks and call it with that job's dead argument. (The index is read
   * as an int, hence 0x7fffffff.) */
  atomic_store(&p->quit, 1);
  atomic_store(&p->word, (((atomic_load(&p->word) >> 32) + 1) << 32) | 0x7fffffffull);
  pthread_cond_broadcast(&p->cv);
  pthread_mutex_unlock(&p->mu);
  for (int i = 1; i < p->n; i++) pthread_join(p->th[i], NULL);
  pthread_mutex_destroy(&p->mu);
  pthread_cond_destroy(&p->cv);
  free(p->th);
  free(p);
}

int rt_pool_size(const rt_pool *p) { return p ? p->n : 1; }

void rt_pool_run(rt_pool *p, int nchunks, rt_task_fn fn, void *arg) {
  if (nchunks <= 0) return;
  if (!p || nchunks == 1) {
    for (int k = 0; k < nchunks; k++) fn(arg, k, 0);
    return;
  }
  unsigned long long seq = (atomic_load(&p->word) >> 32) + 1;
  job *j = &p->jobs[seq & 1];
  atomic_store_explicit(&j->fn, fn, memory_order_relaxed);
  atomic_store_explicit(&j->arg, arg, memory_order_relaxed);
  atomic_store_explicit(&j->n, nchunks, memory_order_relaxed);
  atomic_store_explicit(&j->done, 0, memory_order_relaxed);
  atomic_store(&p->word, seq << 32); /* publishes the descriptor (sequentially consistent with the sleeping check) */
  if (atomic_load(&p->sleeping)) {
    pthread_mutex_lock(&p->mu);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
  }
  work(p, 0);
  while (atomic_load_explicit(&j->done, memory_order_acquire) < nchunks) {
#if defined(__aarch64__)
    __asm__ volatile("yield");
#endif
  }
}
