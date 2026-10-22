// mg_pool.cpp — persistent worker pool behind parallel_for / parallel_items
// (spin, then park). Starting fresh threads on every loop cost more than
// the loop itself on small graphs.
// SPDX-License-Identifier: Apache-2.0

#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include "mg_common.hpp"

namespace mg::detail {
namespace {

inline void cpu_relax() {
#if defined(__aarch64__)
  __asm__ __volatile__("yield");
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#endif
}

// How long a worker spins for its next job after finishing one before it
// waits: covers the gaps between an iterative op's back-to-back loops. The
// wait itself (libc++ atomic::wait) polls, yielding after 4 us, and only
// parks after 64 us, so an idle worker stays busy for ~165 us after each
// job. M4 Max, rmat14/15 CPU PageRank, HITS and PPR: no spin was 15-25%
// slower; 40 us cost HITS up to 30% in some rounds.
constexpr auto k_spin = std::chrono::microseconds(100);

// Set on pool workers, fallback threads, and a caller mid-dispatch: a
// nested parallel loop runs inline (the outer one already uses the threads).
thread_local bool t_in_region = false;

// One dispatch: tasks [0, n) claimed one index at a time by the caller and
// every helper that took the job.
struct alignas(128) Job {
  TaskFn task;
  const void* ctx;
  std::size_t n;
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr error{};  // written once, by the thread that set failed
};

// Never throws: a task's exception is stored and ends further claims.
void run_claims(Job& job) {
  for (;;) {
    const std::size_t i = job.next.fetch_add(1, std::memory_order_relaxed);
    if (i >= job.n) return;
    try {
      job.task(job.ctx, i);
    } catch (...) {
      if (!job.failed.exchange(true, std::memory_order_relaxed))
        job.error = std::current_exception();
      job.next.store(job.n, std::memory_order_relaxed);
      return;
    }
  }
}

// A worker's mailbox. The caller posts a job (kIdle -> kPosted); the worker
// takes it (kPosted -> kTaken) and returns the slot to kIdle once its claims
// run dry. A caller that runs out of tasks first revokes a job still posted
// (kPosted -> kIdle), so a worker that wakes late never touches a finished
// job and the caller never waits for a wakeup. Both transitions out of
// kPosted are CASes: exactly one side wins.
enum : uint64_t { kIdle = 0, kPosted = 1, kTaken = 2 };

struct alignas(128) Slot {
  std::atomic<uint64_t> state{kIdle};  // 64-bit: libc++ waits on it natively
  Job* job = nullptr;  // written while kIdle; read only by the taker
};

// Workers beyond this are not started; the caller and the started workers
// claim their share. Static, so a worker's slot and its children's always
// exist.
constexpr std::size_t k_max_workers = 256;
Slot g_slots[k_max_workers];

// Parked workers are woken down a binary tree: the caller wakes slot 0, and
// a worker that finds its slot posted wakes slots 2i+1 and 2i+2 before
// taking it. The caller pays for at most one wakeup (waking 15 parked
// workers from the caller took 20-40 us), and a notify finds no waiter, so
// makes no syscall, when the child is spinning. Jobs are posted from the
// highest slot down, so a worker that sees its own post sees its children's.
void wake_children(std::size_t i) {
  for (std::size_t c = 2 * i + 1; c <= 2 * i + 2 && c < k_max_workers; ++c)
    if (g_slots[c].state.load(std::memory_order_relaxed) == kPosted)
      g_slots[c].state.notify_one();
}

// Returns once slot i is (or was just) kPosted: true if seen while spinning
// (up to k_spin, when `spin`), false if seen in wait() (which polls for up
// to 64 us before parking). wait() rechecks the value, so a post between the
// load and the wait is never missed. The acquire load orders the children's
// (earlier) posts before wake_children.
bool wait_posted(std::size_t i, bool spin) {
  Slot& s = g_slots[i];
  if (spin) {
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned n = 1;; ++n) {
      cpu_relax();
      if (s.state.load(std::memory_order_acquire) == kPosted) return true;
      if ((n & 63) == 0 && std::chrono::steady_clock::now() - t0 > k_spin)
        break;
    }
  }
  for (;;) {
    const uint64_t st = s.state.load(std::memory_order_acquire);
    if (st == kPosted) return false;
    s.state.wait(st, std::memory_order_relaxed);
  }
}

void worker_main(std::size_t i) {
  t_in_region = true;
#if defined(__APPLE__)
  pthread_setname_np("mg-parallel");
#endif
  Slot& s = g_slots[i];
  // Spin after a job; a worker woken from park into an already revoked job
  // parks again straight away.
  bool spin = true;
  for (;;) {
    const bool hot = wait_posted(i, spin);
    wake_children(i);
    uint64_t expect = kPosted;
    if (!s.state.compare_exchange_strong(expect, kTaken,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed)) {
      spin = hot;  // revoked by a caller that ran out of tasks
      continue;
    }
    run_claims(*s.job);
    s.state.store(kIdle, std::memory_order_release);
    spin = true;
  }
}

class Pool {
 public:
  // Runs job on the caller plus up to `helpers` workers. False, with
  // nothing run, when another thread is mid-dispatch or this is a forked
  // child (the workers did not survive the fork).
  bool run(Job& job, std::size_t helpers) {
    if (getpid() != pid_) return false;
    std::unique_lock<std::mutex> lk(run_mutex_, std::try_to_lock);
    if (!lk.owns_lock()) return false;
    grow(std::min(helpers, k_max_workers));
    const std::size_t used = std::min(helpers, started_);
    for (std::size_t i = used; i-- > 0;) {
      g_slots[i].job = &job;
      g_slots[i].state.store(kPosted, std::memory_order_release);
    }
    if (used > 0) g_slots[0].state.notify_one();
    t_in_region = true;
    run_claims(job);
    for (std::size_t i = 0; i < used; ++i) retire(g_slots[i]);
    t_in_region = false;
    return true;
  }

 private:
  // Starts workers until `want` exist. A failure to start one leaves the
  // pool smaller: the caller claims whatever no worker takes.
  void grow(std::size_t want) {
    try {
      for (; started_ < want; ++started_)
        std::thread(worker_main, started_).detach();
    } catch (...) {
    }
  }

  // Revokes s's job if no worker took it; else waits until the worker is
  // done with it. No claims remain, but the worker may still be inside its
  // last task, so the wait lasts as long as the slowest in-flight task
  // (one dense PPR query, say). Spins, then yields: a worker preempted by
  // other load gets the core. Blocking instead (atomic wait + a notify from
  // every finishing worker) made short dispatches 12-60% slower: most
  // notifies then cost a wake syscall.
  static void retire(Slot& s) {
    uint64_t expect = kPosted;
    if (s.state.compare_exchange_strong(expect, kIdle,
                                        std::memory_order_relaxed))
      return;
    unsigned spins = 0;
    while (s.state.load(std::memory_order_acquire) != kIdle) {
      if (++spins < 4096)
        cpu_relax();
      else
        std::this_thread::yield();
    }
  }

  const pid_t pid_ = getpid();
  std::mutex run_mutex_;
  std::size_t started_ = 0;  // workers on g_slots[0, started_)
};

// Concurrent callers and forked children: fresh threads for this call only.
void run_spawned(Job& job, std::size_t helpers) {
  std::vector<std::thread> ts;
  try {
    ts.reserve(helpers);
    for (std::size_t h = 0; h < helpers; ++h)
      ts.emplace_back([&job] {
        t_in_region = true;
        run_claims(job);
      });
  } catch (...) {  // fewer threads: the caller claims what they leave
  }
  t_in_region = true;
  run_claims(job);
  t_in_region = false;
  for (auto& t : ts) t.join();
}

}  // namespace

void run_tasks(std::size_t n, std::size_t helpers, TaskFn task,
               const void* ctx) {
  if (t_in_region || helpers == 0 || n <= 1) {
    for (std::size_t i = 0; i < n; ++i) task(ctx, i);
    return;
  }
  Job job{task, ctx, n};
  helpers = std::min(helpers, n - 1);
  static Pool* const pool = new Pool;  // leaked: workers outlive static dtors
  if (!pool->run(job, helpers)) run_spawned(job, helpers);
  if (job.failed.load(std::memory_order_relaxed))
    std::rethrow_exception(job.error);
}

}  // namespace mg::detail
