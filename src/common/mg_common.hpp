// mg_common.hpp — shared host-side basics: errors, env helpers, threading.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mg {

// Error codes mirror mg_status in include/mg.h (keep in sync).
enum class ErrorCode : int {
  ok = 0,
  invalid_argument = 1,
  out_of_memory = 2,
  no_gpu = 3,
  internal = 4,
  not_implemented = 5,
};

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string& msg)
      : std::runtime_error(msg), code_(code) {}
  ErrorCode code() const { return code_; }

 private:
  ErrorCode code_;
};

[[noreturn]] inline void throw_invalid(const std::string& msg) {
  throw Error(ErrorCode::invalid_argument, msg);
}

inline const char* env_str(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return (v && *v) ? v : fallback;
}

inline long env_long(const char* name, long fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  char* end = nullptr;
  long r = std::strtol(v, &end, 10);
  return (end && *end == '\0') ? r : fallback;
}

inline bool env_flag(const char* name, bool fallback = false) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  return !(v[0] == '0' && v[1] == '\0');
}

inline unsigned num_threads() {
  long t = env_long("MG_THREADS", 0);
  if (t > 0) return static_cast<unsigned>(t);
  unsigned hc = std::thread::hardware_concurrency();
  return hc ? hc : 4;
}

// Work units (one O(1) item, or one CSR edge) each parallel_for chunk must
// carry before handing it to another thread pays off. The old fixed
// 4096-item grain spawned fresh threads for any loop over 4096 items, so
// small graphs ran slower with all threads than with MG_THREADS=1.
constexpr std::size_t k_par_work_grain = std::size_t{1} << 15;

// Chunks per thread for loops whose chunks vary in cost (degree-skewed
// rows, targets already visited, efficiency cores): the chunks are claimed
// one at a time, so faster threads take more of them. M4 Max CPU BFS on
// rmat18-22: 8 per thread cut time 6-30% against 1; 16 was no better.
constexpr std::size_t k_par_balance = 8;

// Chunks parallel_for splits n items into: at most per_thread *
// num_threads(), each of at least `grain` items. 1 means the loop runs
// inline on the caller.
inline std::size_t parallel_chunks(std::size_t n,
                                   std::size_t grain = k_par_work_grain,
                                   std::size_t per_thread = 1) {
  const std::size_t chunks = std::min<std::size_t>(
      per_thread * num_threads(), n / std::max<std::size_t>(grain, 1));
  return std::max<std::size_t>(chunks, 1);
}

// parallel_for grain for n items doing `work` units in total (rows plus
// their edges, say), so each chunk carries ~k_par_work_grain units.
inline std::size_t work_grain(std::size_t n, uint64_t work) {
  if (work <= k_par_work_grain) return std::max<std::size_t>(n, 1);  // inline
  const uint64_t g = uint64_t(n) * k_par_work_grain / work;
  return static_cast<std::size_t>(std::max<uint64_t>(g, 1));
}

namespace detail {
using TaskFn = void (*)(const void* ctx, std::size_t i);
// Calls task(ctx, i) once for every i in [0, n): the calling thread and up
// to `helpers` other threads claim indices from a shared counter, so which
// thread runs an index is unspecified. Helpers are persistent pool workers
// (mg_pool.cpp). A call from inside a task runs inline; a call the pool
// cannot take (another thread mid-dispatch, or a forked child) runs on
// freshly started threads. The first exception a task throws stops further
// claims and is rethrown here once every helper has finished.
void run_tasks(std::size_t n, std::size_t helpers, TaskFn task,
               const void* ctx);

// fn(begin, end) over `chunks` equal chunks of [0, n), claimed by the caller
// and pool workers; inline for one chunk.
inline void run_chunks(std::size_t n, std::size_t chunks,
                       const std::function<void(std::size_t, std::size_t)>& fn) {
  if (chunks <= 1) {
    fn(0, n);
    return;
  }
  struct Ctx {
    const std::function<void(std::size_t, std::size_t)>* fn;
    std::size_t n, per;
  } ctx{&fn, n, (n + chunks - 1) / chunks};
  run_tasks(
      chunks, std::min<std::size_t>(chunks, num_threads()) - 1,
      [](const void* p, std::size_t c) {
        const Ctx& x = *static_cast<const Ctx*>(p);
        const std::size_t b = c * x.per;
        if (b < x.n) (*x.fn)(b, std::min(x.n, b + x.per));
      },
      &ctx);
}
}  // namespace detail

// Chunked parallel for over [0, n): fn(begin, end) per chunk, chunks claimed
// by the caller and pool workers. `grain` is the minimum items per chunk;
// the default suits O(1) items, and loops with heavier items pass
// work_grain(). Runs inline when there is under two chunks' work or one
// thread is available.
inline void parallel_for(std::size_t n,
                         const std::function<void(std::size_t, std::size_t)>& fn,
                         std::size_t grain = k_par_work_grain) {
  if (n == 0) return;
  detail::run_chunks(n, parallel_chunks(n, grain), fn);
}

// parallel_for split into up to k_par_balance chunks per thread, for loops
// whose chunks vary in cost.
inline void parallel_for_balanced(
    std::size_t n, const std::function<void(std::size_t, std::size_t)>& fn,
    std::size_t grain = k_par_work_grain) {
  if (n == 0) return;
  detail::run_chunks(n, parallel_chunks(n, grain, k_par_balance), fn);
}

// Dynamically scheduled loop over items [0, n): fn(i) per item. The caller
// and up to num_threads()-1 pool workers claim items one at a time from a
// shared counter, so faster (performance) cores take more items. Results
// must not depend on which thread runs an item. Runs inline when n <= 1 or
// one thread is available.
inline void parallel_items(std::size_t n,
                           const std::function<void(std::size_t)>& fn) {
  const unsigned nt = num_threads();
  if (nt <= 1 || n <= 1) {
    for (std::size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  detail::run_tasks(
      n, std::min<std::size_t>(nt, n) - 1,
      [](const void* p, std::size_t i) {
        (*static_cast<const std::function<void(std::size_t)>*>(p))(i);
      },
      &fn);
}

inline uint32_t ceil_div_u32(uint64_t a, uint64_t b) {
  return static_cast<uint32_t>((a + b - 1) / b);
}

}  // namespace mg
