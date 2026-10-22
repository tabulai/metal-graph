// pool_stress.cpp — stress test of the persistent worker pool behind
// parallel_for / parallel_items (src/common/mg_pool.cpp): every index runs
// once, in the documented chunks, on at most num_threads() threads; nested
// loops run inline; a task's exception reaches the caller only once no task
// is running; idle gaps cross the workers' spin/park switch; MG_THREADS
// changes (beyond the 256 pool workers too); concurrent callers and forked
// children fall back to fresh threads. A watchdog turns a hang into a
// failure. Optional argument: work scale (default 1).
// SPDX-License-Identifier: Apache-2.0
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "common/mg_common.hpp"

namespace {

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "pool_stress FAILED: %s\n", what);
    std::fflush(stderr);
    std::_Exit(1);
  }
}

std::atomic<long> g_progress{0};
// Tasks of caller `who` currently running (who < kMaxCallers).
constexpr int kMaxCallers = 8;
std::atomic<int> g_running[kMaxCallers];

void busy_us(unsigned us) {
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - t0 <
         std::chrono::microseconds(us)) {
  }
}

// No gap, one inside the workers' spin, or one long enough to park them.
void gap(std::mt19937& rng) {
  const unsigned g = rng() % 8;
  if (g < 2) return;
  if (g < 6)
    busy_us(rng() % 260);
  else
    std::this_thread::sleep_for(std::chrono::microseconds(rng() % 400));
}

void coverage(std::mt19937& rng) {
  const std::size_t n = 1 + rng() % 100000;
  const std::size_t grain = 1 + rng() % 4000;
  const bool balanced = rng() % 2;
  std::vector<uint8_t> hit(n, 0);
  std::mutex mtx;
  std::set<std::pair<std::size_t, std::size_t>> chunks;
  std::set<std::thread::id> ids;
  auto body = [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) ++hit[i];
    std::lock_guard<std::mutex> lk(mtx);
    chunks.emplace(b, e);
    ids.insert(std::this_thread::get_id());
  };
  if (balanced)
    mg::parallel_for_balanced(n, body, grain);
  else
    mg::parallel_for(n, body, grain);
  for (uint8_t h : hit) check(h == 1, "parallel_for runs every index once");
  // Chunk boundaries depend only on (n, grain, num_threads()).
  const std::size_t c =
      mg::parallel_chunks(n, grain, balanced ? mg::k_par_balance : 1);
  const std::size_t per = (n + c - 1) / c;
  std::set<std::pair<std::size_t, std::size_t>> want;
  for (std::size_t b = 0; b < n; b += per) want.emplace(b, std::min(n, b + per));
  check(chunks == want, "parallel_for chunk boundaries");
  check(ids.size() <= std::min<std::size_t>(c, mg::num_threads()),
        "parallel_for thread count");

  const std::size_t m = 1 + rng() % 300;
  std::vector<unsigned> cost(m);
  for (auto& x : cost) x = rng() % 4 == 0 ? rng() % 20 : 0;
  std::vector<uint8_t> item_hit(m, 0);
  ids.clear();
  mg::parallel_items(m, [&](std::size_t i) {
    busy_us(cost[i]);
    ++item_hit[i];
    std::lock_guard<std::mutex> lk(mtx);
    ids.insert(std::this_thread::get_id());
  });
  for (uint8_t h : item_hit) check(h == 1, "parallel_items runs every item once");
  check(ids.size() <= std::min<std::size_t>(mg::num_threads(), m),
        "parallel_items thread count");
}

// Loops called from inside a task (on a worker, or on the caller mid-region)
// run inline on that thread.
void nested(std::mt19937& rng) {
  const std::size_t outer = 2 + rng() % 40;
  std::vector<long> sums(outer, 0);
  std::atomic<int> moved{0};
  mg::parallel_items(outer, [&](std::size_t q) {
    const auto self = std::this_thread::get_id();
    long s = 0;
    mg::parallel_for(
        5000 + q,
        [&](std::size_t b, std::size_t e) {
          if (std::this_thread::get_id() != self) moved.fetch_add(1);
          for (std::size_t i = b; i < e; ++i) s += long(i);
        },
        16);
    mg::parallel_items(8, [&](std::size_t i) {
      if (std::this_thread::get_id() != self) moved.fetch_add(1);
      s += long(i);
    });
    sums[q] = s;
  });
  std::atomic<long> total{0};
  mg::parallel_for(
      1 << 16,
      [&](std::size_t b, std::size_t e) {
        const auto self = std::this_thread::get_id();
        mg::parallel_items(e - b, [&](std::size_t i) {
          if (std::this_thread::get_id() != self) moved.fetch_add(1);
          total.fetch_add(long(b + i), std::memory_order_relaxed);
        });
      },
      1024);
  check(moved.load() == 0, "nested loops run inline");
  for (std::size_t q = 0; q < outer; ++q) {
    const long len = long(5000 + q);
    check(sums[q] == len * (len - 1) / 2 + 28, "nested loop results");
  }
  check(total.load() == (1L << 16) * ((1L << 16) - 1) / 2, "nested items");
}

// Tasks throw while others still run: one exception reaches the caller, and
// only after every task has returned.
void exceptions(std::mt19937& rng, int who) {
  std::atomic<int>& running = g_running[who];
  const std::size_t n = 2 + rng() % 100;
  const std::size_t bad = rng() % n;
  int caught = 0;
  try {
    mg::parallel_items(n, [&](std::size_t i) {
      running.fetch_add(1);
      const bool fail = i == bad || i % 7 == 3;
      if (!fail) busy_us(20);
      running.fetch_sub(1);
      if (i == bad) throw std::runtime_error("item");
      if (fail) throw std::logic_error("other item");
    });
  } catch (const std::runtime_error&) {
    ++caught;
  } catch (const std::logic_error&) {
    ++caught;
  }
  check(caught == 1, "parallel_items rethrows a task's exception");
  check(running.load() == 0, "no task runs after parallel_items throws");

  // Thrown from a loop nested (inline) in the last chunk.
  const std::size_t total = std::size_t{1} << 20;
  caught = 0;
  try {
    mg::parallel_for(total, [&](std::size_t b, std::size_t e) {
      running.fetch_add(1);
      mg::parallel_for(
          e - b,
          [&](std::size_t, std::size_t) {
            if (e == total) {
              running.fetch_sub(1);
              throw std::out_of_range("nested");
            }
          },
          1);
      busy_us(5);
      running.fetch_sub(1);
    });
  } catch (const std::out_of_range&) {
    ++caught;
  }
  check(caught == 1, "parallel_for rethrows a nested exception");
  check(running.load() == 0, "no task runs after parallel_for throws");

  // Loops after an exception run normally.
  std::vector<uint8_t> hit(n, 0);
  mg::parallel_items(n, [&](std::size_t i) { ++hit[i]; });
  for (uint8_t h : hit) check(h == 1, "parallel_items after an exception");
}

void mixed(int rounds, unsigned seed, int who) {
  std::mt19937 rng(seed);
  for (int r = 0; r < rounds; ++r) {
    coverage(rng);
    gap(rng);
    if (r % 4 == 0) nested(rng);
    if (r % 3 == 0) exceptions(rng, who);
    gap(rng);
    g_progress.fetch_add(1);
  }
}

// Many tiny dispatches with gaps around the spin/park switch.
void tiny(int rounds, unsigned seed) {
  std::mt19937 rng(seed);
  for (int r = 0; r < rounds; ++r) {
    const std::size_t n = 2 + rng() % 40;
    std::vector<std::atomic<int>> hit(n);
    for (auto& h : hit) h.store(0, std::memory_order_relaxed);
    const unsigned cost = rng() % 4 == 0 ? rng() % 30 : 0;
    mg::parallel_items(n, [&](std::size_t i) {
      busy_us(cost);
      hit[i].fetch_add(1, std::memory_order_relaxed);
    });
    for (auto& h : hit) check(h.load() == 1, "tiny dispatch coverage");
    gap(rng);
    g_progress.fetch_add(1);
  }
}

// The workers do not survive fork(): a child's loops run on fresh threads,
// also when another parent thread was mid-dispatch at the fork.
void forks(bool busy_parent) {
  std::atomic<bool> stop{false};
  std::thread busy;
  if (busy_parent) {
    busy = std::thread([&] {
      while (!stop.load())
        mg::parallel_items(32, [](std::size_t) { busy_us(200); });
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  for (int k = 0; k < 3; ++k) {
    const pid_t pid = fork();
    check(pid >= 0, "fork");
    if (pid == 0) {
      mixed(12, 100 + k, 0);
      tiny(100, 200 + k);
      std::_Exit(0);
    }
    int status = 0;
    for (int t = 0;; ++t) {
      if (waitpid(pid, &status, WNOHANG) == pid) break;
      if (t == 2000) {  // 20 s
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        check(false, "forked child hung");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "forked child");
    g_progress.fetch_add(1);
  }
  stop.store(true);
  if (busy.joinable()) busy.join();
}

}  // namespace

int main(int argc, char** argv) {
  const int scale = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
  std::thread([] {
    for (long last = -1;;) {
      std::this_thread::sleep_for(std::chrono::seconds(20));
      const long p = g_progress.load();
      if (p == last) {
        std::fprintf(stderr, "pool_stress FAILED: no progress in 20 s\n");
        std::_Exit(2);
      }
      last = p;
    }
  }).detach();

  unsetenv("MG_THREADS");
  mixed(150 * scale, 1, 0);
  tiny(3000 * scale, 2);
  {
    // One caller owns the pool at a time; the others run on fresh threads.
    std::vector<std::thread> callers;
    for (int c = 1; c < 7; ++c)
      callers.emplace_back([c, scale] {
        if (c % 2)
          mixed(40 * scale, 10 + c, c);
        else
          tiny(600 * scale, 10 + c);
      });
    for (auto& t : callers) t.join();
  }
  forks(false);
  forks(true);
  // Only this thread calls getenv from here on (setenv is not thread-safe).
  for (const char* t : {"1", "2", "3", "16", "17", "40", "300"}) {
    setenv("MG_THREADS", t, 1);
    mixed(8 * scale, 300, 0);
    tiny(200 * scale, 301);
  }
  std::printf("pool_stress OK\n");
  return 0;
}
