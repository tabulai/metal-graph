// cpu_khop.cpp — deterministic truncated multi-source BFS for k-hop vertex
// discovery (CANONICAL space). Level-by-level; when max_vertices would be
// exceeded mid-level, candidates are sorted by ASCENDING USER INDEX and
// admitted up to the cap, then discovery stops. Seeds count toward the cap.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "../algos.hpp"

namespace mg::cpu {

namespace {

struct Csr {
  const uint32_t* row = nullptr;
  const uint32_t* col = nullptr;
};

Csr csr_of(const Orientation& o) {
  return {o.row_offsets->as<uint32_t>(), o.col_indices->as<uint32_t>()};
}

}  // namespace

std::vector<uint32_t> khop_vertices(Graph& g, const uint32_t* seeds_canon,
                                    uint32_t n_seeds, uint32_t k, BfsDir dir,
                                    uint32_t max_vertices) {
  std::vector<uint32_t> reached;
  const uint32_t V = g.V;
  if (V == 0 || n_seeds == 0) return reached;

  Csr fwd{}, rev{};
  bool two_sided = false;
  if (dir == BfsDir::out) {
    fwd = csr_of(g.orientation(Dir::out));
  } else if (dir == BfsDir::in) {
    fwd = csr_of(g.orientation(Dir::in));
  } else {
    fwd = csr_of(g.orientation(Dir::out));
    if (g.directed) {
      rev = csr_of(g.orientation(Dir::in));
      two_sided = true;
    }
  }

  // seen[v] != 0 once claimed as seed or level candidate. A claimed-but-not-
  // admitted vertex exists only when the cap binds, and the cap binding
  // terminates discovery, so a single marker is enough. Value-initialized
  // (zeroed) at allocation: a threaded clear would spawn threads for every
  // query, which dominates small k-hop neighborhoods.
  std::unique_ptr<std::atomic<uint8_t>[]> seen(new std::atomic<uint8_t>[V]());

  const bool capped = max_vertices > 0;
  const std::size_t cap = capped ? static_cast<std::size_t>(max_vertices) : 0;

  // Admit candidates; truncate by ascending USER index when the cap would
  // be exceeded. Returns true when discovery must stop (cap reached).
  auto admit = [&](std::vector<uint32_t>& cand) -> bool {
    if (capped && reached.size() + cand.size() > cap) {
      std::sort(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) {
        return g.user_of_canon[a] < g.user_of_canon[b];
      });
      cand.resize(cap - reached.size());
      reached.insert(reached.end(), cand.begin(), cand.end());
      return true;
    }
    reached.insert(reached.end(), cand.begin(), cand.end());
    return capped && reached.size() >= cap;
  };

  std::vector<uint32_t> frontier;  // deduped seeds, depth 0
  frontier.reserve(n_seeds);
  for (uint32_t i = 0; i < n_seeds; ++i) {
    const uint32_t s = seeds_canon[i];
    uint8_t expect = 0;
    if (seen[s].compare_exchange_strong(expect, 1,
                                        std::memory_order_relaxed))
      frontier.push_back(s);
  }
  if (admit(frontier)) return reached;

  for (uint32_t level = 1; level <= k; ++level) {
    if (frontier.empty()) break;
    std::vector<uint32_t> cand;
    std::mutex mtx;
    // Edge-balanced partition: each thread owns an equal slice of the
    // frontier's concatenated edge ranges (fwd then rev per vertex), so hub
    // rows and degree-skewed frontiers (canonical order is degree-descending)
    // are split across threads. The per-level candidate SET is partition-
    // invariant (seen[] is claimed once); only within-level order varies.
    const std::size_t F = frontier.size();
    std::vector<uint64_t> pre(F + 1);
    pre[0] = 0;
    for (std::size_t i = 0; i < F; ++i) {
      const uint32_t v = frontier[i];
      uint64_t d = fwd.row[v + 1] - fwd.row[v];
      if (two_sided) d += rev.row[v + 1] - rev.row[v];
      pre[i + 1] = pre[i] + d;
    }
    const uint64_t total = pre[F];
    // Below this many frontier edges: one inline slice. Each extra slice
    // costs a thread spawn, which outweighs splitting 8k-65k-edge levels.
    constexpr uint64_t kSliceEdges = 65536;
    std::size_t chunks = static_cast<std::size_t>(std::min<uint64_t>(
        num_threads(), (total + kSliceEdges - 1) / kSliceEdges));
    if (chunks == 0) chunks = 1;
    const uint64_t per = (total + chunks - 1) / chunks;
    parallel_for(
        chunks,
        [&](std::size_t cb, std::size_t ce) {
          std::vector<uint32_t> local;
          auto scan = [&](const uint32_t* col, uint32_t lo, uint32_t hi) {
            for (uint32_t j = lo; j < hi; ++j) {
              const uint32_t u = col[j];
              uint8_t expect = 0;
              if (seen[u].load(std::memory_order_relaxed) == 0 &&
                  seen[u].compare_exchange_strong(
                      expect, 1, std::memory_order_relaxed))
                local.push_back(u);
            }
          };
          for (std::size_t c = cb; c < ce; ++c) {
            uint64_t cur = c * per;
            const uint64_t hi = std::min(total, cur + per);
            if (cur >= hi) continue;
            // Last frontier slot whose edge range starts at or before cur.
            std::size_t i = static_cast<std::size_t>(
                std::upper_bound(pre.begin(), pre.end(), cur) - pre.begin() -
                1);
            for (; cur < hi && i < F; ++i) {
              const uint32_t v = frontier[i];
              const uint64_t a = cur - pre[i];                  // slice start
              const uint64_t b = std::min(hi, pre[i + 1]) - pre[i];  // end
              const uint32_t f0 = fwd.row[v];
              const uint64_t dfwd = fwd.row[v + 1] - f0;
              if (a < dfwd)
                scan(fwd.col, static_cast<uint32_t>(f0 + a),
                     static_cast<uint32_t>(f0 + std::min(b, dfwd)));
              if (two_sided && b > dfwd) {
                const uint32_t r0 = rev.row[v];
                scan(rev.col,
                     static_cast<uint32_t>(r0 + (std::max(a, dfwd) - dfwd)),
                     static_cast<uint32_t>(r0 + (b - dfwd)));
              }
              cur = pre[i] + b;
            }
          }
          if (!local.empty()) {
            std::lock_guard<std::mutex> lk(mtx);
            cand.insert(cand.end(), local.begin(), local.end());
          }
        },
        1);
    const bool stop = admit(cand);
    frontier.swap(cand);
    if (stop) break;
  }
  return reached;
}

}  // namespace mg::cpu
