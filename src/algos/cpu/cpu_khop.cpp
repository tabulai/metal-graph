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
  // terminates discovery, so a single marker is enough.
  std::unique_ptr<std::atomic<uint8_t>[]> seen(new std::atomic<uint8_t>[V]);
  parallel_for(V, [&](std::size_t b, std::size_t e) {
    for (std::size_t v = b; v < e; ++v)
      seen[v].store(0, std::memory_order_relaxed);
  });

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
    parallel_for(
        frontier.size(),
        [&](std::size_t b, std::size_t e) {
          std::vector<uint32_t> local;
          auto expand = [&](const Csr& c, uint32_t v) {
            for (uint32_t j = c.row[v]; j < c.row[v + 1]; ++j) {
              const uint32_t u = c.col[j];
              uint8_t expect = 0;
              if (seen[u].load(std::memory_order_relaxed) == 0 &&
                  seen[u].compare_exchange_strong(
                      expect, 1, std::memory_order_relaxed))
                local.push_back(u);
            }
          };
          for (std::size_t i = b; i < e; ++i) {
            expand(fwd, frontier[i]);
            if (two_sided) expand(rev, frontier[i]);
          }
          if (!local.empty()) {
            std::lock_guard<std::mutex> lk(mtx);
            cand.insert(cand.end(), local.begin(), local.end());
          }
        },
        1024);
    const bool stop = admit(cand);
    frontier.swap(cand);
    if (stop) break;
  }
  return reached;
}

}  // namespace mg::cpu
