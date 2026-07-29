// cpu_bfs.cpp — threaded level-synchronous multi-source BFS in CANONICAL
// space (oracle + planner fallback). Vertices are claimed with an atomic
// compare-exchange on dist (-1 -> depth); only the claiming thread writes
// parent, so parent choice among equal-depth candidates is thread-timing
// dependent (acceptable: validated structurally, same as the GPU path).
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

int bfs_sparse_user(Graph& g, const uint32_t* sources_canon,
                    uint32_t n_sources, BfsDir dir, uint32_t max_vertices,
                    uint64_t max_edges, int32_t* dist_user,
                    int32_t* parent_user) {
  const uint32_t V = g.V;
  std::fill_n(dist_user, V, int32_t{-1});
  std::fill_n(parent_user, V, int32_t{-1});
  if (V == 0 || n_sources == 0) return 0;

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

  // One queue is enough for level-order traversal: items appended while a
  // level is processed form the next level. Canonical ids keep CSR access
  // direct; dist_user is also the visited set after a single canonical->user
  // lookup per neighbor.
  std::vector<uint32_t> queue;
  queue.reserve(std::min<uint32_t>(max_vertices, 256u));
  for (uint32_t i = 0; i < n_sources; ++i) {
    const uint32_t c = sources_canon[i];
    const uint32_t u = g.user_of_canon[c];
    if (dist_user[u] == -1) {
      if (queue.size() >= max_vertices) return -1;
      dist_user[u] = 0;
      queue.push_back(c);
    }
  }

  uint64_t scanned_edges = 0;
  std::size_t level_begin = 0;
  int32_t depth = 0;
  int levels = 0;
  while (level_begin < queue.size()) {
    const std::size_t level_end = queue.size();
    ++levels;
    const int32_t next_depth = depth + 1;
    for (std::size_t i = level_begin; i < level_end; ++i) {
      const uint32_t c = queue[i];
      const uint32_t parent_u = g.user_of_canon[c];
      auto expand = [&](const Csr& csr) -> bool {
        const uint32_t begin = csr.row[c];
        const uint32_t end = csr.row[c + 1];
        const uint64_t degree = uint64_t(end) - begin;
        if (degree > max_edges - scanned_edges) return false;
        scanned_edges += degree;
        for (uint32_t edge = begin; edge < end; ++edge) {
          const uint32_t next_c = csr.col[edge];
          const uint32_t next_u = g.user_of_canon[next_c];
          if (dist_user[next_u] != -1) continue;
          if (queue.size() >= max_vertices) return false;
          dist_user[next_u] = next_depth;
          parent_user[next_u] = static_cast<int32_t>(parent_u);
          queue.push_back(next_c);
        }
        return true;
      };
      if (!expand(fwd) || (two_sided && !expand(rev))) return -1;
    }
    level_begin = level_end;
    depth = next_depth;
  }
  return levels;
}

int bfs(Graph& g, const uint32_t* sources_canon, uint32_t n_sources,
        BfsDir dir, int32_t* dist_canon, int32_t* parent_canon) {
  const uint32_t V = g.V;
  if (V == 0) return 0;

  // dir out/in pick one orientation (Graph maps in == out when undirected);
  // dir both on a DIRECTED graph unions neighbors from both orientations.
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

  std::unique_ptr<std::atomic<int32_t>[]> dist(new std::atomic<int32_t>[V]);
  parallel_for(V, [&](std::size_t b, std::size_t e) {
    for (std::size_t v = b; v < e; ++v) {
      dist[v].store(-1, std::memory_order_relaxed);
      parent_canon[v] = -1;
    }
  });

  std::vector<uint32_t> frontier;
  frontier.reserve(n_sources);
  for (uint32_t i = 0; i < n_sources; ++i) {
    const uint32_t s = sources_canon[i];
    int32_t expect = -1;  // duplicate sources fail the exchange -> deduped
    if (dist[s].compare_exchange_strong(expect, 0,
                                        std::memory_order_relaxed))
      frontier.push_back(s);
  }

  // Levels executed = number of non-empty frontiers processed
  // (= max assigned depth + 1; an isolated source counts as 1 level).
  int levels = 0;
  int32_t next_depth = 1;
  while (!frontier.empty()) {
    ++levels;
    std::vector<uint32_t> next;
    std::mutex mtx;
    parallel_for(
        frontier.size(),
        [&](std::size_t b, std::size_t e) {
          std::vector<uint32_t> local;
          auto expand = [&](const Csr& c, uint32_t v) {
            for (uint32_t k = c.row[v]; k < c.row[v + 1]; ++k) {
              const uint32_t u = c.col[k];
              int32_t expect = -1;
              if (dist[u].load(std::memory_order_relaxed) == -1 &&
                  dist[u].compare_exchange_strong(
                      expect, next_depth, std::memory_order_relaxed)) {
                parent_canon[u] = static_cast<int32_t>(v);
                local.push_back(u);
              }
            }
          };
          for (std::size_t i = b; i < e; ++i) {
            const uint32_t v = frontier[i];
            expand(fwd, v);
            if (two_sided) expand(rev, v);
          }
          if (!local.empty()) {
            std::lock_guard<std::mutex> lk(mtx);
            next.insert(next.end(), local.begin(), local.end());
          }
        },
        1024);
    frontier.swap(next);
    ++next_depth;
  }

  parallel_for(V, [&](std::size_t b, std::size_t e) {
    for (std::size_t v = b; v < e; ++v)
      dist_canon[v] = dist[v].load(std::memory_order_relaxed);
  });
  return levels;
}

}  // namespace mg::cpu
