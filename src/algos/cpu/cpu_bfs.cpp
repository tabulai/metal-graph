// cpu_bfs.cpp — threaded level-synchronous multi-source BFS in CANONICAL
// space (oracle + planner fallback). Vertices are claimed with an atomic
// compare-exchange on dist (-1 -> depth); only the claiming thread writes
// parent, so parent choice among equal-depth candidates is thread-timing
// dependent (acceptable: validated structurally, same as the GPU path).
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

#include "../algos.hpp"
#include "../boundary.hpp"

namespace mg::cpu {

namespace {

struct Csr {
  const uint32_t* row = nullptr;
  const uint32_t* col = nullptr;
};

Csr csr_of(const Orientation& o) {
  return {o.row_offsets->as<uint32_t>(), o.col_indices->as<uint32_t>()};
}

// A bounded serial attempt succeeds only after expanding every vertex it
// reaches in full, so the edges it scans are exactly the summed degrees of
// the vertices it enqueues. Charging that degree at enqueue time rejects an
// attempt whose reach cannot fit max_edges as soon as the offending vertex
// is discovered, instead of after scanning everything queued ahead of it.
// Which attempts succeed, and every output they produce, is unchanged.
struct EdgeBudget {
  Csr fwd;
  Csr rev;
  bool two_sided;
  uint64_t max_edges;
  uint64_t committed = 0;

  uint64_t degree(uint32_t c) const {
    uint64_t d = uint64_t(fwd.row[c + 1]) - fwd.row[c];
    if (two_sided) d += uint64_t(rev.row[c + 1]) - rev.row[c];
    return d;
  }

  bool admit(uint32_t c) {
    const uint64_t d = degree(c);
    if (d > max_edges - committed) return false;
    committed += d;
    return true;
  }
};

}  // namespace

int bfs_sparse_collect(Graph& g, const uint32_t* sources_canon,
                       uint32_t n_sources, BfsDir dir, uint32_t max_vertices,
                       uint64_t max_edges, std::vector<uint32_t>& out_user,
                       std::vector<int32_t>& out_dist,
                       std::vector<int32_t>& out_parent) {
  const uint32_t V = g.V;
  out_user.clear();
  out_dist.clear();
  out_parent.clear();
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

  // calloc: the kernel hands back lazily-zeroed pages, so a traversal that
  // touches R vertices faults in O(R) pages of this array, not O(V) — the
  // whole point of the sparse output path.
  std::unique_ptr<unsigned char, decltype(&std::free)> visited(
      static_cast<unsigned char*>(std::calloc(V, 1)), &std::free);
  if (!visited) throw std::bad_alloc();
  unsigned char* seen = visited.get();

  EdgeBudget budget{fwd, rev, two_sided, max_edges};
  std::vector<uint32_t> queue;  // canonical ids, aligned with out_* entries
  queue.reserve(std::min<uint32_t>(max_vertices, 256u));
  for (uint32_t i = 0; i < n_sources; ++i) {
    const uint32_t c = sources_canon[i];
    if (seen[c]) continue;
    if (queue.size() >= max_vertices || !budget.admit(c)) return -1;
    seen[c] = 1;
    queue.push_back(c);
    out_user.push_back(g.user_of_canon[c]);
    out_dist.push_back(0);
    out_parent.push_back(-1);
  }

  std::size_t level_begin = 0;
  int32_t depth = 0;
  int levels = 0;
  while (level_begin < queue.size()) {
    const std::size_t level_end = queue.size();
    ++levels;
    const int32_t next_depth = depth + 1;
    for (std::size_t i = level_begin; i < level_end; ++i) {
      const uint32_t c = queue[i];
      const int32_t parent_u = static_cast<int32_t>(g.user_of_canon[c]);
      auto expand = [&](const Csr& csr) -> bool {
        // Register copies keep the already-seen fast path free of reloads
        // through the closure (the byte stores below may alias anything).
        const uint32_t* const col = csr.col;
        unsigned char* const seen_map = seen;
        const uint32_t begin = csr.row[c];
        const uint32_t end = csr.row[c + 1];
        for (uint32_t edge = begin; edge < end; ++edge) {
          const uint32_t next_c = col[edge];
          if (seen_map[next_c]) continue;
          if (queue.size() >= max_vertices || !budget.admit(next_c))
            return false;
          seen_map[next_c] = 1;
          queue.push_back(next_c);
          out_user.push_back(g.user_of_canon[next_c]);
          out_dist.push_back(next_depth);
          out_parent.push_back(parent_u);
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

int bfs_sparse_user(Graph& g, const uint32_t* sources_canon,
                    uint32_t n_sources, BfsDir dir, uint32_t max_vertices,
                    uint64_t max_edges, int32_t* dist_user,
                    int32_t* parent_user) {
  const uint32_t V = g.V;
  if (V == 0) return 0;
  if (n_sources == 0) {
    std::fill_n(dist_user, V, int32_t{-1});
    std::fill_n(parent_user, V, int32_t{-1});
    return 0;
  }

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

  EdgeBudget budget{fwd, rev, two_sided, max_edges};
  // Every source must eventually be expanded. If one source alone exceeds
  // the edge budget, the attempt cannot succeed, so reject before touching
  // either V-length output array. The fallback overwrites all output slots.
  // Check individual degrees only: duplicate sources must not be charged
  // twice, and the normal visited/queue loop below handles deduplication.
  for (uint32_t i = 0; i < n_sources; ++i)
    if (budget.degree(sources_canon[i]) > max_edges) return -1;
  std::fill_n(dist_user, V, int32_t{-1});
  std::fill_n(parent_user, V, int32_t{-1});

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
      if (queue.size() >= max_vertices || !budget.admit(c)) return -1;
      dist_user[u] = 0;
      queue.push_back(c);
    }
  }

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
        for (uint32_t edge = begin; edge < end; ++edge) {
          const uint32_t next_c = csr.col[edge];
          const uint32_t next_u = g.user_of_canon[next_c];
          if (dist_user[next_u] != -1) continue;
          if (queue.size() >= max_vertices || !budget.admit(next_c))
            return false;
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

namespace {

// The traversal and parent claims are identical for both output layouts.
// Dense USER output gathers atomic distances and canonical parents in one
// pass, avoiding the intermediate distance vector and a second threaded
// translation pass. Sparse-output compaction still uses CANONICAL output.
template <bool UserOutput>
int bfs_impl(Graph& g, const uint32_t* sources_canon, uint32_t n_sources,
             BfsDir dir, int32_t* dist_out, int32_t* parent_canon,
             int32_t* parent_user = nullptr) {
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
  // Edges each one-sided frontier scans, counted as its vertices are
  // claimed: early frontiers hold the hubs (rmat18 level 1: 173 edges per
  // vertex against an average of 16), so a per-vertex grain under-threads
  // them. The count only sizes the fan-out, so it stops at count_cap, enough
  // work for every balanced chunk (exact below it, so the chunk count matches
  // a full count); an uncapped degree read per claim cost large graphs 7-15%.
  // Skipped when even the whole graph is under two chunks of work. Threaded
  // two-sided traversals partition by edges, and sum the edges of the small
  // frontiers they still split by vertex below.
  const bool count_edges =
      !two_sided && parallel_chunks(uint64_t(V) + fwd.row[V]) > 1;
  const uint64_t count_cap =
      uint64_t(k_par_balance) * num_threads() * k_par_work_grain;
  auto degree = [&](uint32_t v) -> uint64_t {
    return fwd.row[v + 1] - fwd.row[v];
  };
  uint64_t frontier_edges = 0;
  if (count_edges)
    for (uint32_t v : frontier) {
      frontier_edges += degree(v);
      if (frontier_edges >= count_cap) break;
    }
  while (!frontier.empty()) {
    ++levels;
    std::vector<uint32_t> next;
    std::atomic<uint64_t> next_edges{0};  // saturates at count_cap
    std::mutex mtx;
    constexpr uint64_t kSliceEdges = 65536;
    // Directed both-direction traversals can have one huge adjacency or
    // degree-skewed frontiers. Keep the established vertex partition for
    // other directions, single-thread calls, and small serial frontiers.
    bool vertex_partitioned = !two_sided || num_threads() <= 1;
    if (!vertex_partitioned && frontier.size() <= 1024) {
      uint64_t edges = 0;
      for (uint32_t v : frontier) {
        edges += fwd.row[v + 1] - fwd.row[v];
        if (two_sided) edges += rev.row[v + 1] - rev.row[v];
        if (edges > kSliceEdges) break;
      }
      vertex_partitioned = edges <= kSliceEdges;
      frontier_edges = edges;  // exact when vertex_partitioned
    }
    if (vertex_partitioned) {
      parallel_for_balanced(
          frontier.size(),
          [&](std::size_t b, std::size_t e) {
            std::vector<uint32_t> local;
            bool counting =
                count_edges &&
                next_edges.load(std::memory_order_relaxed) < count_cap;
            uint64_t local_edges = 0;
            unsigned pending = 0;
            auto expand = [&](const Csr& c, uint32_t v) {
              for (uint32_t k = c.row[v]; k < c.row[v + 1]; ++k) {
                const uint32_t u = c.col[k];
                int32_t expect = -1;
                if (dist[u].load(std::memory_order_relaxed) == -1 &&
                    dist[u].compare_exchange_strong(
                        expect, next_depth, std::memory_order_relaxed)) {
                  parent_canon[u] = static_cast<int32_t>(v);
                  local.push_back(u);
                  if (counting) {
                    local_edges += degree(u);
                    if (++pending == 64) {
                      const uint64_t total =
                          next_edges.fetch_add(local_edges,
                                               std::memory_order_relaxed) +
                          local_edges;
                      counting = total < count_cap;
                      local_edges = 0;
                      pending = 0;
                    }
                  }
                }
              }
            };
            for (std::size_t i = b; i < e; ++i) {
              const uint32_t v = frontier[i];
              expand(fwd, v);
              if (two_sided) expand(rev, v);
            }
            if (local_edges != 0)
              next_edges.fetch_add(local_edges, std::memory_order_relaxed);
            if (!local.empty()) {
              std::lock_guard<std::mutex> lk(mtx);
              next.insert(next.end(), local.begin(), local.end());
            }
          },
          work_grain(frontier.size(), frontier.size() + frontier_edges));
    } else {
      // Partition the concatenated adjacency by edge count, splitting a
      // single hub as well as skewed frontiers while retaining the level
      // barrier. Successful atomic claims still uniquely own parent writes.
      const std::size_t F = frontier.size();
      std::vector<uint64_t> pre(F + 1);
      pre[0] = 0;
      constexpr std::size_t kParallelPrefixMin = 262144;
      constexpr std::size_t kPrefixBlockVertices = 65536;
      if (F < kParallelPrefixMin) {
        for (std::size_t i = 0; i < F; ++i) {
          const uint32_t v = frontier[i];
          uint64_t d = fwd.row[v + 1] - fwd.row[v];
          if (two_sided) d += rev.row[v + 1] - rev.row[v];
          pre[i + 1] = pre[i] + d;
        }
      } else {
        // Large frontiers must not serialize all irregular CSR degree
        // reads before expansion. Independent block scans followed by
        // block-offset addition produce the identical integer prefix;
        // the adjacency slices and atomic claims below stay unchanged.
        const std::size_t blocks = std::min<std::size_t>(
            num_threads(), (F + kPrefixBlockVertices - 1) /
                               kPrefixBlockVertices);
        const std::size_t block_size = (F + blocks - 1) / blocks;
        std::vector<uint64_t> offsets(blocks + 1, 0);
        parallel_for(blocks, [&](std::size_t cb, std::size_t ce) {
          for (std::size_t c = cb; c < ce; ++c) {
            const std::size_t begin = c * block_size;
            const std::size_t end = std::min(F, begin + block_size);
            uint64_t sum = 0;
            for (std::size_t i = begin; i < end; ++i) {
              const uint32_t v = frontier[i];
              uint64_t d = fwd.row[v + 1] - fwd.row[v];
              if (two_sided) d += rev.row[v + 1] - rev.row[v];
              sum += d;
              pre[i + 1] = sum;
            }
            offsets[c + 1] = sum;
          }
        }, 1);
        for (std::size_t c = 0; c < blocks; ++c)
          offsets[c + 1] += offsets[c];
        parallel_for(blocks, [&](std::size_t cb, std::size_t ce) {
          for (std::size_t c = cb; c < ce; ++c) {
            const uint64_t offset = offsets[c];
            if (offset == 0) continue;
            const std::size_t begin = c * block_size;
            const std::size_t end = std::min(F, begin + block_size);
            for (std::size_t i = begin; i < end; ++i)
              pre[i + 1] += offset;
          }
        }, 1);
      }
      const uint64_t total = pre[F];
      // Several slices per thread, each its own balanced chunk: slices into
      // visited targets or on efficiency cores finish later, and faster
      // threads take more slices.
      std::size_t chunks = static_cast<std::size_t>(std::min<uint64_t>(
          k_par_balance * num_threads(),
          (total + kSliceEdges - 1) / kSliceEdges));
      if (chunks == 0) chunks = 1;
      const uint64_t per = (total + chunks - 1) / chunks;
      parallel_for_balanced(
          chunks,
          [&](std::size_t cb, std::size_t ce) {
            std::vector<uint32_t> local;
            auto scan = [&](const uint32_t* col, uint32_t lo, uint32_t hi,
                            uint32_t parent) {
              for (uint32_t j = lo; j < hi; ++j) {
                const uint32_t u = col[j];
                int32_t expect = -1;
                if (dist[u].load(std::memory_order_relaxed) == -1 &&
                    dist[u].compare_exchange_strong(
                        expect, next_depth, std::memory_order_relaxed)) {
                  parent_canon[u] = static_cast<int32_t>(parent);
                  local.push_back(u);
                }
              }
            };
            for (std::size_t c = cb; c < ce; ++c) {
              uint64_t cur = c * per;
              const uint64_t hi = std::min(total, cur + per);
              if (cur >= hi) continue;
              std::size_t i = static_cast<std::size_t>(
                  std::upper_bound(pre.begin(), pre.end(), cur) - pre.begin() - 1);
              for (; cur < hi && i < F; ++i) {
                const uint32_t v = frontier[i];
                const uint64_t a = cur - pre[i];
                const uint64_t b = std::min(hi, pre[i + 1]) - pre[i];
                const uint32_t f0 = fwd.row[v];
                const uint64_t dfwd = fwd.row[v + 1] - f0;
                if (a < dfwd)
                  scan(fwd.col, static_cast<uint32_t>(f0 + a),
                       static_cast<uint32_t>(f0 + std::min(b, dfwd)), v);
                if (two_sided && b > dfwd) {
                  const uint32_t r0 = rev.row[v];
                  scan(rev.col,
                       static_cast<uint32_t>(r0 + (std::max(a, dfwd) - dfwd)),
                       static_cast<uint32_t>(r0 + (b - dfwd)), v);
                }
                cur = pre[i] + b;
              }
            }
            if (!local.empty()) {
              std::lock_guard<std::mutex> lk(mtx);
              next.insert(next.end(), local.begin(), local.end());
            }
          },
          1);
    }
    frontier.swap(next);
    frontier_edges = next_edges.load(std::memory_order_relaxed);
    ++next_depth;
  }

  if constexpr (UserOutput) {
    parallel_for(V, [&](std::size_t b, std::size_t e) {
      for (std::size_t u = b; u < e; ++u) {
        const uint32_t c = g.canon_of_user[u];
        dist_out[u] = dist[c].load(std::memory_order_relaxed);
        parent_user[u] = boundary::canon_value_to_user(g, parent_canon[c]);
      }
    });
  } else {
    parallel_for(V, [&](std::size_t b, std::size_t e) {
      for (std::size_t v = b; v < e; ++v)
        dist_out[v] = dist[v].load(std::memory_order_relaxed);
    });
  }
  return levels;
}

}  // namespace

int bfs(Graph& g, const uint32_t* sources_canon, uint32_t n_sources,
        BfsDir dir, int32_t* dist_canon, int32_t* parent_canon) {
  return bfs_impl<false>(g, sources_canon, n_sources, dir, dist_canon,
                          parent_canon);
}

int bfs_user(Graph& g, const uint32_t* sources_canon, uint32_t n_sources,
             BfsDir dir, int32_t* dist_user, int32_t* parent_user) {
  // bfs_impl initializes every parent alongside the atomic distances.
  std::unique_ptr<int32_t[]> parent_canon(new int32_t[g.V]);
  return bfs_impl<true>(g, sources_canon, n_sources, dir, dist_user,
                         parent_canon.get(), parent_user);
}

}  // namespace mg::cpu
