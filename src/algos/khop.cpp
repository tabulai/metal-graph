// khop.cpp — mg::k_hop entry point and mg::gpu::khop_edges. Cap-bounded
// runs force the deterministic CPU path (v0.1 semantics); induced-edge
// extraction is always over the OUT-orientation CSR (it stores every edge
// once per direction copy, and edge_orig dedupes undirected twins).
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#include "../engines/frontier_engine.hpp"
#include "algos.hpp"
#include "boundary.hpp"

namespace mg {

namespace {

// CPU induced edges among `reached` (unique canonical ids): two-pass
// (count, prefix, fill) so the parallel fill is deterministic.
std::vector<uint32_t> induced_edges_cpu(Graph& g,
                                        const std::vector<uint32_t>& reached) {
  std::vector<uint32_t> edges;
  if (reached.empty() || g.V == 0) return edges;
  const Orientation& o = g.orientation(Dir::out);
  if (o.e_stored == 0) return edges;
  const uint32_t* row = o.row_offsets->as<uint32_t>();
  const uint32_t* col = o.col_indices->as<uint32_t>();
  const uint32_t* eorig = o.edge_orig->as<uint32_t>();

  std::vector<uint32_t> bm((std::size_t(g.V) + 31u) / 32u, 0u);
  for (uint32_t c : reached) bm[c >> 5] |= 1u << (c & 31u);

  const std::size_t n = reached.size();
  std::vector<uint64_t> cnt(n + 1, 0);
  parallel_for(n, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const uint32_t v = reached[i];
      uint64_t c2 = 0;
      for (uint32_t s = row[v]; s < row[v + 1]; ++s) {
        const uint32_t u = col[s];
        if (bm[u >> 5] & (1u << (u & 31u))) ++c2;
      }
      cnt[i + 1] = c2;
    }
  });
  for (std::size_t i = 0; i < n; ++i) cnt[i + 1] += cnt[i];
  edges.resize(static_cast<std::size_t>(cnt[n]));
  parallel_for(n, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const uint32_t v = reached[i];
      std::size_t w = static_cast<std::size_t>(cnt[i]);
      for (uint32_t s = row[v]; s < row[v + 1]; ++s) {
        const uint32_t u = col[s];
        if (bm[u >> 5] & (1u << (u & 31u))) edges[w++] = eorig[s];
      }
    }
  });
  return edges;
}

// cpu::khop_vertices reports no level count; use the depth budget.
int depth_budget(uint32_t k, uint32_t v) {
  const uint64_t b = std::min<uint64_t>(k, v);
  return static_cast<int>(std::min<uint64_t>(b, 0x7FFFFFFFull));
}

}  // namespace

namespace gpu {

std::vector<uint32_t> khop_edges(Graph& g,
                                 const std::vector<uint32_t>& reached_canon) {
  FrontierEngine eng(Runtime::instance());
  return eng.khop_edges(g, reached_canon);
}

}  // namespace gpu

KhopResult k_hop(Graph& g, const uint32_t* seeds, uint32_t n_seeds,
                 const KhopOpts& o) {
  if (n_seeds > 0 && !seeds) throw_invalid("k_hop: seeds is null");
  const auto t0 = std::chrono::steady_clock::now();
  Runtime& rt = Runtime::instance();
  std::vector<uint32_t> seeds_canon =
      boundary::to_canon(g, seeds, n_seeds, "k_hop seeds");

  ExecPath path = rt.plan(g.stored_edges());
  const bool caps = o.max_vertices != 0 || o.max_edges != 0;
  if (caps) path = ExecPath::cpu;  // deterministic cap semantics (v0.1)
  if (o.dir == BfsDir::both && g.directed && o.k > 0)
    path = ExecPath::cpu;  // documented v0.1 routing (see gpu::bfs) —
                           // recorded honestly instead of a gpu/cpu hybrid

  KhopResult res;
  std::vector<uint32_t> reached;  // unique canonical ids
  int iterations = 0;

  if (path == ExecPath::cpu) {
    reached = cpu::khop_vertices(g, seeds_canon.data(), n_seeds, o.k, o.dir,
                                 o.max_vertices);
    iterations = depth_budget(o.k, g.V);
    res.edges = induced_edges_cpu(g, reached);
  } else {
    if (o.k == 0) {
      reached = seeds_canon;  // depth 0: the seeds themselves
      std::sort(reached.begin(), reached.end());
      reached.erase(std::unique(reached.begin(), reached.end()),
                    reached.end());
    } else {
      std::vector<int32_t> dist(g.V), parent(g.V);
      iterations = gpu::bfs(g, seeds_canon.data(), n_seeds, o.dir,
                            dist.data(), parent.data(), o.k);
      reached.reserve(seeds_canon.size());
      for (uint32_t c = 0; c < g.V; ++c)
        if (dist[c] != -1) reached.push_back(c);
    }
    res.edges = gpu::khop_edges(g, reached);
  }

  // Shared output contract: edges = unique original ids ascending, cap
  // applied AFTER sorting; vertices = user indices ascending.
  std::sort(res.edges.begin(), res.edges.end());
  res.edges.erase(std::unique(res.edges.begin(), res.edges.end()),
                  res.edges.end());
  if (o.max_edges != 0 && res.edges.size() > o.max_edges)
    res.edges.resize(static_cast<std::size_t>(o.max_edges));

  res.vertices.resize(reached.size());
  for (std::size_t i = 0; i < reached.size(); ++i)
    res.vertices[i] = g.user_of_canon[reached[i]];
  std::sort(res.vertices.begin(), res.vertices.end());

  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  RunInfo info;
  info.op = "k_hop";
  info.path = path;
  info.iterations = iterations;
  info.elapsed_ms = ms;
  rt.record_run(info);
  return res;
}

}  // namespace mg
