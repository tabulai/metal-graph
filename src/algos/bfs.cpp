// bfs.cpp — mg::bfs entry point (planner + boundary translation) and
// mg::gpu::bfs (frontier-engine driver).
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>

#include "../engines/frontier_engine.hpp"
#include "algos.hpp"
#include "boundary.hpp"

namespace mg {

namespace gpu {

int bfs(Graph& g, const uint32_t* sources_canon, uint32_t n_sources,
        BfsDir dir, int32_t* dist_canon, int32_t* parent_canon,
        uint32_t max_levels) {
  if (dir == BfsDir::both && g.directed) {
    // v0.1: direction=both on DIRECTED graphs runs the CPU implementation
    // (the level protocol has no bin-count re-zero between per-direction
    // sub-passes). Entry points route this before picking the GPU path;
    // this branch is defensive.
    if (max_levels != 0)
      throw Error(ErrorCode::internal,
                  "gpu::bfs: truncated direction=both is CPU-routed in v0.1");
    return cpu::bfs(g, sources_canon, n_sources, dir, dist_canon,
                    parent_canon);
  }
  // Undirected graphs: in == out, so 'both' is just 'out'.
  const BfsDir d = (dir == BfsDir::both) ? BfsDir::out : dir;
  const Dir orient = (d == BfsDir::in) ? Dir::in : Dir::out;
  const bool bottomup = env_flag("MG_BFS_BOTTOMUP", true);
  // Dedup sources so decide's level-1 unvisited subtraction is exact.
  std::vector<uint32_t> srcs(sources_canon, sources_canon + n_sources);
  std::sort(srcs.begin(), srcs.end());
  srcs.erase(std::unique(srcs.begin(), srcs.end()), srcs.end());
  FrontierEngine eng(Runtime::instance());
  return eng.run_bfs(g, srcs, orient, bottomup, max_levels, dist_canon,
                     parent_canon);
}

}  // namespace gpu

int bfs(Graph& g, const uint32_t* sources, uint32_t n_sources, BfsDir dir,
        int32_t* out_dist, int32_t* out_parent) {
  if (!out_dist || !out_parent) throw_invalid("bfs: output arrays are null");
  if (n_sources > 0 && !sources) throw_invalid("bfs: sources is null");
  const auto t0 = std::chrono::steady_clock::now();
  Runtime& rt = Runtime::instance();
  std::vector<uint32_t> src_canon =
      boundary::to_canon(g, sources, n_sources, "bfs sources");

  ExecPath path = rt.plan(g.stored_edges());
  if (dir == BfsDir::both && g.directed)
    path = ExecPath::cpu;  // documented v0.1 routing (see gpu::bfs)

  const uint32_t V = g.V;
  int levels = 0;
  bool output_ready = false;

  // A full GPU launch is the wrong latency tradeoff for a tiny reachable
  // component, even when the graph itself is large. In auto/CPU mode, first
  // try a tightly bounded serial traversal that writes USER-order outputs
  // directly. Dense outputs still preserve the public BFS contract. Forced
  // GPU mode remains an exact opt-out for testing and benchmarking kernels.
  const long sparse_vertices =
      env_long("MG_BFS_SPARSE_MAX_VERTICES", 1024);
  const long sparse_edges = env_long("MG_BFS_SPARSE_MAX_EDGES", 8192);
  if (V > 0 && rt.mode() != ExecMode::gpu && sparse_vertices > 0 &&
      sparse_edges > 0) {
    const uint32_t vertex_cap = static_cast<uint32_t>(
        std::min<long>(sparse_vertices,
                       static_cast<long>(std::numeric_limits<uint32_t>::max())));
    const uint64_t edge_cap = static_cast<uint64_t>(sparse_edges);
    const int sparse_levels = cpu::bfs_sparse_user(
        g, src_canon.data(), static_cast<uint32_t>(src_canon.size()), dir,
        vertex_cap, edge_cap, out_dist, out_parent);
    if (sparse_levels >= 0) {
      levels = sparse_levels;
      path = ExecPath::cpu;
      output_ready = true;
    }
  }

  if (!output_ready) {
    std::vector<int32_t> dist_c(V, -1), parent_c(V, -1);
    if (V > 0) {
      if (path == ExecPath::gpu)
        levels = gpu::bfs(g, src_canon.data(),
                          static_cast<uint32_t>(src_canon.size()), dir,
                          dist_c.data(), parent_c.data(), 0);
      else
        levels = cpu::bfs(g, src_canon.data(),
                          static_cast<uint32_t>(src_canon.size()), dir,
                          dist_c.data(), parent_c.data());
    }
    parallel_for(V, [&](std::size_t b, std::size_t e) {
      for (std::size_t u = b; u < e; ++u) {
        const uint32_t c = g.canon_of_user[u];
        out_dist[u] = dist_c[c];
        out_parent[u] = boundary::canon_value_to_user(g, parent_c[c]);
      }
    });
  }

  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  RunInfo info;
  info.op = output_ready ? "bfs_sparse" : "bfs";
  info.path = path;
  info.iterations = levels;
  info.elapsed_ms = ms;
  rt.record_run(info);
  return levels;
}

}  // namespace mg
