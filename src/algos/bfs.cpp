// bfs.cpp — mg::bfs entry point (planner + boundary translation) and
// mg::gpu::bfs (frontier-engine driver).
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <numeric>
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

namespace {

// Full (threaded CPU or GPU) traversal in canonical space, translated to
// dense USER-order outputs — the shared fallback behind both the dense
// entry and the sparse-output entry.
int run_dense_user(Graph& g, const std::vector<uint32_t>& src_canon,
                   BfsDir dir, ExecPath path, int32_t* out_dist,
                   int32_t* out_parent) {
  const uint32_t V = g.V;
  int levels = 0;
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
  return levels;
}

}  // namespace

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

  if (!output_ready)
    levels = run_dense_user(g, src_canon, dir, path, out_dist, out_parent);

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

int bfs_sparse(Graph& g, const uint32_t* sources, uint32_t n_sources,
               BfsDir dir, SparseBfsResult& out) {
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
  bool collected = false;
  out.vertices.clear();
  out.dist.clear();
  out.parent.clear();

  // Same bounded serial attempt (and the same opt-outs) as the dense
  // preflight — but collecting O(|reached|) output directly, which is the
  // entire point of this API: a tiny traversal never touches O(V) memory.
  const long sparse_vertices = env_long("MG_BFS_SPARSE_MAX_VERTICES", 1024);
  const long sparse_edges = env_long("MG_BFS_SPARSE_MAX_EDGES", 8192);
  if (V > 0 && rt.mode() != ExecMode::gpu && sparse_vertices > 0 &&
      sparse_edges > 0) {
    const uint32_t vertex_cap = static_cast<uint32_t>(std::min<long>(
        sparse_vertices,
        static_cast<long>(std::numeric_limits<uint32_t>::max())));
    const int r = cpu::bfs_sparse_collect(
        g, src_canon.data(), static_cast<uint32_t>(src_canon.size()), dir,
        vertex_cap, static_cast<uint64_t>(sparse_edges), out.vertices,
        out.dist, out.parent);
    if (r >= 0) {
      levels = r;
      path = ExecPath::cpu;
      collected = true;
      // Contract: ascending user index on every path. The collector emits
      // traversal order; apply the sorting permutation (R is cap-bounded).
      const std::size_t n = out.vertices.size();
      std::vector<uint32_t> idx(n);
      std::iota(idx.begin(), idx.end(), 0u);
      std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
        return out.vertices[a] < out.vertices[b];
      });
      SparseBfsResult sorted;
      sorted.vertices.reserve(n);
      sorted.dist.reserve(n);
      sorted.parent.reserve(n);
      for (uint32_t i : idx) {
        sorted.vertices.push_back(out.vertices[i]);
        sorted.dist.push_back(out.dist[i]);
        sorted.parent.push_back(out.parent[i]);
      }
      out = std::move(sorted);
    } else {
      out.vertices.clear();
      out.dist.clear();
      out.parent.clear();
    }
  }

  if (!collected && V > 0) {
    // Cap overflow or forced GPU: run the full machinery densely, then
    // compact — naturally ascending in user order.
    std::vector<int32_t> dist_u(V), parent_u(V);
    levels = run_dense_user(g, src_canon, dir, path, dist_u.data(),
                            parent_u.data());
    std::size_t reached = 0;
    for (uint32_t u = 0; u < V; ++u) reached += (dist_u[u] >= 0);
    out.vertices.reserve(reached);
    out.dist.reserve(reached);
    out.parent.reserve(reached);
    for (uint32_t u = 0; u < V; ++u) {
      if (dist_u[u] >= 0) {
        out.vertices.push_back(u);
        out.dist.push_back(dist_u[u]);
        out.parent.push_back(parent_u[u]);
      }
    }
  }

  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  RunInfo info;
  info.op = collected ? "bfs_sparse" : "bfs";
  info.path = path;
  info.iterations = levels;
  info.elapsed_ms = ms;
  rt.record_run(info);
  return levels;
}

}  // namespace mg
