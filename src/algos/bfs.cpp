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

// Caps of the bounded serial BFS preflight (auto/CPU modes), shared by the
// dense and sparse-output entries. The attempt competes with whatever runs
// when it overflows, so the defaults depend on that fallback:
//  * GPU: reach, not graph size, decides whether the GPU's fixed launch
//    floor (~0.7 ms) pays off, so mid-size reaches (a few thousand
//    vertices, e.g. behind a heavily duplicated hub adjacency) finish
//    serially: 8192 vertices / 524288 scanned edges.
//  * threaded CPU: that path has no launch floor and beats the serial walk
//    on mid-size dense reaches (10-60% on 4k-8k-vertex graphs), so the
//    tight 1024 / 8192 caps stay.
// MG_BFS_SPARSE_MAX_VERTICES / MG_BFS_SPARSE_MAX_EDGES override both
// (0 disables). Returns false when the preflight is disabled.
bool sparse_caps(ExecPath fallback, uint32_t& max_vertices,
                 uint64_t& max_edges) {
  const bool gpu = fallback == ExecPath::gpu;
  const long v = env_long("MG_BFS_SPARSE_MAX_VERTICES", gpu ? 8192 : 1024);
  const long e = env_long("MG_BFS_SPARSE_MAX_EDGES", gpu ? 524288 : 8192);
  if (v <= 0 || e <= 0) return false;
  max_vertices = static_cast<uint32_t>(
      std::min<long>(v, std::numeric_limits<uint32_t>::max()));
  max_edges = static_cast<uint64_t>(e);
  return true;
}

// Full (threaded CPU or GPU) traversal in canonical space — the shared
// fallback behind both the dense entry and the sparse-output entry.
int run_dense_canon(Graph& g, const std::vector<uint32_t>& src_canon,
                    BfsDir dir, ExecPath path, std::vector<int32_t>& dist_c,
                    std::vector<int32_t>& parent_c) {
  const uint32_t V = g.V;
  dist_c.assign(V, -1);
  parent_c.assign(V, -1);
  if (V == 0) return 0;
  if (path == ExecPath::gpu)
    return gpu::bfs(g, src_canon.data(),
                    static_cast<uint32_t>(src_canon.size()), dir,
                    dist_c.data(), parent_c.data(), 0);
  return cpu::bfs(g, src_canon.data(), static_cast<uint32_t>(src_canon.size()),
                  dir, dist_c.data(), parent_c.data());
}

// Dense traversal translated to dense USER-order outputs.
int run_dense_user(Graph& g, const std::vector<uint32_t>& src_canon,
                   BfsDir dir, ExecPath path, int32_t* out_dist,
                   int32_t* out_parent) {
  if (path == ExecPath::cpu)
    return cpu::bfs_user(g, src_canon.data(),
                         static_cast<uint32_t>(src_canon.size()), dir,
                         out_dist, out_parent);
  const uint32_t V = g.V;
  std::vector<int32_t> dist_c, parent_c;
  const int levels = run_dense_canon(g, src_canon, dir, path, dist_c, parent_c);
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

  ExecPath path = rt.plan(g.stored_edges(), PlanOp::bfs);
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
  uint32_t vertex_cap = 0;
  uint64_t edge_cap = 0;
  if (V > 0 && rt.mode() != ExecMode::gpu &&
      sparse_caps(path, vertex_cap, edge_cap)) {
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

  ExecPath path = rt.plan(g.stored_edges(), PlanOp::bfs);
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
  uint32_t vertex_cap = 0;
  uint64_t edge_cap = 0;
  if (V > 0 && rt.mode() != ExecMode::gpu &&
      sparse_caps(path, vertex_cap, edge_cap)) {
    const int r = cpu::bfs_sparse_collect(
        g, src_canon.data(), static_cast<uint32_t>(src_canon.size()), dir,
        vertex_cap, edge_cap, out.vertices, out.dist, out.parent);
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
    // Cap overflow or forced GPU: run the full machinery densely in
    // canonical space, then compact straight into USER order (no dense
    // USER-order intermediate). Two passes over fixed USER-index chunks —
    // per-chunk reached counts, an exclusive prefix, one resize, then each
    // chunk fills its own slice — so the output is ascending in user order
    // and independent of the thread count (chunking depends only on V).
    std::vector<int32_t> dist_c, parent_c;
    levels = run_dense_canon(g, src_canon, dir, path, dist_c, parent_c);
    const uint32_t* cou = g.canon_of_user.data();
    const int32_t* dc = dist_c.data();
    const int32_t* pc = parent_c.data();
    constexpr std::size_t kChunk = 16384;  // user indices per chunk
    const std::size_t nchunks =
        (static_cast<std::size_t>(V) + kChunk - 1) / kChunk;
    // Handing chunks to other threads only pays off once the two O(V)
    // gathers are large; below the threshold run both inline (measured on
    // the persistent pool: serial wins at V=2^15, parallel from V=2^16).
    constexpr std::size_t kParMinV = std::size_t{1} << 16;
    const std::size_t pgrain = V >= kParMinV ? 1 : nchunks;
    std::vector<std::size_t> off(nchunks + 1, 0);
    parallel_for(
        nchunks,
        [&](std::size_t b, std::size_t e) {
          for (std::size_t c = b; c < e; ++c) {
            const std::size_t lo = c * kChunk;
            const std::size_t hi = std::min<std::size_t>(V, lo + kChunk);
            std::size_t n = 0;
            for (std::size_t u = lo; u < hi; ++u) n += (dc[cou[u]] >= 0);
            off[c + 1] = n;
          }
        },
        pgrain);
    for (std::size_t c = 0; c < nchunks; ++c) off[c + 1] += off[c];
    const std::size_t reached = off[nchunks];
    out.vertices.resize(reached);
    out.dist.resize(reached);
    out.parent.resize(reached);
    uint32_t* ov = out.vertices.data();
    int32_t* od = out.dist.data();
    int32_t* op = out.parent.data();
    parallel_for(
        nchunks,
        [&](std::size_t b, std::size_t e) {
          for (std::size_t c = b; c < e; ++c) {
            const std::size_t lo = c * kChunk;
            const std::size_t hi = std::min<std::size_t>(V, lo + kChunk);
            std::size_t w = off[c];
            for (std::size_t u = lo; u < hi; ++u) {
              const uint32_t cv = cou[u];
              const int32_t d = dc[cv];
              if (d >= 0) {
                ov[w] = static_cast<uint32_t>(u);
                od[w] = d;
                op[w] = boundary::canon_value_to_user(g, pc[cv]);
                ++w;
              }
            }
          }
        },
        pgrain);
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
