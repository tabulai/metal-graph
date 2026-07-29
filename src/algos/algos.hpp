// algos.hpp — public algorithm entry points (USER index space in/out) and
// the CPU oracle/fallback namespace (CANONICAL space, called by entry
// points after planner dispatch).
//
// SEMANTICS (authoritative; tests/README follow this):
//
// PageRank (pagerank / ppr_topk):
//   Iteration:  r'[v] = (1-a)*p[v] + a * (sum_{u->v} r[u]*w(u,v)/ows[u]
//                                         + D * p[v])
//   where ows[u] = sum of outgoing weights (out-degree if unweighted),
//   D = sum of r[u] over dangling vertices (ows==0), p = teleport vector
//   (uniform 1/V, or normalized personalization / seed weights). Dangling
//   mass is redistributed by p (NetworkX-compatible). Start: r0 = p.
//   Convergence: L1(r' - r) < V * tol, checked every audit_interval
//   iterations (default 5, env MG_PR_AUDIT_INTERVAL) in fp64 on the CPU;
//   iteration count may overshoot to the audit boundary (deterministic).
//   max_iter is a budget, not an error: hitting it returns the current
//   iterate (documented divergence from NetworkX's raise).
//   Vertices with all-zero outgoing weight but nonzero degree are treated
//   as dangling (documented).
//
// ppr_topk: B queries, each a set of (seed, finite weight>=0) with a finite
//   sum > 0 (normalized per query; duplicate seeds within a query are summed).
//   Runs tiles of MG_PPR_TILE lanes; per-lane convergence masking freezes
//   converged lanes at audit boundaries. Selection: top-k by score, ties
//   broken by ascending USER index; if k > V, rows are padded with
//   id = -1 / score = 0.
//
// BFS: multi-source, unit depth. direction: out = follow edges as stored,
//   in = reverse orientation, both = treat edges as undirected (union of
//   both orientations per level). dist = -1 unreachable (int32), parent =
//   -1 for sources/unreachable. Parent choice among equal-depth candidates
//   is nondeterministic on the GPU path (validated structurally).
//
// k_hop: vertices within <= k hops of any seed (seeds included, depth 0),
//   plus the induced edges among reached vertices... precisely: every
//   stored edge (u,v) with BOTH endpoints reached, reported as original
//   input edge ids (deduplicated: undirected symmetrization reports the
//   input edge once). direction as BFS. Caps: max_vertices truncates
//   discovery (BFS order, deterministic on CPU path — cap runs force the
//   CPU path in v0.1, documented); max_edges truncates the edge list after
//   sorting. Outputs sorted ascending: vs by user index, es by input edge id.
//
// WCC: undirected connected components over the stored edges (directed
//   edges treated as undirected by hooking both endpoints). Output:
//   comp[user_idx] = component id; components numbered by first occurrence
//   in USER index order (canonical-partition semantics).
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "../graph/graph.hpp"

namespace mg {

struct PagerankOpts {
  double alpha = 0.85;
  double tol = 1e-6;
  int max_iter = 100;
  // Optional dense personalization, USER order, length V, finite entries
  // >= 0, finite sum > 0 (normalized internally). Empty => uniform.
  const float* personalization = nullptr;
};

// out_rank: caller-allocated float[V], USER order.
// Returns iterations executed.
int pagerank(Graph& g, const PagerankOpts& o, float* out_rank);

struct PprTopkOpts {
  double alpha = 0.85;
  double tol = 1e-6;
  int max_iter = 50;
  uint32_t k = 64;
};

// seeds/seed_weights packed CSR-style: query q owns [offsets[q], offsets[q+1]).
// Seed weights must be finite and >= 0 with a positive sum per query.
// B = n_queries. out_ids: int32[B*k] (USER indices, -1 padding), out_scores:
// float[B*k]. Returns max iterations executed over all queries.
int ppr_topk(Graph& g, const uint32_t* seeds, const float* seed_weights,
             const uint64_t* offsets, uint32_t n_queries,
             const PprTopkOpts& o, int32_t* out_ids, float* out_scores);

enum class BfsDir : int { out = 0, in = 1, both = 2 };

// sources: USER indices. out_dist/out_parent: caller-allocated int32[V],
// USER order. Returns number of BFS levels executed.
int bfs(Graph& g, const uint32_t* sources, uint32_t n_sources, BfsDir dir,
        int32_t* out_dist, int32_t* out_parent);

// Sparse-output BFS (opt-in; the dense contract above is unchanged).
// vertices: reached USER indices ascending (sources included, depth 0);
// dist/parent aligned with vertices; parent is -1 for sources and a USER
// index otherwise. Reachability is identical to dense bfs; only the output
// shape differs (O(|reached|) instead of O(V)). Deterministic: ascending
// order is part of the contract on every execution path.
struct SparseBfsResult {
  std::vector<uint32_t> vertices;
  std::vector<int32_t> dist;
  std::vector<int32_t> parent;
};

// Returns BFS levels executed (same convention as bfs()).
int bfs_sparse(Graph& g, const uint32_t* sources, uint32_t n_sources,
               BfsDir dir, SparseBfsResult& out);

struct KhopOpts {
  uint32_t k = 2;
  BfsDir dir = BfsDir::both;
  uint32_t max_vertices = 0;  // 0 = unlimited
  uint64_t max_edges = 0;     // 0 = unlimited
};

struct KhopResult {
  std::vector<uint32_t> vertices;  // USER indices, ascending
  std::vector<uint32_t> edges;     // original input edge ids, ascending
};

KhopResult k_hop(Graph& g, const uint32_t* seeds, uint32_t n_seeds,
                 const KhopOpts& o);

// out_comp: caller-allocated int32[V], USER order. Returns #components.
int wcc(Graph& g, int32_t* out_comp);

// ---------------------------------------------------------------------------
// CPU implementations (threaded; oracles + planner fallback).
// They operate in CANONICAL space and entry points translate, except the
// explicitly named bfs_sparse_user latency path, which writes USER order
// directly to avoid a dense boundary scatter.
// pvec_canon: normalized teleport vector in canonical order (always dense
// here; length V). rank_canon out is canonical order.
// ---------------------------------------------------------------------------
namespace cpu {

int pagerank(Graph& g, const double* pvec_canon, double alpha, double tol,
             int max_iter, int audit_interval, double* rank_canon);

// Batched PPR CPU path: same tiling/masking semantics as GPU (audit-interval
// convergence, per-query freeze) so results are comparable; per-query dense
// iteration in fp32 to mirror GPU numerics is NOT required — CPU path is
// fp64 (documented: CPU and GPU paths agree to ~1e-6 rel, not bitwise).
// seeds_canon packed like the public API but already canonical/deduped.
// sink(q, rank_canon) is invoked exactly once per query with that query's
// FINAL dense rank vector (canonical order, length V), valid only for the
// duration of the call. Queries run in parallel, so the sink may be called
// CONCURRENTLY for DISTINCT q (never twice for the same q); it must be safe
// under that concurrency. Never called when V == 0 or n_queries == 0.
int ppr(Graph& g, const uint32_t* seeds_canon, const float* seed_w,
        const uint64_t* offsets, uint32_t n_queries, double alpha, double tol,
        int max_iter, int audit_interval,
        const std::function<void(uint32_t, const double*)>& sink);

int bfs(Graph& g, const uint32_t* sources_canon, uint32_t n_sources,
        BfsDir dir, int32_t* dist_canon, int32_t* parent_canon);

// Serial, sparse-collecting BFS for the sparse-output API: appends each
// reached vertex (USER index, incl. sources) with its dist/parent to the
// out vectors in TRAVERSAL order (caller sorts). The dense visited scratch
// is calloc-backed so an untouched graph costs O(pages touched), not O(V).
// Returns BFS levels, or -1 once max_vertices/max_edges is exceeded (out
// vectors left partial; caller discards). Limits must be positive.
int bfs_sparse_collect(Graph& g, const uint32_t* sources_canon,
                       uint32_t n_sources, BfsDir dir, uint32_t max_vertices,
                       uint64_t max_edges, std::vector<uint32_t>& out_user,
                       std::vector<int32_t>& out_dist,
                       std::vector<int32_t>& out_parent);

// Serial, output-direct BFS for latency-sensitive traversals that touch only
// a small component. The traversal still uses canonical CSR internally, but
// writes dist/parent directly in USER order so a successful attempt avoids
// two dense canonical scratch arrays and the dense boundary scatter.
//
// Returns BFS levels on success, or -1 after max_vertices/max_edges is
// exceeded. A failed attempt may leave partial output; callers must overwrite
// it with the regular CPU/GPU path. Limits must be positive.
int bfs_sparse_user(Graph& g, const uint32_t* sources_canon,
                    uint32_t n_sources, BfsDir dir, uint32_t max_vertices,
                    uint64_t max_edges, int32_t* dist_user,
                    int32_t* parent_user);

// Truncated multi-source BFS honoring caps; returns reached canonical ids
// in discovery order (deterministic: level by level; when max_vertices
// binds mid-level, candidates are admitted in ascending USER index order).
std::vector<uint32_t> khop_vertices(Graph& g, const uint32_t* seeds_canon,
                                    uint32_t n_seeds, uint32_t k, BfsDir dir,
                                    uint32_t max_vertices);

int wcc(Graph& g, uint32_t* labels_canon);  // labels = canonical min-id per comp

}  // namespace cpu

// GPU implementations (engines layer). Same canonical-space contracts as
// cpu::. Defined in src/algos/*.cpp + src/engines/*.
namespace gpu {

int pagerank(Graph& g, const double* pvec_canon, double alpha, double tol,
             int max_iter, int audit_interval, double* rank_canon);

// sink(q, rank_canon) is invoked exactly once per query with that query's
// FINAL fp32 dense rank vector (canonical order, length V), deinterleaved
// from tile state as each tile completes. Calls are SEQUENTIAL (driver
// thread) and the buffer is a scratch reused across calls — consume or copy
// before returning. Never called when V == 0 or n_queries == 0. Only one
// tile's worth of dense state is live at a time.
int ppr(Graph& g, const uint32_t* seeds_canon, const float* seed_w,
        const uint64_t* offsets, uint32_t n_queries, double alpha, double tol,
        int max_iter, int audit_interval,
        const std::function<void(uint32_t, const float*)>& sink);

// max_levels: 0 = unbounded; otherwise stop after assigning depth
// max_levels (k-hop truncation).
int bfs(Graph& g, const uint32_t* sources_canon, uint32_t n_sources,
        BfsDir dir, int32_t* dist_canon, int32_t* parent_canon,
        uint32_t max_levels = 0);

// Induced-edge extraction on GPU given the reached set (bitmap built by
// caller); returns original input edge ids, unsorted.
std::vector<uint32_t> khop_edges(Graph& g,
                                 const std::vector<uint32_t>& reached_canon);

int wcc(Graph& g, uint32_t* labels_canon);

}  // namespace gpu

}  // namespace mg
