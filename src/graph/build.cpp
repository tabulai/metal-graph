// build.cpp — Graph::build (plan §5): validate, symmetrize (undirected),
// canonical renumber by out-degree descending, deterministic CSR fill,
// per-orientation worklists, dangling/out_weight_sum, lazy IN transpose.
// SPDX-License-Identifier: Apache-2.0

#include "graph.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "../kernels/mg_params.h"

namespace mg {
namespace {

constexpr uint64_t k_max_stored_edges = (1ull << 31) - 1;  // int32 id ceiling
constexpr uint64_t k_max_vertices = (1ull << 31) - 1;  // int32 outputs (ids,
                                                       // parents, comp)

// Parallel degree histogram over `key` (atomics: determinism of counts is
// order-independent). Result copied to a plain vector.
std::vector<uint32_t> histogram(uint32_t v_count,
                                const std::vector<uint32_t>& key) {
  std::vector<std::atomic<uint32_t>> deg_a(v_count);  // value-init to 0
  parallel_for(key.size(), [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      deg_a[key[i]].fetch_add(1, std::memory_order_relaxed);
  });
  std::vector<uint32_t> deg(v_count);
  for (uint32_t v = 0; v < v_count; ++v)
    deg[v] = deg_a[v].load(std::memory_order_relaxed);
  return deg;
}

// Build one Orientation from the stored canonical edge list. `key` is the
// CSR row vertex (canon_src for OUT, canon_dst for IN), `val` the stored
// column. Deterministic: row_offsets from an exclusive prefix over degrees,
// then a SEQUENTIAL fill in stored-edge order (stable edge order is what
// PageRank determinism rests on — no atomic cursors).
Orientation build_orientation(uint32_t v_count,
                              const std::vector<uint32_t>& key,
                              const std::vector<uint32_t>& val,
                              const std::vector<float>& w, bool weighted,
                              const std::vector<uint32_t>* orig,
                              const char* tag) {
  Runtime& rt = Runtime::instance();
  const uint64_t es = key.size();
  const std::string pre = std::string("mg_") + tag + "_";
  Orientation o;
  o.e_stored = es;

  std::vector<uint32_t> deg = histogram(v_count, key);

  o.row_offsets = rt.alloc(sizeof(uint32_t) * (uint64_t(v_count) + 1),
                           (pre + "row_offsets").c_str());
  uint32_t* row = o.row_offsets->as<uint32_t>();
  uint32_t acc = 0;
  for (uint32_t v = 0; v < v_count; ++v) {
    row[v] = acc;
    acc += deg[v];
  }
  row[v_count] = acc;  // == es

  o.col_indices =
      rt.alloc(sizeof(uint32_t) * es, (pre + "col_indices").c_str());
  uint32_t* col = o.col_indices->as<uint32_t>();
  float* wp = nullptr;
  if (weighted) {  // allocate even when es == 0 so weights is never null
    o.weights = rt.alloc(sizeof(float) * es, (pre + "weights").c_str());
    wp = o.weights->as<float>();
  }
  uint32_t* op = nullptr;
  if (orig) {
    o.edge_orig = rt.alloc(sizeof(uint32_t) * es, (pre + "edge_orig").c_str());
    op = o.edge_orig->as<uint32_t>();
  }

  std::vector<uint32_t> cursor(row, row + v_count);
  for (uint64_t i = 0; i < es; ++i) {
    const uint32_t slot = cursor[key[i]]++;
    col[slot] = val[i];
    if (wp) wp[slot] = w[i];
    if (op) op[slot] = (*orig)[i];
  }

  // Degree-bin worklists, ascending canonical id within each bin.
  uint32_t ng = 0, nh = 0, nm = 0, nl = 0, nz = 0;
  uint64_t n_tiles64 = 0;
  for (uint32_t v = 0; v < v_count; ++v) {
    const uint32_t d = deg[v];
    if (d >= MG_BIN_HUGE_MIN) {
      ++ng;
      n_tiles64 += (uint64_t(d) + MG_HUGE_TILE_EDGES - 1) / MG_HUGE_TILE_EDGES;
    } else if (d >= MG_BIN_HIGH_MIN)
      ++nh;
    else if (d >= MG_BIN_MID_MIN)
      ++nm;
    else if (d >= 1)
      ++nl;
    else
      ++nz;
  }
  const uint32_t nt = static_cast<uint32_t>(n_tiles64);  // <= E/TILE + n_huge
  o.wl_high = rt.alloc(sizeof(uint32_t) * nh, (pre + "wl_high").c_str());
  o.wl_mid = rt.alloc(sizeof(uint32_t) * nm, (pre + "wl_mid").c_str());
  o.wl_low = rt.alloc(sizeof(uint32_t) * nl, (pre + "wl_low").c_str());
  o.zero_list = rt.alloc(sizeof(uint32_t) * nz, (pre + "zero_list").c_str());
  o.wl_huge = rt.alloc(sizeof(uint32_t) * ng, (pre + "wl_huge").c_str());
  o.tile_owner = rt.alloc(sizeof(uint32_t) * nt, (pre + "tile_owner").c_str());
  o.tile_first_edge =
      rt.alloc(sizeof(uint32_t) * nt, (pre + "tile_first_edge").c_str());
  o.huge_tile_off = rt.alloc(sizeof(uint32_t) * (uint64_t(ng) + 1),
                             (pre + "huge_tile_off").c_str());
  uint32_t* ph = o.wl_high->as<uint32_t>();
  uint32_t* pm = o.wl_mid->as<uint32_t>();
  uint32_t* pl = o.wl_low->as<uint32_t>();
  uint32_t* pz = o.zero_list->as<uint32_t>();
  uint32_t* pg = o.wl_huge->as<uint32_t>();
  uint32_t* pto = o.tile_owner->as<uint32_t>();
  uint32_t* ptf = o.tile_first_edge->as<uint32_t>();
  uint32_t* pgo = o.huge_tile_off->as<uint32_t>();
  uint32_t gi = 0, ti = 0;
  for (uint32_t v = 0; v < v_count; ++v) {
    const uint32_t d = deg[v];
    if (d >= MG_BIN_HUGE_MIN) {
      pgo[gi] = ti;
      *pg++ = v;
      for (uint32_t first = row[v]; first < row[v + 1];
           first += MG_HUGE_TILE_EDGES) {
        pto[ti] = gi;
        ptf[ti] = first;
        ++ti;
      }
      ++gi;
    } else if (d >= MG_BIN_HIGH_MIN)
      *ph++ = v;
    else if (d >= MG_BIN_MID_MIN)
      *pm++ = v;
    else if (d >= 1)
      *pl++ = v;
    else
      *pz++ = v;
  }
  pgo[gi] = ti;
  o.n_high = nh;
  o.n_mid = nm;
  o.n_low = nl;
  o.n_zero = nz;
  o.n_huge = ng;
  o.n_tiles = nt;
  return o;
}

}  // namespace

std::shared_ptr<Graph> Graph::build(const uint32_t* src, const uint32_t* dst,
                                    const float* weights, uint64_t n_edges,
                                    uint32_t n_vertices, bool directed) {
  if (n_vertices > k_max_vertices)
    throw_invalid("num_vertices exceeds 2^31-1 (v0.1 int32 id limit)");
  if (n_edges > k_max_stored_edges)
    throw_invalid("input edge count exceeds 2^31-1 (v0.1 int32 id limit)");
  // Validation (parallel; flags, not throws, from worker threads).
  std::atomic<bool> bad_index{false}, bad_weight{false};
  parallel_for(n_edges, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      if (src[i] >= n_vertices || dst[i] >= n_vertices) {
        bad_index.store(true, std::memory_order_relaxed);
        return;
      }
      if (weights) {
        const float w = weights[i];
        if (!std::isfinite(w) || w < 0.0f) {
          bad_weight.store(true, std::memory_order_relaxed);
          return;
        }
      }
    }
  });
  if (bad_index.load())
    throw_invalid("edge endpoint index out of range (>= num_vertices)");
  if (bad_weight.load())
    throw_invalid("edge weights must be finite and >= 0");

  auto g = std::make_shared<Graph>();
  g->V = n_vertices;
  g->E_input = n_edges;
  g->directed = directed;
  g->weighted = (weights != nullptr);
  const uint32_t V = n_vertices;

  // USER-space out-degrees over the STORED (symmetrized) edge set.
  std::vector<std::atomic<uint32_t>> deg_a(V);
  parallel_for(n_edges, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      deg_a[src[i]].fetch_add(1, std::memory_order_relaxed);
      if (!directed && src[i] != dst[i])
        deg_a[dst[i]].fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::vector<uint32_t> deg_user(V);
  for (uint32_t u = 0; u < V; ++u)
    deg_user[u] = deg_a[u].load(std::memory_order_relaxed);

  // Canonical order: stable sort by out-degree DESCENDING (stable => equal
  // degrees keep ascending user index).
  g->user_of_canon.resize(V);
  std::iota(g->user_of_canon.begin(), g->user_of_canon.end(), 0u);
  std::stable_sort(
      g->user_of_canon.begin(), g->user_of_canon.end(),
      [&](uint32_t a, uint32_t b) { return deg_user[a] > deg_user[b]; });
  g->canon_of_user.resize(V);
  for (uint32_t c = 0; c < V; ++c) g->canon_of_user[g->user_of_canon[c]] = c;

  // Stored canonical edge list (kept for the lazy transpose). Undirected:
  // u!=v stored both ways (forward then reverse, input order), self-loops
  // once; the ORIGINAL input edge id rides along.
  uint64_t e_stored = n_edges;
  if (!directed) {
    uint64_t n_self = 0;
    for (uint64_t i = 0; i < n_edges; ++i) n_self += (src[i] == dst[i]);
    e_stored = 2 * n_edges - n_self;
  }
  if (e_stored > k_max_stored_edges)
    throw_invalid("stored edge count exceeds 2^31-1 (v0.1 int32 id limit)");

  g->edges_src_canon_.resize(e_stored);
  g->edges_dst_canon_.resize(e_stored);
  if (g->weighted) g->edges_w_.resize(e_stored);
  g->edges_orig_id_.resize(e_stored);
  if (directed) {
    parallel_for(n_edges, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) {
        g->edges_src_canon_[i] = g->canon_of_user[src[i]];
        g->edges_dst_canon_[i] = g->canon_of_user[dst[i]];
        if (weights) g->edges_w_[i] = weights[i];
        g->edges_orig_id_[i] = static_cast<uint32_t>(i);
      }
    });
  } else {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n_edges; ++i) {
      const uint32_t cu = g->canon_of_user[src[i]];
      const uint32_t cv = g->canon_of_user[dst[i]];
      g->edges_src_canon_[s] = cu;
      g->edges_dst_canon_[s] = cv;
      if (weights) g->edges_w_[s] = weights[i];
      g->edges_orig_id_[s] = static_cast<uint32_t>(i);
      ++s;
      if (cu != cv) {
        g->edges_src_canon_[s] = cv;
        g->edges_dst_canon_[s] = cu;
        if (weights) g->edges_w_[s] = weights[i];
        g->edges_orig_id_[s] = static_cast<uint32_t>(i);
        ++s;
      }
    }
  }

  g->out_ = build_orientation(V, g->edges_src_canon_, g->edges_dst_canon_,
                              g->edges_w_, g->weighted, &g->edges_orig_id_,
                              "out");

  // out_weight_sum (canonical): CSR row sums, or out-degree when unweighted.
  Runtime& rt = Runtime::instance();
  g->out_weight_sum = rt.alloc(sizeof(float) * uint64_t(V),
                               "mg_out_weight_sum");
  float* ows = g->out_weight_sum->as<float>();
  const uint32_t* row = g->out_.row_offsets->as<uint32_t>();
  if (g->weighted) {
    const float* wp = g->out_.weights->as<float>();
    std::atomic<bool> bad_weight_sum{false};
    parallel_for(V, [&](std::size_t b, std::size_t e) {
      for (std::size_t c = b; c < e; ++c) {
        float s = 0.0f;
        for (uint32_t k = row[c]; k < row[c + 1]; ++k) s += wp[k];
        if (!std::isfinite(s)) {
          bad_weight_sum.store(true, std::memory_order_relaxed);
          ows[c] = 0.0f;
          continue;
        }
        // Flush subnormal sums to zero: GPU fp32 runs flush-to-zero, so a
        // subnormal ows would make kernels drop the vertex's contribution
        // while the host skips it in dangling_list — silent mass loss.
        // Flushing here classifies such vertices as dangling on BOTH paths.
        ows[c] = (s < FLT_MIN) ? 0.0f : s;
      }
    });
    if (bad_weight_sum.load(std::memory_order_relaxed))
      throw_invalid(
          "outgoing edge weights must have a finite float32 sum per vertex");
  } else {
    parallel_for(V, [&](std::size_t b, std::size_t e) {
      for (std::size_t c = b; c < e; ++c)
        ows[c] = static_cast<float>(row[c + 1] - row[c]);
    });
  }

  // dangling_list: canonical ids with out_weight_sum == 0, ascending.
  // (Weighted vertices whose outgoing weights sum to 0 count as dangling.)
  uint32_t nd = 0;
  for (uint32_t c = 0; c < V; ++c) nd += (ows[c] == 0.0f);
  g->dangling_list = rt.alloc(sizeof(uint32_t) * nd, "mg_dangling_list");
  uint32_t* dl = g->dangling_list->as<uint32_t>();
  for (uint32_t c = 0; c < V; ++c)
    if (ows[c] == 0.0f) *dl++ = c;
  g->n_dangling = nd;

  return g;
}

const Orientation& Graph::orientation(Dir d) {
  if (d == Dir::out || !directed) return out_;
  std::call_once(in_once_, [this] { build_in_orientation(); });
  return *in_;
}

// Lazy transpose: same deterministic counting sort keyed by canon_dst.
// Weights carried; edge_orig NOT materialized for IN.
void Graph::build_in_orientation() {
  in_ = std::make_unique<Orientation>(build_orientation(
      V, edges_dst_canon_, edges_src_canon_, edges_w_, weighted, nullptr,
      "in"));
}

}  // namespace mg
