// cpu_pagerank.cpp — threaded CPU PageRank + batched PPR (fp64 oracles and
// planner fallback). Pull iteration over the IN orientation in CANONICAL
// space; semantics per src/algos/algos.hpp. Convergence is audited every
// audit_interval iterations (L1 of the LAST iteration only), matching the
// GPU command-batch protocol so iteration counts agree across paths.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "../algos.hpp"
#include "../boundary.hpp"
#include "cpu_gather.hpp"

namespace mg::cpu {

namespace {

struct PullCsr {
  const uint32_t* row = nullptr;
  const uint32_t* col = nullptr;
  const float* w = nullptr;  // null when unweighted
  const CpuGatherPlan* plan = nullptr;  // edge-balanced items (cached)
};

PullCsr in_csr(Graph& g) {
  const Orientation& o = g.orientation(Dir::in);
  PullCsr c;
  c.row = o.row_offsets->as<uint32_t>();
  c.col = o.col_indices->as<uint32_t>();
  if (g.weighted && o.weights) c.w = o.weights->as<float>();
  c.plan = &g.cpu_gather_plan(Dir::in);
  return c;
}

// Serial fp64 sum over the dangling list: deterministic, O(n_dangling),
// negligible next to the gather.
double dangling_mass(Graph& g, const double* r) {
  if (g.n_dangling == 0 || !g.dangling_list) return 0.0;
  const uint32_t* dl = g.dangling_list->as<uint32_t>();
  double d = 0.0;
  for (uint32_t i = 0; i < g.n_dangling; ++i) d += r[dl[i]];
  return d;
}

// fn(begin, end) over [0, n): threaded or inline. PPR runs queries in
// parallel, so per-query loops must not nest a second thread fan-out.
void for_range(std::size_t n, bool threaded,
               const std::function<void(std::size_t, std::size_t)>& fn) {
  if (threaded) parallel_for(n, fn);
  else if (n > 0) fn(0, n);
}

// contrib[u] = r[u] / ows[u], 0 for dangling (ows == 0).
void contrib_pass(uint32_t V, const float* ows, const double* r,
                  double* contrib, bool threaded) {
  for_range(V, threaded, [&](std::size_t b, std::size_t e) {
    for (std::size_t u = b; u < e; ++u)
      contrib[u] = ows[u] > 0.0f ? r[u] / static_cast<double>(ows[u]) : 0.0;
  });
}

// One pull iteration: next[v] = alpha * sum_in(contrib * w) + base_v where
// base_v = ((1-a) + a*dmass) * p[v] for a dense teleport vector p, or 0 when
// p == null (sparse teleport: the caller adds it at the seeds afterwards).
// Edge-balanced items claimed dynamically (threaded) or run inline in plan
// order (per-query PPR workers); same bits either way. tile_part: scratch
// of c.plan->n_slots doubles.
void gather_pass(const PullCsr& c, const double* contrib, const double* p,
                 double alpha, double one_minus_a, double dmass, double* next,
                 double* tile_part, bool threaded) {
  plan_gather(*c.plan, c.row, c.col, c.w, contrib, tile_part, threaded,
              [&](uint32_t v, double acc) {
                double x = alpha * acc;
                if (p) x += (one_minus_a + alpha * dmass) * p[v];
                next[v] = x;
              });
}

double l1_diff(uint32_t V, const double* a, const double* b) {
  double err = 0.0;
  for (uint32_t v = 0; v < V; ++v) err += std::fabs(a[v] - b[v]);
  return err;
}

// Open-addressing map canonical id -> local index, sized for at most `cap`
// keys (load <= 1/2). O(cap) memory regardless of V.
class LocalIndex {
 public:
  static constexpr uint32_t kEmpty = 0xFFFFFFFFu;  // never a canonical id
  explicit LocalIndex(uint32_t cap) {
    std::size_t n = 16;
    while (n < 2 * static_cast<std::size_t>(cap) + 2) n <<= 1;
    shift_ = 64;
    for (std::size_t m = n; m > 1; m >>= 1) --shift_;
    mask_ = n - 1;
    keys_.assign(n, kEmpty);
    vals_.resize(n);
  }
  // Slot for key; *inserted reports whether it was absent (value unset).
  uint32_t* insert(uint32_t key, bool* inserted) {
    std::size_t i = hash(key);
    while (keys_[i] != kEmpty && keys_[i] != key) i = (i + 1) & mask_;
    *inserted = keys_[i] == kEmpty;
    keys_[i] = key;
    return &vals_[i];
  }
  // Value for a key known to be present.
  uint32_t& at(uint32_t key) {
    std::size_t i = hash(key);
    while (keys_[i] != key) i = (i + 1) & mask_;
    return vals_[i];
  }

 private:
  std::size_t hash(uint32_t key) const {
    return static_cast<std::size_t>(
        (uint64_t{key} * 0x9E3779B97F4A7C15ull) >> shift_);
  }
  std::vector<uint32_t> keys_, vals_;
  std::size_t mask_ = 0;
  int shift_ = 60;
};

}  // namespace

int pagerank(Graph& g, const double* pvec_canon, double alpha, double tol,
             int max_iter, int audit_interval, double* rank_canon) {
  const uint32_t V = g.V;
  if (V == 0) return 0;
  if (audit_interval < 1) audit_interval = 1;
  if (max_iter < 0) max_iter = 0;
  const PullCsr c = in_csr(g);
  const float* ows = g.out_weight_sum->as<float>();
  const double one_minus_a = 1.0 - alpha;

  std::vector<double> r(pvec_canon, pvec_canon + V);  // r0 = pvec
  std::vector<double> r_prev(V, 0.0);
  std::vector<double> contrib(V, 0.0);
  std::vector<double> tile_part(c.plan->n_slots);
  int done = 0;
  while (true) {
    const int batch = boundary::next_audit_batch(done, max_iter,
                                                 audit_interval);
    if (batch <= 0) break;
    for (int it = 0; it < batch; ++it) {
      contrib_pass(V, ows, r.data(), contrib.data(), true);
      const double dmass = dangling_mass(g, r.data());
      gather_pass(c, contrib.data(), pvec_canon, alpha, one_minus_a, dmass,
                  r_prev.data(), tile_part.data(), true);
      r.swap(r_prev);  // r = new iterate; r_prev = previous
    }
    done += batch;
    const double err = l1_diff(V, r.data(), r_prev.data());
    if (err < static_cast<double>(V) * tol) break;
  }
  std::copy(r.begin(), r.end(), rank_canon);
  return done;
}

int ppr(Graph& g, const uint32_t* seeds_canon, const float* seed_w,
        const uint64_t* offsets, uint32_t n_queries, double alpha, double tol,
        int max_iter, int audit_interval,
        const std::function<void(uint32_t, const double*)>& sink,
        const std::function<int(uint32_t)>& pre) {
  if (n_queries == 0) return 0;
  const uint32_t V = g.V;
  if (V == 0) return 0;
  if (audit_interval < 1) audit_interval = 1;
  if (max_iter < 0) max_iter = 0;
  const PullCsr c = in_csr(g);
  const float* ows = g.out_weight_sum->as<float>();
  const double one_minus_a = 1.0 - alpha;

  // Validate seed-weight sums up front: worker threads must not throw.
  std::vector<double> wsum(n_queries, 0.0);
  for (uint32_t q = 0; q < n_queries; ++q) {
    for (uint64_t i = offsets[q]; i < offsets[q + 1]; ++i) {
      const double w = static_cast<double>(seed_w[i]);
      if (!std::isfinite(w) || w < 0.0)
        throw_invalid("ppr: seed weights must be finite and >= 0");
      wsum[q] += w;
    }
    if (!(wsum[q] > 0.0) || !std::isfinite(wsum[q]))
      throw_invalid(
          "ppr: seed weights must have a finite positive sum per query");
  }

  std::vector<int> iters(n_queries, 0);
  auto run_query = [&](uint32_t q, bool threaded) {
    if (pre) {
      const int it = pre(q);
      if (it >= 0) {
        iters[q] = it;
        return;
      }
    }
    const uint64_t s0 = offsets[q], s1 = offsets[q + 1];
    const std::size_t ns = static_cast<std::size_t>(s1 - s0);
    std::vector<uint32_t> sv(ns);
    std::vector<double> sw(ns);  // normalized teleport weight per seed
    for (std::size_t i = 0; i < ns; ++i) {
      sv[i] = seeds_canon[s0 + i];
      sw[i] = static_cast<double>(seed_w[s0 + i]) / wsum[q];
    }
    std::vector<double> r(V, 0.0), r_prev(V, 0.0), contrib(V, 0.0);
    std::vector<double> tile_part(c.plan->n_slots);
    for (std::size_t i = 0; i < ns; ++i) r[sv[i]] += sw[i];  // r0 = p
    int done = 0;
    while (true) {
      const int batch = boundary::next_audit_batch(done, max_iter,
                                                   audit_interval);
      if (batch <= 0) break;
      for (int it = 0; it < batch; ++it) {
        contrib_pass(V, ows, r.data(), contrib.data(), threaded);
        const double dmass = dangling_mass(g, r.data());
        gather_pass(c, contrib.data(), nullptr, alpha, one_minus_a, dmass,
                    r_prev.data(), tile_part.data(), threaded);
        // Sparse teleport + dangling term at the seeds only (p[v] = 0
        // elsewhere); mirrors mg_pr_seed_scatter_b.
        for (std::size_t i = 0; i < ns; ++i)
          r_prev[sv[i]] += (one_minus_a + alpha * dmass) * sw[i];
        r.swap(r_prev);
      }
      done += batch;
      if (l1_diff(V, r.data(), r_prev.data()) < static_cast<double>(V) * tol)
        break;
    }
    iters[q] = done;
    // Deliver the final vector to the sink from inside the worker (no
    // serialization): the batched entry point's sink writes only the
    // disjoint output rows for q, so concurrent calls for DISTINCT q are
    // safe by contract (see algos.hpp). Only this worker's dense state is
    // live — never a B x V accumulation.
    sink(q, r.data());
  };

  if (n_queries == 1) {
    run_query(0, true);
  } else {
    // Queries claimed one at a time (P/E-core speed differences and
    // n_queries not divisible by the thread count no longer idle threads).
    parallel_items(n_queries, [&](std::size_t q) {
      run_query(static_cast<uint32_t>(q), false);
    });
  }
  return *std::max_element(iters.begin(), iters.end());
}

int ppr_topk_sparse(Graph& g, const uint32_t* seeds_canon,
                    const float* seed_w, uint32_t n_seeds, double alpha,
                    double tol, int max_iter, int audit_interval,
                    uint32_t max_vertices, uint64_t max_edges, uint32_t k,
                    int32_t* out_ids, float* out_scores) {
  const uint32_t V = g.V;
  if (V == 0 || n_seeds == 0 || max_vertices == 0 || max_edges == 0)
    return -1;
  if (audit_interval < 1) audit_interval = 1;
  if (max_iter < 0) max_iter = 0;
  const Orientation& oo = g.orientation(Dir::out);
  if (!oo.edge_orig) return -1;
  const uint32_t* row = oo.row_offsets->as<uint32_t>();
  const uint32_t* col = oo.col_indices->as<uint32_t>();
  const uint32_t* orig = oo.edge_orig->as<uint32_t>();
  const float* wt =
      (g.weighted && oo.weights) ? oo.weights->as<float>() : nullptr;
  const float* ows_g = g.out_weight_sum->as<float>();

  // 1) Bounded OUT-reach of the seeds (BFS order). R is closed under
  // out-edges, so r == 0 exactly outside R on every iteration.
  const uint32_t vcap = std::min(max_vertices, V);
  LocalIndex idx(vcap);
  std::vector<uint32_t> reach;
  reach.reserve(std::min<uint32_t>(vcap, 256u));
  bool fresh = false;
  for (uint32_t i = 0; i < n_seeds; ++i) {
    idx.insert(seeds_canon[i], &fresh);
    if (!fresh) continue;
    if (reach.size() >= vcap) return -1;
    reach.push_back(seeds_canon[i]);
  }
  uint64_t scanned = 0;
  for (std::size_t i = 0; i < reach.size(); ++i) {
    const uint32_t u = reach[i];
    const uint64_t deg = uint64_t{row[u + 1]} - row[u];
    if (deg > max_edges - scanned) return -1;
    scanned += deg;
    for (uint32_t e = row[u]; e < row[u + 1]; ++e) {
      idx.insert(col[e], &fresh);
      if (!fresh) continue;
      if (reach.size() >= vcap) return -1;
      reach.push_back(col[e]);
    }
  }

  // 2) Local system in ascending canonical order (matches the dense sums:
  // dangling list and audit are ascending canonical; zero terms outside R
  // do not change an fp64 sum).
  std::sort(reach.begin(), reach.end());
  const uint32_t n = static_cast<uint32_t>(reach.size());
  for (uint32_t i = 0; i < n; ++i) idx.at(reach[i]) = i;

  // Local IN-CSR whose rows list in-edges in ORIGINAL input-edge order:
  // that is the stored order of the global IN rows (and of the symmetrized
  // rows of an undirected graph), so each gather sums in cpu::ppr's order.
  struct LEdge {
    uint32_t dst, orig, src;
    float w;
  };
  std::vector<LEdge> le;
  le.reserve(static_cast<std::size_t>(scanned));
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t u = reach[i];
    for (uint32_t e = row[u]; e < row[u + 1]; ++e)
      le.push_back({idx.at(col[e]), orig[e], i, wt ? wt[e] : 1.0f});
  }
  std::sort(le.begin(), le.end(), [](const LEdge& a, const LEdge& b) {
    return a.dst != b.dst ? a.dst < b.dst : a.orig < b.orig;
  });
  std::vector<uint32_t> in_row(std::size_t{n} + 1, 0);
  for (const LEdge& x : le) ++in_row[x.dst + 1];
  for (uint32_t i = 0; i < n; ++i) in_row[i + 1] += in_row[i];

  std::vector<float> ows(n);
  std::vector<uint32_t> dang;  // local ids, ascending canonical
  for (uint32_t i = 0; i < n; ++i) {
    ows[i] = ows_g[reach[i]];
    if (ows[i] == 0.0f) dang.push_back(i);
  }

  // Teleport: identical normalization to cpu::ppr.
  double wsum = 0.0;
  for (uint32_t i = 0; i < n_seeds; ++i) wsum += static_cast<double>(seed_w[i]);
  if (!(wsum > 0.0) || !std::isfinite(wsum)) return -1;
  std::vector<uint32_t> sl(n_seeds);
  std::vector<double> sw(n_seeds);
  for (uint32_t i = 0; i < n_seeds; ++i) {
    sl[i] = idx.at(seeds_canon[i]);
    sw[i] = static_cast<double>(seed_w[i]) / wsum;
  }

  // 3) The cpu::ppr iteration restricted to R.
  const double one_minus_a = 1.0 - alpha;
  const double thresh = static_cast<double>(V) * tol;
  std::vector<double> r(n, 0.0), r_prev(n, 0.0), contrib(n, 0.0);
  for (uint32_t i = 0; i < n_seeds; ++i) r[sl[i]] += sw[i];
  int done = 0;
  while (true) {
    const int batch =
        boundary::next_audit_batch(done, max_iter, audit_interval);
    if (batch <= 0) break;
    for (int it = 0; it < batch; ++it) {
      for (uint32_t u = 0; u < n; ++u)
        contrib[u] =
            ows[u] > 0.0f ? r[u] / static_cast<double>(ows[u]) : 0.0;
      double dmass = 0.0;
      for (uint32_t d : dang) dmass += r[d];
      for (uint32_t v = 0; v < n; ++v) {
        double acc = 0.0;
        if (wt) {
          for (uint32_t j = in_row[v]; j < in_row[v + 1]; ++j)
            acc += contrib[le[j].src] * static_cast<double>(le[j].w);
        } else {
          for (uint32_t j = in_row[v]; j < in_row[v + 1]; ++j)
            acc += contrib[le[j].src];
        }
        r_prev[v] = alpha * acc;
      }
      for (uint32_t i = 0; i < n_seeds; ++i)
        r_prev[sl[i]] += (one_minus_a + alpha * dmass) * sw[i];
      r.swap(r_prev);
    }
    done += batch;
    if (l1_diff(n, r.data(), r_prev.data()) < thresh) break;
  }

  // 4) Top-k with boundary::topk_user's order (score desc in fp64, ties by
  // ascending USER index). Every vertex outside R scores exactly 0, so the
  // ranked prefix is R's positive scores followed by the ascending-user
  // pool of all zero-score vertices.
  std::vector<std::pair<double, uint32_t>> pos;
  pos.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    if (r[i] > 0.0)
      pos.emplace_back(r[i], g.user_of_canon[reach[i]]);
    else if (!(r[i] == 0.0))
      return -1;  // unreachable for valid weights; defer to the dense path
  }
  const uint32_t kk = std::min<uint32_t>(k, V);
  const std::size_t npos = std::min<std::size_t>(pos.size(), kk);
  std::partial_sort(pos.begin(), pos.begin() + npos, pos.end(),
                    [](const std::pair<double, uint32_t>& a,
                       const std::pair<double, uint32_t>& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                    });
  uint32_t o = 0;
  for (; o < npos; ++o) {
    out_ids[o] = static_cast<int32_t>(pos[o].second);
    out_scores[o] = static_cast<float>(pos[o].first);
  }
  if (o < kk) {
    // Every positive vertex is already emitted; skip them in the zero pool.
    std::vector<uint32_t> skip(pos.size());
    for (std::size_t i = 0; i < pos.size(); ++i) skip[i] = pos[i].second;
    std::sort(skip.begin(), skip.end());
    std::size_t s = 0;
    for (uint32_t u = 0; o < kk; ++u) {
      if (s < skip.size() && skip[s] == u) {
        ++s;
        continue;
      }
      out_ids[o] = static_cast<int32_t>(u);
      out_scores[o] = 0.0f;
      ++o;
    }
  }
  for (; o < k; ++o) {
    out_ids[o] = -1;
    out_scores[o] = 0.0f;
  }
  return done;
}

}  // namespace mg::cpu
