// cpu_pagerank.cpp — threaded CPU PageRank + batched PPR (fp64 oracles and
// planner fallback). Pull iteration over the IN orientation in CANONICAL
// space; semantics per src/algos/algos.hpp. Convergence is audited every
// audit_interval iterations (L1 of the LAST iteration only), matching the
// GPU command-batch protocol so iteration counts agree across paths.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../algos.hpp"
#include "../boundary.hpp"

namespace mg::cpu {

namespace {

struct PullCsr {
  const uint32_t* row = nullptr;
  const uint32_t* col = nullptr;
  const float* w = nullptr;  // null when unweighted
};

PullCsr in_csr(Graph& g) {
  const Orientation& o = g.orientation(Dir::in);
  PullCsr c;
  c.row = o.row_offsets->as<uint32_t>();
  c.col = o.col_indices->as<uint32_t>();
  if (g.weighted && o.weights) c.w = o.weights->as<float>();
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
void gather_pass(uint32_t V, const PullCsr& c, const double* contrib,
                 const double* p, double alpha, double one_minus_a,
                 double dmass, double* next, bool threaded) {
  for_range(V, threaded, [&](std::size_t b, std::size_t e) {
    for (std::size_t v = b; v < e; ++v) {
      double acc = 0.0;
      const uint32_t lo = c.row[v], hi = c.row[v + 1];
      if (c.w) {
        for (uint32_t k = lo; k < hi; ++k)
          acc += contrib[c.col[k]] * static_cast<double>(c.w[k]);
      } else {
        for (uint32_t k = lo; k < hi; ++k) acc += contrib[c.col[k]];
      }
      double x = alpha * acc;
      if (p) x += (one_minus_a + alpha * dmass) * p[v];
      next[v] = x;
    }
  });
}

double l1_diff(uint32_t V, const double* a, const double* b) {
  double err = 0.0;
  for (uint32_t v = 0; v < V; ++v) err += std::fabs(a[v] - b[v]);
  return err;
}

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
  int done = 0;
  while (true) {
    const int batch = boundary::next_audit_batch(done, max_iter,
                                                 audit_interval);
    if (batch <= 0) break;
    for (int it = 0; it < batch; ++it) {
      contrib_pass(V, ows, r.data(), contrib.data(), true);
      const double dmass = dangling_mass(g, r.data());
      gather_pass(V, c, contrib.data(), pvec_canon, alpha, one_minus_a,
                  dmass, r_prev.data(), true);
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
        const std::function<void(uint32_t, const double*)>& sink) {
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
    const uint64_t s0 = offsets[q], s1 = offsets[q + 1];
    const std::size_t ns = static_cast<std::size_t>(s1 - s0);
    std::vector<uint32_t> sv(ns);
    std::vector<double> sw(ns);  // normalized teleport weight per seed
    for (std::size_t i = 0; i < ns; ++i) {
      sv[i] = seeds_canon[s0 + i];
      sw[i] = static_cast<double>(seed_w[s0 + i]) / wsum[q];
    }
    std::vector<double> r(V, 0.0), r_prev(V, 0.0), contrib(V, 0.0);
    for (std::size_t i = 0; i < ns; ++i) r[sv[i]] += sw[i];  // r0 = p
    int done = 0;
    while (true) {
      const int batch = boundary::next_audit_batch(done, max_iter,
                                                   audit_interval);
      if (batch <= 0) break;
      for (int it = 0; it < batch; ++it) {
        contrib_pass(V, ows, r.data(), contrib.data(), threaded);
        const double dmass = dangling_mass(g, r.data());
        gather_pass(V, c, contrib.data(), nullptr, alpha, one_minus_a,
                    dmass, r_prev.data(), threaded);
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
    parallel_for(
        n_queries,
        [&](std::size_t b, std::size_t e) {
          for (std::size_t q = b; q < e; ++q)
            run_query(static_cast<uint32_t>(q), false);
        },
        1);
  }
  return *std::max_element(iters.begin(), iters.end());
}

}  // namespace mg::cpu
