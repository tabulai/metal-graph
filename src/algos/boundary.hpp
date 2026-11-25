// boundary.hpp — USER <-> CANONICAL translation helpers shared by the
// algorithm entry points (header-only).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../common/mg_common.hpp"
#include "../graph/graph.hpp"

namespace mg::boundary {

// Validate USER indices and map to canonical.
inline std::vector<uint32_t> to_canon(const Graph& g, const uint32_t* user,
                                      uint32_t n, const char* what) {
  std::vector<uint32_t> out(n);
  for (uint32_t i = 0; i < n; ++i) {
    if (user[i] >= g.V)
      throw_invalid(std::string(what) + ": vertex index out of range");
    out[i] = g.canon_of_user[user[i]];
  }
  return out;
}

// Validate a dense USER-order personalization and return its fp64 sum
// (deterministic fixed chunks, independent of MG_THREADS). Rejects
// non-finite/negative and all-zero vectors.
inline double personalization_sum(const Graph& g, const float* p_user) {
  const uint32_t V = g.V;
  constexpr std::size_t kChunk = std::size_t{1} << 16;
  const std::size_t chunks = (std::size_t{V} + kChunk - 1) / kChunk;
  if (chunks <= 1) {
    double s = 0.0;
    for (uint32_t u = 0; u < V; ++u) {
      const double v = p_user[u];
      if (!std::isfinite(v) || v < 0.0)
        throw_invalid("personalization: entries must be finite and >= 0");
      s += v;
    }
    if (!(s > 0.0) || !std::isfinite(s))
      throw_invalid("personalization: sum must be finite and > 0");
    return s;
  }
  std::vector<double> part(chunks, 0.0);
  std::atomic<bool> bad{false};
  parallel_for(
      chunks,
      [&](std::size_t cb, std::size_t ce) {
        for (std::size_t c = cb; c < ce; ++c) {
          const std::size_t lo = c * kChunk;
          const std::size_t hi = std::min<std::size_t>(V, lo + kChunk);
          double s = 0.0;
          for (std::size_t u = lo; u < hi; ++u) {
            const double v = p_user[u];
            if (!std::isfinite(v) || v < 0.0) {
              bad.store(true, std::memory_order_relaxed);
              return;
            }
            s += v;
          }
          part[c] = s;
        }
      },
      1);
  if (bad.load(std::memory_order_relaxed))
    throw_invalid("personalization: entries must be finite and >= 0");
  double s = 0.0;
  for (double x : part) s += x;
  if (!(s > 0.0) || !std::isfinite(s))
    throw_invalid("personalization: sum must be finite and > 0");
  return s;
}

// Dense personalization (USER order, may be null => uniform) -> normalized
// canonical fp64 vector: p[canon_of_user[u]] = p_user[u] / p_sum, where p_sum
// is personalization_sum(g, p_user) (validated by the caller).
inline std::vector<double> pvec_to_canon(const Graph& g, const float* p_user,
                                         double p_sum) {
  std::vector<double> p(g.V);
  if (!p_user) {
    const double u_val = 1.0 / static_cast<double>(g.V);
    parallel_for(g.V, [&](std::size_t b, std::size_t e) {
      std::fill(p.data() + b, p.data() + e, u_val);
    });
    return p;
  }
  const uint32_t* user_of = g.user_of_canon.data();
  parallel_for(g.V, [&](std::size_t b, std::size_t e) {
    for (std::size_t c = b; c < e; ++c)
      p[c] = static_cast<double>(p_user[user_of[c]]) / p_sum;
  });
  return p;
}

// Scatter a canonical-order vector into a USER-order output array.
template <typename TSrc, typename TDst>
inline void canon_to_user(const Graph& g, const TSrc* canon, TDst* user_out) {
  const uint32_t* canon_of = g.canon_of_user.data();
  parallel_for(g.V, [&](std::size_t b, std::size_t e) {
    for (std::size_t u = b; u < e; ++u)
      user_out[u] = static_cast<TDst>(canon[canon_of[u]]);
  });
}

// Translate a canonical parent/id VALUE array (already in USER row order)
// value-wise: canonical vertex values -> user indices; v < 0 passes through.
inline int32_t canon_value_to_user(const Graph& g, int32_t canon_value) {
  return canon_value < 0 ? canon_value
                         : static_cast<int32_t>(g.user_of_canon[canon_value]);
}

// Top-k selection over a canonical-order score vector. Ties break by
// ascending USER index; rows pad with id = -1 / score = 0 when k > V.
// Deterministic and shared by CPU/GPU ppr paths (the "CPU selection
// fallback behind the same API" of the plan).
template <typename T>
inline void topk_user(const Graph& g, const T* rank_canon, uint32_t stride,
                      uint32_t lane, uint32_t k, int32_t* out_ids,
                      float* out_scores) {
  const uint32_t V = g.V;
  std::vector<std::pair<T, uint32_t>> items(V);  // (score, user idx)
  for (uint32_t c = 0; c < V; ++c) {
    T s = rank_canon[static_cast<size_t>(c) * stride + lane];
    items[c] = {s, g.user_of_canon[c]};
  }
  auto cmp = [](const std::pair<T, uint32_t>& a,
                const std::pair<T, uint32_t>& b) {
    if (a.first != b.first) return a.first > b.first;
    return a.second < b.second;
  };
  uint32_t kk = std::min<uint32_t>(k, V);
  std::partial_sort(items.begin(), items.begin() + kk, items.end(), cmp);
  for (uint32_t i = 0; i < kk; ++i) {
    out_ids[i] = static_cast<int32_t>(items[i].second);
    out_scores[i] = static_cast<float>(items[i].first);
  }
  for (uint32_t i = kk; i < k; ++i) {
    out_ids[i] = -1;
    out_scores[i] = 0.0f;
  }
}

// Iteration batching: iterations to run before the next fp64 audit.
inline int next_audit_batch(int done, int max_iter, int audit_interval) {
  int remaining = max_iter - done;
  return remaining < audit_interval ? remaining : audit_interval;
}

}  // namespace mg::boundary
