// cpu_hits.cpp — threaded fp64 HITS oracle/fallback (CANONICAL space).
// Edge weights are intentionally ignored: every stored edge contributes one
// link, matching the public HITS contract and cuGraph's unweighted semantics.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../algos.hpp"

namespace mg::cpu {
namespace {

void gather(const Orientation& o, const double* input, double* output,
            uint32_t v_count) {
  const uint32_t* row = o.row_offsets->as<uint32_t>();
  const uint32_t* col = o.col_indices->as<uint32_t>();
  parallel_for(v_count, [&](std::size_t begin, std::size_t end) {
    for (std::size_t v = begin; v < end; ++v) {
      double sum = 0.0;
      for (uint32_t e = row[v]; e < row[v + 1]; ++e) sum += input[col[e]];
      output[v] = sum;
    }
  });
}

double normalize_l1(double* values, uint32_t count) {
  double norm = 0.0;
  for (uint32_t i = 0; i < count; ++i) norm += values[i];
  if (!(norm > 0.0) || !std::isfinite(norm))
    throw Error(ErrorCode::internal,
                "hits: encountered a degenerate normalization");
  const double inv = 1.0 / norm;
  parallel_for(count, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) values[i] *= inv;
  });
  return norm;
}

double l1_diff(const double* a, const double* b, uint32_t count) {
  double sum = 0.0;
  for (uint32_t i = 0; i < count; ++i) sum += std::fabs(a[i] - b[i]);
  return sum;
}

}  // namespace

int hits(Graph& g, double tol, int max_iter, int audit_interval,
         double* hubs_canon, double* authorities_canon) {
  const uint32_t V = g.V;
  if (V == 0) return 0;
  if (g.stored_edges() == 0) {
    std::fill_n(hubs_canon, V, 0.0);
    std::fill_n(authorities_canon, V, 0.0);
    return 0;
  }
  audit_interval = std::max(audit_interval, 1);

  const Orientation& in_o = g.orientation(Dir::in);
  const Orientation& out_o = g.orientation(Dir::out);
  std::vector<double> hubs[2] = {
      std::vector<double>(V, 1.0 / static_cast<double>(V)),
      std::vector<double>(V, 0.0)};
  std::vector<double> authorities(V, 0.0);

  int cur = 0;
  int done = 0;
  while (done < max_iter) {
    gather(in_o, hubs[cur].data(), authorities.data(), V);
    normalize_l1(authorities.data(), V);
    gather(out_o, authorities.data(), hubs[cur ^ 1].data(), V);
    normalize_l1(hubs[cur ^ 1].data(), V);
    ++done;

    const bool audit = (done % audit_interval == 0) || (done == max_iter);
    const double err = audit
                           ? l1_diff(hubs[cur ^ 1].data(), hubs[cur].data(), V)
                           : 0.0;
    cur ^= 1;
    if (audit && err < static_cast<double>(V) * tol) break;
  }

  std::copy_n(hubs[cur].data(), V, hubs_canon);
  std::copy_n(authorities.data(), V, authorities_canon);
  return done;
}

}  // namespace mg::cpu
