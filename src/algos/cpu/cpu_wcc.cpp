// cpu_wcc.cpp — sequential union-find WCC oracle (CANONICAL space).
// Path halving + union by smaller root id, so each component's
// representative is its MINIMUM canonical id. Deterministic. Directed edges
// are treated as undirected by uniting both endpoints of every stored OUT
// edge (matches the GPU hook-both-endpoints semantics).
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <numeric>
#include <vector>

#include "../algos.hpp"

namespace mg::cpu {

int wcc(Graph& g, uint32_t* labels_canon) {
  const uint32_t V = g.V;
  if (V == 0) return 0;
  const Orientation& o = g.orientation(Dir::out);
  const uint32_t* row = o.row_offsets->as<uint32_t>();
  const uint32_t* col = o.col_indices->as<uint32_t>();

  std::vector<uint32_t> parent(V);
  std::iota(parent.begin(), parent.end(), 0u);
  auto find = [&](uint32_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];  // path halving
      x = parent[x];
    }
    return x;
  };

  for (uint32_t v = 0; v < V; ++v) {
    for (uint32_t e = row[v]; e < row[v + 1]; ++e) {
      const uint32_t ru = find(v);
      const uint32_t rv = find(col[e]);
      if (ru == rv) continue;
      if (ru < rv)
        parent[rv] = ru;  // smaller root wins -> min-id representative
      else
        parent[ru] = rv;
    }
  }

  for (uint32_t v = 0; v < V; ++v) labels_canon[v] = find(v);
  int n_components = 0;
  for (uint32_t v = 0; v < V; ++v)
    if (labels_canon[v] == v) ++n_components;
  return n_components;
}

}  // namespace mg::cpu
