// frontier_engine.hpp — host driver for the GPU-resident BFS level
// protocol and the k-hop induced-edge kernels (frontier.metal). Internal
// to the algorithm entry points; all ids are CANONICAL.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

#include "../graph/graph.hpp"
#include "../runtime/runtime.hpp"

namespace mg {

class FrontierEngine {
 public:
  explicit FrontierEngine(Runtime& rt);  // throws no_gpu without a device

  // GPU-resident multi-source BFS over orientation `dir`. sources_canon
  // must be deduplicated (unvisited counting depends on it). Writes
  // dist/parent (int32, canonical order, -1 = unvisited/none).
  // max_levels: 0 = unbounded, else at most that many levels are assigned
  // (k-hop truncation). Returns the number of BFS levels executed
  // (non-empty frontier expansions).
  int run_bfs(Graph& g, const std::vector<uint32_t>& sources_canon, Dir dir,
              bool enable_bottomup, uint32_t max_levels, int32_t* dist_canon,
              int32_t* parent_canon);

  // Induced-edge extraction over the OUT-orientation CSR for the reached
  // set (unique canonical ids). Returns original input edge ids, unsorted;
  // undirected symmetrization yields duplicates (caller sorts + uniques).
  std::vector<uint32_t> khop_edges(Graph& g,
                                   const std::vector<uint32_t>& reached_canon);

 private:
  Runtime& rt_;
};

}  // namespace mg
