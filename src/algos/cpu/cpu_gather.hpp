// cpu_gather.hpp — edge-balanced pull gather over a CpuGatherPlan (see
// src/graph/graph.hpp) for the threaded CPU PageRank/PPR/HITS paths.
// Per-vertex fp64 sums run serially in stored-edge order; tiled (huge) rows
// sum fixed MG_HUGE_TILE_EDGES tiles, and the tile partials are added in
// slot order after the barrier. The bits therefore never depend on the
// thread count, on which thread claims an item, or on whether the gather
// runs threaded or inline (per-query PPR workers).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common/mg_common.hpp"
#include "../../graph/graph.hpp"

namespace mg::cpu {

// Plans with fewer items gather inline even when threading is allowed. On
// the persistent pool two 16k-cost items already break even, and more gain
// quickly (M4 Max: rmat11, 3 items, PageRank/HITS/PPR 1.6x faster threaded;
// rmat12 2.4x). Spawning threads for every pass had needed 8.
constexpr std::size_t kMinThreadedGatherItems = 2;

// Serial fp64 sum of x[col[k]] (* w[k] when weighted) over [lo, hi). Each
// product is rounded before it is added: under clang's default
// -ffp-contract=on, `acc += x * w` may or may not become one fused
// multiply-add depending on how the loop is unrolled (tile and row loops
// differed), and the bounded-reach PPR path must reproduce these bits.
inline double row_sum(const uint32_t* col, const float* w, const double* x,
                      uint32_t lo, uint32_t hi) {
#pragma clang fp contract(off)
  double acc = 0.0;
  if (w) {
    for (uint32_t k = lo; k < hi; ++k)
      acc += x[col[k]] * static_cast<double>(w[k]);
  } else {
    for (uint32_t k = lo; k < hi; ++k) acc += x[col[k]];
  }
  return acc;
}

// Calls finish(v, sum_in(v)) exactly once per vertex covered by `plan`
// (every vertex of the orientation). `tile_part` is caller scratch of at
// least plan.n_slots doubles (private to this call).
template <class Finish>
void plan_gather(const CpuGatherPlan& plan, const uint32_t* row,
                 const uint32_t* col, const float* w, const double* x,
                 double* tile_part, bool threaded, Finish&& finish) {
  auto run_item = [&](std::size_t i) {
    const CpuGatherItem& it = plan.items[i];
    if (it.slot >= 0) {
      tile_part[it.slot] = row_sum(col, w, x, it.e0, it.e1);
      return;
    }
    for (uint32_t v = it.v0; v < it.v1; ++v)
      finish(v, row_sum(col, w, x, row[v], row[v + 1]));
  };
  if (threaded && plan.items.size() >= kMinThreadedGatherItems) {
    parallel_items(plan.items.size(), run_item);
  } else {
    for (std::size_t i = 0; i < plan.items.size(); ++i) run_item(i);
  }
  for (std::size_t h = 0; h < plan.huge.size(); ++h) {
    double acc = 0.0;
    for (uint32_t s = plan.huge_off[h]; s < plan.huge_off[h + 1]; ++s)
      acc += tile_part[s];
    finish(plan.huge[h], acc);
  }
}

}  // namespace mg::cpu
