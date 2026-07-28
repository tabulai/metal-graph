// reduce_engine.hpp — host dispatch driver for the list-sum reduce pattern
// (reduce.metal): per-threadgroup partial sums over a vertex list, then a
// one-threadgroup finalize into a scalar slot. Used for the per-iteration
// dangling-mass reduction of PageRank / batched PPR. Bindings follow the
// tables in src/kernels/mg_params.h.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "../kernels/mg_params.h"
#include "../runtime/runtime.hpp"

namespace mg {

class ReduceEngine {
 public:
  // batched=false caches mg_pr_dangling_partials + mg_reduce_finalize,
  // batched=true the _b variants. Requires a live GPU.
  ReduceEngine(Runtime& rt, bool batched);

  // Threadgroups of a partials pass over `count` items. Size the partials
  // buffer as n_blocks(count) floats — times MG_PPR_TILE for the batched
  // variant (layout partials[block * MG_PPR_TILE + lane]).
  static uint32_t n_blocks(uint32_t count) {
    return ceil_div_u32(count, MG_TG_SIZE);
  }

  // dst[dst_index] = sum over the list of values[list[i]].
  // `base` carries the iteration params; count is set to list_count here.
  // No-op when list_count == 0.
  void encode_list_sum(CommandBatch& cb, MGPrParams base, uint32_t list_count,
                       Buffer* list, Buffer* values, Buffer* partials,
                       Buffer* dst, uint32_t dst_index);

  // Per-lane variant: dst[lane] = sum over the list of
  // values_b[list[i] * MG_PPR_TILE + lane], for every lane.
  void encode_list_sum_b(CommandBatch& cb, MGPprParams base,
                         uint32_t list_count, Buffer* list, Buffer* values_b,
                         Buffer* partials, Buffer* dst);

 private:
  MTL::ComputePipelineState* dangling_ = nullptr;
  MTL::ComputePipelineState* finalize_ = nullptr;
  MTL::ComputePipelineState* dangling_b_ = nullptr;
  MTL::ComputePipelineState* finalize_b_ = nullptr;
};

}  // namespace mg
