// reduce_engine.hpp — host dispatch driver for grid-stride list-sum,
// dangling-mass finalize, and GPU residual audit reductions (reduce.metal).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>

#include "../kernels/mg_params.h"
#include "../runtime/runtime.hpp"

namespace mg {

class ReduceEngine {
 public:
  static constexpr uint32_t kMaxBlocks = 1024u;
  static constexpr uint32_t kItemsPerThread = 4u;
  static constexpr uint32_t kSingleTgMax = 1024u;

  ReduceEngine(Runtime& rt, bool batched);

  static uint32_t n_blocks(uint32_t count) {
    if (count == 0) return 0;
    return std::min(kMaxBlocks, ceil_div_u32(count, MG_TG_SIZE * kItemsPerThread));
  }

  // Single-query PageRank dangling mass + isolated-scale advance.
  // No-op when both n_active_dangling == 0 and n_iso == 0.
  void encode_pr_dangling(CommandBatch& cb, float alpha, float one_minus_alpha,
                          float iso_p_sum, uint32_t n_active_dangling,
                          uint32_t n_iso, Buffer* active_list, Buffer* values,
                          Buffer* partials, Buffer* scalars);

  // dst[dst_index] = sum of values[0..count) (HITS L1 norm).
  void encode_sum(CommandBatch& cb, MGPrParams base, uint32_t count,
                  Buffer* values, Buffer* partials, Buffer* dst,
                  uint32_t dst_index);

  // Single-query GPU L1 residual partials over vertices[0..count).
  // Returns the number of partials written to `partials` (0 when count == 0).
  uint32_t encode_residual(CommandBatch& cb, uint32_t count, bool rescale_ows,
                           Buffer* vertices, Buffer* buf_a, Buffer* buf_b,
                           Buffer* out_weight_sum, Buffer* partials);

  // Batched PPR dangling mass over active_dangling + isolated seeds.
  void encode_pr_dangling_b(CommandBatch& cb, MGPprParams base,
                            uint32_t n_active_dangling, bool has_iso_seed,
                            Buffer* active_list, Buffer* values_b,
                            Buffer* partials_b, Buffer* seed_vertex,
                            Buffer* seed_lane, Buffer* dst);

  // Batched PPR GPU L1 residual partials over vertices[0..count).
  // Returns the number of block partials written per lane (0 when count == 0).
  uint32_t encode_residual_b(CommandBatch& cb, uint32_t count,
                             uint32_t active_mask, bool rescale_ows,
                             Buffer* vertices, Buffer* buf_a, Buffer* buf_b,
                             Buffer* out_weight_sum, Buffer* partials_b);

 private:
  Runtime& rt_;
  MTL::ComputePipelineState* dangling_ = nullptr;
  MTL::ComputePipelineState* dangling_fin_ = nullptr;
  MTL::ComputePipelineState* sum_ = nullptr;
  MTL::ComputePipelineState* finalize_ = nullptr;
  MTL::ComputePipelineState* residual_ = nullptr;
  MTL::ComputePipelineState* dangling_b_ = nullptr;
  MTL::ComputePipelineState* finalize_b_ = nullptr;
  MTL::ComputePipelineState* residual_b_ = nullptr;
};

}  // namespace mg
