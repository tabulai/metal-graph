// spmv_engine.hpp — host dispatch driver for the fused single-dispatch
// PageRank / HITS / batched-PPR gather pipelines (spmv.metal).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "../graph/graph.hpp"
#include "../kernels/mg_params.h"
#include "../runtime/runtime.hpp"

namespace mg {

class SpmvEngine {
 public:
  // Caches the pipelines for one configuration. batched=false builds the
  // single-query set (MG_FC_WEIGHTED + MG_FC_UNIFORM_P), batched=true the _b
  // set (MG_FC_WEIGHTED only). Requires a live GPU.
  SpmvEngine(Runtime& rt, bool weighted, bool uniform_p, bool batched);

  // One fused single-query PageRank iteration.
  // When rescale_ows == true: `in_buf` already holds contrib (rank/ows for
  // ows>0, rank for dangling), no prepare runs, and `out_buf` receives the
  // next contrib. When rescale_ows == false (extreme weights): runs
  // mg_pr_prepare into `contrib_scratch` and writes raw rank to `out_buf`.
  // `include_iso`: when true, writes isolated vertices too (audit boundary).
  void encode_iteration(CommandBatch& cb, const Orientation& in_o,
                        const GpuGatherPlan& plan, MGPrParams base,
                        Buffer* in_buf, Buffer* contrib_scratch,
                        Buffer* out_buf, Buffer* iter_scalars, Buffer* pvec,
                        Buffer* out_weight_sum, Buffer* weights,
                        bool rescale_ows, bool include_iso);

  // Raw prepared adjacency-vector multiply for HITS (writes acc directly for
  // every vertex including zero-degree vertices, no alpha/pvec/rescale).
  void encode_prepared_gather(CommandBatch& cb, const Orientation& orientation,
                              const GpuGatherPlan& plan, MGPrParams base,
                              Buffer* input, Buffer* output, Buffer* weights);

  // One fused batched PPR tile iteration (8 lanes).
  // When rescale_ows == true: `in_buf_b` holds contrib_b, `contrib_scratch_b`
  // is unused, and `out_buf_b` receives next contrib_b.
  void encode_iteration_b(CommandBatch& cb, const Orientation& in_o,
                          const GpuGatherPlan& plan, MGPprParams base,
                          Buffer* in_buf_b, Buffer* contrib_scratch_b,
                          Buffer* out_buf_b, Buffer* iter_scalars_b,
                          Buffer* out_weight_sum, Buffer* weights,
                          Buffer* seed_vertex, Buffer* seed_lane,
                          Buffer* seed_weight, bool rescale_ows);

 private:
  Buffer* tile_scratch(const GpuGatherPlan& plan, bool batched);

  Runtime& rt_;
  MTL::ComputePipelineState* prepare_ = nullptr;
  MTL::ComputePipelineState* gather_fused_ = nullptr;
  MTL::ComputePipelineState* tiled_finalize_ = nullptr;
  MTL::ComputePipelineState* prepare_b_ = nullptr;
  MTL::ComputePipelineState* gather_fused_b_ = nullptr;
  MTL::ComputePipelineState* tiled_finalize_b_ = nullptr;
  MTL::ComputePipelineState* seed_scatter_b_ = nullptr;
  BufferPtr tile_scratch_;
  uint32_t sgs_per_tg_ = 8;
  uint32_t sgs_per_tg_b_ = 8;
};

}  // namespace mg
