// spmv_engine.hpp — host dispatch driver for the PageRank / batched-PPR
// gather pipelines (spmv.metal). Encodes one full power iteration per call
// (prepare -> degree-binned gather over the IN worklists -> zero-fill, plus
// seed scatter on the batched tile path); the dangling reduce is
// ReduceEngine's job. Internal engine layer: algo entry points own buffers,
// ping-pong, and iteration/audit policy. Bindings follow the tables in
// src/kernels/mg_params.h (params at buffer 0, listed buffers from 1).
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
  // single-query set (gather uses MG_FC_WEIGHTED + MG_FC_UNIFORM_P),
  // batched=true the _b set (MG_FC_WEIGHTED only; teleport and dangling
  // mass go through mg_pr_seed_scatter_b). Requires a live GPU.
  SpmvEngine(Runtime& rt, bool weighted, bool uniform_p, bool batched);

  // One single-query iteration: mg_pr_prepare -> mg_pr_gather_{tg,sg,thread}
  // (zero-count bins skipped) -> mg_pr_zero_fill. `base` carries
  // alpha/one_minus_alpha/p_uniform/v_count; count is set per dispatch.
  // pvec must be Runtime::dummy() when uniform_p, weights when unweighted.
  void encode_iteration(CommandBatch& cb, const Orientation& in_o,
                        MGPrParams base, Buffer* rank_cur, Buffer* contrib,
                        Buffer* rank_next, Buffer* iter_scalars, Buffer* pvec,
                        Buffer* out_weight_sum, Buffer* weights);

  // One batched tile iteration: mg_pr_prepare_b -> gather _b bins ->
  // mg_pr_zero_fill_b -> mg_pr_seed_scatter_b. `base` carries alpha/
  // one_minus_alpha/v_count/active_mask/seed_total. Seeds must be
  // pre-deduplicated per (vertex, lane) — single writer per slot.
  void encode_iteration_b(CommandBatch& cb, const Orientation& in_o,
                          MGPprParams base, Buffer* rank_cur_b,
                          Buffer* contrib_b, Buffer* rank_next_b,
                          Buffer* iter_scalars_b, Buffer* out_weight_sum,
                          Buffer* weights, Buffer* seed_vertex,
                          Buffer* seed_lane, Buffer* seed_weight);

 private:
  // Huge-bin tile partials scratch, sized on first encode for the
  // orientation in use (engines are per-call; orientation is fixed).
  Buffer* huge_scratch(const Orientation& o, bool batched);

  Runtime& rt_;
  MTL::ComputePipelineState* prepare_ = nullptr;
  MTL::ComputePipelineState* gather_tg_ = nullptr;
  MTL::ComputePipelineState* gather_sg_ = nullptr;
  MTL::ComputePipelineState* gather_th_ = nullptr;
  MTL::ComputePipelineState* zero_fill_ = nullptr;
  MTL::ComputePipelineState* huge_partials_ = nullptr;
  MTL::ComputePipelineState* huge_finalize_ = nullptr;
  MTL::ComputePipelineState* prepare_b_ = nullptr;
  MTL::ComputePipelineState* gather_tg_b_ = nullptr;
  MTL::ComputePipelineState* gather_sg_b_ = nullptr;
  MTL::ComputePipelineState* gather_th_b_ = nullptr;
  MTL::ComputePipelineState* zero_fill_b_ = nullptr;
  MTL::ComputePipelineState* seed_scatter_b_ = nullptr;
  MTL::ComputePipelineState* huge_partials_b_ = nullptr;
  MTL::ComputePipelineState* huge_finalize_b_ = nullptr;
  BufferPtr huge_scratch_;
  uint32_t sgs_per_tg_ = 8;    // MG_TG_SIZE / exec_width(gather_sg)
  uint32_t sgs_per_tg_b_ = 8;  // MG_TG_SIZE / exec_width(gather_sg_b)
};

}  // namespace mg
