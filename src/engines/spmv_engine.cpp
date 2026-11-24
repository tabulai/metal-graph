// spmv_engine.cpp — encode fused single-dispatch PageRank / HITS / batched-PPR
// iterations (spmv.metal).
// SPDX-License-Identifier: Apache-2.0
#include "spmv_engine.hpp"

#include <algorithm>

namespace mg {

namespace {

inline uint32_t tg_count(uint32_t items) {
  return ceil_div_u32(items, MG_TG_SIZE);
}

inline uint32_t sgs_per_tg_of(Runtime& rt, MTL::ComputePipelineState* pso) {
  const uint32_t ew = rt.exec_width(pso);
  return ew ? std::max(1u, MG_TG_SIZE / ew) : 8u;
}

template <class Params>
uint32_t fill_bin_layout(Params& p, const GpuGatherPlan& plan,
                         uint32_t sgs_per_tg, bool include_iso) {
  p.n_tiles = plan.n_tiles;
  p.n_tiled = plan.n_tiled;
  p.off_tg = plan.off_tg;
  p.n_tg = plan.n_tg;
  p.off_sg32 = plan.off_sg32;
  p.n_sg32 = plan.n_sg32;
  p.sg32_tgs = ceil_div_u32(plan.n_sg32, sgs_per_tg);
  p.off_sg16 = plan.off_sg16;
  p.n_sg16 = plan.n_sg16;
  p.sg16_tgs = ceil_div_u32(plan.n_sg16, MG_TG_SIZE / 16u);
  p.off_sg8 = plan.off_sg8;
  p.n_sg8 = plan.n_sg8;
  p.sg8_tgs = ceil_div_u32(plan.n_sg8, MG_TG_SIZE / 8u);
  p.off_sg4 = plan.off_sg4;
  p.n_sg4 = plan.n_sg4;
  p.sg4_tgs = ceil_div_u32(plan.n_sg4, MG_TG_SIZE / 4u);
  p.off_th = plan.off_th;
  p.n_th = plan.n_th;
  p.th_tgs = tg_count(plan.n_th);
  p.off_zero = plan.off_zact;
  p.n_zero = plan.n_zact + (include_iso ? plan.n_iso : 0u);
  p.zero_tgs = tg_count(p.n_zero);
  return p.n_tiles + p.n_tg + p.sg32_tgs + p.sg16_tgs + p.sg8_tgs + p.sg4_tgs +
         p.th_tgs + p.zero_tgs;
}

}  // namespace

SpmvEngine::SpmvEngine(Runtime& rt, bool weighted, bool uniform_p,
                       bool batched)
    : rt_(rt) {
  if (!batched) {
    prepare_ = rt.pipeline("mg_pr_prepare");
    gather_fused_ = rt.pipeline("mg_pr_gather_fused", weighted, uniform_p);
    tiled_finalize_ = rt.pipeline("mg_pr_tiled_finalize", false, uniform_p);
    sgs_per_tg_ = sgs_per_tg_of(rt, gather_fused_);
  } else {
    prepare_b_ = rt.pipeline("mg_pr_prepare_b");
    gather_fused_b_ = rt.pipeline("mg_pr_gather_fused_b", weighted);
    tiled_finalize_b_ = rt.pipeline("mg_pr_tiled_finalize_b");
    seed_scatter_b_ = rt.pipeline("mg_pr_seed_scatter_b");
    sgs_per_tg_b_ = sgs_per_tg_of(rt, gather_fused_b_);
  }
}

Buffer* SpmvEngine::tile_scratch(const GpuGatherPlan& plan, bool batched) {
  if (plan.n_tiles == 0) return rt_.dummy().get();
  if (!tile_scratch_) {
    const std::size_t n = std::size_t{plan.n_tiles} *
                          (batched ? MG_PPR_TILE : 1u) * sizeof(float);
    tile_scratch_ = rt_.alloc_scratch(n, "pr.tile_partials");
  }
  return tile_scratch_.get();
}

void SpmvEngine::encode_iteration(CommandBatch& cb, const Orientation& in_o,
                                  const GpuGatherPlan& plan, MGPrParams base,
                                  Buffer* in_buf, Buffer* contrib_scratch,
                                  Buffer* out_buf, Buffer* iter_scalars,
                                  Buffer* pvec, Buffer* out_weight_sum,
                                  Buffer* weights, bool rescale_ows,
                                  bool include_iso) {
  Buffer* gather_in = in_buf;
  if (!rescale_ows) {
    MGPrParams pp = base;
    pp.count = base.v_count;
    cb.dispatch(prepare_, &pp, sizeof(pp),
                {in_buf, out_weight_sum, contrib_scratch}, tg_count(pp.count),
                MG_TG_SIZE);
    gather_in = contrib_scratch;
  }

  MGFusedPrParams fp{};
  fp.alpha = base.alpha;
  fp.one_minus_alpha = base.one_minus_alpha;
  fp.p_uniform = base.p_uniform;
  fp.flags = rescale_ows ? MG_PR_FLAG_RESCALE_OWS : 0u;
  const uint32_t total_tgs =
      fill_bin_layout(fp, plan, sgs_per_tg_, include_iso);
  Buffer* partials = tile_scratch(plan, /*batched=*/false);

  if (total_tgs > 0) {
    cb.dispatch(gather_fused_, &fp, sizeof(fp),
                {plan.vertices.get(), plan.tile_owner.get(),
                 plan.tile_first_edge.get(), in_o.row_offsets.get(),
                 in_o.col_indices.get(), gather_in, out_buf, out_weight_sum,
                 iter_scalars, pvec, weights, partials},
                total_tgs, MG_TG_SIZE);
  }
  if (plan.n_tiled > 0) {
    cb.dispatch(tiled_finalize_, &fp, sizeof(fp),
                {plan.vertices.get(), plan.tiled_tile_off.get(), partials,
                 out_buf, out_weight_sum, iter_scalars, pvec},
                ceil_div_u32(plan.n_tiled, sgs_per_tg_), MG_TG_SIZE);
  }
}

void SpmvEngine::encode_prepared_gather(
    CommandBatch& cb, const Orientation& orientation, const GpuGatherPlan& plan,
    MGPrParams base, Buffer* input, Buffer* output, Buffer* weights) {
  Buffer* dummy = rt_.dummy().get();
  MGFusedPrParams fp{};
  fp.alpha = base.alpha;
  fp.one_minus_alpha = base.one_minus_alpha;
  fp.p_uniform = base.p_uniform;
  fp.flags = MG_PR_FLAG_RAW_GATHER;
  const uint32_t total_tgs =
      fill_bin_layout(fp, plan, sgs_per_tg_, /*include_iso=*/true);
  Buffer* partials = tile_scratch(plan, /*batched=*/false);

  if (total_tgs > 0) {
    cb.dispatch(gather_fused_, &fp, sizeof(fp),
                {plan.vertices.get(), plan.tile_owner.get(),
                 plan.tile_first_edge.get(), orientation.row_offsets.get(),
                 orientation.col_indices.get(), input, output, dummy, dummy,
                 dummy, weights, partials},
                total_tgs, MG_TG_SIZE);
  }
  if (plan.n_tiled > 0) {
    cb.dispatch(tiled_finalize_, &fp, sizeof(fp),
                {plan.vertices.get(), plan.tiled_tile_off.get(), partials,
                 output, dummy, dummy, dummy},
                ceil_div_u32(plan.n_tiled, sgs_per_tg_), MG_TG_SIZE);
  }
}

void SpmvEngine::encode_iteration_b(CommandBatch& cb, const Orientation& in_o,
                                    const GpuGatherPlan& plan, MGPprParams base,
                                    Buffer* in_buf_b, Buffer* contrib_scratch_b,
                                    Buffer* out_buf_b, Buffer* iter_scalars_b,
                                    Buffer* out_weight_sum, Buffer* weights,
                                    Buffer* seed_vertex, Buffer* seed_lane,
                                    Buffer* seed_weight, bool rescale_ows) {
  Buffer* gather_in = in_buf_b;
  if (!rescale_ows) {
    MGPprParams pp = base;
    pp.count = base.v_count;
    cb.dispatch(prepare_b_, &pp, sizeof(pp),
                {in_buf_b, out_weight_sum, contrib_scratch_b},
                tg_count(pp.count), MG_TG_SIZE);
    gather_in = contrib_scratch_b;
  }

  MGFusedPprParams fp{};
  fp.alpha = base.alpha;
  fp.one_minus_alpha = base.one_minus_alpha;
  fp.active_mask = base.active_mask;
  fp.flags = rescale_ows ? MG_PR_FLAG_RESCALE_OWS : 0u;
  const uint32_t total_tgs =
      fill_bin_layout(fp, plan, sgs_per_tg_b_, /*include_iso=*/false);
  Buffer* partials = tile_scratch(plan, /*batched=*/true);

  if (total_tgs > 0) {
    cb.dispatch(gather_fused_b_, &fp, sizeof(fp),
                {plan.vertices.get(), plan.tile_owner.get(),
                 plan.tile_first_edge.get(), in_o.row_offsets.get(),
                 in_o.col_indices.get(), gather_in, out_buf_b, in_buf_b,
                 out_weight_sum, weights, partials},
                total_tgs, MG_TG_SIZE);
  }
  if (plan.n_tiled > 0) {
    cb.dispatch(tiled_finalize_b_, &fp, sizeof(fp),
                {plan.vertices.get(), plan.tiled_tile_off.get(), partials,
                 out_buf_b, in_buf_b, out_weight_sum},
                ceil_div_u32(plan.n_tiled, sgs_per_tg_b_), MG_TG_SIZE);
  }
  if (base.seed_total > 0) {
    MGPprParams sp = base;
    sp.count = base.seed_total;
    sp.pad0 = rescale_ows ? MG_PR_FLAG_RESCALE_OWS : 0u;
    cb.dispatch(seed_scatter_b_, &sp, sizeof(sp),
                {seed_vertex, seed_lane, seed_weight, iter_scalars_b,
                 out_weight_sum, out_buf_b, in_buf_b},
                tg_count(base.seed_total), MG_TG_SIZE);
  }
}

}  // namespace mg
