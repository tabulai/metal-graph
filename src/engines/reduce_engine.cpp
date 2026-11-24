// reduce_engine.cpp — grid-stride partials + finalize encode patterns
// (reduce.metal).
// SPDX-License-Identifier: Apache-2.0
#include "reduce_engine.hpp"

namespace mg {

ReduceEngine::ReduceEngine(Runtime& rt, bool batched) : rt_(rt) {
  if (!batched) {
    dangling_ = rt.pipeline("mg_pr_dangling_partials");
    dangling_fin_ = rt.pipeline("mg_pr_dangling_finalize");
    sum_ = rt.pipeline("mg_sum_partials");
    finalize_ = rt.pipeline("mg_reduce_finalize");
    residual_ = rt.pipeline("mg_pr_residual_partials");
  } else {
    dangling_b_ = rt.pipeline("mg_pr_dangling_partials_b");
    finalize_b_ = rt.pipeline("mg_reduce_finalize_b");
    residual_b_ = rt.pipeline("mg_pr_residual_partials_b");
  }
}

void ReduceEngine::encode_pr_dangling(CommandBatch& cb, float alpha,
                                      float one_minus_alpha, float iso_p_sum,
                                      uint32_t n_active_dangling,
                                      uint32_t n_iso, Buffer* active_list,
                                      Buffer* values, Buffer* partials,
                                      Buffer* scalars) {
  if (n_active_dangling == 0 && n_iso == 0) return;
  MGPrDangFinParams fp{};
  fp.alpha = alpha;
  fp.one_minus_alpha = one_minus_alpha;
  fp.iso_p_sum = iso_p_sum;
  if (n_active_dangling > kSingleTgMax) {
    MGPrParams p{};
    p.count = n_active_dangling;
    const uint32_t blocks = n_blocks(n_active_dangling);
    cb.dispatch(dangling_, &p, sizeof(p), {active_list, values, partials},
                blocks, MG_TG_SIZE);
    fp.n_partials = blocks;
    fp.n_active = 0;
  } else {
    fp.n_partials = 0;
    fp.n_active = n_active_dangling;
  }
  Buffer* dummy = rt_.dummy().get();
  cb.dispatch(dangling_fin_, &fp, sizeof(fp),
              {partials ? partials : dummy,
               active_list ? active_list : dummy, values, scalars},
              1, MG_TG_SIZE);
}

void ReduceEngine::encode_sum(CommandBatch& cb, MGPrParams base, uint32_t count,
                              Buffer* values, Buffer* partials, Buffer* dst,
                              uint32_t dst_index) {
  if (count == 0) return;
  MGPrParams p = base;
  p.count = count;
  const uint32_t blocks = n_blocks(count);
  cb.dispatch(sum_, &p, sizeof(p), {values, partials}, blocks, MG_TG_SIZE);
  MGReduceParams rp{};
  rp.n_partials = blocks;
  rp.dst_index = dst_index;
  cb.dispatch(finalize_, &rp, sizeof(rp), {partials, dst}, 1, MG_TG_SIZE);
}

uint32_t ReduceEngine::encode_residual(CommandBatch& cb, uint32_t count,
                                       bool rescale_ows, Buffer* vertices,
                                       Buffer* buf_a, Buffer* buf_b,
                                       Buffer* out_weight_sum,
                                       Buffer* partials) {
  if (count == 0) return 0;
  const uint32_t blocks = n_blocks(count);
  MGPrResParams rp{};
  rp.count = count;
  rp.flags = rescale_ows ? MG_PR_FLAG_RESCALE_OWS : 0u;
  cb.dispatch(residual_, &rp, sizeof(rp),
              {vertices, buf_a, buf_b, out_weight_sum, partials}, blocks,
              MG_TG_SIZE);
  return blocks;
}

void ReduceEngine::encode_pr_dangling_b(
    CommandBatch& cb, MGPprParams base, uint32_t n_active_dangling,
    bool has_iso_seed, Buffer* active_list, Buffer* values_b,
    Buffer* partials_b, Buffer* seed_vertex, Buffer* seed_lane, Buffer* dst) {
  if (n_active_dangling == 0 && !has_iso_seed) return;
  MGPprDangFinParams fp{};
  fp.seed_total = has_iso_seed ? base.seed_total : 0u;
  fp.active_mask = base.active_mask;
  if (n_active_dangling > kSingleTgMax) {
    MGPprParams p = base;
    p.count = n_active_dangling;
    const uint32_t blocks = n_blocks(n_active_dangling);
    cb.dispatch(dangling_b_, &p, sizeof(p), {active_list, values_b, partials_b},
                blocks, MG_TG_SIZE);
    fp.n_partials = blocks;
    fp.n_active = 0;
  } else {
    fp.n_partials = 0;
    fp.n_active = n_active_dangling;
  }
  Buffer* dummy = rt_.dummy().get();
  cb.dispatch(finalize_b_, &fp, sizeof(fp),
              {partials_b ? partials_b : dummy,
               active_list ? active_list : dummy, seed_vertex, seed_lane,
               values_b, dst},
              1, MG_TG_SIZE);
}

uint32_t ReduceEngine::encode_residual_b(
    CommandBatch& cb, uint32_t count, uint32_t active_mask, bool rescale_ows,
    Buffer* vertices, Buffer* buf_a, Buffer* buf_b, Buffer* out_weight_sum,
    Buffer* partials_b) {
  if (count == 0) return 0;
  const uint32_t blocks = n_blocks(count);
  MGPrResParams rp{};
  rp.count = count;
  rp.flags = rescale_ows ? MG_PR_FLAG_RESCALE_OWS : 0u;
  rp.active_mask = active_mask;
  cb.dispatch(residual_b_, &rp, sizeof(rp),
              {vertices, buf_a, buf_b, out_weight_sum, partials_b}, blocks,
              MG_TG_SIZE);
  return blocks;
}

}  // namespace mg
