// reduce_engine.cpp — partials + finalize encode pattern (reduce.metal).
// SPDX-License-Identifier: Apache-2.0
#include "reduce_engine.hpp"

namespace mg {

ReduceEngine::ReduceEngine(Runtime& rt, bool batched) {
  if (!batched) {
    dangling_ = rt.pipeline("mg_pr_dangling_partials");
    finalize_ = rt.pipeline("mg_reduce_finalize");
  } else {
    dangling_b_ = rt.pipeline("mg_pr_dangling_partials_b");
    finalize_b_ = rt.pipeline("mg_reduce_finalize_b");
  }
}

void ReduceEngine::encode_list_sum(CommandBatch& cb, MGPrParams base,
                                   uint32_t list_count, Buffer* list,
                                   Buffer* values, Buffer* partials,
                                   Buffer* dst, uint32_t dst_index) {
  if (list_count == 0) return;
  MGPrParams p = base;
  p.count = list_count;
  const uint32_t blocks = n_blocks(list_count);
  cb.dispatch(dangling_, &p, sizeof(p), {list, values, partials}, blocks,
              MG_TG_SIZE);
  MGReduceParams rp{};
  rp.n_partials = blocks;
  rp.dst_index = dst_index;
  cb.dispatch(finalize_, &rp, sizeof(rp), {partials, dst}, 1, MG_TG_SIZE);
}

void ReduceEngine::encode_list_sum_b(CommandBatch& cb, MGPprParams base,
                                     uint32_t list_count, Buffer* list,
                                     Buffer* values_b, Buffer* partials,
                                     Buffer* dst) {
  if (list_count == 0) return;
  MGPprParams p = base;
  p.count = list_count;
  const uint32_t blocks = n_blocks(list_count);
  cb.dispatch(dangling_b_, &p, sizeof(p), {list, values_b, partials}, blocks,
              MG_TG_SIZE);
  MGReduceParams rp{};
  rp.n_partials = blocks;
  rp.dst_index = 0;  // ignored by the _b finalize
  cb.dispatch(finalize_b_, &rp, sizeof(rp), {partials, dst}, 1, MG_TG_SIZE);
}

}  // namespace mg
