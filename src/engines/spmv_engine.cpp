// spmv_engine.cpp — encode PageRank / batched-PPR iterations per the
// mg_params.h binding tables.
// SPDX-License-Identifier: Apache-2.0
#include "spmv_engine.hpp"

#include <algorithm>

namespace mg {

namespace {

inline uint32_t tg_count(uint32_t items) {
  return ceil_div_u32(items, MG_TG_SIZE);
}

inline uint32_t sgs_per_tg_of(Runtime& rt, MTL::ComputePipelineState* pso) {
  uint32_t ew = rt.exec_width(pso);
  return ew ? std::max(1u, MG_TG_SIZE / ew) : 8u;
}

}  // namespace

SpmvEngine::SpmvEngine(Runtime& rt, bool weighted, bool uniform_p,
                       bool batched)
    : rt_(rt) {
  if (!batched) {
    prepare_ = rt.pipeline("mg_pr_prepare");
    gather_tg_ = rt.pipeline("mg_pr_gather_tg", weighted, uniform_p);
    gather_sg_ = rt.pipeline("mg_pr_gather_sg", weighted, uniform_p);
    gather_th_ = rt.pipeline("mg_pr_gather_thread", weighted, uniform_p);
    zero_fill_ = rt.pipeline("mg_pr_zero_fill", false, uniform_p);
    huge_partials_ = rt.pipeline("mg_pr_gather_huge_partials", weighted);
    huge_finalize_ = rt.pipeline("mg_pr_gather_huge_finalize", false,
                                 uniform_p);
    sgs_per_tg_ = sgs_per_tg_of(rt, gather_sg_);
  } else {
    prepare_b_ = rt.pipeline("mg_pr_prepare_b");
    gather_tg_b_ = rt.pipeline("mg_pr_gather_tg_b", weighted);
    gather_sg_b_ = rt.pipeline("mg_pr_gather_sg_b", weighted);
    gather_th_b_ = rt.pipeline("mg_pr_gather_thread_b", weighted);
    zero_fill_b_ = rt.pipeline("mg_pr_zero_fill_b");
    seed_scatter_b_ = rt.pipeline("mg_pr_seed_scatter_b");
    huge_partials_b_ = rt.pipeline("mg_pr_gather_huge_partials_b", weighted);
    huge_finalize_b_ = rt.pipeline("mg_pr_gather_huge_finalize_b");
    sgs_per_tg_b_ = sgs_per_tg_of(rt, gather_sg_b_);
  }
}

Buffer* SpmvEngine::huge_scratch(const Orientation& o, bool batched) {
  if (!huge_scratch_) {
    const std::size_t n = std::size_t{o.n_tiles} *
                          (batched ? MG_PPR_TILE : 1u) * sizeof(float);
    huge_scratch_ = rt_.alloc(n, "pr.huge_tile_partials");
  }
  return huge_scratch_.get();
}

void SpmvEngine::encode_iteration(CommandBatch& cb, const Orientation& in_o,
                                  MGPrParams base, Buffer* rank_cur,
                                  Buffer* contrib, Buffer* rank_next,
                                  Buffer* iter_scalars, Buffer* pvec,
                                  Buffer* out_weight_sum, Buffer* weights) {
  MGPrParams p = base;

  // contrib[v] = ows[v] > 0 ? rank[v] / ows[v] : 0  (over all V)
  p.count = base.v_count;
  cb.dispatch(prepare_, &p, sizeof(p), {rank_cur, out_weight_sum, contrib},
              tg_count(p.count), MG_TG_SIZE);

  auto gather = [&](MTL::ComputePipelineState* pso, Buffer* wl, uint32_t n,
                    uint32_t tgs) {
    if (n == 0) return;
    p.count = n;
    cb.dispatch(pso, &p, sizeof(p),
                {wl, in_o.row_offsets.get(), in_o.col_indices.get(), contrib,
                 rank_next, iter_scalars, pvec, weights},
                tgs, MG_TG_SIZE);
  };
  gather(gather_tg_, in_o.wl_high.get(), in_o.n_high, in_o.n_high);
  gather(gather_sg_, in_o.wl_mid.get(), in_o.n_mid,
         ceil_div_u32(in_o.n_mid, sgs_per_tg_));
  gather(gather_th_, in_o.wl_low.get(), in_o.n_low, tg_count(in_o.n_low));

  if (in_o.n_huge) {
    Buffer* partials = huge_scratch(in_o, /*batched=*/false);
    p.count = in_o.n_tiles;
    cb.dispatch(huge_partials_, &p, sizeof(p),
                {in_o.tile_owner.get(), in_o.tile_first_edge.get(),
                 in_o.wl_huge.get(), in_o.row_offsets.get(),
                 in_o.col_indices.get(), contrib, weights, partials},
                in_o.n_tiles, MG_TG_SIZE);
    p.count = in_o.n_huge;
    cb.dispatch(huge_finalize_, &p, sizeof(p),
                {in_o.wl_huge.get(), in_o.huge_tile_off.get(), partials,
                 rank_next, iter_scalars, pvec},
                tg_count(in_o.n_huge), MG_TG_SIZE);
  }

  if (in_o.n_zero) {
    p.count = in_o.n_zero;
    cb.dispatch(zero_fill_, &p, sizeof(p),
                {in_o.zero_list.get(), rank_next, iter_scalars, pvec},
                tg_count(in_o.n_zero), MG_TG_SIZE);
  }
}

void SpmvEngine::encode_iteration_b(CommandBatch& cb, const Orientation& in_o,
                                    MGPprParams base, Buffer* rank_cur_b,
                                    Buffer* contrib_b, Buffer* rank_next_b,
                                    Buffer* iter_scalars_b,
                                    Buffer* out_weight_sum, Buffer* weights,
                                    Buffer* seed_vertex, Buffer* seed_lane,
                                    Buffer* seed_weight) {
  MGPprParams p = base;

  p.count = base.v_count;
  cb.dispatch(prepare_b_, &p, sizeof(p),
              {rank_cur_b, out_weight_sum, contrib_b}, tg_count(p.count),
              MG_TG_SIZE);

  auto gather = [&](MTL::ComputePipelineState* pso, Buffer* wl, uint32_t n,
                    uint32_t tgs) {
    if (n == 0) return;
    p.count = n;
    cb.dispatch(pso, &p, sizeof(p),
                {wl, in_o.row_offsets.get(), in_o.col_indices.get(),
                 contrib_b, rank_next_b, rank_cur_b, weights},
                tgs, MG_TG_SIZE);
  };
  gather(gather_tg_b_, in_o.wl_high.get(), in_o.n_high, in_o.n_high);
  gather(gather_sg_b_, in_o.wl_mid.get(), in_o.n_mid,
         ceil_div_u32(in_o.n_mid, sgs_per_tg_b_));
  gather(gather_th_b_, in_o.wl_low.get(), in_o.n_low, tg_count(in_o.n_low));

  if (in_o.n_huge) {
    Buffer* partials = huge_scratch(in_o, /*batched=*/true);
    p.count = in_o.n_tiles;
    cb.dispatch(huge_partials_b_, &p, sizeof(p),
                {in_o.tile_owner.get(), in_o.tile_first_edge.get(),
                 in_o.wl_huge.get(), in_o.row_offsets.get(),
                 in_o.col_indices.get(), contrib_b, weights, partials},
                in_o.n_tiles, MG_TG_SIZE);
    p.count = in_o.n_huge;
    cb.dispatch(huge_finalize_b_, &p, sizeof(p),
                {in_o.wl_huge.get(), in_o.huge_tile_off.get(), partials,
                 rank_next_b, rank_cur_b},
                tg_count(in_o.n_huge), MG_TG_SIZE);
  }

  if (in_o.n_zero) {
    p.count = in_o.n_zero;
    cb.dispatch(zero_fill_b_, &p, sizeof(p),
                {in_o.zero_list.get(), rank_next_b, rank_cur_b},
                tg_count(in_o.n_zero), MG_TG_SIZE);
  }

  if (base.seed_total) {
    p.count = base.seed_total;
    cb.dispatch(seed_scatter_b_, &p, sizeof(p),
                {seed_vertex, seed_lane, seed_weight, iter_scalars_b,
                 rank_next_b},
                tg_count(base.seed_total), MG_TG_SIZE);
  }
}

}  // namespace mg
