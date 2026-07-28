// wcc.cpp — mg::wcc entry point and mg::gpu::wcc (hook + pointer-jump
// rounds over the OUT-orientation worklists; labels converge to the
// minimum canonical id per component).
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../kernels/mg_params.h"
#include "algos.hpp"

namespace mg {

namespace gpu {

int wcc(Graph& g, uint32_t* labels_canon) {
  Runtime& rt = Runtime::instance();
  if (!rt.has_gpu())
    throw Error(ErrorCode::no_gpu, "gpu::wcc requires a Metal device");
  const uint32_t V = g.V;
  if (V == 0) return 0;

  const Orientation& out_o = g.orientation(Dir::out);
  MTL::ComputePipelineState* pso_iota = rt.pipeline("mg_iota_u32");
  MTL::ComputePipelineState* pso_fill = rt.pipeline("mg_fill_u32");
  MTL::ComputePipelineState* pso_hook_tg = rt.pipeline("mg_wcc_hook_tg");
  MTL::ComputePipelineState* pso_hook_sg = rt.pipeline("mg_wcc_hook_sg");
  MTL::ComputePipelineState* pso_hook_th = rt.pipeline("mg_wcc_hook_thread");
  MTL::ComputePipelineState* pso_hook_huge = rt.pipeline("mg_wcc_hook_huge");
  MTL::ComputePipelineState* pso_jump = rt.pipeline("mg_wcc_jump");

  BufferPtr labels = rt.alloc(std::size_t(V) * sizeof(uint32_t),
                              "wcc.labels");
  BufferPtr changed = rt.alloc(sizeof(uint32_t), "wcc.changed");

  const uint32_t sgs_per_tg =
      std::max(1u, MG_TG_SIZE / std::max(1u, rt.exec_width(pso_hook_sg)));
  // Default 4 (was 8): convergence is only observable at batch granularity,
  // and on skewed graphs each wasted post-convergence round costs real
  // milliseconds — one extra sync is cheaper.
  const long rpb = env_long("MG_WCC_ROUNDS_PER_BATCH", 4);
  const int rounds_per_batch = rpb < 1 ? 1 : static_cast<int>(rpb);

  {
    CommandBatch cb(rt);
    MGFillParams ip{0.0f, 0u, V, 0u};
    cb.dispatch(pso_iota, &ip, sizeof(ip), {labels.get()},
                ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
    cb.commit_and_wait();
  }

  const MGFillParams zero{0.0f, 0u, 1u, 0u};
  const MGWccParams p_high{V, out_o.n_high, 0u, 0u};
  const MGWccParams p_mid{V, out_o.n_mid, 0u, 0u};
  const MGWccParams p_low{V, out_o.n_low, 0u, 0u};
  const MGWccParams p_huge{V, out_o.n_tiles, 0u, 0u};
  const MGWccParams p_all{V, V, 0u, 0u};

  const uint32_t* changed_host = changed->as<uint32_t>();
  int total_rounds = 0;
  for (;;) {
    // Serial encoder, no indirect args -> no encoder breaks needed. Each
    // round re-zeroes the flag, so after the batch it reflects the LAST
    // round only.
    CommandBatch cb(rt);
    for (int r = 0; r < rounds_per_batch; ++r) {
      cb.dispatch(pso_fill, &zero, sizeof(zero), {changed.get()}, 1u,
                  MG_TG_SIZE);
      if (out_o.n_high != 0)
        cb.dispatch(pso_hook_tg, &p_high, sizeof(p_high),
                    {out_o.wl_high.get(), out_o.row_offsets.get(),
                     out_o.col_indices.get(), labels.get(), changed.get()},
                    out_o.n_high, MG_TG_SIZE);
      if (out_o.n_mid != 0)
        cb.dispatch(pso_hook_sg, &p_mid, sizeof(p_mid),
                    {out_o.wl_mid.get(), out_o.row_offsets.get(),
                     out_o.col_indices.get(), labels.get(), changed.get()},
                    ceil_div_u32(out_o.n_mid, sgs_per_tg), MG_TG_SIZE);
      if (out_o.n_low != 0)
        cb.dispatch(pso_hook_th, &p_low, sizeof(p_low),
                    {out_o.wl_low.get(), out_o.row_offsets.get(),
                     out_o.col_indices.get(), labels.get(), changed.get()},
                    ceil_div_u32(out_o.n_low, MG_TG_SIZE), MG_TG_SIZE);
      if (out_o.n_huge != 0)
        cb.dispatch(pso_hook_huge, &p_huge, sizeof(p_huge),
                    {out_o.tile_owner.get(), out_o.tile_first_edge.get(),
                     out_o.wl_huge.get(), out_o.row_offsets.get(),
                     out_o.col_indices.get(), labels.get(), changed.get()},
                    out_o.n_tiles, MG_TG_SIZE);
      cb.dispatch(pso_jump, &p_all, sizeof(p_all),
                  {labels.get(), changed.get()},
                  ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
    }
    cb.commit_and_wait();
    total_rounds += rounds_per_batch;
    if (*changed_host == 0u) break;
    if (static_cast<uint64_t>(total_rounds) > uint64_t(V) + 64u)
      throw Error(ErrorCode::internal, "gpu::wcc failed to converge");
  }

  std::memcpy(labels_canon, labels->data(),
              std::size_t(V) * sizeof(uint32_t));
  return total_rounds;
}

}  // namespace gpu

int wcc(Graph& g, int32_t* out_comp) {
  const auto t0 = std::chrono::steady_clock::now();
  Runtime& rt = Runtime::instance();
  const uint32_t V = g.V;
  if (V > 0 && !out_comp) throw_invalid("wcc: output array is null");

  int rounds = 0;
  int n_components = 0;
  ExecPath path = ExecPath::cpu;
  if (V > 0) {
    path = rt.plan(g.stored_edges());
    std::vector<uint32_t> labels(V);
    rounds = (path == ExecPath::gpu) ? gpu::wcc(g, labels.data())
                                     : cpu::wcc(g, labels.data());
    // Canonical-partition ids: components numbered by first occurrence
    // scanning USER index order (algos.hpp).
    std::vector<int32_t> comp_of_root(V, -1);
    for (uint32_t u = 0; u < V; ++u) {
      const uint32_t root = labels[g.canon_of_user[u]];
      if (root >= V)
        throw Error(ErrorCode::internal, "wcc: label out of range");
      int32_t& slot = comp_of_root[root];
      if (slot < 0) slot = n_components++;
      out_comp[u] = slot;
    }
  }

  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  RunInfo info;
  info.op = "wcc";
  info.path = path;
  info.iterations = rounds;
  info.elapsed_ms = ms;
  rt.record_run(info);
  return n_components;
}

}  // namespace mg
