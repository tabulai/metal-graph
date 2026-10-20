// wcc.cpp — mg::wcc entry point and mg::gpu::wcc (one concurrent
// union-find link pass over the OUT-orientation worklists plus one compress
// pass; labels are the minimum canonical id per component).
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
  MTL::ComputePipelineState* pso_link_tg = rt.pipeline("mg_wcc_link_tg");
  MTL::ComputePipelineState* pso_link_sg = rt.pipeline("mg_wcc_link_sg");
  MTL::ComputePipelineState* pso_link_th = rt.pipeline("mg_wcc_link_thread");
  MTL::ComputePipelineState* pso_link_huge = rt.pipeline("mg_wcc_link_huge");
  MTL::ComputePipelineState* pso_compress = rt.pipeline("mg_wcc_compress");

  BufferPtr labels = rt.alloc(std::size_t(V) * sizeof(uint32_t),
                              "wcc.labels");

  const uint32_t sgs_per_tg =
      std::max(1u, MG_TG_SIZE / std::max(1u, rt.exec_width(pso_link_sg)));
  // Undirected graphs store every u != v edge both ways; one copy suffices.
  const uint32_t flags = g.directed ? 0u : MG_WCC_SKIP_TWINS;
  const MGWccParams p_high{V, out_o.n_high, flags, 0u};
  const MGWccParams p_mid{V, out_o.n_mid, flags, 0u};
  const MGWccParams p_low{V, out_o.n_low, flags, 0u};
  const MGWccParams p_huge{V, out_o.n_tiles, flags, 0u};
  const MGWccParams p_all{V, V, 0u, 0u};

  // Single batch, serial encoder (each dispatch sees the previous one's
  // writes): iota -> one union-find link pass over every stored edge ->
  // one compress pass. Concurrent union-find needs no convergence rounds:
  // after the link pass every edge's endpoints share a tree rooted at the
  // component's minimum canonical id, and compress flattens each vertex to
  // that root. (MG_WCC_ROUNDS_PER_BATCH no longer applies; it is ignored.)
  CommandBatch cb(rt);
  const MGFillParams ip{0.0f, 0u, V, 0u};
  cb.dispatch(pso_iota, &ip, sizeof(ip), {labels.get()},
              ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
  if (out_o.n_huge != 0)
    cb.dispatch(pso_link_huge, &p_huge, sizeof(p_huge),
                {out_o.tile_owner.get(), out_o.tile_first_edge.get(),
                 out_o.wl_huge.get(), out_o.row_offsets.get(),
                 out_o.col_indices.get(), labels.get()},
                out_o.n_tiles, MG_TG_SIZE);
  if (out_o.n_high != 0)
    cb.dispatch(pso_link_tg, &p_high, sizeof(p_high),
                {out_o.wl_high.get(), out_o.row_offsets.get(),
                 out_o.col_indices.get(), labels.get()},
                out_o.n_high, MG_TG_SIZE);
  if (out_o.n_mid != 0)
    cb.dispatch(pso_link_sg, &p_mid, sizeof(p_mid),
                {out_o.wl_mid.get(), out_o.row_offsets.get(),
                 out_o.col_indices.get(), labels.get()},
                ceil_div_u32(out_o.n_mid, sgs_per_tg), MG_TG_SIZE);
  if (out_o.n_low != 0)
    cb.dispatch(pso_link_th, &p_low, sizeof(p_low),
                {out_o.wl_low.get(), out_o.row_offsets.get(),
                 out_o.col_indices.get(), labels.get()},
                ceil_div_u32(out_o.n_low, MG_TG_SIZE), MG_TG_SIZE);
  cb.dispatch(pso_compress, &p_all, sizeof(p_all), {labels.get()},
              ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
  cb.commit_and_wait();

  std::memcpy(labels_canon, labels->data(),
              std::size_t(V) * sizeof(uint32_t));
  return 1;  // one link + compress pass
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
    path = rt.plan(g.stored_edges(), PlanOp::wcc);
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
