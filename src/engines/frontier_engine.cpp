// frontier_engine.cpp — BFS level protocol (mg_params.h, authoritative)
// and k-hop count/emit. Per level, three encoders:
//   prepare | frontier_bin + decide | expands (+ bottom-up probe)
// because frontier_bin consumes args[BIN] written by prepare, and the
// expand kernels consume args[EXPAND_*]/args[BOTTOMUP] written by decide;
// indirect args written on the GPU must be consumed in a LATER encoder.
// SPDX-License-Identifier: Apache-2.0
#include "frontier_engine.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>

#include "../kernels/mg_params.h"
#include "scan_engine.hpp"

namespace mg {

namespace {
inline std::size_t args_offset(uint32_t slot) {
  return std::size_t(slot) * 3u * sizeof(uint32_t);
}
}  // namespace

FrontierEngine::FrontierEngine(Runtime& rt) : rt_(rt) {
  if (!rt_.has_gpu())
    throw Error(ErrorCode::no_gpu, "frontier engine requires a Metal device");
}

int FrontierEngine::run_bfs(Graph& g,
                            const std::vector<uint32_t>& sources_canon,
                            Dir dir, bool enable_bottomup, uint32_t max_levels,
                            int32_t* dist_canon, int32_t* parent_canon) {
  const uint32_t V = g.V;
  if (V == 0) return 0;
  if (sources_canon.empty()) {
    std::memset(dist_canon, 0xFF, std::size_t(V) * sizeof(int32_t));
    std::memset(parent_canon, 0xFF, std::size_t(V) * sizeof(int32_t));
    return 0;
  }

  const Orientation& fwd = g.orientation(dir);
  const Orientation* rev = nullptr;
  if (enable_bottomup)
    rev = &g.orientation(dir == Dir::out ? Dir::in : Dir::out);

  MTL::ComputePipelineState* pso_prepare =
      rt_.pipeline("mg_bfs_prepare_level");
  MTL::ComputePipelineState* pso_bin = rt_.pipeline("mg_bfs_frontier_bin");
  MTL::ComputePipelineState* pso_decide = rt_.pipeline("mg_bfs_decide");
  MTL::ComputePipelineState* pso_exp_tg = rt_.pipeline("mg_bfs_expand_tg");
  MTL::ComputePipelineState* pso_exp_sg = rt_.pipeline("mg_bfs_expand_sg");
  MTL::ComputePipelineState* pso_exp_th = rt_.pipeline("mg_bfs_expand_thread");
  MTL::ComputePipelineState* pso_bu = rt_.pipeline("mg_bfs_bottomup_probe");

  BufferPtr control = rt_.alloc(sizeof(MGBfsControl), "bfs.control");
  BufferPtr f0 = rt_.alloc(std::size_t(V) * 4, "bfs.frontier0");
  BufferPtr f1 = rt_.alloc(std::size_t(V) * 4, "bfs.frontier1");
  BufferPtr dist_b = rt_.alloc(std::size_t(V) * 4, "bfs.dist");
  BufferPtr parent_b = rt_.alloc(std::size_t(V) * 4, "bfs.parent");
  BufferPtr visited =
      rt_.alloc(std::size_t((V + 31u) / 32u) * 4, "bfs.visited");
  BufferPtr bins = rt_.alloc(std::size_t(V) * 3u * 4, "bfs.bins");
  BufferPtr args_b =
      rt_.alloc(std::size_t(MG_BFS_ARGS_SLOTS) * 3u * 4, "bfs.args");
  BufferPtr dmy = rt_.dummy();

  // Host init per the level protocol: dist/parent = -1; sources at dist 0,
  // visited bits set, "next" list = sources; the first prepare turns them
  // into the frontier of level 1. visited/bins/args are zero-initialized
  // by Runtime::alloc.
  std::memset(dist_b->data(), 0xFF, dist_b->byte_size());
  std::memset(parent_b->data(), 0xFF, parent_b->byte_size());
  const uint32_t n_src = static_cast<uint32_t>(sources_canon.size());
  int32_t* dist_host = dist_b->as<int32_t>();
  uint32_t* vis_host = visited->as<uint32_t>();
  uint32_t* next_host = f0->as<uint32_t>();
  for (uint32_t i = 0; i < n_src; ++i) {
    const uint32_t c = sources_canon[i];
    dist_host[c] = 0;
    vis_host[c >> 5] |= 1u << (c & 31u);
    next_host[i] = c;
  }
  MGBfsControl* ctrl = control->as<MGBfsControl>();
  std::memset(ctrl, 0, sizeof(MGBfsControl));
  ctrl->next_count = n_src;
  ctrl->mode = MG_BFS_MODE_TOPDOWN;
  // V, not V - n_src: decide subtracts each frontier as it is consumed and
  // the sources ARE the first frontier — seeding V - n_src would subtract
  // them twice (and wrap below zero on full-coverage traversals).
  ctrl->unvisited = V;

  MGBfsParams p{};
  p.v_count = V;
  p.e_count = static_cast<uint32_t>(fwd.e_stored);
  p.tg_size = MG_TG_SIZE;
  p.sgs_per_tg =
      std::max(1u, MG_TG_SIZE / std::max(1u, rt_.exec_width(pso_exp_sg)));
  p.alpha_cut = 14;
  p.beta_cut = 24;
  p.enable_bottomup = enable_bottomup ? 1u : 0u;

  Buffer* row = fwd.row_offsets.get();
  Buffer* col = fwd.col_indices.get();
  Buffer* rrow = rev ? rev->row_offsets.get() : dmy.get();
  Buffer* rcol = rev ? rev->col_indices.get() : dmy.get();

  long pb = env_long("MG_BFS_LEVELS_PER_BATCH", 8);
  const uint32_t per_batch = pb < 1 ? 1u : static_cast<uint32_t>(pb);
  uint32_t encoded = 0;  // total prepares issued; parity = encoded & 1
  bool done = false;

  while (!done) {
    uint32_t batch_n = per_batch;
    if (max_levels != 0 && max_levels - encoded < batch_n)
      batch_n = max_levels - encoded;
    if (batch_n == 0) break;
    CommandBatch cb(rt_);
    for (uint32_t i = 0; i < batch_n; ++i) {
      const bool even = ((encoded + i) & 1u) == 0u;
      Buffer* cur = even ? f0.get() : f1.get();
      Buffer* nxt = even ? f1.get() : f0.get();
      // Uniform b1..b12 binding set for every BFS kernel (b0 = params via
      // setBytes); frontier/next physical buffers swap by level parity.
      const std::initializer_list<Buffer*> bufs = {
          control.get(), row,           col,         cur,
          nxt,           dist_b.get(),  parent_b.get(), visited.get(),
          bins.get(),    args_b.get(),  rrow,        rcol};
      cb.dispatch(pso_prepare, &p, sizeof(p), bufs, 1u, MG_TG_SIZE);
      cb.break_encoder();
      cb.dispatch_indirect(pso_bin, &p, sizeof(p), bufs, args_b.get(),
                           args_offset(MG_BFS_ARG_BIN), MG_TG_SIZE);
      cb.dispatch(pso_decide, &p, sizeof(p), bufs, 1u, MG_TG_SIZE);
      cb.break_encoder();
      cb.dispatch_indirect(pso_exp_tg, &p, sizeof(p), bufs, args_b.get(),
                           args_offset(MG_BFS_ARG_EXPAND_TG), MG_TG_SIZE);
      cb.dispatch_indirect(pso_exp_sg, &p, sizeof(p), bufs, args_b.get(),
                           args_offset(MG_BFS_ARG_EXPAND_SG), MG_TG_SIZE);
      cb.dispatch_indirect(pso_exp_th, &p, sizeof(p), bufs, args_b.get(),
                           args_offset(MG_BFS_ARG_EXPAND_TH), MG_TG_SIZE);
      if (enable_bottomup)
        cb.dispatch_indirect(pso_bu, &p, sizeof(p), bufs, args_b.get(),
                             args_offset(MG_BFS_ARG_BOTTOMUP), MG_TG_SIZE);
      cb.break_encoder();
    }
    cb.commit_and_wait();
    encoded += batch_n;
    done = ctrl->done != 0;
    if (!done && max_levels != 0 && encoded >= max_levels) break;
    if (!done && uint64_t(encoded) > uint64_t(V) + 2u + per_batch)
      throw Error(ErrorCode::internal,
                  "bfs: level protocol failed to terminate");
  }

  std::memcpy(dist_canon, dist_b->data(), std::size_t(V) * sizeof(int32_t));
  std::memcpy(parent_canon, parent_b->data(),
              std::size_t(V) * sizeof(int32_t));
  if (!done) return static_cast<int>(encoded);  // max_levels bound
  // done: levels executed = deepest assigned level + 1 (the level-counter
  // in control may overshoot within a batch; derive from dist instead).
  std::atomic<int32_t> amax{0};
  parallel_for(V, [&](std::size_t b, std::size_t e) {
    int32_t m = 0;
    for (std::size_t i = b; i < e; ++i)
      if (dist_canon[i] > m) m = dist_canon[i];
    int32_t cur = amax.load(std::memory_order_relaxed);
    while (m > cur &&
           !amax.compare_exchange_weak(cur, m, std::memory_order_relaxed)) {
    }
  });
  return static_cast<int>(amax.load(std::memory_order_relaxed)) + 1;
}

std::vector<uint32_t> FrontierEngine::khop_edges(
    Graph& g, const std::vector<uint32_t>& reached_canon) {
  std::vector<uint32_t> out;
  const uint32_t n = static_cast<uint32_t>(reached_canon.size());
  if (n == 0 || g.V == 0) return out;
  const Orientation& o = g.orientation(Dir::out);
  if (o.e_stored == 0) return out;

  MTL::ComputePipelineState* pso_count = rt_.pipeline("mg_khop_count");
  MTL::ComputePipelineState* pso_emit = rt_.pipeline("mg_khop_emit");

  const uint32_t words = (g.V + 31u) / 32u;
  BufferPtr list = rt_.alloc(std::size_t(n) * 4, "khop.reached");
  BufferPtr bitmap = rt_.alloc(std::size_t(words) * 4, "khop.bitmap");
  BufferPtr counts = rt_.alloc(std::size_t(n) * 4, "khop.counts");
  BufferPtr offsets = rt_.alloc(std::size_t(n) * 4, "khop.offsets");

  std::memcpy(list->data(), reached_canon.data(), std::size_t(n) * 4);
  uint32_t* bm = bitmap->as<uint32_t>();  // zero-initialized by alloc
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t c = reached_canon[i];
    bm[c >> 5] |= 1u << (c & 31u);
  }

  MGKhopParams kp{n, 0u, 0u, 0u};
  const uint32_t tgs = ceil_div_u32(n, MG_TG_SIZE);
  {
    CommandBatch cb(rt_);
    cb.dispatch(pso_count, &kp, sizeof(kp),
                {list.get(), o.row_offsets.get(), o.col_indices.get(),
                 bitmap.get(), counts.get()},
                tgs, MG_TG_SIZE);
    cb.commit_and_wait();
  }

  ScanEngine scan(rt_);
  const uint64_t total = scan.exclusive_scan(*counts, *offsets, n);
  if (total == 0) return out;

  BufferPtr edges = rt_.alloc(std::size_t(total) * 4, "khop.edges");
  {
    CommandBatch cb(rt_);
    cb.dispatch(pso_emit, &kp, sizeof(kp),
                {list.get(), o.row_offsets.get(), o.col_indices.get(),
                 bitmap.get(), offsets.get(), o.edge_orig.get(),
                 edges.get()},
                tgs, MG_TG_SIZE);
    cb.commit_and_wait();
  }
  out.resize(static_cast<std::size_t>(total));
  std::memcpy(out.data(), edges->data(), std::size_t(total) * 4);
  return out;
}

}  // namespace mg
