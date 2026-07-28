// spmv.metal — PageRank pull-gather kernels, single-query and batched PPR.
// Binding tables and write contracts: src/kernels/mg_params.h (authoritative).
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

constant bool mg_weighted  [[function_constant(MG_FC_WEIGHTED)]];
constant bool mg_uniform_p [[function_constant(MG_FC_UNIFORM_P)]];

// ---------------------------------------------------------------------------
// Single-query PageRank
// ---------------------------------------------------------------------------

// rank_next[v] = (1-a)*p[v] + a*(acc + D*p[v]), D = iter_scalars[DANGLING]
static inline void mg_pr_write(device float* rank_next, uint v, float acc,
                               constant MGPrParams& p,
                               device const float* iter_scalars,
                               device const float* pvec) {
  const float pv = mg_uniform_p ? p.p_uniform : pvec[v];
  rank_next[v] = p.one_minus_alpha * pv +
                 p.alpha * (acc + iter_scalars[MG_SCALAR_DANGLING] * pv);
}

// contrib[v] = ows[v] > 0 ? rank[v]/ows[v] : 0 — one thread per vertex.
kernel void mg_pr_prepare(
    constant MGPrParams& p            [[buffer(0)]],
    device const float* rank_cur      [[buffer(1)]],
    device const float* out_weight_sum [[buffer(2)]],
    device float* contrib             [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const float w = out_weight_sum[gid];
  contrib[gid] = (w > 0.0f) ? rank_cur[gid] / w : 0.0f;
}

// High bin: one threadgroup per worklist vertex, edge loop strided by
// threads_per_threadgroup, threadgroup reduction.
kernel void mg_pr_gather_tg(
    constant MGPrParams& p           [[buffer(0)]],
    device const uint* worklist      [[buffer(1)]],
    device const uint* row_off       [[buffer(2)]],
    device const uint* col_idx       [[buffer(3)]],
    device const float* contrib      [[buffer(4)]],
    device float* rank_next          [[buffer(5)]],
    device const float* iter_scalars [[buffer(6)]],
    device const float* pvec         [[buffer(7)]],
    device const float* weights      [[buffer(8)]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  if (tg_id >= p.count) return;  // threadgroup-uniform
  const uint v = worklist[tg_id];
  const uint end = row_off[v + 1u];
  float acc = 0.0f;
  for (uint e = row_off[v] + tid; e < end; e += tg_size) {
    float c = contrib[col_idx[e]];
    if (mg_weighted) c *= weights[e];
    acc += c;
  }
  threadgroup float scratch[MG_TG_SCRATCH];
  const float total =
      mg_tg_reduce_sum(acc, scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) mg_pr_write(rank_next, v, total, p, iter_scalars, pvec);
}

// Mid bin: one simdgroup per worklist vertex (plan §6 sketch).
kernel void mg_pr_gather_sg(
    constant MGPrParams& p           [[buffer(0)]],
    device const uint* worklist      [[buffer(1)]],
    device const uint* row_off       [[buffer(2)]],
    device const uint* col_idx       [[buffer(3)]],
    device const float* contrib      [[buffer(4)]],
    device float* rank_next          [[buffer(5)]],
    device const float* iter_scalars [[buffer(6)]],
    device const float* pvec         [[buffer(7)]],
    device const float* weights      [[buffer(8)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  const uint wi = tg_id * sgs_per_tg + sg_in_tg;  // global simdgroup index
  if (wi >= p.count) return;
  const uint v = worklist[wi];
  const uint end = row_off[v + 1u];
  float acc = 0.0f;
  for (uint e = row_off[v] + lane; e < end; e += simd_sz) {
    float c = contrib[col_idx[e]];
    if (mg_weighted) c *= weights[e];
    acc += c;
  }
  acc = simd_sum(acc);
  if (lane == 0u) mg_pr_write(rank_next, v, acc, p, iter_scalars, pvec);
}

// Low bin: one thread per worklist vertex, sequential edge loop.
kernel void mg_pr_gather_thread(
    constant MGPrParams& p           [[buffer(0)]],
    device const uint* worklist      [[buffer(1)]],
    device const uint* row_off       [[buffer(2)]],
    device const uint* col_idx       [[buffer(3)]],
    device const float* contrib      [[buffer(4)]],
    device float* rank_next          [[buffer(5)]],
    device const float* iter_scalars [[buffer(6)]],
    device const float* pvec         [[buffer(7)]],
    device const float* weights      [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint v = worklist[gid];
  const uint end = row_off[v + 1u];
  float acc = 0.0f;
  for (uint e = row_off[v]; e < end; ++e) {
    float c = contrib[col_idx[e]];
    if (mg_weighted) c *= weights[e];
    acc += c;
  }
  mg_pr_write(rank_next, v, acc, p, iter_scalars, pvec);
}

// Zero-in-degree vertices: same formula with acc = 0 (must be written
// every iteration).
kernel void mg_pr_zero_fill(
    constant MGPrParams& p           [[buffer(0)]],
    device const uint* zero_list     [[buffer(1)]],
    device float* rank_next          [[buffer(2)]],
    device const float* iter_scalars [[buffer(3)]],
    device const float* pvec         [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  mg_pr_write(rank_next, zero_list[gid], 0.0f, p, iter_scalars, pvec);
}

// ---------------------------------------------------------------------------
// Batched PPR (tile of MG_PPR_TILE query lanes; state vertex-major
// interleaved: state[v * MG_PPR_TILE + lane]).
// ---------------------------------------------------------------------------

// Batched write contract: ASSIGN alpha*acc for active lanes, COPY rank_cur
// for inactive lanes (teleport + dangling added later by seed_scatter).
static inline void mg_pr_write_b(device float* rank_next_b,
                                 device const float* rank_cur_b, uint v,
                                 thread const float* acc,
                                 constant MGPprParams& p) {
  const ulong base = ulong(v) * MG_PPR_TILE;
  for (uint t = 0u; t < MG_PPR_TILE; ++t) {
    const bool active = ((p.active_mask >> t) & 1u) != 0u;
    rank_next_b[base + t] = active ? p.alpha * acc[t] : rank_cur_b[base + t];
  }
}

kernel void mg_pr_prepare_b(
    constant MGPprParams& p            [[buffer(0)]],
    device const float* rank_cur_b     [[buffer(1)]],
    device const float* out_weight_sum [[buffer(2)]],
    device float* contrib_b            [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const float w = out_weight_sum[gid];
  const float inv = (w > 0.0f) ? (1.0f / w) : 0.0f;
  const ulong base = ulong(gid) * MG_PPR_TILE;
  for (uint t = 0u; t < MG_PPR_TILE; ++t)
    contrib_b[base + t] = rank_cur_b[base + t] * inv;
}

// High bin, batched: one threadgroup per vertex; MG_PPR_TILE register
// accumulators; col_indices streamed once per thread stride.
kernel void mg_pr_gather_tg_b(
    constant MGPprParams& p        [[buffer(0)]],
    device const uint* worklist    [[buffer(1)]],
    device const uint* row_off     [[buffer(2)]],
    device const uint* col_idx     [[buffer(3)]],
    device const float* contrib_b  [[buffer(4)]],
    device float* rank_next_b      [[buffer(5)]],
    device const float* rank_cur_b [[buffer(6)]],
    device const float* weights    [[buffer(7)]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  if (tg_id >= p.count) return;  // threadgroup-uniform
  const uint v = worklist[tg_id];
  const uint end = row_off[v + 1u];
  float acc[MG_PPR_TILE];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] = 0.0f;
  for (uint e = row_off[v] + tid; e < end; e += tg_size) {
    const ulong ub = ulong(col_idx[e]) * MG_PPR_TILE;
    const float w = mg_weighted ? weights[e] : 1.0f;
    for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] += contrib_b[ub + t] * w;
  }
  threadgroup float scratch[MG_TG_SCRATCH];
  float tot[MG_PPR_TILE];
  for (uint t = 0u; t < MG_PPR_TILE; ++t)
    tot[t] = mg_tg_reduce_sum(acc[t], scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) mg_pr_write_b(rank_next_b, rank_cur_b, v, tot, p);
}

// Mid bin, batched: one simdgroup per vertex.
kernel void mg_pr_gather_sg_b(
    constant MGPprParams& p        [[buffer(0)]],
    device const uint* worklist    [[buffer(1)]],
    device const uint* row_off     [[buffer(2)]],
    device const uint* col_idx     [[buffer(3)]],
    device const float* contrib_b  [[buffer(4)]],
    device float* rank_next_b      [[buffer(5)]],
    device const float* rank_cur_b [[buffer(6)]],
    device const float* weights    [[buffer(7)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  const uint wi = tg_id * sgs_per_tg + sg_in_tg;
  if (wi >= p.count) return;
  const uint v = worklist[wi];
  const uint end = row_off[v + 1u];
  float acc[MG_PPR_TILE];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] = 0.0f;
  for (uint e = row_off[v] + lane; e < end; e += simd_sz) {
    const ulong ub = ulong(col_idx[e]) * MG_PPR_TILE;
    const float w = mg_weighted ? weights[e] : 1.0f;
    for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] += contrib_b[ub + t] * w;
  }
  for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] = simd_sum(acc[t]);
  if (lane == 0u) mg_pr_write_b(rank_next_b, rank_cur_b, v, acc, p);
}

// Low bin, batched: one thread per vertex.
kernel void mg_pr_gather_thread_b(
    constant MGPprParams& p        [[buffer(0)]],
    device const uint* worklist    [[buffer(1)]],
    device const uint* row_off     [[buffer(2)]],
    device const uint* col_idx     [[buffer(3)]],
    device const float* contrib_b  [[buffer(4)]],
    device float* rank_next_b      [[buffer(5)]],
    device const float* rank_cur_b [[buffer(6)]],
    device const float* weights    [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint v = worklist[gid];
  const uint end = row_off[v + 1u];
  float acc[MG_PPR_TILE];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] = 0.0f;
  for (uint e = row_off[v]; e < end; ++e) {
    const ulong ub = ulong(col_idx[e]) * MG_PPR_TILE;
    const float w = mg_weighted ? weights[e] : 1.0f;
    for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] += contrib_b[ub + t] * w;
  }
  mg_pr_write_b(rank_next_b, rank_cur_b, v, acc, p);
}

// Zero-in-degree vertices, batched: assign 0 on active lanes, copy on
// inactive.
kernel void mg_pr_zero_fill_b(
    constant MGPprParams& p        [[buffer(0)]],
    device const uint* zero_list   [[buffer(1)]],
    device float* rank_next_b      [[buffer(2)]],
    device const float* rank_cur_b [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const ulong base = ulong(zero_list[gid]) * MG_PPR_TILE;
  for (uint t = 0u; t < MG_PPR_TILE; ++t) {
    const bool active = ((p.active_mask >> t) & 1u) != 0u;
    rank_next_b[base + t] = active ? 0.0f : rank_cur_b[base + t];
  }
}

// Seed scatter: rank_next_b[v*TILE+t] += (1-a)*w + a*D_t*w for active
// lanes. Seeds pre-deduplicated per (v, lane) on host — no write conflicts.
kernel void mg_pr_seed_scatter_b(
    constant MGPprParams& p           [[buffer(0)]],
    device const uint* seed_vertex    [[buffer(1)]],
    device const uint* seed_lane      [[buffer(2)]],
    device const float* seed_weight   [[buffer(3)]],
    device const float* iter_scalars_b [[buffer(4)]],
    device float* rank_next_b         [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint t = seed_lane[gid];
  if (((p.active_mask >> t) & 1u) == 0u) return;
  const float w = seed_weight[gid];
  rank_next_b[ulong(seed_vertex[gid]) * MG_PPR_TILE + t] +=
      p.one_minus_alpha * w + p.alpha * iter_scalars_b[t] * w;
}

// ---------------------------------------------------------------------------
// Huge bin (degree >= MG_BIN_HUGE_MIN): fixed edge tiles across many
// threadgroups — a mega-hub must not serialize on one threadgroup.
// Deterministic two-pass: per-tile threadgroup partials, then a per-vertex
// finalize summing tiles in fixed order. Tiling tables: mg_params.h.
// ---------------------------------------------------------------------------

kernel void mg_pr_gather_huge_partials(
    constant MGPrParams& p            [[buffer(0)]],
    device const uint* tile_owner     [[buffer(1)]],
    device const uint* tile_first     [[buffer(2)]],
    device const uint* wl_huge        [[buffer(3)]],
    device const uint* row_off        [[buffer(4)]],
    device const uint* col_idx        [[buffer(5)]],
    device const float* contrib       [[buffer(6)]],
    device const float* weights       [[buffer(7)]],
    device float* partials            [[buffer(8)]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  if (tg_id >= p.count) return;  // threadgroup-uniform
  const uint v = wl_huge[tile_owner[tg_id]];
  const uint begin = tile_first[tg_id];
  const uint end = min(row_off[v + 1u], begin + MG_HUGE_TILE_EDGES);
  float acc = 0.0f;
  for (uint e = begin + tid; e < end; e += tg_size) {
    float c = contrib[col_idx[e]];
    if (mg_weighted) c *= weights[e];
    acc += c;
  }
  threadgroup float scratch[MG_TG_SCRATCH];
  const float total =
      mg_tg_reduce_sum(acc, scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) partials[tg_id] = total;
}

kernel void mg_pr_gather_huge_finalize(
    constant MGPrParams& p           [[buffer(0)]],
    device const uint* wl_huge       [[buffer(1)]],
    device const uint* tile_off      [[buffer(2)]],
    device const float* partials     [[buffer(3)]],
    device float* rank_next          [[buffer(4)]],
    device const float* iter_scalars [[buffer(5)]],
    device const float* pvec         [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  float acc = 0.0f;
  const uint end = tile_off[gid + 1u];
  for (uint t = tile_off[gid]; t < end; ++t) acc += partials[t];
  mg_pr_write(rank_next, wl_huge[gid], acc, p, iter_scalars, pvec);
}

kernel void mg_pr_gather_huge_partials_b(
    constant MGPprParams& p       [[buffer(0)]],
    device const uint* tile_owner [[buffer(1)]],
    device const uint* tile_first [[buffer(2)]],
    device const uint* wl_huge    [[buffer(3)]],
    device const uint* row_off    [[buffer(4)]],
    device const uint* col_idx    [[buffer(5)]],
    device const float* contrib_b [[buffer(6)]],
    device const float* weights   [[buffer(7)]],
    device float* partials_b      [[buffer(8)]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  if (tg_id >= p.count) return;  // threadgroup-uniform
  const uint v = wl_huge[tile_owner[tg_id]];
  const uint begin = tile_first[tg_id];
  const uint end = min(row_off[v + 1u], begin + MG_HUGE_TILE_EDGES);
  float acc[MG_PPR_TILE];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] = 0.0f;
  for (uint e = begin + tid; e < end; e += tg_size) {
    const ulong ub = ulong(col_idx[e]) * MG_PPR_TILE;
    const float w = mg_weighted ? weights[e] : 1.0f;
    for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] += contrib_b[ub + t] * w;
  }
  threadgroup float scratch[MG_TG_SCRATCH];
  const ulong pb = ulong(tg_id) * MG_PPR_TILE;
  for (uint t = 0u; t < MG_PPR_TILE; ++t) {
    const float total =
        mg_tg_reduce_sum(acc[t], scratch, sg_in_tg, lane, sgs_per_tg);
    if (tid == 0u) partials_b[pb + t] = total;
  }
}

kernel void mg_pr_gather_huge_finalize_b(
    constant MGPprParams& p        [[buffer(0)]],
    device const uint* wl_huge     [[buffer(1)]],
    device const uint* tile_off    [[buffer(2)]],
    device const float* partials_b [[buffer(3)]],
    device float* rank_next_b      [[buffer(4)]],
    device const float* rank_cur_b [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  float acc[MG_PPR_TILE];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] = 0.0f;
  const uint end = tile_off[gid + 1u];
  for (uint tl = tile_off[gid]; tl < end; ++tl) {
    const ulong pb = ulong(tl) * MG_PPR_TILE;
    for (uint t = 0u; t < MG_PPR_TILE; ++t) acc[t] += partials_b[pb + t];
  }
  mg_pr_write_b(rank_next_b, rank_cur_b, wl_huge[gid], acc, p);
}
