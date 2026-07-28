// reduce.metal — dangling/residual partial reductions (single + batched),
// reduce finalize, fills, iota.
// Binding tables: src/kernels/mg_params.h (authoritative).
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

// NOTE: kernels below contain threadgroup barriers (mg_tg_reduce_sum), so
// out-of-range threads contribute 0 instead of returning early.

// ---------------------------------------------------------------------------
// Single-query partials
// ---------------------------------------------------------------------------

// partials[tg] = sum of rank[dangling_list[i]] over this threadgroup.
kernel void mg_pr_dangling_partials(
    constant MGPrParams& p           [[buffer(0)]],
    device const uint* dangling_list [[buffer(1)]],
    device const float* rank         [[buffer(2)]],
    device float* partials           [[buffer(3)]],
    uint gid        [[thread_position_in_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  const float x = (gid < p.count) ? rank[dangling_list[gid]] : 0.0f;
  threadgroup float scratch[MG_TG_SCRATCH];
  const float total = mg_tg_reduce_sum(x, scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) partials[tg_id] = total;
}

// partials[tg] = sum of |rank_a - rank_b| over this threadgroup.
kernel void mg_pr_residual_partials(
    constant MGPrParams& p     [[buffer(0)]],
    device const float* rank_a [[buffer(1)]],
    device const float* rank_b [[buffer(2)]],
    device float* partials     [[buffer(3)]],
    uint gid        [[thread_position_in_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  const float x = (gid < p.count) ? fabs(rank_a[gid] - rank_b[gid]) : 0.0f;
  threadgroup float scratch[MG_TG_SCRATCH];
  const float total = mg_tg_reduce_sum(x, scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) partials[tg_id] = total;
}

// ONE threadgroup: dst[dst_index] = sum(partials[0..n_partials)).
kernel void mg_reduce_finalize(
    constant MGReduceParams& p   [[buffer(0)]],
    device const float* partials [[buffer(1)]],
    device float* dst            [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  float x = 0.0f;
  for (uint i = tid; i < p.n_partials; i += tg_size) x += partials[i];
  threadgroup float scratch[MG_TG_SCRATCH];
  const float total = mg_tg_reduce_sum(x, scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) dst[p.dst_index] = total;
}

// ---------------------------------------------------------------------------
// Batched partials (partials laid out [block * MG_PPR_TILE + lane])
// ---------------------------------------------------------------------------

kernel void mg_pr_dangling_partials_b(
    constant MGPprParams& p          [[buffer(0)]],
    device const uint* dangling_list [[buffer(1)]],
    device const float* rank_b       [[buffer(2)]],
    device float* partials           [[buffer(3)]],
    uint gid        [[thread_position_in_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  const bool in_range = gid < p.count;
  const ulong base = in_range ? ulong(dangling_list[gid]) * MG_PPR_TILE : 0ul;
  threadgroup float scratch[MG_TG_SCRATCH];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) {
    const float x = in_range ? rank_b[base + t] : 0.0f;
    const float total =
        mg_tg_reduce_sum(x, scratch, sg_in_tg, lane, sgs_per_tg);
    if (tid == 0u) partials[tg_id * MG_PPR_TILE + t] = total;
  }
}

kernel void mg_pr_residual_partials_b(
    constant MGPprParams& p      [[buffer(0)]],
    device const float* rank_a_b [[buffer(1)]],
    device const float* rank_b_b [[buffer(2)]],
    device float* partials       [[buffer(3)]],
    uint gid        [[thread_position_in_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  const bool in_range = gid < p.count;
  const ulong base = ulong(gid) * MG_PPR_TILE;
  threadgroup float scratch[MG_TG_SCRATCH];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) {
    const float x =
        in_range ? fabs(rank_a_b[base + t] - rank_b_b[base + t]) : 0.0f;
    const float total =
        mg_tg_reduce_sum(x, scratch, sg_in_tg, lane, sgs_per_tg);
    if (tid == 0u) partials[tg_id * MG_PPR_TILE + t] = total;
  }
}

// ONE threadgroup: dst[t] = sum over blocks of partials[b*TILE + t].
kernel void mg_reduce_finalize_b(
    constant MGReduceParams& p   [[buffer(0)]],
    device const float* partials [[buffer(1)]],
    device float* dst            [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  threadgroup float scratch[MG_TG_SCRATCH];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) {
    float x = 0.0f;
    for (uint i = tid; i < p.n_partials; i += tg_size)
      x += partials[i * MG_PPR_TILE + t];
    const float total =
        mg_tg_reduce_sum(x, scratch, sg_in_tg, lane, sgs_per_tg);
    if (tid == 0u) dst[t] = total;
  }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

kernel void mg_fill_f32(
    constant MGFillParams& p [[buffer(0)]],
    device float* dst        [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < p.count) dst[gid] = p.f_value;
}

kernel void mg_fill_u32(
    constant MGFillParams& p [[buffer(0)]],
    device uint* dst         [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < p.count) dst[gid] = p.u_value;
}

kernel void mg_iota_u32(
    constant MGFillParams& p [[buffer(0)]],
    device uint* dst         [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < p.count) dst[gid] = gid;
}
