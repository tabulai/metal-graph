// reduce.metal — dangling/residual partial reductions (single + batched),
// reduce finalize, fills, iota, scalar normalization.
// Grid-stride partials + 1-barrier threadgroup reductions; batched variants
// use float4 loads and single-barrier 8-lane reductions.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

// NOTE: kernels below contain threadgroup barriers, so out-of-range threads
// contribute 0 instead of returning early.

// ---------------------------------------------------------------------------
// Single-query reductions
// ---------------------------------------------------------------------------

// Grid-stride partials over a vertex list: partials[tg] = sum of
// values[list[i]] over this threadgroup's stride.
kernel void mg_pr_dangling_partials(
    constant MGPrParams& p           [[buffer(0)]],
    device const uint* dangling_list [[buffer(1)]],
    device const float* rank         [[buffer(2)]],
    device float* partials           [[buffer(3)]],
    uint gid        [[thread_position_in_grid]],
    uint grid_size  [[threads_per_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  float x = 0.0f;
  for (uint i = gid; i < p.count; i += grid_size) x += rank[dangling_list[i]];
  threadgroup float scratch[MG_TG_SCRATCH];
  const float total =
      mg_tg_reduce_sum_tid0(x, scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) partials[tg_id] = total;
}

// ONE threadgroup: finalize single-query PageRank dangling mass AND advance
// isolated-vertex scale scalars:
//   * n_partials > 0 : sums partials[0..n_partials)
//   * n_partials == 0 && n_active > 0 : single-TG direct sum over active_list
//   * n_partials == 0 && n_active == 0 : d_active = 0 (isolated-only update)
kernel void mg_pr_dangling_finalize(
    constant MGPrDangFinParams& p  [[buffer(0)]],
    device const float* partials   [[buffer(1)]],
    device const uint* active_list [[buffer(2)]],
    device const float* values     [[buffer(3)]],
    device float* scalars          [[buffer(4)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  float x = 0.0f;
  if (p.n_partials > 0u) {
    for (uint i = tid; i < p.n_partials; i += tg_size) x += partials[i];
  } else if (p.n_active > 0u) {
    for (uint i = tid; i < p.n_active; i += tg_size)
      x += values[active_list[i]];
  }
  threadgroup float scratch[MG_TG_SCRATCH];
  const float d_active =
      mg_tg_reduce_sum_tid0(x, scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) {
    const float d_prev = scalars[MG_SCALAR_DANGLING];
    const float s_cur = p.one_minus_alpha + p.alpha * d_prev;
    scalars[MG_SCALAR_ISO_SCALE_PREV] = scalars[MG_SCALAR_ISO_SCALE];
    scalars[MG_SCALAR_ISO_SCALE] = s_cur;
    scalars[MG_SCALAR_DANGLING] = d_active + s_cur * p.iso_p_sum;
  }
}

// Grid-stride partials over values[0..count) (HITS L1 norm).
kernel void mg_sum_partials(
    constant MGPrParams& p     [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* partials     [[buffer(2)]],
    uint gid        [[thread_position_in_grid]],
    uint grid_size  [[threads_per_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  float x = 0.0f;
  for (uint i = gid; i < p.count; i += grid_size) x += values[i];
  threadgroup float scratch[MG_TG_SCRATCH];
  const float total =
      mg_tg_reduce_sum_tid0(x, scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) partials[tg_id] = total;
}

// Grid-stride L1 residual partials over vertices[0..count), optionally
// rescaling contrib back to rank via out_weight_sum.
kernel void mg_pr_residual_partials(
    constant MGPrResParams& p          [[buffer(0)]],
    device const uint* vertices        [[buffer(1)]],
    device const float* buf_a          [[buffer(2)]],
    device const float* buf_b          [[buffer(3)]],
    device const float* out_weight_sum [[buffer(4)]],
    device float* partials             [[buffer(5)]],
    uint gid        [[thread_position_in_grid]],
    uint grid_size  [[threads_per_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  const bool rescale = (p.flags & MG_PR_FLAG_RESCALE_OWS) != 0u;
  float x = 0.0f;
  for (uint i = gid; i < p.count; i += grid_size) {
    const uint v = vertices[i];
    if (rescale) {
      const float ow = out_weight_sum[v];
      const float m = (ow > 0.0f) ? ow : 1.0f;
      x += fabs(buf_a[v] * m - buf_b[v] * m);
    } else {
      x += fabs(buf_a[v] - buf_b[v]);
    }
  }
  threadgroup float scratch[MG_TG_SCRATCH];
  const float total =
      mg_tg_reduce_sum_tid0(x, scratch, sg_in_tg, lane, sgs_per_tg);
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
  const float total =
      mg_tg_reduce_sum_tid0(x, scratch, sg_in_tg, lane, sgs_per_tg);
  if (tid == 0u) dst[p.dst_index] = total;
}

// ---------------------------------------------------------------------------
// Batched partials (float4 loads + 1-barrier 8-lane reductions)
// ---------------------------------------------------------------------------

kernel void mg_pr_dangling_partials_b(
    constant MGPprParams& p          [[buffer(0)]],
    device const uint* dangling_list [[buffer(1)]],
    device const float4* buf4_b      [[buffer(2)]],
    device float* partials           [[buffer(3)]],
    uint gid        [[thread_position_in_grid]],
    uint grid_size  [[threads_per_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  const bool active_lo = (p.active_mask & 0x0Fu) != 0u;
  const bool active_hi = (p.active_mask & 0xF0u) != 0u;
  float4 lo = float4(0.0f), hi = float4(0.0f);
  if (active_lo && active_hi) {
    for (uint i = gid; i < p.count; i += grid_size) {
      const ulong v2 = ulong(dangling_list[i]) * 2ul;
      lo += buf4_b[v2];
      hi += buf4_b[v2 + 1ul];
    }
  } else if (active_lo) {
    for (uint i = gid; i < p.count; i += grid_size)
      lo += buf4_b[ulong(dangling_list[i]) * 2ul];
  } else if (active_hi) {
    for (uint i = gid; i < p.count; i += grid_size)
      hi += buf4_b[ulong(dangling_list[i]) * 2ul + 1ul];
  }
  threadgroup float scratch[MG_PPR_TILE * MG_TG_SCRATCH];
  const float total = mg_tg_reduce_sum_b_lane(lo, hi, scratch, sg_in_tg, lane,
                                              sgs_per_tg);
  if (sg_in_tg == 0u && lane < MG_PPR_TILE)
    partials[ulong(tg_id) * MG_PPR_TILE + lane] = total;
}

kernel void mg_pr_residual_partials_b(
    constant MGPrResParams& p          [[buffer(0)]],
    device const uint* vertices        [[buffer(1)]],
    device const float4* buf4_a        [[buffer(2)]],
    device const float4* buf4_b        [[buffer(3)]],
    device const float* out_weight_sum [[buffer(4)]],
    device float* partials             [[buffer(5)]],
    uint gid        [[thread_position_in_grid]],
    uint grid_size  [[threads_per_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  const bool rescale = (p.flags & MG_PR_FLAG_RESCALE_OWS) != 0u;
  const bool active_lo = (p.active_mask & 0x0Fu) != 0u;
  const bool active_hi = (p.active_mask & 0xF0u) != 0u;
  float4 lo = float4(0.0f), hi = float4(0.0f);
  if (active_lo && active_hi) {
    for (uint i = gid; i < p.count; i += grid_size) {
      const uint v = vertices[i];
      const ulong v2 = ulong(v) * 2ul;
      float m = 1.0f;
      if (rescale) {
        const float ow = out_weight_sum[v];
        if (ow > 0.0f) m = ow;
      }
      lo += fabs(buf4_a[v2] * m - buf4_b[v2] * m);
      hi += fabs(buf4_a[v2 + 1ul] * m - buf4_b[v2 + 1ul] * m);
    }
  } else if (active_lo) {
    for (uint i = gid; i < p.count; i += grid_size) {
      const uint v = vertices[i];
      float m = 1.0f;
      if (rescale) {
        const float ow = out_weight_sum[v];
        if (ow > 0.0f) m = ow;
      }
      const ulong v2 = ulong(v) * 2ul;
      lo += fabs(buf4_a[v2] * m - buf4_b[v2] * m);
    }
  } else if (active_hi) {
    for (uint i = gid; i < p.count; i += grid_size) {
      const uint v = vertices[i];
      float m = 1.0f;
      if (rescale) {
        const float ow = out_weight_sum[v];
        if (ow > 0.0f) m = ow;
      }
      const ulong v2 = ulong(v) * 2ul + 1ul;
      hi += fabs(buf4_a[v2] * m - buf4_b[v2] * m);
    }
  }
  threadgroup float scratch[MG_PPR_TILE * MG_TG_SCRATCH];
  const float total = mg_tg_reduce_sum_b_lane(lo, hi, scratch, sg_in_tg, lane,
                                              sgs_per_tg);
  if (sg_in_tg == 0u && lane < MG_PPR_TILE)
    partials[ulong(tg_id) * MG_PPR_TILE + lane] = total;
}

// ONE threadgroup: finalize batched PPR dangling mass:
//   * n_partials > 0 : sums partials4[0..n_partials)
//   * n_partials == 0 && n_active > 0 : single-TG direct sum over active_list
//   * plus any isolated seeds (MG_PPR_SEED_ISO_BIT) for each active lane.
kernel void mg_reduce_finalize_b(
    constant MGPprDangFinParams& p  [[buffer(0)]],
    device const float4* partials4  [[buffer(1)]],
    device const uint* active_list  [[buffer(2)]],
    device const uint* seed_vertex  [[buffer(3)]],
    device const uint* seed_lane    [[buffer(4)]],
    device const float4* buf4_b     [[buffer(5)]],
    device float* dst               [[buffer(6)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  const bool active_lo = (p.active_mask & 0x0Fu) != 0u;
  const bool active_hi = (p.active_mask & 0xF0u) != 0u;
  float4 lo = float4(0.0f), hi = float4(0.0f);
  if (p.n_partials > 0u) {
    for (uint i = tid; i < p.n_partials; i += tg_size) {
      const ulong i2 = ulong(i) * 2ul;
      if (active_lo) lo += partials4[i2];
      if (active_hi) hi += partials4[i2 + 1ul];
    }
  } else if (p.n_active > 0u) {
    for (uint i = tid; i < p.n_active; i += tg_size) {
      const ulong v2 = ulong(active_list[i]) * 2ul;
      if (active_lo) lo += buf4_b[v2];
      if (active_hi) hi += buf4_b[v2 + 1ul];
    }
  }
  threadgroup float scratch[MG_PPR_TILE * MG_TG_SCRATCH];
  float total = mg_tg_reduce_sum_b_lane(lo, hi, scratch, sg_in_tg, lane,
                                        sgs_per_tg);
  if (sg_in_tg == 0u && lane < MG_PPR_TILE) {
    if (((p.active_mask >> lane) & 1u) != 0u) {
      device const float* buf_b = (device const float*)buf4_b;
      for (uint j = 0u; j < p.seed_total; ++j) {
        const uint raw = seed_lane[j];
        if ((raw & MG_PPR_SEED_ISO_BIT) != 0u && (raw & 0xFFu) == lane)
          total += buf_b[ulong(seed_vertex[j]) * MG_PPR_TILE + lane];
      }
      dst[lane] = total;
    }
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

kernel void mg_scale_f32_by_scalar(
    constant MGFillParams& p [[buffer(0)]],
    device float* values     [[buffer(1)]],
    device const float* norm [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const float d = norm[0];
  values[gid] = (d > 0.0f && isfinite(d)) ? values[gid] / d : 0.0f;
}
