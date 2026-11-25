// select.metal — GPU radix-select for batched-PPR top-k (plan M3).
// Binding tables: src/kernels/mg_params.h (authoritative).
//
// Scores are PageRank probabilities (fp32, >= 0), so their raw IEEE bit
// patterns compared as uint32 order exactly like the values do — no
// sign-flip mapping needed, and bit 31 is always 0.
//
// Optimizations:
//   * Pass 0 optionally unscales contrib (rank/ows -> rank) in place in the
//     same coalesced sequential pass, eliminating a separate RMW pass.
//   * Per-lane early termination: as soon as (k - rem) + class_size <= cap
//     (often already at pass 0 or pass 1!), that lane leaves pending_mask so
//     subsequent passes skip it (or exit immediately if pending_mask == 0).
//   * Exact O(k) tie-flood handling: if a lane still exceeds cap after pass 3,
//     thresholds[lane] gets MG_TOPK_STRICT_GT_BIT set so compact collects only
//     strictly-greater items (< k <= cap, zero tie-flood atomics) and the host
//     fills the tie tail in O(k) ascending user order.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

static_assert(MG_PPR_TILE == 8u, "a vertex's lanes are two float4s");

kernel void mg_topk_hist_b(
    constant MGTopkParams& p           [[buffer(0)]],
    device float4* rank4_b             [[buffer(1)]],
    device const float* out_weight_sum [[buffer(2)]],
    device const uint* thresholds      [[buffer(3)]],
    device const uint* select_state    [[buffer(4)]],
    device atomic_uint* hist           [[buffer(5)]],
    uint gid       [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]]) {
  const uint mask = select_state[MG_TOPK_STATE_PENDING];
  if (mask == 0u) return;

  threadgroup atomic_uint h[MG_PPR_TILE * 256u];
  for (uint i = tid; i < MG_PPR_TILE * 256u; i += tg_size)
    atomic_store_explicit(&h[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint shift = 24u - 8u * p.pass;
  const uint pmask = (p.pass == 0u) ? 0u : (0xFFFFFFFFu << (shift + 8u));
  const bool active_lo = (mask & 0x0Fu) != 0u;
  const bool active_hi = (mask & 0xF0u) != 0u;
  const bool do_unscale =
      (p.pass == 0u) && ((p.flags & MG_TOPK_FLAG_UNSCALE_OWS) != 0u);
  uint prefix[MG_PPR_TILE];
  if (p.pass != 0u) {
    for (uint t = 0u; t < MG_PPR_TILE; ++t) prefix[t] = thresholds[t];
  }

  if (p.pass == 0u) {
    for (uint v = gid; v < p.v_count; v += grid_size) {
      const float ow = do_unscale ? out_weight_sum[v] : 0.0f;
      if (active_lo) {
        float4 lo = rank4_b[2ul * v];
        if (ow > 0.0f) {
          lo *= ow;
          rank4_b[2ul * v] = lo;
        }
        const uint b[4] = {as_type<uint>(lo.x), as_type<uint>(lo.y),
                           as_type<uint>(lo.z), as_type<uint>(lo.w)};
        for (uint t = 0u; t < 4u; ++t) {
          if (((mask >> t) & 1u) == 0u) continue;
          const uint digit = b[t] >> 24u;
          if (digit != 0u)
            atomic_fetch_add_explicit(&h[t * 256u + digit], 1u,
                                      memory_order_relaxed);
        }
      }
      if (active_hi) {
        float4 hi = rank4_b[2ul * v + 1ul];
        if (ow > 0.0f) {
          hi *= ow;
          rank4_b[2ul * v + 1ul] = hi;
        }
        const uint b[4] = {as_type<uint>(hi.x), as_type<uint>(hi.y),
                           as_type<uint>(hi.z), as_type<uint>(hi.w)};
        for (uint t = 0u; t < 4u; ++t) {
          const uint lane = t + 4u;
          if (((mask >> lane) & 1u) == 0u) continue;
          const uint digit = b[t] >> 24u;
          if (digit != 0u)
            atomic_fetch_add_explicit(&h[lane * 256u + digit], 1u,
                                      memory_order_relaxed);
        }
      }
    }
  } else {
    for (uint v = gid; v < p.v_count; v += grid_size) {
      if (active_lo) {
        const float4 lo = rank4_b[2ul * v];
        const uint b[4] = {as_type<uint>(lo.x), as_type<uint>(lo.y),
                           as_type<uint>(lo.z), as_type<uint>(lo.w)};
        for (uint t = 0u; t < 4u; ++t) {
          if (((mask >> t) & 1u) == 0u || b[t] == 0u) continue;
          const uint digit = (b[t] >> shift) & 0xFFu;
          if (digit != 0u && (b[t] & pmask) == prefix[t])
            atomic_fetch_add_explicit(&h[t * 256u + digit], 1u,
                                      memory_order_relaxed);
        }
      }
      if (active_hi) {
        const float4 hi = rank4_b[2ul * v + 1ul];
        const uint b[4] = {as_type<uint>(hi.x), as_type<uint>(hi.y),
                           as_type<uint>(hi.z), as_type<uint>(hi.w)};
        for (uint t = 0u; t < 4u; ++t) {
          const uint lane = t + 4u;
          if (((mask >> lane) & 1u) == 0u || b[t] == 0u) continue;
          const uint digit = (b[t] >> shift) & 0xFFu;
          if (digit != 0u && (b[t] & pmask) == prefix[lane])
            atomic_fetch_add_explicit(&h[lane * 256u + digit], 1u,
                                      memory_order_relaxed);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < MG_PPR_TILE * 256u; i += tg_size) {
    const uint c = atomic_load_explicit(&h[i], memory_order_relaxed);
    if (c != 0u)
      atomic_fetch_add_explicit(&hist[i], c, memory_order_relaxed);
  }
}

// ONE threadgroup: each pending lane selects its next radix byte. If the
// >=-threshold class already fits in p.cap, the lane finishes immediately and
// drops out of pending_mask; if it still exceeds p.cap on pass 3, bit 31
// (MG_TOPK_STRICT_GT_BIT) is set on thresholds[lane].
kernel void mg_topk_digit_b(
    constant MGTopkParams& p     [[buffer(0)]],
    device atomic_uint* hist     [[buffer(1)]],
    device uint* thresholds      [[buffer(2)]],
    device uint* select_state    [[buffer(3)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]]) {
  const uint pending = select_state[MG_TOPK_STATE_PENDING];
  if (pending == 0u) return;

  uint my_pending = 0u;
  if (tid < MG_PPR_TILE && ((pending >> tid) & 1u) != 0u) {
    const uint l = tid;
    const uint shift = 24u - 8u * p.pass;
    device uint* remaining = select_state;
    device uint* class_size = select_state + MG_PPR_TILE;
    const uint total = class_size[l];
    uint rem = remaining[l];
    uint acc = 0u;
    uint digit = 0u;
    uint count = 0u;
    for (int b = 255; b >= 0; --b) {
      const uint c = (b > 0) ? atomic_load_explicit(
                                   &hist[l * 256u + uint(b)],
                                   memory_order_relaxed)
                             : total - acc;
      if (acc + c >= rem) {
        digit = uint(b);
        rem -= acc;
        count = c;
        break;
      }
      acc += c;
    }
    remaining[l] = rem;
    class_size[l] = count;
    const uint thr = thresholds[l] | (digit << shift);
    const uint total_ge = (p.k - rem) + count;
    if (total_ge <= p.cap) {
      thresholds[l] = thr;
    } else if (p.pass == 3u) {
      thresholds[l] = thr | MG_TOPK_STRICT_GT_BIT;
    } else {
      thresholds[l] = thr;
      my_pending = 1u << l;
    }
  }
  if (sg_in_tg == 0u) {
    // Disjoint bit per lane 0..7 => simd_sum == bitwise OR.
    const uint next_pending = simd_sum(my_pending);
    if (lane == 0u) select_state[MG_TOPK_STATE_PENDING] = next_pending;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = tid; i < MG_PPR_TILE * 256u; i += tg_size)
    atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
}

// Thread per vertex: vectorized float4 loads + SIMD-group compaction (at most
// 1 global atomic per simdgroup per lane). Normal lanes compact bits >= thr;
// tie-flood lanes (MG_TOPK_STRICT_GT_BIT) compact bits > thr (< k <= cap).
kernel void mg_topk_compact_b(
    constant MGTopkParams& p      [[buffer(0)]],
    device const float4* rank4_b  [[buffer(1)]],
    device const uint* thresholds [[buffer(2)]],
    device atomic_uint* counters  [[buffer(3)]],
    device uint* candidates       [[buffer(4)]],
    uint gid    [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  uint thr[MG_PPR_TILE];
  bool strict_gt[MG_PPR_TILE];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) {
    const uint raw = thresholds[t];
    strict_gt[t] = (raw & MG_TOPK_STRICT_GT_BIT) != 0u;
    thr[t] = raw & 0x7FFFFFFFu;
  }

  const bool in_range = gid < p.v_count;
  uint bits[MG_PPR_TILE] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
  if (in_range) {
    if ((p.active_mask & 0x0Fu) != 0u) {
      const float4 lo = rank4_b[2ul * gid];
      bits[0] = as_type<uint>(lo.x);
      bits[1] = as_type<uint>(lo.y);
      bits[2] = as_type<uint>(lo.z);
      bits[3] = as_type<uint>(lo.w);
    }
    if ((p.active_mask & 0xF0u) != 0u) {
      const float4 hi = rank4_b[2ul * gid + 1ul];
      bits[4] = as_type<uint>(hi.x);
      bits[5] = as_type<uint>(hi.y);
      bits[6] = as_type<uint>(hi.z);
      bits[7] = as_type<uint>(hi.w);
    }
  }

  for (uint t = 0u; t < MG_PPR_TILE; ++t) {
    if (((p.active_mask >> t) & 1u) == 0u) continue;
    const bool hit = in_range &&
                     (strict_gt[t] ? (bits[t] > thr[t]) : (bits[t] >= thr[t]));
    const uint sg_cnt = simd_sum(uint(hit));
    if (sg_cnt == 0u) continue;
    const uint r = simd_prefix_exclusive_sum(uint(hit));
    uint base = 0u;
    if (lane == 0u)
      base = atomic_fetch_add_explicit(&counters[t], sg_cnt,
                                       memory_order_relaxed);
    base = simd_broadcast_first(base);
    if (hit && (base + r) < p.cap)
      candidates[ulong(t) * p.cap + (base + r)] = gid;
  }
}
