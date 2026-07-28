// select.metal — GPU radix-select for batched-PPR top-k (plan M3).
// Binding tables: src/kernels/mg_params.h (authoritative).
//
// Scores are PageRank probabilities (fp32, >= 0), so their raw IEEE bit
// patterns compared as uint32 order exactly like the values do — no
// sign-flip mapping needed.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

// One threadgroup per query lane: MSB-first 8-bit histogram refinement.
// Four passes; after pass p the top 8*(p+1) bits of the k-th largest score
// are fixed in `prefix`. Elements are re-scanned each pass and counted only
// when their already-decided high bits match the prefix (no compaction —
// V*4 strided reads per lane are trivial next to the PPR iterations).
kernel void mg_topk_threshold_b(
    constant MGTopkParams& p   [[buffer(0)]],
    device const float* rank_b [[buffer(1)]],
    device uint* thresholds    [[buffer(2)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  const uint lane = tg_id;
  if (lane >= MG_PPR_TILE) return;                  // threadgroup-uniform
  if (((p.active_mask >> lane) & 1u) == 0u) return; // threadgroup-uniform

  threadgroup atomic_uint hist[256];
  threadgroup uint tg_prefix;
  threadgroup uint tg_remaining;
  if (tid == 0u) {
    tg_prefix = 0u;
    tg_remaining = p.k;  // 1-based rank still wanted inside the prefix class
  }

  for (uint pass = 0u; pass < 4u; ++pass) {
    const uint shift = 24u - pass * 8u;
    for (uint i = tid; i < 256u; i += tg_size)
      atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint prefix = tg_prefix;
    const uint pmask = (pass == 0u) ? 0u : (0xFFFFFFFFu << (shift + 8u));
    for (uint v = tid; v < p.v_count; v += tg_size) {
      const uint bits = as_type<uint>(rank_b[ulong(v) * MG_PPR_TILE + lane]);
      if ((bits & pmask) == prefix)
        atomic_fetch_add_explicit(&hist[(bits >> shift) & 0xFFu], 1u,
                                  memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
      // Walk buckets from the top: the k-th largest lives in the first
      // bucket where the running count reaches the remaining rank.
      uint rem = tg_remaining;
      uint acc = 0u;
      uint digit = 0u;
      for (int b = 255; b >= 0; --b) {
        const uint c =
            atomic_load_explicit(&hist[uint(b)], memory_order_relaxed);
        if (acc + c >= rem) {
          digit = uint(b);
          rem -= acc;
          break;
        }
        acc += c;
      }
      tg_remaining = rem;
      tg_prefix = prefix | (digit << shift);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0u) thresholds[lane] = tg_prefix;
}

// Thread per vertex: append every candidate with bits >= threshold to its
// lane's list. Counters keep counting past cap (overflow indicator for the
// host); writes are capped.
kernel void mg_topk_compact_b(
    constant MGTopkParams& p      [[buffer(0)]],
    device const float* rank_b    [[buffer(1)]],
    device const uint* thresholds [[buffer(2)]],
    device atomic_uint* counters  [[buffer(3)]],
    device uint* candidates       [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.v_count) return;
  const ulong base = ulong(gid) * MG_PPR_TILE;
  for (uint t = 0u; t < MG_PPR_TILE; ++t) {
    if (((p.active_mask >> t) & 1u) == 0u) continue;
    const uint bits = as_type<uint>(rank_b[base + t]);
    if (bits >= thresholds[t]) {
      const uint idx =
          atomic_fetch_add_explicit(&counters[t], 1u, memory_order_relaxed);
      if (idx < p.cap) candidates[ulong(t) * p.cap + idx] = gid;
    }
  }
}
