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

// MSB-first 8-bit radix select over every active lane of a tile at once:
// four passes, each mg_topk_hist_b (whole grid) then mg_topk_digit_b (one
// threadgroup). After pass p the top 8*(p+1) bits of each lane's k-th
// largest score are fixed in thresholds[lane]. Counts are exact integers,
// so the result is the same bit pattern any exact selection would return.
//
// Digit 0 is never counted: it is derived as class_size - (digits 1..255),
// where class_size is the number of scores matching the decided prefix (V
// in pass 0). Exact zeros (unreached vertices) would otherwise put most of
// the tile on one threadgroup atomic.

// Thread per vertex (grid-stride): every active lane's score whose decided
// high bits match thresholds[lane] counts toward its next digit, first in a
// threadgroup histogram, then flushed to hist with one device atomic per
// nonzero bucket.
static_assert(MG_PPR_TILE == 8u, "a vertex's lanes are two float4s");
kernel void mg_topk_hist_b(
    constant MGTopkParams& p      [[buffer(0)]],
    device const float4* rank4_b  [[buffer(1)]],
    device const uint* thresholds [[buffer(2)]],
    device atomic_uint* hist      [[buffer(3)]],
    uint gid       [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]]) {
  threadgroup atomic_uint h[MG_PPR_TILE * 256u];
  for (uint i = tid; i < MG_PPR_TILE * 256u; i += tg_size)
    atomic_store_explicit(&h[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint shift = 24u - 8u * p.pass;
  const uint pmask = (p.pass == 0u) ? 0u : (0xFFFFFFFFu << (shift + 8u));
  uint prefix[MG_PPR_TILE];
  for (uint t = 0u; t < MG_PPR_TILE; ++t) prefix[t] = thresholds[t];
  for (uint v = gid; v < p.v_count; v += grid_size) {
    const float4 lo = rank4_b[2ul * v];
    const float4 hi = rank4_b[2ul * v + 1ul];
    const float s[MG_PPR_TILE] = {lo.x, lo.y, lo.z, lo.w,
                                  hi.x, hi.y, hi.z, hi.w};
    for (uint t = 0u; t < MG_PPR_TILE; ++t) {
      if (((p.active_mask >> t) & 1u) == 0u) continue;
      const uint bits = as_type<uint>(s[t]);
      const uint digit = (bits >> shift) & 0xFFu;
      if (digit != 0u && (bits & pmask) == prefix[t])
        atomic_fetch_add_explicit(&h[t * 256u + digit], 1u,
                                  memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < MG_PPR_TILE * 256u; i += tg_size) {
    const uint c = atomic_load_explicit(&h[i], memory_order_relaxed);
    if (c != 0u)
      atomic_fetch_add_explicit(&hist[i], c, memory_order_relaxed);
  }
}

// ONE threadgroup: thread `lane` walks its lane's buckets from the top; the
// k-th largest lives in the first bucket where the running count reaches
// the remaining rank. Then every thread clears hist for the next pass.
kernel void mg_topk_digit_b(
    constant MGTopkParams& p      [[buffer(0)]],
    device atomic_uint* hist      [[buffer(1)]],
    device uint* thresholds       [[buffer(2)]],
    device uint* select_state     [[buffer(3)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  if (tid < MG_PPR_TILE && ((p.active_mask >> tid) & 1u) != 0u) {
    const uint lane = tid;
    const uint shift = 24u - 8u * p.pass;
    device uint* remaining = select_state;
    device uint* class_size = select_state + MG_PPR_TILE;
    const uint total = class_size[lane];
    uint rem = remaining[lane];
    uint acc = 0u;
    uint digit = 0u;
    uint count = 0u;
    for (int b = 255; b >= 0; --b) {
      // acc == sum of buckets 255..1 when b reaches 0.
      const uint c = (b > 0) ? atomic_load_explicit(
                                   &hist[lane * 256u + uint(b)],
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
    remaining[lane] = rem;
    class_size[lane] = count;
    thresholds[lane] |= digit << shift;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint i = tid; i < MG_PPR_TILE * 256u; i += tg_size)
    atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
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
