// mg_common_msl.h — MSL-only helpers shared by metal-graph kernels.
// Include order (contract): <metal_stdlib>, using namespace metal,
// "mg_params.h", then this header.
// SPDX-License-Identifier: Apache-2.0
#ifndef MG_COMMON_MSL_H
#define MG_COMMON_MSL_H

#include "mg_params.h"

// Scratch slots for cross-simdgroup reductions. MG_TG_SIZE (256) threads
// with SIMD width >= 8 gives at most 32 simdgroups per threadgroup.
#define MG_TG_SCRATCH 32u

// Threadgroup-wide float sum over MG_TG_SIZE threads: simd_sum per
// simdgroup, then across simdgroups via threadgroup memory. Every thread
// of the threadgroup must call this (barriers inside — keep control flow
// threadgroup-uniform). Requires sgs_per_tg <= threads_per_simdgroup
// (true for MG_TG_SIZE == 256 with SIMD width >= 16). Returns the total
// to ALL threads; `scratch` (>= MG_TG_SCRATCH floats) is reusable
// immediately after return.
static inline float mg_tg_reduce_sum(float x, threadgroup float* scratch,
                                     ushort sg_in_tg, ushort lane,
                                     ushort sgs_per_tg) {
  x = simd_sum(x);
  if (lane == 0) scratch[sg_in_tg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_in_tg == 0) {
    float y = (lane < sgs_per_tg) ? scratch[lane] : 0.0f;
    y = simd_sum(y);
    if (lane == 0) scratch[0] = y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// Exactly-once visited-bitmap claim (plan §6 sketch): cheap atomic peek,
// then fetch_or. True iff THIS thread set the bit.
static inline bool mg_visited_claim(device atomic_uint* visited, uint v) {
  device atomic_uint* w = &visited[v >> 5u];
  const uint bit = 1u << (v & 31u);
  if ((atomic_load_explicit(w, memory_order_relaxed) & bit) != 0u)
    return false;
  return (atomic_fetch_or_explicit(w, bit, memory_order_relaxed) & bit) == 0u;
}

// Non-atomic bitmap test (read-only bitmaps, e.g. the k-hop reached set).
static inline bool mg_bitmap_test(device const uint* bitmap, uint v) {
  return (bitmap[v >> 5u] & (1u << (v & 31u))) != 0u;
}

// Device-side view of MGBfsControl (mg_params.h) — field-for-field mirror
// with atomic_uint for the atomically updated fields. atomic_uint has the
// size and alignment of uint, so the layouts are identical.
typedef struct MGBfsControlDev {
  uint frontier_count;         // single-writer (prepare_level)
  atomic_uint next_count;      // atomic append counter
  uint level;                  // single-writer (prepare_level)
  uint done;                   // single-writer (prepare_level)
  uint mode;                   // single-writer (decide)
  atomic_uint bin_count_high;  // atomic (frontier_bin)
  atomic_uint bin_count_mid;   // atomic (frontier_bin)
  atomic_uint bin_count_low;   // atomic (frontier_bin)
  atomic_uint frontier_edges;  // atomic (frontier_bin)
  uint unvisited;              // single-writer (decide)
  uint pad0;
  uint pad1;
} MGBfsControlDev;

#endif  // MG_COMMON_MSL_H
