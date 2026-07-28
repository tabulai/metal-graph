// wcc.metal — WCC hook (min over both endpoints, per stored edge) and
// pointer jumping. Worklists are the OUT-orientation degree bins.
// Binding tables: src/kernels/mg_params.h (authoritative).
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

// For edge (v, u): m = min(labels[v], labels[u]); atomic_min both
// endpoints. True iff either fetch_min actually lowered a label
// (compare returned old value).
static inline bool mg_wcc_hook_edge(device atomic_uint* labels, uint v,
                                    uint u) {
  const uint lv = atomic_load_explicit(&labels[v], memory_order_relaxed);
  const uint lu = atomic_load_explicit(&labels[u], memory_order_relaxed);
  const uint m = min(lv, lu);
  const uint old_u =
      atomic_fetch_min_explicit(&labels[u], m, memory_order_relaxed);
  const uint old_v =
      atomic_fetch_min_explicit(&labels[v], m, memory_order_relaxed);
  return (old_u > m) || (old_v > m);
}

// High bin: one threadgroup per worklist vertex, edges strided by
// threads_per_threadgroup.
kernel void mg_wcc_hook_tg(
    constant MGWccParams& p     [[buffer(0)]],
    device const uint* worklist [[buffer(1)]],
    device const uint* row_off  [[buffer(2)]],
    device const uint* col_idx  [[buffer(3)]],
    device atomic_uint* labels  [[buffer(4)]],
    device atomic_uint* changed [[buffer(5)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  if (tg_id >= p.count) return;
  const uint v = worklist[tg_id];
  const uint end = row_off[v + 1u];
  bool any = false;
  for (uint e = row_off[v] + tid; e < end; e += tg_size)
    any = mg_wcc_hook_edge(labels, v, col_idx[e]) || any;
  if (any) atomic_store_explicit(changed, 1u, memory_order_relaxed);
}

// Mid bin: one simdgroup per worklist vertex.
kernel void mg_wcc_hook_sg(
    constant MGWccParams& p     [[buffer(0)]],
    device const uint* worklist [[buffer(1)]],
    device const uint* row_off  [[buffer(2)]],
    device const uint* col_idx  [[buffer(3)]],
    device atomic_uint* labels  [[buffer(4)]],
    device atomic_uint* changed [[buffer(5)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  const uint wi = tg_id * sgs_per_tg + sg_in_tg;  // global simdgroup index
  if (wi >= p.count) return;
  const uint v = worklist[wi];
  const uint end = row_off[v + 1u];
  bool any = false;
  for (uint e = row_off[v] + lane; e < end; e += simd_sz)
    any = mg_wcc_hook_edge(labels, v, col_idx[e]) || any;
  if (any) atomic_store_explicit(changed, 1u, memory_order_relaxed);
}

// Low bin: one thread per worklist vertex, sequential edge loop.
kernel void mg_wcc_hook_thread(
    constant MGWccParams& p     [[buffer(0)]],
    device const uint* worklist [[buffer(1)]],
    device const uint* row_off  [[buffer(2)]],
    device const uint* col_idx  [[buffer(3)]],
    device atomic_uint* labels  [[buffer(4)]],
    device atomic_uint* changed [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint v = worklist[gid];
  const uint end = row_off[v + 1u];
  bool any = false;
  for (uint e = row_off[v]; e < end; ++e)
    any = mg_wcc_hook_edge(labels, v, col_idx[e]) || any;
  if (any) atomic_store_explicit(changed, 1u, memory_order_relaxed);
}

// Pointer jump: l = labels[v]; gp = labels[l]; if gp < l, lower labels[v].
kernel void mg_wcc_jump(
    constant MGWccParams& p     [[buffer(0)]],
    device atomic_uint* labels  [[buffer(1)]],
    device atomic_uint* changed [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint l = atomic_load_explicit(&labels[gid], memory_order_relaxed);
  const uint gp = atomic_load_explicit(&labels[l], memory_order_relaxed);
  if (gp < l) {
    atomic_fetch_min_explicit(&labels[gid], gp, memory_order_relaxed);
    atomic_store_explicit(changed, 1u, memory_order_relaxed);
  }
}

// Huge bin: one threadgroup per fixed edge tile of a mega-degree vertex
// (tiling tables in mg_params.h) — hooks are per-edge independent, so no
// reduction is needed; only the changed-flag store is folded per thread.
kernel void mg_wcc_hook_huge(
    constant MGWccParams& p       [[buffer(0)]],
    device const uint* tile_owner [[buffer(1)]],
    device const uint* tile_first [[buffer(2)]],
    device const uint* wl_huge    [[buffer(3)]],
    device const uint* row_off    [[buffer(4)]],
    device const uint* col_idx    [[buffer(5)]],
    device atomic_uint* labels    [[buffer(6)]],
    device atomic_uint* changed   [[buffer(7)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  if (tg_id >= p.count) return;
  const uint v = wl_huge[tile_owner[tg_id]];
  const uint begin = tile_first[tg_id];
  const uint end = min(row_off[v + 1u], begin + MG_HUGE_TILE_EDGES);
  bool any = false;
  for (uint e = begin + tid; e < end; e += tg_size)
    any = mg_wcc_hook_edge(labels, v, col_idx[e]) || any;
  if (any) atomic_store_explicit(changed, 1u, memory_order_relaxed);
}
