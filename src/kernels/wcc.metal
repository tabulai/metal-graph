// wcc.metal — WCC as one concurrent union-find LINK pass over the stored
// OUT edges (ECL-CC style CAS hooking) followed by one COMPRESS pass.
// Worklists are the OUT-orientation degree bins.
// Binding tables: src/kernels/mg_params.h (authoritative).
//
// Invariant: labels[x] <= x for every x at all times (iota start; a link
// only ever stores a smaller id into a ROOT, compression only stores an
// ancestor). Parent chains therefore strictly decrease, every tree is
// rooted at its minimum canonical id, and no loop below can cycle: each
// retry observes a strictly smaller id. No thread ever waits on another.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

// Parent loads are PLAIN (non-atomic) reads of the atomic array: atomic
// loads are served at the coherence point and serialize on hot roots
// (measured ~3x slower link pass). A plain read may return a stale value,
// which is harmless here: every value labels[x] ever holds is x itself or
// an ancestor of x (trees only merge, never split), so a stale read only
// yields a less-compressed path or a stale "root" — equal finds still imply
// the same set, and correctness rests solely on the CAS (performed on the
// coherent copy) plus the strict decrease that bounds every loop.
static inline uint mg_wcc_ld(device atomic_uint* labels, uint i) {
  return reinterpret_cast<device const uint*>(labels)[i];
}

// Root of x's tree as of the loads performed. Path halving: a non-root x
// is re-pointed at its grandparent (an ancestor, so the store is safe
// against any concurrent link — links only target roots, and x is
// permanently a non-root once labels[x] < x was observed). Vertex 0 is
// always a root (nothing is smaller), so reaching it ends the walk.
static inline uint mg_wcc_find(device atomic_uint* labels, uint x) {
  uint p = mg_wcc_ld(labels, x);
  while (p < x) {
    if (p == 0u) return 0u;
    const uint gp = mg_wcc_ld(labels, p);
    if (gp < p)
      atomic_store_explicit(&labels[x], gp, memory_order_relaxed);
    x = p;
    p = gp;
  }
  return x;
}

// Unite the sets containing a and b (a, b: representatives obtained by
// find, possibly stale). The larger id is hooked under the smaller one
// with a CAS that succeeds only while the larger id is still a root
// (expected == itself); on failure it was hooked elsewhere meanwhile, so
// re-find from its new parent and retry. Returns a member of the merged
// set (the smaller id) for the caller to reuse as its representative.
static inline uint mg_wcc_link(device atomic_uint* labels, uint a, uint b) {
  while (a != b) {
    const uint hi = max(a, b);
    const uint lo = min(a, b);
    uint expected = hi;
    if (atomic_compare_exchange_weak_explicit(&labels[hi], &expected, lo,
                                              memory_order_relaxed,
                                              memory_order_relaxed))
      return lo;
    // expected == labels[hi] now (< hi unless the weak CAS failed
    // spuriously, in which case find(hi) == hi and we simply retry).
    a = mg_wcc_find(labels, expected);
    b = mg_wcc_find(labels, lo);
  }
  return a;
}

// Link every edge (v, u) of [begin, end) with the given stride; find(v) is
// hoisted out of the edge loop and refreshed only when a link is needed.
// skip_twins: undirected storage holds each u != v edge both ways; only
// the (v, u > v) copy is linked.
static inline void mg_wcc_link_edges(device const uint* col_idx,
                                     device atomic_uint* labels, uint v,
                                     uint begin, uint end, uint stride,
                                     bool skip_twins) {
  uint rv = mg_wcc_find(labels, v);
  for (uint e = begin; e < end; e += stride) {
    const uint u = col_idx[e];
    if (skip_twins && u <= v) continue;
    const uint ru = mg_wcc_find(labels, u);
    if (ru == rv) continue;
    rv = mg_wcc_find(labels, rv);  // rv may have been hooked meanwhile
    if (ru != rv) rv = mg_wcc_link(labels, rv, ru);
  }
}

// High bin: one threadgroup per worklist vertex, edges strided by
// threads_per_threadgroup.
kernel void mg_wcc_link_tg(
    constant MGWccParams& p     [[buffer(0)]],
    device const uint* worklist [[buffer(1)]],
    device const uint* row_off  [[buffer(2)]],
    device const uint* col_idx  [[buffer(3)]],
    device atomic_uint* labels  [[buffer(4)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  if (tg_id >= p.count) return;
  const uint v = worklist[tg_id];
  mg_wcc_link_edges(col_idx, labels, v, row_off[v] + tid, row_off[v + 1u],
                    tg_size, (p.flags & MG_WCC_SKIP_TWINS) != 0u);
}

// Mid bin: one simdgroup per worklist vertex.
kernel void mg_wcc_link_sg(
    constant MGWccParams& p     [[buffer(0)]],
    device const uint* worklist [[buffer(1)]],
    device const uint* row_off  [[buffer(2)]],
    device const uint* col_idx  [[buffer(3)]],
    device atomic_uint* labels  [[buffer(4)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  const uint wi = tg_id * sgs_per_tg + sg_in_tg;  // global simdgroup index
  if (wi >= p.count) return;
  const uint v = worklist[wi];
  mg_wcc_link_edges(col_idx, labels, v, row_off[v] + lane, row_off[v + 1u],
                    simd_sz, (p.flags & MG_WCC_SKIP_TWINS) != 0u);
}

// Low bin: one thread per worklist vertex, sequential edge loop.
kernel void mg_wcc_link_thread(
    constant MGWccParams& p     [[buffer(0)]],
    device const uint* worklist [[buffer(1)]],
    device const uint* row_off  [[buffer(2)]],
    device const uint* col_idx  [[buffer(3)]],
    device atomic_uint* labels  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint v = worklist[gid];
  mg_wcc_link_edges(col_idx, labels, v, row_off[v], row_off[v + 1u], 1u,
                    (p.flags & MG_WCC_SKIP_TWINS) != 0u);
}

// Huge bin: one threadgroup per fixed edge tile of a mega-degree vertex
// (tiling tables in mg_params.h) — links are per-edge independent.
kernel void mg_wcc_link_huge(
    constant MGWccParams& p       [[buffer(0)]],
    device const uint* tile_owner [[buffer(1)]],
    device const uint* tile_first [[buffer(2)]],
    device const uint* wl_huge    [[buffer(3)]],
    device const uint* row_off    [[buffer(4)]],
    device const uint* col_idx    [[buffer(5)]],
    device atomic_uint* labels    [[buffer(6)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  if (tg_id >= p.count) return;
  const uint v = wl_huge[tile_owner[tg_id]];
  const uint begin = tile_first[tg_id];
  const uint end = min(row_off[v + 1u], begin + MG_HUGE_TILE_EDGES);
  mg_wcc_link_edges(col_idx, labels, v, begin + tid, end, tg_size,
                    (p.flags & MG_WCC_SKIP_TWINS) != 0u);
}

// Compress: labels[v] = root(v). Runs after every link has completed
// (serial encoder), so roots are final and each thread's walk ends at its
// component minimum; concurrent stores only ever write roots.
kernel void mg_wcc_compress(
    constant MGWccParams& p     [[buffer(0)]],
    device atomic_uint* labels  [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint l0 = mg_wcc_ld(labels, gid);
  uint l = l0;
  uint nx = mg_wcc_ld(labels, l);
  while (nx < l) {
    l = nx;
    nx = mg_wcc_ld(labels, l);
  }
  if (l != l0) atomic_store_explicit(&labels[gid], l, memory_order_relaxed);
}
