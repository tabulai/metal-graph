// wcc.metal — WCC as concurrent union-find (ECL-CC style CAS hooking)
// behind an Afforest-style snapshot filter, in one serial batch:
//   1. sample:   link every vertex with its first MG_WCC_SAMPLE_EDGES OUT
//                neighbors, which already joins most of the giant component;
//   2. snapshot: roots[v] = root(v), plus a bitmap of vertex 0's set (the
//                "giant set": vertex 0 is always a root);
//   3. link:     every stored OUT edge (all degree bins in one dispatch)
//                whose snapshot roots differ; on undirected storage the
//                giant set's vertices are not scanned at all (Afforest);
//   4. compress: roots[v] = root(roots[v]), the final labels.
// Binding tables: src/kernels/mg_params.h (authoritative).
//
// Invariant: labels[x] <= x for every x at all times (iota start; a link
// only ever stores a smaller id into a ROOT, path halving only stores an
// ancestor). Parent walks therefore strictly decrease and cannot cycle;
// every tree is rooted at its minimum canonical id. A failed non-spurious
// CAS observes a smaller parent; a spurious failure simply retries.
// No thread ever waits on another.
//
// Memory model: labels is accessed only atomically in the dispatches that
// write it (sample, link); snapshot and compress only READ it, so they use
// plain loads. roots and giant are written only by their owner threads
// (snapshot, compress) and are read-only during link, where plain loads of
// them replace the atomic finds of every edge inside one snapshot set. The
// serial encoder orders the dispatches.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

// Parent loads while other threads may store into labels. Relaxed ordering
// is sufficient: union-find needs coherent parent values but publishes no
// other data.
static inline uint mg_wcc_ld(device atomic_uint* labels, uint i) {
  return atomic_load_explicit(&labels[i], memory_order_relaxed);
}

// Root of x's tree as of the loads performed. Path halving: a non-root x
// is re-pointed at its grandparent (an ancestor, so the store is safe
// against any concurrent link — links only target roots, and x is
// permanently a non-root once labels[x] < x was observed). Vertex 0 is
// always a root (nothing is smaller), so reaching it ends the walk without
// loading its hot slot.
static inline uint mg_wcc_find(device atomic_uint* labels, uint x) {
  if (x == 0u) return 0u;
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

// Unite the sets containing a and b (any members, e.g. possibly stale find
// results). The larger id is hooked under the smaller one with a CAS that
// succeeds only while the larger id is still a root (expected == itself);
// a root is its tree's minimum, so the smaller id lies in another tree. On
// failure the larger id is not (or no longer) a root, so re-find both and
// retry. Returns a member of the merged set (the smaller id) for the
// caller to reuse as its representative.
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

// Sample: one thread per vertex with out-degree >= 1 (canonical ids are
// out-degree descending, so they are exactly [0, count)); links its first
// MG_WCC_SAMPLE_EDGES stored edges. Linking v itself rather than find(v)
// lets the common first link succeed with one CAS on v's own slot (v is
// still a root and its neighbor, often a hub, has the smaller id) without
// loading hot hub slots.
kernel void mg_wcc_sample(
    constant MGWccParams& p     [[buffer(0)]],
    device const uint* row_off  [[buffer(1)]],
    device const uint* col_idx  [[buffer(2)]],
    device atomic_uint* labels  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint begin = row_off[gid];
  const uint end = min(row_off[gid + 1u], begin + MG_WCC_SAMPLE_EDGES);
  for (uint e = begin; e < end; ++e) mg_wcc_link(labels, gid, col_idx[e]);
}

// Snapshot giant set: bit x of giant[] is set iff x's snapshot root is 0
// (vertex 0 is always a root, so that set's root never changes).
static inline bool mg_wcc_in_giant(device const uint* giant, uint x) {
  return ((giant[x >> 5] >> (x & 31u)) & 1u) != 0u;
}

// Link every edge (v, u) of [begin, end) with the given stride whose
// snapshot roots differ: equal snapshot roots already share a set (sets
// only merge). Giant membership is a bitmap probe; roots[] is read only
// for vertices outside the giant set. For crossing edges the finds start
// at the snapshot roots, which were roots when taken and stay in their
// vertices' trees.
// MG_WCC_SKIP_GIANT (undirected storage): giant-set vertices are skipped,
// so an edge between the giant set and another set is linked from its
// other endpoint's stored copy instead. MG_WCC_SKIP_TWINS (undirected
// storage): an edge between two scanned vertices is linked only from its
// (v, u > v) copy.
static inline void mg_wcc_link_edges(device const uint* col_idx,
                                     device const uint* roots,
                                     device const uint* giant,
                                     device atomic_uint* labels, uint v,
                                     uint begin, uint end, uint stride,
                                     uint flags) {
  const bool gv = mg_wcc_in_giant(giant, v);
  if (gv && (flags & MG_WCC_SKIP_GIANT) != 0u) return;
  const bool skip_twins = (flags & MG_WCC_SKIP_TWINS) != 0u;
  const uint sv = gv ? 0u : roots[v];
  uint rv = sv;
  for (uint e = begin; e < end; e += stride) {
    const uint u = col_idx[e];
    uint su = 0u;
    if (mg_wcc_in_giant(giant, u)) {
      if (gv) continue;
    } else {
      su = roots[u];
      if (su == sv || (skip_twins && u < v)) continue;
    }
    const uint ru = mg_wcc_find(labels, su);
    rv = mg_wcc_find(labels, rv);  // rv may have been hooked meanwhile
    if (ru != rv) rv = mg_wcc_link(labels, rv, ru);
  }
}

// Every bin in ONE dispatch: no drain between bins, and the latency-bound
// loops over long rows overlap the short rows. Threadgroup ranges, in
// order: huge tiles, high vertices (one threadgroup each), mid vertices
// (one simdgroup each), low vertices (one thread each).
kernel void mg_wcc_link(
    constant MGWccLinkParams& p   [[buffer(0)]],
    device const uint* tile_owner [[buffer(1)]],
    device const uint* tile_first [[buffer(2)]],
    device const uint* wl_huge    [[buffer(3)]],
    device const uint* wl_high    [[buffer(4)]],
    device const uint* wl_mid     [[buffer(5)]],
    device const uint* wl_low     [[buffer(6)]],
    device const uint* row_off    [[buffer(7)]],
    device const uint* col_idx    [[buffer(8)]],
    device const uint* roots      [[buffer(9)]],
    device const uint* giant      [[buffer(10)]],
    device atomic_uint* labels    [[buffer(11)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid          [[thread_index_in_threadgroup]],
    uint tg_size      [[threads_per_threadgroup]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  uint t = tg_id;
  if (t < p.n_tiles) {
    const uint v = wl_huge[tile_owner[t]];
    const uint begin = tile_first[t];
    const uint end = min(row_off[v + 1u], begin + MG_HUGE_TILE_EDGES);
    mg_wcc_link_edges(col_idx, roots, giant, labels, v, begin + tid, end,
                      tg_size, p.flags);
    return;
  }
  t -= p.n_tiles;
  if (t < p.n_high) {
    const uint v = wl_high[t];
    mg_wcc_link_edges(col_idx, roots, giant, labels, v, row_off[v] + tid,
                      row_off[v + 1u], tg_size, p.flags);
    return;
  }
  t -= p.n_high;
  if (t < p.mid_tgs) {
    const uint wi = t * sgs_per_tg + sg_in_tg;  // global simdgroup index
    if (wi >= p.n_mid) return;
    const uint v = wl_mid[wi];
    mg_wcc_link_edges(col_idx, roots, giant, labels, v, row_off[v] + lane,
                      row_off[v + 1u], simd_sz, p.flags);
    return;
  }
  t -= p.mid_tgs;
  const uint i = t * tg_size + tid;
  if (i >= p.n_low) return;
  const uint v = wl_low[i];
  mg_wcc_link_edges(col_idx, roots, giant, labels, v, row_off[v],
                    row_off[v + 1u], 1u, p.flags);
}

// Snapshot: roots[v] = root of v's tree, recorded for vertices outside the
// giant set, plus the giant bitmap (bit v <=> root 0) assembled from one
// simd_ballot per 32 vertices (SIMD width is a multiple of 32 — checked
// by the host — and threadgroups start at multiples of it, so each word
// has exactly one writer). No thread writes labels in this dispatch, so
// its plain loads are race free.
kernel void mg_wcc_snapshot(
    constant MGWccParams& p     [[buffer(0)]],
    device const uint* labels   [[buffer(1)]],
    device uint* roots          [[buffer(2)]],
    device uint* giant          [[buffer(3)]],
    uint gid    [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  if (gid >= p.count) return;
  uint l = gid;
  uint nx = labels[l];
  while (nx < l) {
    l = nx;
    nx = labels[l];
  }
  if (l != 0u) roots[gid] = l;
  const ulong vote = static_cast<ulong>(simd_ballot(l == 0u));
  if ((gid & 31u) == 0u)
    giant[gid >> 5] = static_cast<uint>(vote >> (lane & ~ushort(31)));
}

// Compress: roots[v] = final root (0 for the giant set; otherwise the root
// of roots[v]'s tree, which holds v). labels and giant are read-only in
// this dispatch; each thread reads and writes only its own roots slot.
kernel void mg_wcc_compress(
    constant MGWccParams& p     [[buffer(0)]],
    device const uint* labels   [[buffer(1)]],
    device const uint* giant    [[buffer(2)]],
    device uint* roots          [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  if (mg_wcc_in_giant(giant, gid)) {
    roots[gid] = 0u;
    return;
  }
  const uint l0 = roots[gid];
  uint l = l0;
  uint nx = labels[l];
  while (nx < l) {
    l = nx;
    nx = labels[l];
  }
  if (l != l0) roots[gid] = l;
}
