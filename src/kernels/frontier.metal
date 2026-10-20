// frontier.metal — GPU-resident BFS machinery (level protocol of
// mg_params.h); k-hop reuses it for vertex discovery.
// Binding tables and level protocol: src/kernels/mg_params.h (authoritative).
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

// Every BFS kernel binds the same 13 buffers in this exact order so the
// host encoders stay uniform (mg_params.h BFS table). Kernels always read
// the frontier from b4 and append to b5; the host swaps those bindings
// each level.
#define MG_BFS_BUFFERS                                \
  constant MGBfsParams& prm      [[buffer(0)]],       \
  device MGBfsControlDev& ctl    [[buffer(1)]],       \
  device const uint* row_off     [[buffer(2)]],       \
  device const uint* col_idx     [[buffer(3)]],       \
  device const uint* frontier    [[buffer(4)]],       \
  device uint* next              [[buffer(5)]],       \
  device mg_i32* dist            [[buffer(6)]],       \
  device mg_i32* parent          [[buffer(7)]],       \
  device atomic_uint* visited    [[buffer(8)]],       \
  device uint* bins              [[buffer(9)]],       \
  device uint* args              [[buffer(10)]],      \
  device const uint* rev_row_off [[buffer(11)]],      \
  device const uint* rev_col_idx [[buffer(12)]]

// ---------------------------------------------------------------------------
// Level bookkeeping
// ---------------------------------------------------------------------------

// One threadgroup; thread 0 does the bookkeeping. ALWAYS increments level
// (first prepare makes level 1: sources have dist 0).
kernel void mg_bfs_prepare_level(
    MG_BFS_BUFFERS,
    uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0u) return;
  const uint fc = atomic_load_explicit(&ctl.next_count, memory_order_relaxed);
  ctl.frontier_count = fc;
  atomic_store_explicit(&ctl.next_count, 0u, memory_order_relaxed);
  if (fc == 0u) ctl.done = 1u;
  ctl.level += 1u;
  atomic_store_explicit(&ctl.max_high_deg, 0u, memory_order_relaxed);
  atomic_store_explicit(&ctl.bin_count_high, 0u, memory_order_relaxed);
  atomic_store_explicit(&ctl.bin_count_mid, 0u, memory_order_relaxed);
  atomic_store_explicit(&ctl.bin_count_low, 0u, memory_order_relaxed);
  atomic_store_explicit(&ctl.frontier_edges, 0u, memory_order_relaxed);
  const uint groups =
      (ctl.done == 1u) ? 0u : (fc + prm.tg_size - 1u) / prm.tg_size;
  args[MG_BFS_ARG_BIN * 3u + 0u] = groups;
  args[MG_BFS_ARG_BIN * 3u + 1u] = 1u;
  args[MG_BFS_ARG_BIN * 3u + 2u] = 1u;
}

// Indirect ARG_BIN. Thread i < frontier_count bins frontier[i] by degree
// and accumulates frontier_edges (and max_high_deg over the high bin).
// Must no-op entirely when done == 1.
// Degree-0 frontier vertices have nothing to expand and are not binned.
kernel void mg_bfs_frontier_bin(
    MG_BFS_BUFFERS,
    uint gid [[thread_position_in_grid]]) {
  if (ctl.done == 1u) return;
  if (gid >= ctl.frontier_count) return;
  const uint v = frontier[gid];
  const uint d = row_off[v + 1u] - row_off[v];
  if (d >= MG_BIN_HIGH_MIN) {
    const uint i =
        atomic_fetch_add_explicit(&ctl.bin_count_high, 1u, memory_order_relaxed);
    bins[i] = v;
    atomic_fetch_max_explicit(&ctl.max_high_deg, d, memory_order_relaxed);
  } else if (d >= MG_BIN_MID_MIN) {
    const uint i =
        atomic_fetch_add_explicit(&ctl.bin_count_mid, 1u, memory_order_relaxed);
    bins[prm.v_count + i] = v;
  } else if (d > 0u) {
    const uint i =
        atomic_fetch_add_explicit(&ctl.bin_count_low, 1u, memory_order_relaxed);
    bins[2u * prm.v_count + i] = v;
  }
  if (d > 0u)
    atomic_fetch_add_explicit(&ctl.frontier_edges, d, memory_order_relaxed);
}

// One threadgroup; thread 0 picks the traversal mode and writes the
// expand/bottom-up indirect args ({tgX, 1, 1}; tgX = 0 for unused paths
// and for everything when done). The top-down high-bin slot is 2D:
// {bin_count_high, high_split, 1} (see mg_bfs_expand_tg).
kernel void mg_bfs_decide(
    MG_BFS_BUFFERS,
    uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0u) return;
  if (ctl.done == 1u) {
    ctl.mode = MG_BFS_MODE_TOPDOWN;
    for (uint s = MG_BFS_ARG_EXPAND_TG; s <= MG_BFS_ARG_BOTTOMUP; ++s) {
      args[s * 3u + 0u] = 0u;
      args[s * 3u + 1u] = 1u;
      args[s * 3u + 2u] = 1u;
    }
    return;
  }
  ctl.unvisited -= ctl.frontier_count;
  const uint fe =
      atomic_load_explicit(&ctl.frontier_edges, memory_order_relaxed);
  // frontier_edges * alpha_cut > e_count, overflow-free:
  // fe * ac > ec  <=>  fe > ec / ac  (integer division, ac > 0).
  uint mode = MG_BFS_MODE_TOPDOWN;
  // Edge pressure alone is insufficient on highly skewed graphs: a single
  // hub can own a large fraction of E while reaching only a tiny fraction of
  // V. Bottom-up would then scan nearly every unvisited vertex to expand one
  // sparse frontier. Require both Beamer-style edge pressure and a frontier
  // dense enough to cover at least ~V / beta_cut vertices.
  if (prm.enable_bottomup != 0u && prm.alpha_cut != 0u &&
      prm.beta_cut != 0u && fe > prm.e_count / prm.alpha_cut &&
      ctl.frontier_count > prm.v_count / prm.beta_cut)
    mode = MG_BFS_MODE_BOTTOMUP;
  ctl.mode = mode;
  const uint bh =
      atomic_load_explicit(&ctl.bin_count_high, memory_order_relaxed);
  const uint bm =
      atomic_load_explicit(&ctl.bin_count_mid, memory_order_relaxed);
  const uint bl =
      atomic_load_explicit(&ctl.bin_count_low, memory_order_relaxed);
  const bool td = (mode == MG_BFS_MODE_TOPDOWN);
  // High-bin split: each high vertex's edge range is cut into K slices,
  // one threadgroup each, so a lone mega-hub frontier vertex no longer
  // serializes on one threadgroup. K = min(ceil(max_high_deg / 1024),
  // max(1, MG_BFS_HIGH_SPLIT_TARGET_TGS / bin_count_high), high_split_max):
  // slices of >= ~1024 edges, and wide high bins (which already fill the
  // GPU) stay at K = 1.
  const uint md =
      atomic_load_explicit(&ctl.max_high_deg, memory_order_relaxed);
  uint ks = min((md + MG_BIN_HIGH_MIN - 1u) / MG_BIN_HIGH_MIN,
                max(1u, MG_BFS_HIGH_SPLIT_TARGET_TGS / max(1u, bh)));
  ks = max(1u, min(ks, max(1u, prm.high_split_max)));
  ctl.high_split = ks;
  args[MG_BFS_ARG_EXPAND_TG * 3u + 0u] = td ? bh : 0u;
  args[MG_BFS_ARG_EXPAND_TG * 3u + 1u] = td ? ks : 1u;
  args[MG_BFS_ARG_EXPAND_TG * 3u + 2u] = 1u;
  args[MG_BFS_ARG_EXPAND_SG * 3u + 0u] =
      td ? (bm + prm.sgs_per_tg - 1u) / prm.sgs_per_tg : 0u;
  args[MG_BFS_ARG_EXPAND_SG * 3u + 1u] = 1u;
  args[MG_BFS_ARG_EXPAND_SG * 3u + 2u] = 1u;
  args[MG_BFS_ARG_EXPAND_TH * 3u + 0u] =
      td ? (bl + prm.tg_size - 1u) / prm.tg_size : 0u;
  args[MG_BFS_ARG_EXPAND_TH * 3u + 1u] = 1u;
  args[MG_BFS_ARG_EXPAND_TH * 3u + 2u] = 1u;
  args[MG_BFS_ARG_BOTTOMUP * 3u + 0u] =
      td ? 0u : (prm.v_count + prm.tg_size - 1u) / prm.tg_size;
  args[MG_BFS_ARG_BOTTOMUP * 3u + 1u] = 1u;
  args[MG_BFS_ARG_BOTTOMUP * 3u + 2u] = 1u;
}

// ---------------------------------------------------------------------------
// Top-down expansion (claim via visited bitmap, append via simd compaction)
// ---------------------------------------------------------------------------

// High bin: 2D grid {bin_count_high, high_split}. Threadgroup (x, y)
// expands slice y of vertex bins[x]'s edge range (slices of
// ceil(deg / high_split) edges rounded up to a tg_size multiple; trailing
// slices of lower-degree vertices are empty); edge loop strided by
// threads_per_threadgroup; each simdgroup compacts its own discoveries.
kernel void mg_bfs_expand_tg(
    MG_BFS_BUFFERS,
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint2 tg_dim [[threads_per_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
  const uint tg_size = tg_dim.x;
  const uint n =
      atomic_load_explicit(&ctl.bin_count_high, memory_order_relaxed);
  if (tg_pos.x >= n) return;  // threadgroup-uniform
  const uint v = bins[tg_pos.x];
  const mg_i32 lvl = (mg_i32)ctl.level;
  const uint row_start = row_off[v];
  const uint deg = row_off[v + 1u] - row_start;
  const uint ks = max(1u, ctl.high_split);
  const uint per = ((deg + ks - 1u) / ks + tg_size - 1u) / tg_size * tg_size;
  const uint off = tg_pos.y * per;
  if (off >= deg) return;  // threadgroup-uniform (empty slice)
  const uint start = row_start + off;
  const uint end = row_start + min(deg, off + per);
  for (uint base_e = start; base_e < end; base_e += tg_size) {
    const uint e = base_e + tid;
    bool discovered = false;
    uint nbr = 0u;
    if (e < end) {
      nbr = col_idx[e];
      discovered = mg_visited_claim(visited, nbr);
    }
    const uint rank = simd_prefix_exclusive_sum((uint)discovered);
    const uint cnt = simd_sum((uint)discovered);
    uint base = 0u;
    if (lane == 0u && cnt != 0u)
      base = atomic_fetch_add_explicit(&ctl.next_count, cnt,
                                       memory_order_relaxed);
    base = simd_broadcast_first(base);
    if (discovered) {
      next[base + rank] = nbr;
      dist[nbr] = lvl;
      parent[nbr] = (mg_i32)v;
    }
  }
}

// Mid bin: one simdgroup per vertex.
kernel void mg_bfs_expand_sg(
    MG_BFS_BUFFERS,
    uint tg_id        [[threadgroup_position_in_grid]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  const uint wi = tg_id * sgs_per_tg + sg_in_tg;  // global simdgroup index
  const uint n =
      atomic_load_explicit(&ctl.bin_count_mid, memory_order_relaxed);
  if (wi >= n) return;  // simdgroup-uniform
  const uint v = bins[prm.v_count + wi];
  const mg_i32 lvl = (mg_i32)ctl.level;
  const uint start = row_off[v], end = row_off[v + 1u];
  for (uint base_e = start; base_e < end; base_e += simd_sz) {
    const uint e = base_e + lane;
    bool discovered = false;
    uint nbr = 0u;
    if (e < end) {
      nbr = col_idx[e];
      discovered = mg_visited_claim(visited, nbr);
    }
    const uint rank = simd_prefix_exclusive_sum((uint)discovered);
    const uint cnt = simd_sum((uint)discovered);
    uint base = 0u;
    if (lane == 0u && cnt != 0u)
      base = atomic_fetch_add_explicit(&ctl.next_count, cnt,
                                       memory_order_relaxed);
    base = simd_broadcast_first(base);
    if (discovered) {
      next[base + rank] = nbr;
      dist[nbr] = lvl;
      parent[nbr] = (mg_i32)v;
    }
  }
}

// Low bin: one thread per vertex; one atomic append per discovery.
kernel void mg_bfs_expand_thread(
    MG_BFS_BUFFERS,
    uint gid [[thread_position_in_grid]]) {
  const uint n =
      atomic_load_explicit(&ctl.bin_count_low, memory_order_relaxed);
  if (gid >= n) return;
  const uint v = bins[2u * prm.v_count + gid];
  const mg_i32 lvl = (mg_i32)ctl.level;
  const uint end = row_off[v + 1u];
  for (uint e = row_off[v]; e < end; ++e) {
    const uint nbr = col_idx[e];
    if (mg_visited_claim(visited, nbr)) {
      dist[nbr] = lvl;
      parent[nbr] = (mg_i32)v;
      const uint slot = atomic_fetch_add_explicit(&ctl.next_count, 1u,
                                                  memory_order_relaxed);
      next[slot] = nbr;
    }
  }
}

// ---------------------------------------------------------------------------
// Bottom-up probe
// ---------------------------------------------------------------------------

// Indirect ARG_BOTTOMUP (covers all V). Thread per vertex v: unvisited v
// scans REVERSE-orientation neighbors for dist == level-1; on first hit
// claims v without atomics on dist/parent (this thread owns v), sets the
// visited bit atomically, appends v to next.
kernel void mg_bfs_bottomup_probe(
    MG_BFS_BUFFERS,
    uint gid [[thread_position_in_grid]]) {
  const uint v = gid;
  if (v >= prm.v_count) return;
  if (dist[v] != -1) return;
  const mg_i32 lvl = (mg_i32)ctl.level;
  const uint end = rev_row_off[v + 1u];
  for (uint e = rev_row_off[v]; e < end; ++e) {
    const uint u = rev_col_idx[e];
    if (dist[u] == lvl - 1) {
      parent[v] = (mg_i32)u;
      dist[v] = lvl;
      atomic_fetch_or_explicit(&visited[v >> 5u], 1u << (v & 31u),
                               memory_order_relaxed);
      const uint slot = atomic_fetch_add_explicit(&ctl.next_count, 1u,
                                                  memory_order_relaxed);
      next[slot] = v;
      return;
    }
  }
}
