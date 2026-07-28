// mg_params.h — THE kernel <-> host contract for metal-graph.
//
// This header is included by BOTH Metal Shading Language kernels (.metal) and
// C++ host code. It is the single source of truth for:
//   * parameter structs passed via `constant ... &` bindings
//   * buffer binding indices for every kernel (documented tables below)
//   * threadgroup sizes, degree-bin thresholds, and the PPR batch tile width
//   * function-constant indices
//   * the canonical list of kernel entry-point names
//
// Rules:
//   * Fixed-width types only. Structs must have identical layout in MSL and
//     C++ (no pointers, no bool, 4-byte scalars only; keep fields 4-byte
//     aligned and sizes a multiple of 4).
//   * Changing anything here requires updating both the kernels and the host
//     encoders. Do not change silently.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MG_PARAMS_H
#define MG_PARAMS_H

#ifdef __METAL_VERSION__
#define MG_NS metal
typedef metal::uint32_t mg_u32;
typedef metal::int32_t mg_i32;
#else
#include <cstdint>
typedef uint32_t mg_u32;
typedef int32_t mg_i32;
#endif

// ---------------------------------------------------------------------------
// Global tuning constants (shared by host and device)
// ---------------------------------------------------------------------------

// Threadgroup size for all 1-thread-per-item kernels and for the tg-variant
// (one threadgroup per high-degree vertex) gather/expand kernels.
#define MG_TG_SIZE 256u

// Degree-bin thresholds (Part 1.3 of the assessment; retuned in M4).
//   huge: degree >= MG_BIN_HUGE_MIN      -> fixed edge TILES across many
//                                           threadgroups (see below)
//   high: MG_BIN_HIGH_MIN <= d < HUGE    -> one threadgroup per vertex
//   mid:  MG_BIN_MID_MIN <= d < HIGH_MIN -> one simdgroup per vertex
//   low:  1 <= d < MG_BIN_MID_MIN        -> one thread per vertex
//   zero: d == 0                          -> zero-fill list (per orientation)
#define MG_BIN_HIGH_MIN 1024u
#define MG_BIN_MID_MIN 32u

// Huge bin (plan §11 CSR-tiling fallback for power-law skew): a mega-hub's
// edge list must not serialize on one threadgroup — a KG-shape graph puts
// 44% of all in-edges on ONE vertex. Vertices with degree >=
// MG_BIN_HUGE_MIN leave the high bin; their edge ranges are decomposed AT
// BUILD TIME into fixed tiles of MG_HUGE_TILE_EDGES edges. PageRank gathers
// them in a deterministic two-pass (per-tile threadgroup partials, then a
// per-vertex finalize summing tiles in fixed order); WCC hooks tiles
// directly (no reduction needed). Fixed decomposition + ordered finalize
// preserve the determinism guarantee at fixed iteration counts.
#define MG_BIN_HUGE_MIN 16384u
#define MG_HUGE_TILE_EDGES 16384u

// Batched PPR tile width: the batch gather streams col_indices once for
// MG_PPR_TILE query lanes. Rank/contrib state is interleaved vertex-major:
//   state[v * MG_PPR_TILE + lane]
// Batches larger than MG_PPR_TILE run as ceil(B / MG_PPR_TILE) tiles.
#define MG_PPR_TILE 8u

// Scan block: one threadgroup of MG_TG_SIZE threads scans MG_SCAN_BLOCK
// elements (1 element per thread).
#define MG_SCAN_BLOCK 256u

// iter_scalars slot indices (single-query PageRank; float[MG_SCALARS_LEN])
#define MG_SCALAR_DANGLING 0u
#define MG_SCALARS_LEN 4u
// Batched variant uses float[MG_PPR_TILE]: dangling mass per query lane.

// BFS traversal mode written by mg_bfs_decide into MGBfsControl.mode
#define MG_BFS_MODE_TOPDOWN 0u
#define MG_BFS_MODE_BOTTOMUP 1u

// ---------------------------------------------------------------------------
// Function-constant indices (set at pipeline-creation time)
// ---------------------------------------------------------------------------
// fc0: MG_FC_WEIGHTED  (bool) — gather kernels multiply contrib by weights[e].
//      When false the weights buffer is never dereferenced, but the host
//      still binds a 4-byte dummy buffer at that slot.
// fc1: MG_FC_UNIFORM_P (bool) — single-query PageRank kernels use
//      MGPrParams.p_uniform instead of reading the pvec buffer. When true
//      the pvec slot gets the dummy buffer.
#define MG_FC_WEIGHTED 0
#define MG_FC_UNIFORM_P 1

// ---------------------------------------------------------------------------
// Parameter structs
// ---------------------------------------------------------------------------

// --- PageRank (single query) ---
// Used by: mg_pr_prepare, mg_pr_gather_{tg,sg,thread}, mg_pr_zero_fill,
//          mg_pr_dangling_partials, mg_pr_residual_partials
typedef struct MGPrParams {
  float alpha;            // damping
  float one_minus_alpha;  // 1 - alpha
  float p_uniform;        // 1/V when MG_FC_UNIFORM_P, else unused
  mg_u32 v_count;         // V (vertices)
  mg_u32 count;           // #items for this dispatch (worklist len / list len)
  mg_u32 pad0;
  mg_u32 pad1;
  mg_u32 pad2;
} MGPrParams;

// --- Batched PPR (tile of MG_PPR_TILE query lanes) ---
// Used by: mg_pr_prepare_b, mg_pr_gather_{tg,sg,thread}_b, mg_pr_seed_scatter_b,
//          mg_pr_dangling_partials_b, mg_pr_residual_partials_b
// active_mask: bit t set => query lane t still iterating. Kernels must leave
// rank_next of inactive lanes equal to rank_cur (copy-through), so frozen
// lanes survive the ping-pong.
typedef struct MGPprParams {
  float alpha;
  float one_minus_alpha;
  mg_u32 v_count;
  mg_u32 count;        // #items for this dispatch
  mg_u32 active_mask;  // bits [0, MG_PPR_TILE)
  mg_u32 seed_total;   // total packed seeds in this tile (for seed_scatter)
  mg_u32 pad0;
  mg_u32 pad1;
} MGPprParams;

// --- Generic reduce finalize ---
// mg_reduce_finalize: sums `n_partials` floats from partials buffer and
// writes the total into dst[dst_index]. Runs as ONE threadgroup.
// mg_reduce_finalize_b: same but per query lane; partials are laid out
// partials[block * MG_PPR_TILE + lane], writes dst[lane] for each lane.
typedef struct MGReduceParams {
  mg_u32 n_partials;  // number of partial values (per lane for _b)
  mg_u32 dst_index;   // slot in dst (ignored by _b)
  mg_u32 pad0;
  mg_u32 pad1;
} MGReduceParams;

// --- Fill / iota ---
typedef struct MGFillParams {
  float f_value;
  mg_u32 u_value;
  mg_u32 count;
  mg_u32 pad0;
} MGFillParams;

// --- BFS ---
// One MGBfsControl lives in a device buffer; GPU kernels read & update it
// across an encoded batch of levels. Host reads it back after the batch.
typedef struct MGBfsControl {
  mg_u32 frontier_count;  // size of current frontier list
  mg_u32 next_count;      // size of next frontier list (atomic on device)
  mg_u32 level;           // level being expanded (dist assigned = level)
  mg_u32 done;            // 1 => frontier empty; all later levels no-op
  mg_u32 mode;            // MG_BFS_MODE_* for the current level
  mg_u32 bin_count_high;  // written by mg_bfs_frontier_bin (atomics)
  mg_u32 bin_count_mid;
  mg_u32 bin_count_low;
  mg_u32 frontier_edges;  // sum of degrees over current frontier (atomic)
  mg_u32 unvisited;       // V - #visited so far (maintained by decide)
  mg_u32 pad0;
  mg_u32 pad1;
} MGBfsControl;

typedef struct MGBfsParams {
  mg_u32 v_count;         // V
  mg_u32 e_count;         // edge count of the traversal orientation
  mg_u32 tg_size;         // MG_TG_SIZE (so decide can compute groups)
  mg_u32 sgs_per_tg;      // MG_TG_SIZE / threadExecutionWidth (host-computed)
  mg_u32 alpha_cut;       // bottom-up when frontier_edges * alpha_cut > e_unvisited... see decide
  mg_u32 beta_cut;        // back to top-down when frontier_count * beta_cut < v_count
  mg_u32 enable_bottomup; // 0 disables bottom-up entirely
  mg_u32 pad0;
} MGBfsParams;

// Indirect dispatch argument slots (MTLDispatchThreadgroupsIndirectArguments
// = 3 x uint32). The BFS args buffer holds MG_BFS_ARGS_SLOTS slots:
#define MG_BFS_ARG_BIN 0u       // mg_bfs_frontier_bin dispatch
#define MG_BFS_ARG_EXPAND_TG 1u // mg_bfs_expand_tg (one tg per high vertex)
#define MG_BFS_ARG_EXPAND_SG 2u // mg_bfs_expand_sg
#define MG_BFS_ARG_EXPAND_TH 3u // mg_bfs_expand_thread
#define MG_BFS_ARG_BOTTOMUP 4u  // mg_bfs_bottomup_probe (covers all V)
#define MG_BFS_ARGS_SLOTS 5u
// Byte offset of slot i = i * 3 * sizeof(uint32).

// --- k-hop induced-edge extraction ---
typedef struct MGKhopParams {
  mg_u32 n_vertices;  // #reached vertices (list length)
  mg_u32 pad0;
  mg_u32 pad1;
  mg_u32 pad2;
} MGKhopParams;

// --- WCC ---
typedef struct MGWccParams {
  mg_u32 v_count;
  mg_u32 count;  // worklist length for this dispatch
  mg_u32 pad0;
  mg_u32 pad1;
} MGWccParams;

// --- GPU top-k select (batched PPR tiles) ---
// cap = k + MG_TOPK_TIE_SLACK candidate slots per lane; a lane whose
// >=-threshold class exceeds cap is finished on the CPU oracle instead.
#define MG_TOPK_TIE_SLACK 2048u
typedef struct MGTopkParams {
  mg_u32 v_count;
  mg_u32 k;
  mg_u32 cap;
  mg_u32 active_mask;  // lanes carrying queries in this tile
} MGTopkParams;

// --- Scan ---
// Two-pass exclusive scan over uint32:
//   mg_exscan_partial: per-block exclusive scan in place is NOT done; it
//     writes block-scanned values to `out` and per-block totals to `sums`.
//   mg_exscan_apply: adds scanned `sums` (block offsets) to each element.
typedef struct MGScanParams {
  mg_u32 count;
  mg_u32 pad0;
  mg_u32 pad1;
  mg_u32 pad2;
} MGScanParams;

// ---------------------------------------------------------------------------
// KERNEL BINDING TABLES (authoritative)
// ---------------------------------------------------------------------------
// Convention: buffer(0) is always the params struct (constant address space).
// "atomic" means the kernel must access that buffer through atomic_uint /
// atomic ops. All lists/worklists are uint32 canonical vertex ids.
//
// ======================= pagerank single (spmv.metal, reduce.metal) ========
// mg_pr_prepare  — grid: ceil(V / MG_TG_SIZE) tgs, 1 thread/vertex
//   b0 MGPrParams (count = V)
//   b1 rank_cur        (const float, V)
//   b2 out_weight_sum  (const float, V)
//   b3 contrib         (float, V)   contrib[v] = ows[v] > 0 ? rank[v]/ows[v] : 0
//
// mg_pr_gather_tg — grid: n_high tgs of MG_TG_SIZE  [fc: WEIGHTED, UNIFORM_P]
// mg_pr_gather_sg — grid: ceil(n_mid / sgs_per_tg) tgs [same fcs]
// mg_pr_gather_thread — grid: ceil(n_low / MG_TG_SIZE) tgs [same fcs]
//   b0 MGPrParams (count = worklist length)
//   b1 worklist        (const uint32)   in-orientation degree bin
//   b2 row_offsets     (const uint32, V+1)   IN-orientation CSR
//   b3 col_indices     (const uint32, E)
//   b4 contrib         (const float, V)
//   b5 rank_next       (float, V)
//   b6 iter_scalars    (const float, MG_SCALARS_LEN) [MG_SCALAR_DANGLING]
//   b7 pvec            (const float, V; dummy when UNIFORM_P)
//   b8 weights         (const float, E; dummy when !WEIGHTED)
//   writes: rank_next[v] = (1-a)*p[v] + a*(acc + D*p[v])
//
// mg_pr_zero_fill — grid: ceil(n_zero_in / MG_TG_SIZE) [fc: UNIFORM_P]
//   b0 MGPrParams (count = zero-in-degree list length)
//   b1 zero_list       (const uint32)   zero IN-degree vertices
//   b2 rank_next       (float, V)
//   b3 iter_scalars    (const float)
//   b4 pvec            (const float, V; dummy when UNIFORM_P)
//   writes the same formula with acc = 0.
//
// mg_pr_dangling_partials — grid: ceil(n_dangling / MG_TG_SIZE)
//   b0 MGPrParams (count = dangling list length)
//   b1 dangling_list   (const uint32)   zero OUT-weight vertices
//   b2 rank            (const float, V) rank_next of the iteration just done
//   b3 partials        (float, #tgs)    per-threadgroup sums
//
// mg_pr_residual_partials — grid: ceil(V / MG_TG_SIZE)
//   b0 MGPrParams (count = V)
//   b1 rank_a (const float, V)
//   b2 rank_b (const float, V)
//   b3 partials (float, #tgs)           per-tg sums of |a - b|
//
// mg_reduce_finalize — grid: exactly 1 tg of MG_TG_SIZE
//   b0 MGReduceParams
//   b1 partials (const float)
//   b2 dst      (float)                 dst[dst_index] = sum(partials[0..n))
//
// ======================= batched PPR (spmv.metal, reduce.metal) ============
// State layout: rank/contrib are float[V * MG_PPR_TILE], vertex-major
// interleaved. iter_scalars_b is float[MG_PPR_TILE].
//
// mg_pr_prepare_b — grid: ceil(V / MG_TG_SIZE), thread/vertex
//   b0 MGPprParams (count = V)
//   b1 rank_cur_b   (const float, V*TILE)
//   b2 out_weight_sum (const float, V)
//   b3 contrib_b    (float, V*TILE)
//
// mg_pr_gather_tg_b / mg_pr_gather_sg_b / mg_pr_gather_thread_b
//   [fc: WEIGHTED] — same gridding as single variants
//   b0 MGPprParams (count = worklist length)
//   b1 worklist     (const uint32)      IN-orientation bins
//   b2 row_offsets  (const uint32)
//   b3 col_indices  (const uint32)
//   b4 contrib_b    (const float, V*TILE)
//   b5 rank_next_b  (float, V*TILE)
//   b6 rank_cur_b   (const float, V*TILE)  // copy-through for inactive lanes
//   b7 weights      (const float; dummy when !WEIGHTED)
//   writes per active lane t: rank_next_b[v*TILE+t] = alpha * acc_t
//   (teleport & dangling added by seed_scatter; inactive lanes copy rank_cur)
//
// Batched write contract (final): gather kernels ASSIGN alpha*acc for
// worklist vertices on active lanes and COPY rank_cur on inactive lanes;
// mg_pr_zero_fill_b covers zero-in-degree vertices (assign 0 on active
// lanes, copy on inactive); mg_pr_seed_scatter_b then ADDS teleport +
// dangling for active lanes only. Together every (v, lane) slot is written
// exactly once per iteration before seed_scatter adds on top.
//
// mg_pr_zero_fill_b — grid: ceil(n_zero_in / MG_TG_SIZE)
//   b0 MGPprParams (count = zero-in list length)
//   b1 zero_list    (const uint32)
//   b2 rank_next_b  (float, V*TILE)
//   b3 rank_cur_b   (const float, V*TILE)
//   active lanes: rank_next = 0; inactive: rank_next = rank_cur.
//
// mg_pr_seed_scatter_b — grid: ceil(seed_total / MG_TG_SIZE), thread/seed
//   b0 MGPprParams (count = seed_total)
//   b1 seed_vertex  (const uint32, seed_total)  canonical vertex per seed
//   b2 seed_lane    (const uint32, seed_total)  query lane in [0, TILE)
//   b3 seed_weight  (const float,  seed_total)  normalized within each query
//   b4 iter_scalars_b (const float, TILE)       dangling mass per lane
//   b5 rank_next_b  (float, V*TILE)
//   active lane t: rank_next_b[v*TILE+t] += (1-a)*w + a*D_t*w
//   (seeds pre-deduplicated per (v,lane) on host => no write conflicts)
//   inactive lanes: skip.
//
// mg_pr_dangling_partials_b — grid: ceil(n_dangling / MG_TG_SIZE)
//   b0 MGPprParams (count = dangling list length)
//   b1 dangling_list (const uint32)
//   b2 rank_b        (const float, V*TILE)
//   b3 partials      (float, #tgs * TILE)  partials[tg*TILE + lane]
//
// mg_reduce_finalize_b — grid: exactly 1 tg
//   b0 MGReduceParams (n_partials = #tgs of the partials pass)
//   b1 partials (const float, n_partials * TILE)
//   b2 dst      (float, TILE)            dst[lane] = sum over blocks
//
// mg_pr_residual_partials_b — grid: ceil(V / MG_TG_SIZE)
//   b0 MGPprParams (count = V)
//   b1 rank_a_b (const float, V*TILE)
//   b2 rank_b_b (const float, V*TILE)
//   b3 partials (float, #tgs * TILE)     per-lane |a-b| block sums
//
// ======================= huge-bin tiling (spmv.metal, wcc.metal) ===========
// Tile arrays live on the Orientation (built once, deterministic):
//   wl_huge         uint32 n_huge      huge vertices, ascending canonical
//   tile_owner      uint32 n_tiles     index INTO wl_huge per tile
//   tile_first_edge uint32 n_tiles     absolute CSR edge index of tile start
//   huge_tile_off   uint32 n_huge+1    contiguous tile range per huge vertex
// Tile t covers edges [tile_first_edge[t],
//                      min(row_off[v+1], tile_first_edge[t] +
//                          MG_HUGE_TILE_EDGES)) of v = wl_huge[tile_owner[t]].
//
// mg_pr_gather_huge_partials — grid: n_tiles tgs of MG_TG_SIZE [fc: WEIGHTED]
//   b0 MGPrParams (count = n_tiles)
//   b1 tile_owner; b2 tile_first_edge; b3 wl_huge; b4 row_offsets;
//   b5 col_indices; b6 contrib (const float, V);
//   b7 weights (dummy when !WEIGHTED); b8 partials (float, n_tiles)
//   partials[t] = tile's (weighted) contrib sum, tg-reduced.
//
// mg_pr_gather_huge_finalize — grid: ceil(n_huge/MG_TG_SIZE) [fc: UNIFORM_P]
//   b0 MGPrParams (count = n_huge)
//   b1 wl_huge; b2 huge_tile_off; b3 partials (const float);
//   b4 rank_next; b5 iter_scalars; b6 pvec (dummy when UNIFORM_P)
//   thread per huge vertex: acc = sum of its tile partials IN TILE ORDER,
//   then the standard mg_pr_write formula.
//
// mg_pr_gather_huge_partials_b — same grid as _partials [fc: WEIGHTED]
//   b0 MGPprParams (count = n_tiles)
//   b1 tile_owner; b2 tile_first_edge; b3 wl_huge; b4 row_offsets;
//   b5 col_indices; b6 contrib_b (V*TILE); b7 weights (dummy);
//   b8 partials_b (float, n_tiles*TILE)  partials_b[t*TILE + lane]
//
// mg_pr_gather_huge_finalize_b — grid: ceil(n_huge/MG_TG_SIZE)
//   b0 MGPprParams (count = n_huge)
//   b1 wl_huge; b2 huge_tile_off; b3 partials_b; b4 rank_next_b;
//   b5 rank_cur_b
//   batched write contract: active lanes ASSIGN alpha*acc, inactive COPY.
//
// mg_wcc_hook_huge — grid: n_tiles tgs of MG_TG_SIZE
//   b0 MGWccParams (count = n_tiles)
//   b1 tile_owner; b2 tile_first_edge; b3 wl_huge; b4 row_offsets;
//   b5 col_indices; b6 labels (atomic_uint, V); b7 changed (atomic_uint, 1)
//   per-edge hook over the tile; conditional changed store per thread.
//
// ======================= utilities (reduce.metal) ==========================
// mg_fill_f32 — b0 MGFillParams; b1 dst (float)
// mg_fill_u32 — b0 MGFillParams; b1 dst (uint32)
// mg_iota_u32 — b0 MGFillParams; b1 dst (uint32)   dst[i] = i
//
// ======================= BFS (frontier.metal) ===============================
// Shared buffers across the level loop (bound to every BFS kernel in this
// exact order so encoders are uniform):
//   b0 params   : MGBfsParams (constant)
//   b1 control  : MGBfsControl (device, atomic fields where noted)
//   b2 row_off  : traversal-orientation CSR row_offsets (const uint32, V+1)
//   b3 col_idx  : CSR col_indices (const uint32, E)
//   b4 frontier : current frontier list (uint32, cap V)
//   b5 next     : next frontier list (uint32, cap V)
//   b6 dist     : int32, V   (-1 = unvisited)
//   b7 parent   : int32, V   (-1 = none)
//   b8 visited  : atomic_uint bitmap, ceil(V/32)
//   b9 bins     : uint32, 3*V — high at [0,V), mid at [V,2V), low at [2V,3V)
//   b10 args    : indirect args, MG_BFS_ARGS_SLOTS * 3 uint32
//   b11 rev_row_off : reverse-orientation CSR row_offsets (bottom-up; dummy
//                     when enable_bottomup == 0)
//   b12 rev_col_idx : reverse-orientation col_indices (bottom-up; dummy)
//
// Level protocol. control.level == the depth ASSIGNED to vertices
// discovered this round; the current frontier has dist == level - 1.
// Host initialization: dist/parent filled with -1; dist[source] = 0;
// visited bits set for sources; next list = sources; next_count =
// n_sources; frontier_count = 0; level = 0; done = 0; unvisited = V
// (decide subtracts every frontier including the sources, so seeding
// V - n_sources would double-subtract them). mg_bfs_prepare_level ALWAYS
// increments level (the first prepare makes level 1, which is correct:
// sources have dist 0).
//
// Level sequence (encoded per level; encoder breaks marked |):
//   mg_bfs_prepare_level  (1 tg): frontier_count = next_count;
//       next_count = 0; if (frontier_count == 0) done = 1; level += 1;
//       zero bin counts & frontier_edges; write args[BIN] =
//       ceil(frontier_count / tg_size). (Host swaps the frontier/next
//       buffer BINDINGS each level — kernels always read b4, append to b5.)
//   | mg_bfs_frontier_bin (indirect ARG_BIN): thread i < frontier_count reads
//       v = frontier[i], d = degree(v), appends v to the right bin (atomic
//       bin counters in control), atomically adds d to frontier_edges.
//       Must no-op entirely when control.done == 1.
//   mg_bfs_decide (1 tg): unvisited -= frontier_count; mode = BOTTOMUP iff
//       enable_bottomup && frontier_edges * alpha_cut > e_count (alpha_cut
//       seed 14; beta_cut reserved). Write args[EXPAND_*] from bin counts
//       (0 groups for the unused path), args[BOTTOMUP] = ceil(V/tg_size) or
//       0. All args 0 when done == 1.
//   | mg_bfs_expand_tg / _sg / _thread (indirect): top-down expansion of the
//       matching bin; claim neighbors via visited-bitmap atomic_fetch_or;
//       append to next via simd-compaction (sg/tg variants) or per-discovery
//       atomic (thread variant); dist[nbr] = level; parent[nbr] = v.
//   mg_bfs_bottomup_probe (indirect): thread per vertex v: if dist[v] == -1,
//       scan REVERSE-orientation neighbors u looking for dist[u] ==
//       level - 1; on first hit: parent[v] = u; dist[v] = level; set
//       visited bit (thread owns v — no claim race); append v to next.
//
// After K levels host waits, reads control.done / next_count, re-encodes if
// needed. dist/parent are canonical-id indexed; host translates at boundary.
//
// ======================= k-hop (frontier.metal) =============================
// mg_khop_count — grid ceil(n_vertices / MG_TG_SIZE), thread per reached vtx
//   b0 MGKhopParams; b1 reached_list (const uint32); b2 row_off; b3 col_idx;
//   b4 reached_bitmap (const uint32, ceil(V/32));  b5 counts (uint32, n_vertices)
//   counts[i] = # edges (v -> u) with u's bit set, v = reached_list[i].
// mg_khop_emit — same grid
//   b0 MGKhopParams; b1 reached_list; b2 row_off; b3 col_idx; b4 reached_bitmap;
//   b5 offsets (const uint32, exclusive scan of counts); b6 edge_orig (const
//   uint32, E — CSR slot -> original input edge index); b7 out_edges (uint32)
//   deterministic: edges of reached_list[i] written at offsets[i..], in CSR order.
//
// ======================= WCC (wcc.metal) ====================================
// mg_wcc_hook_{tg,sg,thread} — gridding like the gather variants, worklists
// are the OUT orientation bins.
//   b0 MGWccParams (count = worklist len); b1 worklist; b2 row_off; b3 col_idx;
//   b4 labels (atomic_uint, V); b5 changed (atomic_uint, 1)
//   For each edge (v,u): lv = load(labels[v]), lu = load(labels[u]),
//   m = min(lv, lu); atomic_min(labels[u], m); atomic_min(labels[v], m);
//   set *changed = 1 if either min actually lowered a label.
// mg_wcc_jump — grid ceil(V / MG_TG_SIZE)
//   b0 MGWccParams (count = V); b1 labels (atomic_uint, V); b2 changed
//   l = load(labels[v]); gp = load(labels[l]); if gp < l:
//   atomic_min(labels[v], gp); *changed = 1.
//
// ======================= top-k select (select.metal) ========================
// GPU radix-select (plan M3, behind the same API as the CPU oracle): per
// active lane, find the k-th largest fp32 score via MSB-first 4x8-bit
// histogram refinement over raw IEEE bits (PageRank scores are >= 0, so
// unsigned bit order == value order), then compact EVERY candidate with
// bits >= threshold. The host finishes with a tiny exact sort of the
// k..k+ties candidates under the (score desc, user index asc) tie rule; a
// lane whose >=-threshold class overflows cap falls back to the CPU
// selection oracle wholesale. The candidate SET is deterministic (append
// order is not; the host sort restores full output determinism).
//
// mg_topk_threshold_b — grid: exactly MG_PPR_TILE tgs of MG_TG_SIZE
//   b0 MGTopkParams (k <= v_count required)
//   b1 rank_b     (const float, V*TILE)  final tile state
//   b2 thresholds (uint32, TILE)  raw bit pattern of the k-th largest score
// mg_topk_compact_b — grid: ceil(V / MG_TG_SIZE), thread per vertex
//   b0 MGTopkParams
//   b1 rank_b; b2 thresholds (const uint32)
//   b3 counters   (atomic_uint, TILE)  zeroed by host; may exceed cap
//   b4 candidates (uint32, TILE*cap, lane-major: candidates[lane*cap + i])
//
// ======================= scan (scan.metal) ==================================
// mg_exscan_partial — grid ceil(count / MG_SCAN_BLOCK), MG_TG_SIZE threads
//   b0 MGScanParams; b1 in (const uint32); b2 out (uint32); b3 sums (uint32,
//   #blocks)   out = exclusive scan within each block; sums[b] = block total.
// mg_exscan_apply — same grid
//   b0 MGScanParams; b1 out (uint32, in/out); b2 scanned_sums (const uint32)
//   out[i] += scanned_sums[block(i)].
// (Host recurses on `sums` until one block, then applies back up.)
// ---------------------------------------------------------------------------

#endif  // MG_PARAMS_H
