// graph.hpp — immutable graph snapshot: CSR per orientation, worklists,
// dangling/zero lists, canonical<->user permutations.
//
// Identity model (plan §2 "vertex identity", adapted at the Python boundary):
//   * USER index space: 0..V-1, the order the caller sees. All core-API
//     inputs (seeds, sources) and outputs (vectors, id arrays) are in USER
//     index space; external-ID mapping (int64/str -> user index) lives in
//     the Python layer.
//   * CANONICAL space: internal uint32 renumbering by out-degree descending
//     (stable), for gather locality. Kernels and CPU algos run in canonical
//     space; algo entry points translate at the boundary via perm/inv_perm.
//
// Input policy (plan §2): duplicate edges kept; self-loops kept; non-finite
// or negative weights rejected at build; num_vertices >= max_id+1 for isolated
// vertices. Undirected graphs are symmetrized at build (each u!=v edge
// stored both ways; self-loops stored once); E_input remains the caller's
// edge count.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "../runtime/runtime.hpp"

namespace mg {

enum class Dir : int { out = 0, in = 1 };

// One CSR orientation plus its degree-binned worklists.
// Worklist bins follow MG_BIN_* thresholds in src/kernels/mg_params.h and
// are ordered by ascending canonical id within each bin (deterministic).
struct Orientation {
  BufferPtr row_offsets;   // uint32, V+1
  BufferPtr col_indices;   // uint32, Es (stored edges)
  BufferPtr weights;       // float, Es; null when unweighted
  BufferPtr edge_orig;     // uint32, Es: CSR slot -> original input edge id
                           // (only materialized for the OUT orientation)
  BufferPtr wl_high;       // uint32 worklists (may be empty buffers)
  BufferPtr wl_mid;
  BufferPtr wl_low;
  BufferPtr zero_list;     // vertices with zero degree in THIS orientation
  // Huge bin (degree >= MG_BIN_HUGE_MIN, removed from wl_high): fixed
  // MG_HUGE_TILE_EDGES-edge tile decomposition, built deterministically —
  // see the tiling section of src/kernels/mg_params.h.
  BufferPtr wl_huge;          // uint32, n_huge (ascending canonical)
  BufferPtr tile_owner;       // uint32, n_tiles: index into wl_huge
  BufferPtr tile_first_edge;  // uint32, n_tiles: absolute CSR edge index
  BufferPtr huge_tile_off;    // uint32, n_huge+1: tile range per huge vertex
  uint32_t n_high = 0, n_mid = 0, n_low = 0, n_zero = 0;
  uint32_t n_huge = 0, n_tiles = 0;
  uint64_t e_stored = 0;
};

class Graph {
 public:
  // src/dst are USER indices in [0, V). weights nullable. Throws mg::Error
  // on non-finite/negative weights or out-of-range indices.
  static std::shared_ptr<Graph> build(const uint32_t* src, const uint32_t* dst,
                                      const float* weights, uint64_t n_edges,
                                      uint32_t n_vertices, bool directed);

  uint32_t V = 0;
  uint64_t E_input = 0;     // caller-visible edge count
  bool directed = true;
  bool weighted = false;

  // OUT orientation is always built. IN (transpose) is built lazily and
  // cached; for undirected graphs orientation(in) == orientation(out).
  const Orientation& orientation(Dir d);
  uint64_t stored_edges() const { return out_.e_stored; }

  BufferPtr out_weight_sum;  // float, V (sum of outgoing weights, or
                             // out-degree as float when unweighted;
                             // canonical order)
  BufferPtr dangling_list;   // uint32 canonical ids with out_weight_sum == 0
  uint32_t n_dangling = 0;

  // Permutations (host vectors, USER <-> CANONICAL):
  //   canon_of_user[u] = c, user_of_canon[c] = u
  std::vector<uint32_t> canon_of_user;
  std::vector<uint32_t> user_of_canon;

  Graph() = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

 private:
  friend struct GraphBuilder;
  Orientation out_;
  std::unique_ptr<Orientation> in_;  // directed only
  std::once_flag in_once_;
  // Original input edges in canonical ids (kept to build the transpose
  // lazily and deterministically): pairs (canon_src, canon_dst) in input
  // order, plus weights. For undirected graphs these hold the SYMMETRIZED
  // edge list. Freed only if memory pressure demands (v0.2).
  std::vector<uint32_t> edges_src_canon_;
  std::vector<uint32_t> edges_dst_canon_;
  std::vector<float> edges_w_;
  std::vector<uint32_t> edges_orig_id_;
  void build_in_orientation();
};

using GraphPtr = std::shared_ptr<Graph>;

}  // namespace mg
