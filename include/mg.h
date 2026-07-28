/* mg.h — metal-graph stable C ABI (v0.1).
 *
 * The C ABI works in DENSE USER INDEX space: vertices are 0..num_vertices-1.
 * External-ID mapping (int64/str) is a binding-layer concern (see the
 * Python package). All functions return mg_status; mg_last_error_message()
 * gives a thread-local description of the most recent failure.
 *
 * `edge_types` / `time_range` style filters are reserved for v0.2 and have
 * no C entry points yet.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MG_H
#define MG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MG_VERSION_MAJOR 0
#define MG_VERSION_MINOR 1
#define MG_VERSION_PATCH 0

typedef enum mg_status {
  MG_OK = 0,
  MG_INVALID_ARGUMENT = 1,
  MG_OUT_OF_MEMORY = 2,
  MG_NO_GPU = 3,
  MG_INTERNAL_ERROR = 4,
  MG_NOT_IMPLEMENTED = 5,
} mg_status;

typedef enum mg_exec_mode {
  MG_EXEC_AUTO = 0,
  MG_EXEC_GPU = 1,
  MG_EXEC_CPU = 2,
} mg_exec_mode;

typedef enum mg_direction {
  MG_DIR_OUT = 0,
  MG_DIR_IN = 1,
  MG_DIR_BOTH = 2,
} mg_direction;

typedef struct mg_graph mg_graph; /* opaque */

const char* mg_version(void);
const char* mg_last_error_message(void);

/* Global per-operation planner mode (plan §6). */
mg_status mg_set_execution(mg_exec_mode mode);

/* Telemetry for the last operation on this thread: path 0=gpu 1=cpu. */
mg_status mg_last_run(int* out_path, int* out_iterations, double* out_ms);

/* Build an immutable snapshot from COO. weights may be NULL (unweighted).
 * Duplicate edges and self-loops are kept; NaN/negative weights rejected. */
mg_status mg_graph_create_from_coo(const uint32_t* src, const uint32_t* dst,
                                   const float* weights, uint64_t num_edges,
                                   uint32_t num_vertices, int directed,
                                   mg_graph** out_graph);
void mg_graph_destroy(mg_graph* g);

mg_status mg_graph_num_vertices(const mg_graph* g, uint32_t* out);
mg_status mg_graph_num_edges(const mg_graph* g, uint64_t* out);

/* Full-vector PageRank. personalization: NULL or float[V] (>=0, sum>0).
 * out_rank: caller-allocated float[V]. */
mg_status mg_pagerank(mg_graph* g, double alpha, double tol, int max_iter,
                      const float* personalization, float* out_rank,
                      int* out_iterations);

/* Batched personalized PageRank with top-k. Queries packed CSR-style:
 * query q owns seeds[offsets[q]..offsets[q+1]). out_ids: int32[B*k]
 * (row-major, -1 padded), out_scores: float[B*k]. */
mg_status mg_ppr_topk(mg_graph* g, const uint32_t* seeds,
                      const float* seed_weights, const uint64_t* offsets,
                      uint32_t num_queries, uint32_t k, double alpha,
                      double tol, int max_iter, int32_t* out_ids,
                      float* out_scores);

/* Multi-source BFS. out_dist/out_parent: caller-allocated int32[V], -1 for
 * unreachable / no parent. */
mg_status mg_bfs(mg_graph* g, const uint32_t* sources, uint32_t num_sources,
                 mg_direction direction, int32_t* out_dist,
                 int32_t* out_parent);

/* Bounded k-hop. Two-call pattern: pass NULL out buffers to receive counts,
 * then call again with buffers of (at least) that size. Vertices are user
 * indices ascending; edges are original input edge ids ascending.
 * max_vertices/max_edges: 0 = unlimited. */
mg_status mg_khop(mg_graph* g, const uint32_t* seeds, uint32_t num_seeds,
                  uint32_t k, mg_direction direction, uint32_t max_vertices,
                  uint64_t max_edges, uint32_t* out_vertices,
                  uint32_t* inout_num_vertices, uint32_t* out_edges,
                  uint64_t* inout_num_edges);

/* Weakly connected components (experimental tier, plan §7).
 * out_comp: caller-allocated int32[V]; canonical-partition ids. */
mg_status mg_wcc(mg_graph* g, int32_t* out_comp, int* out_num_components);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MG_H */
