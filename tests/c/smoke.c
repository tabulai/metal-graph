// smoke.c — C-ABI smoke test. Uses include/mg.h ONLY; makes no GPU
// assertions, so it can also run under MG_FORCE_CPU=1.
// SPDX-License-Identifier: Apache-2.0
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mg.h"

static void check(int cond, const char* what) {
  if (!cond) {
    fprintf(stderr, "smoke FAILED: %s\n  last error: %s\n", what,
            mg_last_error_message());
    abort();
  }
}

int main(void) {
  check(strcmp(mg_version(), "0.1.0") == 0, "mg_version is 0.1.0");
  check(mg_set_execution(MG_EXEC_AUTO) == MG_OK, "set_execution(auto)");

  /* Oversized edge counts must fail before the core reads edge pointers. */
  {
    const uint32_t dummy = 0;
    mg_graph* too_large = NULL;
    check(mg_graph_create_from_coo(&dummy, &dummy, NULL, UINT64_MAX, 1, 1,
                                   &too_large) == MG_INVALID_ARGUMENT,
          "oversized edge count rejected");
    check(too_large == NULL, "oversized graph handle remains null");
  }

  /* Every non-finite graph weight must be rejected through the public ABI. */
  {
    const uint32_t one_edge = 0;
    const float bad_weights[3] = {NAN, INFINITY, -INFINITY};
    int i;
    for (i = 0; i < 3; ++i) {
      mg_graph* invalid = NULL;
      check(mg_graph_create_from_coo(&one_edge, &one_edge, &bad_weights[i], 1,
                                     1, 1, &invalid) == MG_INVALID_ARGUMENT,
            "non-finite graph weight rejected");
      check(invalid == NULL, "invalid graph handle remains null");
    }
  }

  /* 10 vertices, 11 directed edges; vertex 9 has no out-edges (dangling).
   *   0->1 0->2 1->2 2->3 3->4 4->5 5->6 6->7 7->8 8->9 1->9            */
  enum { V = 10, E = 11 };
  const uint32_t src[E] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1};
  const uint32_t dst[E] = {1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9};

  mg_graph* g = NULL;
  check(mg_graph_create_from_coo(src, dst, NULL, E, V, 1, &g) == MG_OK,
        "graph create");
  check(g != NULL, "graph handle non-null");

  {
    uint32_t nv = 0;
    uint64_t ne = 0;
    check(mg_graph_num_vertices(g, &nv) == MG_OK && nv == V, "num_vertices");
    check(mg_graph_num_edges(g, &ne) == MG_OK && ne == E, "num_edges");
  }

  /* error path: NULL graph must fail with a thread-local message */
  {
    float r[V];
    check(mg_pagerank(NULL, 0.85, 1e-6, 100, NULL, r, NULL) ==
              MG_INVALID_ARGUMENT,
          "pagerank(NULL graph) -> MG_INVALID_ARGUMENT");
    check(mg_last_error_message()[0] != '\0', "error message set");
  }

  /* pagerank: finite, non-negative, sums to ~1 (dangling mass recycled) */
  {
    float rank[V];
    int iters = -1;
    double sum = 0.0;
    int i;
    check(mg_pagerank(g, 0.85, 1e-6, 100, NULL, rank, &iters) == MG_OK,
          "pagerank");
    check(iters >= 1, "pagerank iterations >= 1");
    for (i = 0; i < V; ++i) {
      check(isfinite((double)rank[i]), "pagerank value finite");
      check(rank[i] >= 0.0f, "pagerank value non-negative");
      sum += (double)rank[i];
    }
    check(fabs(sum - 1.0) < 1e-3, "pagerank sums to ~1");
  }

  /* telemetry: path is 0 (gpu) or 1 (cpu); no GPU assertion */
  {
    int path = -1, iters = -1;
    double ms = -1.0;
    check(mg_last_run(&path, &iters, &ms) == MG_OK, "last_run");
    check(path == 0 || path == 1, "last_run path in {0,1}");
    check(ms >= 0.0, "last_run ms >= 0");
  }

  /* bfs from 0 along out edges */
  {
    const uint32_t sources[1] = {0};
    int32_t dist[V], parent[V];
    check(mg_bfs(g, sources, 1, MG_DIR_OUT, dist, parent) == MG_OK, "bfs");
    check(dist[0] == 0, "bfs source dist 0");
    check(parent[0] == -1, "bfs source parent -1");
    check(dist[1] == 1 && dist[2] == 1, "bfs level-1 dists");
    check(dist[9] == 2, "bfs dist[9] == 2 (0->1->9)");
  }

  /* wcc: graph is weakly connected -> one component, ids all 0 */
  {
    int32_t comp[V];
    int n = -1;
    int i;
    check(mg_wcc(g, comp, &n) == MG_OK, "wcc");
    check(n == 1, "wcc single component");
    for (i = 0; i < V; ++i) check(comp[i] == 0, "wcc component ids all 0");
  }

  /* ppr_topk: B=2, k=3; each seed should top its own query */
  {
    enum { B = 2, K = 3 };
    const uint32_t seeds[2] = {0, 3};
    const float seed_w[2] = {1.0f, 1.0f};
    const uint64_t offsets[3] = {0, 1, 2};
    int32_t ids[B * K];
    float scores[B * K];
    int q, i;
    check(mg_ppr_topk(g, seeds, seed_w, offsets, B, K, 0.85, 1e-6, 50, ids,
                      scores) == MG_OK,
          "ppr_topk");
    for (q = 0; q < B; ++q) {
      for (i = 0; i < K; ++i) {
        int32_t id = ids[q * K + i];
        float s = scores[q * K + i];
        check(id >= -1 && id < (int32_t)V, "ppr id in range");
        check(isfinite((double)s) && s >= 0.0f, "ppr score finite/non-neg");
        if (i > 0)
          check(scores[q * K + i - 1] >= s, "ppr scores non-increasing");
      }
    }
    check(ids[0] == 0, "ppr query 0 top id is seed 0");
    check(ids[K] == 3, "ppr query 1 top id is seed 3");
  }

  /* k-hop two-call pattern: seeds {0}, k=1, OUT -> vertices {0,1,2} and
   * induced input edge ids {0,1,2} (1->9 excluded: 9 unreached) */
  {
    const uint32_t seeds[1] = {0};
    /* Deliberately uninitialized: the count-only call must only write them. */
    uint32_t n_vs;
    uint64_t n_es;
    uint32_t vs[3], es[3];
    check(mg_khop(g, seeds, 1, 1, MG_DIR_OUT, 0, 0, NULL, &n_vs, NULL,
                  &n_es) == MG_OK,
          "khop count call");
    check(n_vs == 3 && n_es == 3, "khop counts (3 vertices, 3 edges)");
    check(mg_khop(g, seeds, 1, 1, MG_DIR_OUT, 0, 0, vs, &n_vs, es, &n_es) ==
              MG_OK,
          "khop copy call");
    check(n_vs == 3 && n_es == 3, "khop counts stable");
    check(vs[0] == 0 && vs[1] == 1 && vs[2] == 2, "khop vertices ascending");
    check(es[0] == 0 && es[1] == 1 && es[2] == 2, "khop edge ids ascending");

    /* insufficient capacity is rejected, not truncated */
    {
      uint32_t cap_v = 1;
      uint64_t cap_e = 3;
      uint32_t one_v[1];
      uint32_t es2[3];
      check(mg_khop(g, seeds, 1, 1, MG_DIR_OUT, 0, 0, one_v, &cap_v, es2,
                    &cap_e) == MG_INVALID_ARGUMENT,
            "khop insufficient capacity -> MG_INVALID_ARGUMENT");
    }
  }

  mg_graph_destroy(g);
  printf("smoke: OK\n");
  return 0;
}
