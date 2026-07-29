# metal-graph bench — v01-orkut-bounded — 2026-07-29T12:54:06.836262+00:00

- chip: `Apple M4 Max`  · macOS 26.2 · python 3.13.3 · metal_graph 0.1.0
- source: git `a4a8bc325785` (clean) · Xcode 26.4

| dataset | algo | line item | median ms | p95 ms | peak RSS Δ MB | notes |
|---|---|---|---:|---:|---:|---|
| com-orkut | build | from_edges | 1856.951 | 1856.951 | 4327.8 |  |
| com-orkut | build | transpose_estimate | 17.215 | nan | 199.4 |  |
| com-orkut | pagerank | warm_full_run | 39.501 | 41.915 | 0.3 | path=gpu, iterations=5, achieved_gb_s=128 |
| com-orkut | pagerank | per_iteration | 7.900 | 8.383 | 0.0 | iterations=5 |
| com-orkut | ppr_topk | warm_batch16_k64 | 257.417 | 265.154 | 469.1 | path=gpu, iterations=5, amortized_per_query_ms=16.09 |
| com-orkut | ppr_topk | python_boundary | 0.000 | nan | 0.0 |  |
| com-orkut | ppr_topk | topk_selection_estimate | 1.906 | nan | 0.1 |  |
| com-orkut | bfs | warm_single_source | 8.290 | 8.540 | 0.0 | path=gpu, levels=8, variant=bfs, source=0, source_out_degree=12, reached_vertices=3072441, reachable_edges_scanned=234370166, max_distance=7, frontier_sizes=[1, 12, 469, 26888, 800020, 2226334, 18712, 5] |
| com-orkut | bfs | warm_high_degree_source | 9.482 | 9.725 | 0.0 | path=gpu, levels=7, variant=bfs, source=43607, source_out_degree=33313, reached_vertices=3072441, reachable_edges_scanned=234370166, max_distance=6, frontier_sizes=[1, 33313, 935702, 1076816, 987315, 39191, 103] |
| com-orkut | bfs | warm_64_sources | 8.086 | 8.558 | 0.0 | path=gpu, variant=bfs |
| com-orkut | wcc | warm | 55.942 | 57.855 | 0.0 | path=gpu, rounds=4 |
| com-orkut | k_hop | warm_k2_capped | 0.367 | 0.425 | 0.0 |  |
| com-orkut | pagerank | baseline_networkx | nan | nan | 0.0 | note=skipped: E=117,185,083 > cap 2,500,000 |
| com-orkut | pagerank | baseline_scipy_20iter | 2261.802 | 2282.839 | 5139.8 |  |
| com-orkut | pagerank | baseline_rustworkx | nan | nan | 19006.4 | note=failed: 'PyGraph' object is not an instance of 'PyDiGraph' |
| com-orkut | bfs | baseline_rustworkx | 41158.931 | 44012.201 | 0.0 | source=0, semantics=dense int32 dist+parent |
| com-orkut | bfs | baseline_rustworkx_noop | 41580.829 | 41760.935 | 0.0 | source=0, semantics=traversal only; no returned result |
| com-orkut | bfs | baseline_rustworkx_layers | 14367.501 | 14663.581 | 0.0 | source=0, semantics=sparse layers; no parent array |
| com-orkut | bfs | baseline_rustworkx_high_degree | 42125.959 | 42204.570 | 0.0 | source=43607, source_out_degree=33313, semantics=dense int32 dist+parent |
| com-orkut | wcc | baseline_rustworkx | 13661.075 | 13746.372 | 0.0 |  |
| com-orkut | pagerank | baseline_igraph | 24247.799 | 24440.438 | 4189.6 |  |
| com-orkut | bfs | baseline_igraph | 1263.730 | 1264.789 | 0.0 |  |
| com-orkut | bfs | baseline_igraph_high_degree | 1463.188 | 1496.221 | 0.0 |  |
| com-orkut | wcc | baseline_igraph | 67.772 | 67.969 | 0.0 |  |
| com-orkut | ppr_topk | baseline_igraph_query_loop | nan | nan | 0.0 | note=skipped: unbounded exact-solver comparator exceeded six hours on LiveJournal |
