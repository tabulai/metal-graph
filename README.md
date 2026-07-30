# metal-graph

[![CI](https://github.com/tabulai/metal-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/tabulai/metal-graph/actions/workflows/ci.yml)

**Fast graph analytics for Apple Silicon.**

metal-graph is a Python and C library for ranking, searching, and exploring
graphs on a Mac. Build a graph once from NumPy arrays, then run PageRank,
batched personalized PageRank, breadth-first search, neighborhood extraction,
and connected-components analysis through one consistent API.

The library uses the Metal GPU for work that benefits from it and a threaded
CPU path for smaller jobs. That choice is automatic by default, so using
metal-graph does not require Metal knowledge or device-management code.

It is particularly useful for:

- knowledge-graph and retrieval pipelines;
- entity ranking, recommendations, and repo maps;
- graph-backed agent memory and neighborhood lookup;
- local analysis of large graphs on Apple Silicon.

## Why use metal-graph?

- **A small, practical API.** Inputs and outputs are NumPy arrays, and integer
  or string IDs are supported.
- **Useful graph operations included.** PageRank, batched PPR with top-k, BFS,
  sparse BFS, bounded k-hop extraction, and weakly connected components are
  available today.
- **Good performance across different query sizes.** Large traversals can use
  the GPU, while tiny reachable components can stay on the CPU and avoid GPU
  launch overhead.
- **Transparent execution.** `mg.last_run_info()` reports what ran, where it
  ran, and how long the engine took.
- **Tested against established libraries.** More than 1,000 Python cases cover
  correctness, properties, and integration behavior. CI also checks the C API,
  wheels, and GPU-required jobs.

The main interface is Python (`metal_graph`, with NumPy input and output).
A stable C API is available in
[`include/mg.h`](https://github.com/tabulai/metal-graph/blob/main/include/mg.h),
but it is not
bundled into the Python wheel; install it separately from source with CMake.

## Requirements

- macOS 14 or newer
- Apple Silicon with an M1 or newer
- CPython 3.10–3.14

Installing a prebuilt wheel does not require a compiler. Building from source,
including installing the C API, additionally requires CMake 3.24 or newer and
Xcode with the Metal compiler.

## Install

Install the prebuilt wheel from PyPI:

```bash
python3 -m pip install metal-graph
python3 -c "import metal_graph as mg; print(mg.__version__, mg.has_gpu())"
```

No compiler or Metal setup is needed for the wheel. Source builds, development
setup, native tests, wheel validation, and C API installation are covered in
[CONTRIBUTING.md](https://github.com/tabulai/metal-graph/blob/main/CONTRIBUTING.md).

## Quick start

```python
import numpy as np
import metal_graph as mg

# Build an immutable directed graph.
src = np.array([0, 0, 1, 2], dtype=np.uint32)
dst = np.array([1, 2, 2, 3], dtype=np.uint32)
G = mg.Graph.from_edges(
    src,
    dst,
    directed=True,
    num_vertices=4,
)

# Rank every vertex.
rank = mg.pagerank(G)

# Explore from vertex 0.
dist, parent = mg.bfs(G, sources=[0], direction="out")
vertices, sparse_dist, sparse_parent = mg.bfs(
    G,
    sources=[0],
    direction="out",
    output="sparse",
)
neighbors, edge_ids = mg.k_hop(
    G,
    seeds=[0],
    k=2,
    direction="both",
)

# Find weakly connected components.
components = mg.experimental.wcc(G)
```

For a batch of personalized ranking queries, pack the seeds using offsets:

```python
seeds = np.array([0, 2], dtype=np.uint32)
seed_weights = np.ones(2, dtype=np.float32)
seed_offsets = np.array([0, 1, 2], dtype=np.uint64)

ids, scores = mg.ppr_topk(
    G,
    seeds,
    seed_weights,
    seed_offsets,
    k=4,
)
# ids and scores both have shape (number_of_queries, k)
```

By default, metal-graph chooses the execution path for each call:

```python
mg.set_execution("auto")       # "auto", "gpu", or "cpu"
mg.last_run_info()             # op, path, iterations, and engine time
```

The
[`notebooks/`](https://github.com/tabulai/metal-graph/tree/main/notebooks)
directory contains complete examples for the
core API, batched PPR retrieval, and low-latency BFS.

## Available operations

- `Graph.from_edges(...)` builds an immutable weighted or unweighted graph.
- `pagerank(...)` returns a score for every vertex.
- `ppr_topk(...)` answers a batch of personalized PageRank queries and returns
  only the highest-scoring vertices.
- `bfs(...)` returns dense distances and parents, or compact results with
  `output="sparse"`.
- `k_hop(...)` finds a bounded neighborhood and its induced input edges.
- `experimental.wcc(...)` labels weakly connected components.

## Performance

The results below were measured on an Apple M4 Max running macOS 26.2.
Timings cover the Python call through the returned result, so they include
normal API overhead. Lower is better.

Most cells come from source commit `a4a8bc`; the HippoRAG PPR cell marked ²
comes from the isolated artifact at source commit `7afb4b2`. metal-graph
cells are medians of 20 warm calls. External baselines generally use four
timed calls after one warm-up, with additional samples for sub-2 ms BFS
measurements. The full
[benchmark report](https://github.com/tabulai/metal-graph/blob/main/docs/benchmark-report-2026-07-29.md)
contains p95 values, package versions, and methodology.

### metal-graph latency

| Dataset | PageRank / iteration | `ppr_topk` B=16, k=64 | BFS source 0 | WCC |
|---|---:|---:|---:|---:|
| RMAT-18 (V=262k, E=4.2M) | 0.35 ms | 10.2 ms | 1.5 ms | 3.8 ms |
| HippoRAG-shape KG (V=100k, E=2M, weighted) | 0.21 ms | 10.707 ms² | 0.012 ms¹ | 4.1 ms |
| RMAT-22 (V=4.2M, E=67M) | 3.4 ms | 122 ms | 13.9 ms | 32.4 ms |
| RMAT-24 (V=16.8M, E=268M) | 14.7 ms | 556 ms | 46.1 ms | 119.8 ms |
| soc-LiveJournal1 (V=4.8M, E=69M) | 4.0 ms | 174 ms | 13.4 ms | 24.9 ms |
| com-orkut (V=3.1M, E=117M) | 7.9 ms | 257 ms | 8.3 ms | 55.9 ms |

### Comparison with CPU graph libraries

Each cell shows `metal-graph vs external median (speedup)`. These are
end-to-end API comparisons, not claims that every library performs identical
internal work.

The rustworkx BFS adapter returns the same dense `dist + parent` shape as
metal-graph. PageRank uses the faster completed igraph or rustworkx call for
each dataset, but solver and iteration behavior are not fully normalized.
WCC compares the corresponding operation, although the external libraries
return their native component representation. Treat the PageRank and WCC
ratios as practical API-level context. The recorded versions were rustworkx
0.18.0 and igraph 1.0.0.

| Dataset | PageRank full run | BFS source 0 | WCC |
|---|---:|---:|---:|
| RMAT-18 | 1.759 vs 306.411 ms, rustworkx (174.2×) | 1.538 vs 1,030.735 ms, rustworkx (670.2×) | 3.819 vs 125.525 ms, igraph (32.9×) |
| HippoRAG KG | 3.074 vs 35.893 ms, igraph (11.7×) | 0.012 vs 0.057 ms, rustworkx (4.8×)¹ | 4.090 vs 13.136 ms, igraph (3.2×) |
| RMAT-22 | 16.892 vs 6,346.648 ms, rustworkx (375.7×) | 13.920 vs 17,600.397 ms, rustworkx (1,264.4×) | 32.362 vs 2,173.854 ms, igraph (67.2×) |
| RMAT-24 | 73.464 vs 30,765.861 ms, rustworkx (418.8×) | 46.096 vs 79,206.021 ms, rustworkx (1,718.3×) | 119.823 vs 9,058.173 ms, igraph (75.6×) |
| LiveJournal | 19.773 vs 6,803.438 ms, rustworkx (344.1×) | 13.398 vs 9,417.407 ms, rustworkx (702.9×) | 24.932 vs 1,652.573 ms, igraph (66.3×) |
| Orkut | 39.501 vs 24,247.799 ms, igraph (613.9×) | 8.290 vs 41,158.931 ms, rustworkx (4,965.2×) | 55.942 vs 67.772 ms, igraph (1.21×; below the 2× target) |

Performance varies with the graph and operation. For example, HippoRAG WCC
and source-0 BFS measured 3.2× and 4.8× faster, while Orkut WCC measured
1.21×. Using the benchmark's historical gate definitions, 17 of 18 full-run
PageRank, high-degree BFS, and WCC cells exceeded the documented 2× target.
The source-0 BFS column above is not that exact gate matrix.

### Contextual personalized PageRank comparison

The igraph query loop uses PRPACK and cannot be pinned to metal-graph's
iteration count. This table is useful end-to-end context, not an
identical-work comparison.

| Dataset | metal-graph B=16, k=64 | igraph 16-query loop | Contextual speedup |
|---|---:|---:|---:|
| RMAT-18 | 10.202 ms | 5,972.022 ms | 585.4× |
| HippoRAG KG | 10.707 ms² | 611.357 ms | 57.1× |
| RMAT-22 | 121.993 ms | 106,395.181 ms | 872.1× |
| RMAT-24 | 555.634 ms | 588,510.973 ms | 1,059.2× |
| LiveJournal | 173.635 ms | did not complete after more than six hours | not comparable |
| Orkut | 257.417 ms | skipped after the LiveJournal overrun | not comparable |

<details>
<summary>Notes for the marked benchmark cells</summary>

¹ **Tiny-component BFS.** Small reachable components use a bounded serial CPU
path and are assessed against a 50 µs absolute-latency objective rather than
a ratio alone. Dense result initialization dominates at this scale.
`output="sparse"` avoids that cost when the traversal fits the configured
caps. Overflow or forced-GPU execution remains exact but materializes dense
state before compaction. The bounded collector still reserves a lazily zeroed
V-byte visited map, so sparse output is not an unconditional
O(|reached|)-memory guarantee. The benchmark records whether the objective
passed but does not fail the command when it is missed.

For the HippoRAG high-degree source, which exercises the GPU, metal-graph
measured 1.905 ms versus rustworkx at 50.058 ms (26.3×).

² **Isolated HippoRAG PPR measurement.** The displayed values come from a
clean-process, 20-run artifact:
[JSON](https://github.com/tabulai/metal-graph/blob/main/bench/results/bench-20260729T201007Z.json)
·
[rendered report](https://github.com/tabulai/metal-graph/blob/main/bench/results/bench-20260729T201007Z.md).
It measured
10.707 ms per batch (p95 10.922), or 0.669 ms/query. That meets the
0.7 ms/query objective and misses the 10 ms batch objective by 7%.

This measurement supersedes the interrupted full-suite run's anomalous
15.116 ms reading, which carried a 3.3 ms `python_boundary` residue unique to
that collection window. The same artifact measured high-degree BFS at
1.857 ms versus igraph at 2.321 ms (1.25×) and rustworkx at 49.334 ms
(26.6×), using the corrected equivalent-output adapter.

</details>

<details>
<summary>Benchmark provenance and comparison caveats</summary>

The run's original igraph BFS rows predate the corrected dense-output adapter
and are excluded as non-equivalent. The current harness contains the corrected
adapter but needs a new physical rerun. SciPy and NetworkX comparisons remain
in the full artifacts and benchmark report.

The Orkut results have a strict checked-in
[JSON](https://github.com/tabulai/metal-graph/blob/main/bench/results/bench-20260729T131101Z-orkut-bounded.json)
and
[rendered report](https://github.com/tabulai/metal-graph/blob/main/bench/results/bench-20260729T131101Z-orkut-bounded.md)
pair. Completed rows for RMAT-18 through LiveJournal were reconstructed from
the
[preserved run log](https://github.com/tabulai/metal-graph/blob/main/bench/results/bench-20260729-full-partial.log)
and
remain provisional. The original run was interrupted before its final JSON
was written; LiveJournal's contextual PPR comparison did not complete.

These are workload-specific measurements from one machine, not a blanket
GPU-speedup claim. Orkut's rustworkx PageRank call failed on the undirected
graph, leaving igraph as its completed comparator.

</details>

## How execution is chosen

`mode="auto"` is intended for normal use. The planner starts with the GPU
when a Metal device is available and the graph has at least one million
stored edges; otherwise it starts with the CPU. Individual algorithms can
override that candidate when another path is more appropriate.

BFS, for example, first tries a bounded serial traversal. If the reachable
component fits within its vertex and scanned-edge caps, the CPU answers the
query directly. If it does not fit, the partial attempt is discarded and the
regular CPU or GPU path runs instead.

You can override the planner when testing or profiling:

```python
mg.set_execution("gpu")   # require the GPU path
mg.set_execution("cpu")   # require the CPU path
mg.set_execution("auto")  # restore automatic selection
```

`mg.last_run_info()` reports the operation, chosen path, iteration count, and
engine time. Forced GPU mode raises an error if no device is available.

<details>
<summary>Advanced environment settings</summary>

GPU BFS batches increasing numbers of levels (8, 8, 16, 32, then 64) to keep
shallow traversals responsive while reducing host round-trips on deep graphs.

| Variable | Default | Meaning |
|---|---|---|
| `MG_E_GPU_MIN` | 1000000 | auto-planner GPU threshold (stored edges) |
| `MG_PR_AUDIT_INTERVAL` | 5 | iterations per GPU command batch / fp64 audit |
| `MG_BFS_LEVELS_PER_BATCH` | adaptive 8,8,16,32,64… | BFS levels per command buffer; an integer pins a fixed size (clamped 1–256) |
| `MG_BFS_SPARSE_MAX_VERTICES` | 1024 | auto/CPU BFS latency-path vertex cap (0 disables) |
| `MG_BFS_SPARSE_MAX_EDGES` | 8192 | auto/CPU BFS latency-path scanned-edge cap (0 disables) |
| `MG_WCC_ROUNDS_PER_BATCH` | 4 | WCC hook+jump rounds per command buffer |
| `MG_GPU_TOPK` | 1 | GPU radix-select for `ppr_topk` (0 = CPU oracle) |
| `MG_BFS_BOTTOMUP` | 1 | direction-optimizing bottom-up switch |
| `MG_REQUIRE_GPU` | 0 | hard-fail when no Metal device |
| `MG_FORCE_CPU` | 0 | disable the GPU entirely |
| `MG_THREADS` | all cores | CPU-path thread count |
| `MG_METALLIB` | embedded | override path to a .metallib (dev loop) |

</details>

## Graph IDs and result behavior

External integer or string IDs are mapped to dense user indices in
`np.unique` order. Algorithm inputs and outputs use those user indices.
Use `G.index_of(ids)` to map external IDs to indices and
`G.external_ids[index]` to map back. `G.external_ids` returns a detached,
read-only NumPy snapshot, so changing a caller-owned copy cannot alter the
graph's ID mapping.

Graphs are immutable after construction. Duplicate edges and self-loops are
kept, and PageRank counts them. Non-finite or negative weights are rejected.
Undirected graphs are symmetrized internally, while `G.num_edges` continues
to report the number of input edges.

The most important result conventions are:

- PageRank follows NetworkX-compatible weighted normalization and dangling
  mass handling. `max_iter` is a budget: metal-graph returns the current
  result when it is reached, while NetworkX raises.
- `ppr_topk` normalizes seed weights, combines duplicate seeds, returns
  deterministic top-k ordering, and pads with `id=-1, score=0` when `k > V`.
- BFS supports `direction="out"`, `"in"`, or `"both"`. Unreachable distances
  and missing parents are `-1`. Parent choice can vary when several
  same-depth parents are valid on a parallel path.
- Sparse BFS returns aligned `(vertices, dist, parent)` arrays in ascending
  user-index order.
- `k_hop` returns ascending vertex indices and original input edge IDs.
  `max_vertices` and `max_edges` bound result size, not worst-case runtime,
  scanned edges, or scratch memory. Use `as_graph=True` to materialize the
  returned neighborhood as a new graph.
- WCC treats directed edges as undirected and numbers components by their
  first appearance in user-index order.

<details>
<summary>Exact PageRank convergence behavior</summary>

Convergence is `L1(r_new - r_old) < V * tol`, audited in fp64 every
`MG_PR_AUDIT_INTERVAL` iterations. Iteration counts therefore land on audit
boundaries. Batched PPR queries converge independently and freeze at those
boundaries. Top-k ties are resolved by ascending user index.

</details>

## Current limits

- The library currently supports `V, E < 2^31`; graph values are fp32 and
  graph IDs are int32 internally.
- Graphs are immutable snapshots rather than streaming or mutable graphs.
- `direction="both"` BFS on directed graphs and capped `k_hop` run on the
  deterministic CPU path.
- Batched GPU PPR keeps the CSR plus an eight-query tile in memory. The
  dominant tile state uses roughly 100 bytes per vertex. Oversized workloads
  raise a descriptive working-set error; `mode="cpu"` avoids that GPU limit.
- GPU top-k selection uses its bit-identical CPU fallback for tiny graphs
  (`V < 4096`), `k >= V`, `k > 4096`, and degenerate tie floods.
- Typed and temporal `k_hop` filters are reserved for a future release;
  `edge_types=` and `time_range=` currently raise `NotImplementedError`.

## Benchmarks and reproducibility

`bench/run.py --suite v01` runs the full matrix against igraph, rustworkx,
SciPy, and NetworkX. It uses synthetic RMAT and knowledge-graph-shaped data
plus SHA-pinned SNAP datasets, which are downloaded only with `--fetch`.
See the
[benchmark guide](https://github.com/tabulai/metal-graph/blob/main/bench/README.md)
for commands and artifact format.

New published performance claims require a matched, checked-in JSON and
rendered-report pair from a physical run. The disclosed exception is the
reconstructed RMAT-18 through LiveJournal data described above; only Orkut
has a matched pair for the later full-suite values.

## Project resources

- [Example notebooks](https://github.com/tabulai/metal-graph/tree/main/notebooks)
- [Benchmark guide](https://github.com/tabulai/metal-graph/blob/main/bench/README.md)
- [Contribution and development guide](https://github.com/tabulai/metal-graph/blob/main/CONTRIBUTING.md)
- [C API](https://github.com/tabulai/metal-graph/blob/main/include/mg.h)
- [v0.1 implementation and design rationale](https://github.com/tabulai/metal-graph/blob/main/docs/implementation-plan-v0.1.md)

## Roadmap

Possible v0.2 additions include label propagation, core number, similarity
top-k, typed and temporal neighborhood filters, MLX device-resident results,
and a NetworkX backend preview. Louvain and Leiden community detection are
longer-term candidates.

## License

Apache-2.0
([LICENSE](https://github.com/tabulai/metal-graph/blob/main/LICENSE)). The
vendored
[metal-cpp](https://github.com/tabulai/metal-graph/tree/main/third_party/metal-cpp)
library is Apache-2.0, © Apple Inc.; see
[THIRD_PARTY_NOTICES.md](https://github.com/tabulai/metal-graph/blob/main/THIRD_PARTY_NOTICES.md).
