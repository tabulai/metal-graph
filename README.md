# metal-graph

[![CI](https://github.com/tabulai/metal-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/tabulai/metal-graph/actions/workflows/ci.yml)

**Metal-native graph analytics for Apple Silicon.** metal-graph runs the
graph algorithms that dominate agent retrieval workloads — batched
personalized PageRank with top-k, full-vector PageRank, direction-optimizing
BFS, bounded k-hop extraction, and weakly connected components — as
GPU-resident Metal kernels over unified memory, with threaded CPU
implementations behind the same API and a threshold-based planner that
selects a candidate path per call.

## Highlights

- **Batched PPR with top-k** is the flagship: queries run in tiles of up to
  eight, lanes within a tile share each graph pass, queries converge
  independently, and a GPU radix-select returns `(B, k)` ids/scores. The
  isolated HippoRAG-shaped B=16 artifact run measured 10.707 ms per batch
  (0.669 ms/query, meeting the ≤0.7 ms amortized target; the ≤10 ms batch
  target is missed by 7%).
- **GPU-resident iteration.** PageRank encodes batches of iterations into
  single command buffers with per-iteration dangling-mass reduction on
  device; BFS runs whole levels through GPU-written indirect dispatch with a
  Beamer-style top-down/bottom-up switch; power-law mega-hubs are decomposed
  into fixed edge tiles so one vertex can't serialize a threadgroup.
- **Latency-aware planning.** Tiny traversals skip the GPU entirely via a
  serial path bounded by configured vertex and scanned-edge caps.
  `mg.bfs(..., output="sparse")` always returns |reached|-length arrays;
  only a traversal that finishes inside those caps avoids dense result
  initialization. Telemetry reports the path that actually executed.
- **Deterministic and tested.** PageRank is bit-deterministic at a fixed
  iteration count, batched PPR is bit-identical to sequential single
  queries, and top-k GPU selection is bit-identical to its CPU oracle. More
  than 1,000 Python cases cover correctness, properties, and integration
  behavior; CI separately checks the native C ABI and wheels, including
  GPU-required jobs. NetworkX and independent references are used where
  their contracts align.
- **Three surfaces:** Python (`metal_graph`, NumPy in/out), a stable C ABI
  ([`include/mg.h`](include/mg.h)), and installable wheels.

## Requirements

macOS 14+, Apple Silicon (M1/Apple7 GPU family or newer), CPython
3.10–3.14, CMake 3.24+, and Xcode with the Metal compiler.

## Installation

```bash
python3 -m pip install .
python3 -c "import metal_graph as mg; print(mg.__version__, mg.has_gpu())"
```

Development builds, native tests, wheel validation, and C ABI installation
are covered in [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

```python
import numpy as np
import metal_graph as mg

# Build once — an immutable snapshot. IDs may be any integers or strings;
# they are mapped to dense indices 0..V-1 ("user indices").
src = np.array([0, 0, 1, 2], dtype=np.uint32)
dst = np.array([1, 2, 2, 3], dtype=np.uint32)
G = mg.Graph.from_edges(src, dst, weights=None, directed=True,
                        num_vertices=4, ids="auto")
G.num_vertices, G.num_edges, G.external_ids

# Batched personalized PageRank with top-k.
seeds = np.array([0, 2], dtype=np.uint32)
seed_weights = np.ones(2, dtype=np.float32)
seed_offsets = np.array([0, 1, 2], dtype=np.uint64)
ids, scores = mg.ppr_topk(G, seeds, seed_weights, seed_offsets,
                          k=4, alpha=0.85, tol=1e-6, max_iter=50)
# ids: int32 (B, k) user indices (-1 padding), scores: float32 (B, k)

pr = mg.pagerank(G, alpha=0.85, tol=1e-6, max_iter=100, personalization=None)
dist, parent = mg.bfs(G, sources=[0], direction="out")
vs, dv, pv = mg.bfs(G, sources=[0], direction="out", output="sparse")
vs, es = mg.k_hop(G, seeds=[0], k=2, direction="both",
                  max_vertices=None, max_edges=None)
comp = mg.experimental.wcc(G)

mg.set_execution(mode="auto")   # "auto" | "gpu" | "cpu" (per-operation planner)
mg.last_run_info()              # {"op", "path": "gpu"|"cpu", "iterations", "ms"}
```

## Performance

Measured on an Apple M4 Max (macOS 26.2) at source commit `a4a8bc`, timed
from the Python call to the returned result. Metal-graph cells are medians
of 20 warm calls. External baselines generally use four timed calls after
one warm-up; sub-2 ms BFS baselines receive additional samples. Lower
latency is better. The
[benchmark report](docs/benchmark-report-2026-07-29.md) contains p95s,
methodology, baseline versions, and the limitations summarized below.

### metal-graph medians

| Dataset | PageRank / iteration | `ppr_topk` B=16, k=64 | BFS source 0 | WCC |
|---|---:|---:|---:|---:|
| RMAT-18 (V=262k, E=4.2M) | 0.35 ms | 10.2 ms | 1.5 ms | 3.8 ms |
| HippoRAG-shape KG (V=100k, E=2M, weighted) | 0.21 ms | 15.1 ms² | 0.012 ms¹ | 4.1 ms |
| RMAT-22 (V=4.2M, E=67M) | 3.4 ms | 122 ms | 13.9 ms | 32.4 ms |
| RMAT-24 (V=16.8M, E=268M) | 14.7 ms | 556 ms | 46.1 ms | 119.8 ms |
| soc-LiveJournal1 (V=4.8M, E=69M) | 4.0 ms | 174 ms | 13.4 ms | 24.9 ms |
| com-orkut (V=3.1M, E=117M) | 7.9 ms | 257 ms | 8.3 ms | 55.9 ms |

### External CPU comparisons

These are end-to-end API latency comparisons, not uniform identical-work
measurements. The rustworkx BFS adapter returns the same dense
`dist+parent` shape as metal-graph. PageRank uses the faster completed
igraph or rustworkx call for each dataset, but solver and iteration behavior
are not fully normalized. WCC compares the corresponding operation, but the
external libraries return their native component representation rather than
metal-graph's dense canonicalized array. Treat the PageRank and WCC ratios
as API-level context. The recorded versions were rustworkx 0.18.0 and
igraph 1.0.0. Each cell is
`metal-graph vs external median (speedup)`.
The run's igraph BFS rows predate the corrected dense-output adapter and
are excluded as non-equivalent; the current harness has the correction but
needs a new physical rerun. SciPy and NetworkX context remains in the full
run artifacts and is summarized where relevant in the report.

| Dataset | PageRank full run | BFS source 0 | WCC |
|---|---:|---:|---:|
| RMAT-18 | 1.759 vs 306.411 ms, rustworkx (174.2×) | 1.538 vs 1,030.735 ms, rustworkx (670.2×) | 3.819 vs 125.525 ms, igraph (32.9×) |
| HippoRAG KG | 3.074 vs 35.893 ms, igraph (11.7×) | 0.012 vs 0.057 ms, rustworkx (4.8×)¹ | 4.090 vs 13.136 ms, igraph (3.2×) |
| RMAT-22 | 16.892 vs 6,346.648 ms, rustworkx (375.7×) | 13.920 vs 17,600.397 ms, rustworkx (1,264.4×) | 32.362 vs 2,173.854 ms, igraph (67.2×) |
| RMAT-24 | 73.464 vs 30,765.861 ms, rustworkx (418.8×) | 46.096 vs 79,206.021 ms, rustworkx (1,718.3×) | 119.823 vs 9,058.173 ms, igraph (75.6×) |
| LiveJournal | 19.773 vs 6,803.438 ms, rustworkx (344.1×) | 13.398 vs 9,417.407 ms, rustworkx (702.9×) | 24.932 vs 1,652.573 ms, igraph (66.3×) |
| Orkut | 39.501 vs 24,247.799 ms, igraph (613.9×) | 8.290 vs 41,158.931 ms, rustworkx (4,965.2×) | 55.942 vs 67.772 ms, igraph (1.21×; below the 2× target) |

Using its then-current baseline definitions, the report's historical gate
accounting counted 17 of 18 full-run PageRank, high-degree BFS, and WCC
cells above the documented 2× target. The source-0 BFS column displayed
here is not that exact gate matrix. The displayed ratios are broad rather
than uniformly orders-of-magnitude: HippoRAG WCC and source-0 BFS are 3.2×
and 4.8×, respectively, and Orkut WCC is 1.21×. Orkut's rustworkx PageRank
call failed on the undirected graph, leaving igraph as its only completed
PageRank comparator. These are workload-specific measurements from one
machine, not a blanket GPU-speedup claim.

### Contextual PPR comparison

The igraph query loop uses PRPACK and cannot be pinned to metal-graph's
iteration count, so these ratios are context, not equivalent-work gate
evidence.

| Dataset | metal-graph B=16, k=64 | igraph 16-query loop | Contextual speedup |
|---|---:|---:|---:|
| RMAT-18 | 10.202 ms | 5,972.022 ms | 585.4× |
| HippoRAG KG | 15.116 ms | 663.211 ms | 43.9× |
| RMAT-22 | 121.993 ms | 106,395.181 ms | 872.1× |
| RMAT-24 | 555.634 ms | 588,510.973 ms | 1,059.2× |
| LiveJournal | 173.635 ms | did not complete after more than six hours | not comparable |
| Orkut | 257.417 ms | skipped after the LiveJournal overrun | not comparable |

² The interrupted full-suite run read the HippoRAG PPR batch at
15.116 ms, but its anomalous per-row accounting (a 3.3 ms `python_boundary`
residue unique to that collection window) prompted an isolated clean-process
re-measurement with full provenance
([JSON](bench/results/bench-20260729T201007Z.json) ·
[rendered](bench/results/bench-20260729T201007Z.md), clean source commit,
28 rows including all KG baselines): **10.707 ms per batch (p95 10.922),
0.669 ms/query over 20 warm runs** — the ≤0.7 ms amortized target passes
and the ≤10 ms batch target is missed by 7%. The same artifact re-measured
KG high-degree BFS against the corrected equivalent-output igraph adapter:
metal-graph 1.857 ms vs igraph 2.321 ms (1.25×) and rustworkx 49.334 ms
(26.6×).

¹ Tiny reachable components route to the bounded serial CPU path and are
assessed against an absolute-latency SLO (≤ 50 µs), rather than a ratio
alone. At microsecond scale, dense `int32[V]` output materialization
dominates the measurement. `output="sparse"` avoids that dense-output cost
when the traversal fits the configured sparse caps. Cap overflow or
forced-GPU execution remains exact but materializes dense state before
compaction.
The bounded collector still reserves a lazily zeroed V-byte visited map, so
sparse output is not an unconditional O(|reached|) total-memory guarantee.
For the HippoRAG high-degree source, which exercises the GPU, metal-graph
measured 1.905 ms versus rustworkx at 50.058 ms (26.3×).
The harness records whether the SLO passed; it does not currently fail the
benchmark command when the SLO is missed.

Evidence status matters: the Orkut measurements have a strict checked-in
[JSON](bench/results/bench-20260729T131101Z-orkut-bounded.json) and
[rendered-report](bench/results/bench-20260729T131101Z-orkut-bounded.md)
pair. The displayed completed core and comparator rows for RMAT-18 through
LiveJournal are reconstructed from the
[preserved run log](bench/results/bench-20260729-full-partial.log) and are
therefore provisional; LiveJournal's contextual PPR comparison did not
complete. The report documents the interrupted run and its missing final
JSON.

## Execution model

The base auto planner selects a candidate CPU or GPU path **once per
operation**: GPU when a Metal device exists and the stored edge count is at
least `MG_E_GPU_MIN` (default 1M). Algorithm-specific semantic routes can
override that candidate. BFS first attempts the bounded serial preflight. A
within-cap traversal completes there; overflow discards its partial result
and runs the planned CPU/GPU path. The cited HippoRAG source-0 case measured
12 µs, but that is an observation, not a general latency guarantee.
`mode="gpu"` bypasses the preflight and errors without a device;
`MG_FORCE_CPU=1` hides the GPU; `MG_REQUIRE_GPU=1` turns device absence into
a hard failure (used in CI). `mg.last_run_info()` reports the op (including
the executed variant, e.g. `bfs_sparse`), the path, the iteration count, and
the engine time.

GPU BFS encodes levels in growing command batches (8, 8, 16, 32, then 64),
keeping completion checks close for shallow traversals while bounding host
round-trips on deep-diameter graphs.

### Environment knobs

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

## Semantics reference

**Identity model.** External IDs (int32/int64/str) map to dense user
indices `0..V-1` in `np.unique` order (the identity when your IDs are
already dense ints — the only mode where `num_vertices=` adds isolated tail
vertices). Every algorithm input and output uses user indices;
`G.external_ids[i]` recovers the original ID and `G.index_of(ids)` maps the
other way. Internal renumbering for gather locality is invisible at the API.

**Input policy.** Duplicate edges and self-loops are kept (PageRank counts
them); non-finite or negative weights are rejected at build, including
fp32-overflowing per-vertex weight sums. Undirected graphs are symmetrized
internally; `num_edges` reports the input count.

**PageRank / `ppr_topk`.** NetworkX-compatible iteration: weighted graphs
normalize by outgoing weight sums, dangling mass is redistributed by the
teleport/personalization vector every iteration, on device. Convergence is
`L1(r_new − r_old) < V · tol`, audited in fp64 every `MG_PR_AUDIT_INTERVAL`
iterations, so iteration counts land on audit boundaries; `max_iter` is a
budget (the current iterate is returned, where NetworkX raises). Per-query
seed weights are normalized with duplicates summed; converged queries
freeze at audit boundaries; top-k ties break by ascending user index and
`k > V` rows pad with `id=-1, score=0`.

**BFS / `k_hop`.** Multi-source; `direction="out"|"in"|"both"`; `dist=-1`
unreachable, `parent=-1` for sources and unreachable vertices. Parent
choice among equal-depth candidates is nondeterministic on the parallel
paths and validated structurally. Sparse BFS returns aligned
`(vertices uint32, dist int32, parent int32)` arrays in ascending user-index
order. `k_hop` returns reached vertices and induced original-input edge ids,
sorted ascending. `max_vertices` deterministically truncates admitted
vertices; `max_edges` truncates sorted returned edge IDs only after full
induced-edge extraction. Capped calls force the CPU path and bound result
cardinality, not worst-case runtime, scanned edges, or O(V) scratch state.
`as_graph=True` materializes the subgraph.

**WCC** (`metal_graph.experimental`). Directed edges are treated as
undirected; component ids are numbered by first occurrence in user-index
order.

**Forward compatibility.** `edge_types=` / `time_range=` kwargs are
reserved and raise `NotImplementedError`.

## Current limits

- `V, E < 2^31` (fp32 values, int32 ids); graphs are immutable snapshots.
- `direction="both"` BFS on directed graphs and cap-bounded `k_hop` run on
  the (deterministic) CPU path.
- Batched PPR on GPU keeps the CSR plus a fixed 8-lane tile state resident
  (~100 bytes/vertex); graphs exceeding `recommendedMaxWorkingSetSize`
  raise a descriptive error. The dominant V-scaled tile state is independent
  of B; inputs, per-query metadata, and B×k outputs still scale with batch
  size. `mode="cpu"` avoids that GPU working-set check.
- The GPU top-k radix-select falls back to its bit-identical CPU oracle for
  tiny graphs (`V < 4096`), `k ≥ V`, `k > 4096`, and degenerate tie floods.

## Repository layout

```
include/mg.h            stable C ABI
src/kernels/            MSL kernels + mg_params.h (kernel<->host contract)
src/runtime/            device, queue, pipeline cache, buffers, planner
src/graph/              CPU threaded build: renumber, CSR, worklists
src/engines/            host-side dispatch drivers
src/algos/              entry points + cpu/ threaded oracles
src/c_api/  python/     C ABI impl · nanobind module + package
tests/  bench/          correctness/property tests · benchmark harness
```

## Benchmarking

`bench/run.py --suite v01` runs the full matrix (synthetic RMAT and
KG-shape graphs plus SHA-pinned SNAP datasets, fetched only with `--fetch`)
against igraph, rustworkx, SciPy, and NetworkX baselines with decomposed,
provenance-stamped reporting — see [`bench/README.md`](bench/README.md).
New published performance claims require a matched, checked-in JSON and
rendered-report pair from a physical run. All claims above for RMAT-18
through LiveJournal are a disclosed provisional exception: they are backed
by the preserved physical-run log and reconstruction report, not a
completed machine-readable pair, and should be replaced after a
checkpointed rerun. Only Orkut has a matched pair for the later full-suite
values.

## Roadmap

v0.2 candidates: label propagation, core number, similarity top-k,
typed/temporal k-hop filters (the reserved kwargs), `MTLHeap` allocation,
MLX device-resident results, and a NetworkX backend preview. v0.3 targets
the Louvain/Leiden index path. Design rationale for v0.1 is preserved in
the [implementation plan](docs/implementation-plan-v0.1.md).

## License

Apache-2.0 ([LICENSE](LICENSE)); vendored
[metal-cpp](third_party/metal-cpp/) is Apache-2.0, © Apple Inc. — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
