# metal-graph

[![CI](https://github.com/tabulai/metal-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/tabulai/metal-graph/actions/workflows/ci.yml)

**Metal-native graph analytics for Apple Silicon.** metal-graph runs the
graph algorithms that dominate agent retrieval workloads — batched
personalized PageRank with top-k, full-vector PageRank, direction-optimizing
BFS, bounded k-hop extraction, and weakly connected components — as
GPU-resident Metal kernels over unified memory, with threaded CPU
implementations behind the same API and a planner that picks the faster
path per operation.

## Highlights

- **Batched PPR with top-k** is the flagship: B queries share one streaming
  pass over the graph in 8-lane tiles, converge independently, and return
  `(B, k)` ids/scores selected by a GPU radix-select — the whole HippoRAG /
  aider-style query path in one call, at sub-millisecond amortized cost per
  query at knowledge-graph scale.
- **GPU-resident iteration.** PageRank encodes batches of iterations into
  single command buffers with per-iteration dangling-mass reduction on
  device; BFS runs whole levels through GPU-written indirect dispatch with a
  Beamer-style top-down/bottom-up switch; power-law mega-hubs are decomposed
  into fixed edge tiles so one vertex can't serialize a threadgroup.
- **Latency-aware planning.** Tiny traversals skip the GPU entirely via a
  bounded serial path (microseconds, not milliseconds); `mg.bfs(...,
  output="sparse")` returns O(|reached|) results for
  tiny-neighborhood-on-huge-graph queries. Telemetry reports the path that
  actually executed — there is no silent fallback.
- **Deterministic and tested.** PageRank is bit-deterministic at a fixed
  iteration count, batched PPR is bit-identical to sequential single
  queries, top-k GPU selection is bit-identical to its CPU oracle. 1,000+
  golden tests validate every algorithm against NetworkX on both execution
  paths, enforced in CI with the GPU required.
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

Measured on an Apple M4 Max (macOS 26.2) by `bench/run.py`; medians of 20
warm runs, timed from the Python call to the returned NumPy result. Full
per-cell baselines, provenance, and caveats:
[benchmark report](docs/benchmark-report-2026-07-29.md) and the checked-in
run artifacts (JSON, rendered tables, and the preserved run log) under
[`bench/results/`](bench/results/).

| dataset | PageRank / iter | `ppr_topk` B=16, k=64 | BFS | WCC |
|---|---:|---:|---:|---:|
| RMAT-18 (V=262k, E=4.2M) | 0.35 ms | 10.2 ms | 1.5 ms | 3.8 ms |
| HippoRAG-shape KG (V=100k, E=2M, weighted) | 0.21 ms | 15.1 ms | 0.012 ms¹ | 4.1 ms |
| RMAT-22 (V=4.2M, E=67M) | 3.4 ms | 122 ms | 13.9 ms | 32.4 ms |
| RMAT-24 (V=16.8M, E=268M) | 14.7 ms | 556 ms | 46.1 ms | 119.8 ms |
| soc-LiveJournal1 (V=4.8M, E=69M) | 4.0 ms | 174 ms | 13.4 ms | 24.9 ms |
| com-orkut (V=3.1M, E=117M) | 7.9 ms | 257 ms | 8.3 ms | 55.9 ms |

Against the fastest maintained CPU baselines (igraph / rustworkx) at
equivalent output semantics, the multi-million-edge cells run roughly one
to three orders of magnitude faster; the report carries the per-cell
ratios and the measurement-window variance of this shared workstation.

¹ Tiny reachable components route to the bounded serial CPU path and are
gated by an absolute-latency SLO (≤ 50 µs) rather than a ratio: at
microsecond scale, dense `int32[V]` output materialization dominates the
measurement. `output="sparse"` returns O(|reached|) results and removes
that cost entirely.

## Execution model

The planner picks CPU or GPU **once per operation**: GPU when a Metal
device exists and the stored edge count is at least `MG_E_GPU_MIN` (default
1M). BFS first attempts a bounded serial traversal; when the reachable work
fits the sparse caps it returns directly from the CPU in microseconds,
otherwise it proceeds with the planned path. `mode="gpu"` bypasses the
preflight and errors without a device; `MG_FORCE_CPU=1` hides the GPU;
`MG_REQUIRE_GPU=1` turns device absence into a hard failure (used in CI).
`mg.last_run_info()` reports the op (including the executed variant, e.g.
`bfs_sparse`), the path, the iteration count, and the engine time.

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
paths and validated structurally. `k_hop` returns reached vertices and
induced original-input edge ids, sorted ascending; `max_vertices` /
`max_edges` caps give predictable agent-side latency; `as_graph=True`
materializes the subgraph.

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
  raise a descriptive error — batch size does not change the footprint,
  `mode="cpu"` does.
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
tests/  bench/          golden tests vs NetworkX · benchmark harness
```

## Benchmarking

`bench/run.py --suite v01` runs the full matrix (synthetic RMAT and
KG-shape graphs plus SHA-pinned SNAP datasets, fetched only with `--fetch`)
against igraph, rustworkx, SciPy, and NetworkX baselines with decomposed,
provenance-stamped reporting — see [`bench/README.md`](bench/README.md).
Published performance claims always link a matched, checked-in JSON +
rendered report pair from a physical run.

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
