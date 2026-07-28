# metal-graph

[![CI](https://github.com/tabulai/metal-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/tabulai/metal-graph/actions/workflows/ci.yml)

**A Metal-native graph-analytics engine for Apple Silicon.** v0.1 ships the
few algorithms that accelerate the most on unified memory: batched
personalized PageRank with top-k (the flagship), full-vector PageRank,
direction-optimizing BFS with bounded k-hop extraction, and WCC (shipped
under `experimental` as the atomics canary).

- Python import: `metal_graph` · C ABI: [`include/mg.h`](include/mg.h)
  (`mg_` prefix) · MSL kernels: `mg_` prefix · License: Apache-2.0.
- Requirements: macOS 14+, Apple Silicon (Apple7+/M1 and up), CPython
  3.10–3.13, CMake 3.24+, and Xcode with the Metal compiler.
- Design + rationale: [v0.1 implementation plan](docs/implementation-plan-v0.1.md).

## Quick start

Build and install the native extension from a checkout:

```bash
python3 -m pip install .
python3 -c "import metal_graph as mg; print(mg.__version__, mg.has_gpu())"
```

```python
import numpy as np
import metal_graph as mg

# Build once — immutable snapshot. IDs may be any integers or strings;
# they are mapped to dense indices 0..V-1 ("user indices").
src = np.array([0, 0, 1, 2], dtype=np.uint32)
dst = np.array([1, 2, 2, 3], dtype=np.uint32)
G = mg.Graph.from_edges(src, dst, weights=None, directed=True,
                        num_vertices=4, ids="auto")
G.num_vertices, G.num_edges, G.external_ids

# Flagship: batched personalized PageRank with top-k.
seeds = np.array([0, 2], dtype=np.uint32)
seed_weights = np.ones(2, dtype=np.float32)
seed_offsets = np.array([0, 1, 2], dtype=np.uint64)
ids, scores = mg.ppr_topk(G, seeds, seed_weights, seed_offsets,
                          k=4, alpha=0.85, tol=1e-6, max_iter=50)
# ids: int32 (B, k) user indices (-1 padding), scores: float32 (B, k)

pr = mg.pagerank(G, alpha=0.85, tol=1e-6, max_iter=100, personalization=None)
dist, parent = mg.bfs(G, sources=[0], direction="out")
vs, es = mg.k_hop(G, seeds=[0], k=2, direction="both",
                  max_vertices=None, max_edges=None)
comp = mg.experimental.wcc(G)

mg.set_execution(mode="auto")   # "auto" | "gpu" | "cpu" (per-operation planner)
mg.last_run_info()              # {"op", "path": "gpu"|"cpu", "iterations", "ms"}
```

Development setup, native tests, wheel validation, C ABI installation, and
benchmark rules are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## Identity model

* **External IDs** — whatever you pass to `from_edges` (int32/int64/str).
* **User indices** — dense `0..V-1`; `np.unique` order of the external IDs
  (identity when your IDs are already `0..V-1` ints, which is also the only
  mode where `num_vertices=` may add isolated tail vertices). All algorithm
  inputs (seeds, sources) and outputs (vectors, id arrays) use user indices;
  `G.external_ids[i]` recovers the original ID and `G.index_of(ids)` maps
  the other way.
* Internally vertices are renumbered once (canonical order, out-degree
  descending) for gather locality; this is invisible at the API.

## Semantics (v0.1, tested)

* **Input policy**: duplicate edges kept (parallel edges; PageRank counts
  them), self-loops kept, non-finite/negative weights rejected at build,
  `num_vertices=` accepted for isolated vertices. Undirected graphs are
  symmetrized internally; `num_edges` reports your input count.
* **PageRank**: NetworkX-compatible iteration; weighted graphs normalize by
  the sum of outgoing weights; dangling mass is redistributed by the
  teleport/personalization vector every iteration (on the GPU, on-GPU).
  Convergence is `L1(r_new - r_old) < V * tol`, audited in fp64 every 5
  iterations (`MG_PR_AUDIT_INTERVAL`; the default is the measured cadence-
  sweep winner), so iteration counts land on audit boundaries. `max_iter`
  is a budget: hitting it returns the current iterate (NetworkX raises
  instead — documented divergence). Deterministic at fixed iteration count.
* **ppr_topk**: per-query seed weights are normalized (duplicates summed);
  queries are packed into tiles of 8 lanes sharing one `col_indices` stream;
  converged queries are frozen at audit boundaries. Top-k ties break by
  ascending user index; `k > V` pads with `id=-1, score=0`. Batched results
  are bit-identical to running the same queries one at a time (same path).
* **BFS**: multi-source, `direction="out"|"in"|"both"`; `dist=-1`
  unreachable; `parent=-1` for sources/unreachable. Parent choice among
  equal-depth candidates is nondeterministic on the GPU path (validated
  structurally in tests). Limitation: `direction="both"` on **directed**
  graphs executes on the CPU path in v0.1 (telemetry reports it honestly);
  undirected graphs and `out`/`in` run GPU-resident.
* **k_hop**: vertices within ≤ k hops of any seed plus induced edges (both
  endpoints reached), returned as original input edge ids; outputs sorted
  ascending. `max_vertices`/`max_edges` make agent-side latency predictable;
  cap-bounded runs execute on the (deterministic) CPU path in v0.1.
  `as_graph=True` materializes the subgraph.
* **WCC** (`metal_graph.experimental`): directed edges treated as
  undirected; canonical-partition output (component ids numbered by first
  occurrence in user-index order).
* Reserved kwargs `edge_types=` / `time_range=` raise `NotImplementedError`
  naming v0.2, so callers can write forward-compatible code today.

## Execution planner

`mode="auto"` picks CPU or GPU **once per operation**: GPU when a Metal
device exists and the stored edge count is ≥ `MG_E_GPU_MIN` (default 1M,
placeholder until the M4 sweep). `mode="gpu"` errors without a device;
`MG_REQUIRE_GPU=1` makes device absence a hard failure everywhere (CI).
`MG_FORCE_CPU=1` hides the GPU. Telemetry from `mg.last_run_info()` reports
the path actually executed — CI asserts it, never silently falls back.

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `MG_E_GPU_MIN` | 1000000 | auto-planner GPU threshold (stored edges) |
| `MG_PR_AUDIT_INTERVAL` | 5 | iterations per GPU command batch / fp64 audit |
| `MG_BFS_LEVELS_PER_BATCH` | 16 | BFS levels encoded per command buffer |
| `MG_WCC_ROUNDS_PER_BATCH` | 4 | WCC hook+jump rounds per command buffer |
| `MG_GPU_TOPK` | 1 | GPU radix-select for `ppr_topk` (0 = CPU oracle) |
| `MG_BFS_BOTTOMUP` | 1 | enable direction-optimizing bottom-up switch |
| `MG_REQUIRE_GPU` | 0 | hard-fail when no Metal device |
| `MG_FORCE_CPU` | 0 | disable the GPU entirely |
| `MG_THREADS` | all cores | CPU-path thread count |
| `MG_METALLIB` | embedded | override path to a .metallib (dev JIT loop) |

## Layout

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

## Benchmarks

`bench/run.py --suite v01` (physical hardware only; see `bench/README.md`).
Reporting decomposes build / transpose / pipeline compile / warm kernels /
iteration-count and audit-cadence metadata / top-k / aggregate Python
boundary overhead, with median+p95 over ≥20 runs. It does not claim a
separate convergence-audit timer. Nothing in the README table is asserted —
targets come from measurements.

## First measurements

Apple M4 Max, macOS 26.2, 2026-07-28, `bench/run.py --suite smoke --runs 20`
(medians; canonical [JSON](bench/results/bench-20260728T205737Z.json) and
[rendered table](bench/results/bench-20260728T205737Z.md)). All ratios are
within-run; treat absolute numbers as workstation medians, not tuned-lab
figures.

| graph | operation | metal-graph | gate baseline (igraph/rustworkx) | ratio |
|---|---|---:|---:|---:|
| RMAT-18 (V=262k, E=4.2M) | PageRank warm (5 iters, 137 GB/s) | 1.5 ms | rustworkx 296 · igraph 365 | **204×** |
| RMAT-18 | `ppr_topk` B=16, k=64 | 6.9 ms | igraph query loop 6 292 | **908×** |
| RMAT-18 | BFS single-source | 1.4 ms | igraph 23.7 | **18×** |
| RMAT-18 | WCC | 3.0 ms | igraph 120.0 | **40×** |
| KG-shape (V=100k, E=2M, weighted) | PageRank warm (15 iters, 114 GB/s) | 2.4 ms | igraph 30.6 | **13×** |
| KG-shape | `ppr_topk` B=16, k=64 (20 iters) | 10.3 ms | igraph query loop 571 | **55×** |
| KG-shape | BFS single-source (tiny component) | 0.9 ms | rustworkx 0.05 | 0.06× |
| KG-shape | WCC | 3.9 ms | igraph 13.4 | **3.5×** |

Three performance passes closed the first run's gaps:

1. **Huge-bin edge tiling** (plan-§11 CSR-tiling fallback): the synthetic KG
   puts 44% of all in-edges on ONE vertex, which serialized on a single
   threadgroup. Degree ≥ 16 384 vertices now decompose into fixed 16k-edge
   tiles across many threadgroups (deterministic two-pass gather; tiled WCC
   hook). KG PageRank: 26.4 → 2.2 ms (11.9 → 134 GB/s); WCC: 16.8 → 3.8 ms.
2. **GPU radix-select top-k** (the plan-M3 item, no longer deferred): a
   per-lane MSB-first histogram refinement finds the k-th-largest score's
   bit pattern, a compaction pass collects the ≥-threshold candidates, and
   a tiny exact host sort applies the (score desc, user asc) tie rule —
   bit-identical to the CPU oracle, which stays behind the same API
   (`MG_GPU_TOPK=0`, automatic fallback on degenerate tie floods and tiny
   graphs). KG batch: 13.7 → 10.1 ms; selection line item 3.4 → 0.8 ms.

3. **Audit-cadence tuning**: convergence is observable only at audit
   boundaries, so boundary spacing controls overshoot. The default is 5;
   the canonical artifact records the resulting iteration counts, while
   `MG_PR_AUDIT_INTERVAL` remains available for repeatable local sweeps.

Gate assessment (plan §8, honest):

* **PPR relative-speed sub-gate**: ≥5× the igraph per-query loop — **met**
  on both shapes (55× / 908×; caveat: igraph uses PRPACK, an exact solver
  without matching iteration-count control). The **amortized ≤0.7
  ms/query target is met** (0.65 ms KG, 0.43 ms RMAT-18). The complete PPR
  gate remains **open**: the canonical KG batch is 10.34 ms against the
  ≤10 ms target, and identical-iteration comparison is unavailable with
  this igraph solver. Certification requires a repeat with a comparable
  baseline on a dedicated physical runner.
* **Primary ≥2× gate**: met on **7 of 8** workload×algorithm cells
  (3.5×–908×). The one standing failure: BFS from a tiny reachable component
  loses to CPU baselines — the per-operation planner keys on graph size, but
  BFS cost tracks *traversal* size, which no per-op planner can know in
  advance (v0.2: first-frontier fallback).

## Status

v0.1 scope only: no mutation (immutable snapshots), no Louvain/Leiden (v0.3),
no similarity/sampling, fp32 + int32 ids (`V, E < 2^31`). Batched PPR on GPU
needs the CSR plus a fixed 8-lane tile state resident (~100 bytes/vertex);
graphs exceeding `recommendedMaxWorkingSetSize` raise a descriptive error —
batch size does not change the footprint, `mode="cpu"` does. Top-k runs a
GPU radix-select (`select.metal`); the threaded CPU selection stays behind
the same API as the correctness oracle (`MG_GPU_TOPK=0`, plus automatic
fallback for tiny graphs, `k ≥ V`, and degenerate tie floods) — tests assert
the two are bit-identical.
