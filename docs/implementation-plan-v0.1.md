# metal-graph — v0.1 Implementation Plan (rev B)

**A Metal-native graph-analytics engine for Apple Silicon. First release scoped to the few algorithms that accelerate the most.**

*Prepared 2026-07-10; rev B 2026-07-10 incorporating external technical
review. This retained plan records the design inputs; the implemented and
tested behavior is documented in the repository README.*

**Naming record** — project/repo/PyPI: `metal-graph` · Python import: `metal_graph` · NetworkX backend name: `"metal"` (shipped later as `nx-metal-graph`) · C ABI prefix: `mg_` · MSL kernel prefix: `mg_`.

**Rev B changelog** (disposition of the external review): accepted all seven must-fix technical items — per-iteration GPU dangling-mass handling, corrected MSL sketches (attributes, `simd_vote`, atomic bitmap access, no hard-coded SIMD width), per-orientation degree worklists replacing permutation-based segmentation, GPU-resident BFS via indirect dispatch instead of per-level CPU switching, scan + indirect-args promoted to M0 primitives, tracked `MTLBuffer`s before heaps with explicit sync protocol, tightened WCC semantics. Accepted the product reshape: **batched PPR with top-k is now the flagship API**; k-hop gains `direction`/`max_vertices`/`max_edges` and returns views; WCC is repositioned as the atomics canary shipping under `experimental`. Accepted the benchmark/CI overhaul (workload-level ship gate, timing decomposition, concurrent-LLM contention scenario, GPU-required CI, measured-not-asserted build target, staffing honesty). Consciously deferred, with API reserved: typed/temporal k-hop filters (needs the edge-property/mask subsystem — v0.2) and fully-async MLX device-resident results (v0.2); GPU top-k is the M3 target with a semantically identical CPU selection fallback behind the same API so the contract never depends on which processor selects.

---

## 1. Scope: three algorithm families, chosen for maximum acceleration

v0.1 ships:

1. **Batched personalized PageRank with top-k** (plus full-vector PageRank; Katz as a freebie if green)
2. **BFS** (direction-optimizing, GPU-resident) with **bounded k-hop / ego-subgraph extraction** built on it
3. **WCC** — built early as the atomics correctness canary, shipped as `metal_graph.experimental.wcc`

The selection criterion is *acceleration ceiling on this hardware*: these are pure memory-bandwidth-bound workloads whose inner loops map 1:1 onto Metal capabilities with **no sorts, no hash maps, and no fp64 in the iteration path** — avoiding operations that Metal handles poorly while leaning on unified-memory bandwidth that CPUs on the same machine cannot touch on random access. They need only 32-bit integer atomics, `atomic<float>`, SIMD-group reductions/prefix sums, and indirect dispatch — all present on every Apple Silicon Mac (Metal 3, Apple7+). They also map directly to agent workloads: batched PPR is the HippoRAG/aider query path, k-hop is Graphiti/LightRAG retrieval, and WCC backs entity dedup.

The batching emphasis is deliberate product design, not just engineering convenience: agent loops issue many PPR queries per task, and a batch of B queries shares one streaming pass over `col_indices` (≈4 B/edge once + 4·B B/edge gathered), so per-query cost drops well below the single-query price while fixed dispatch overhead amortizes to noise. Returning `(B × k)` top-k IDs/scores instead of `B × V` vectors keeps the Python boundary cheap at any batch size.

Deliberately **out** of v0.1: Louvain/Leiden (v0.3 centerpiece), similarity/triangles (v0.2), label propagation and core number (v0.2), typed/temporal edge filters (v0.2 — kwargs reserved now), SSSP, sampling/walks, any GPU sort, fp64, graph mutation (v0.1 graphs are **immutable snapshots**; a CPU delta overlay is a v0.2 design item), multi-GPU, Linux/Windows, full NetworkX backend (preview covering v0.1 calls is an M4 stretch).

## 2. Day-1 decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language / GPU API | C++20 + **metal-cpp**, MSL 3.x kernels; no Objective-C | Full control; metal-cpp is Apple-maintained |
| OS / hardware floor | macOS 14+, Apple7 GPU family+ (M1 and up) | Metal 3 baseline: `atomic<float>`, SIMD-group ops, indirect dispatch everywhere |
| Precision | fp32 values, int32 vertex/edge IDs (`E < 2^31` per graph); **all iteration-critical scalars (dangling mass, residuals) reduced on GPU every iteration**; CPU fp64 used only as a *periodic convergence audit* between batches | Correctness fix from review; removes CPU from the iteration loop |
| Buffer management | **Individually tracked `MTLBuffer`s (default hazard tracking) in v0.1**; `MTLHeap` (untracked — manual fences) deferred to v0.2 behind the same allocator interface | Correctness first; heaps are an optimization with a real hazard-protocol cost |
| CPU↔GPU handoff | Shared storage removes copies, **not** ordering: CPU reads results only after command-buffer completion (`waitUntilCompleted`/shared events); handoffs occur only at defined operation boundaries | Per Apple's shared-storage semantics; no "free access" language |
| Vertex identity | One **canonical internal uint32 ID space**; external IDs may be int32/int64/str (host-side map); segmentation never re-permutes IDs per orientation | Review item 3 |
| Python bindings | **nanobind**; NumPy zero-copy views + DLPack; results synchronous in v0.1 (async/device-resident path reserved for MLX interop in v0.2) | Honest about the sync point NumPy implies |
| Build / packaging | CMake; `.metal` → `.metallib` at build (runtime JIT fallback for dev); cibuildwheel macOS-arm64 wheels, Python 3.10–3.13 | Standard |
| License | Apache-2.0 | Ecosystem norm |
| Input policy (defined, tested) | duplicate edges kept (parallel edges; PageRank counts them), self-loops kept (documented per-algorithm semantics), NaN/negative-where-invalid weights rejected at build, `num_vertices=` accepted for isolated vertices | Review "define behavior" item |

## 3. The API surface (all of it)

```python
import numpy as np, metal_graph as mg

# Build once — immutable snapshot; CPU threaded build writes straight into shared MTLBuffers
G = mg.Graph.from_edges(src, dst, weights=None, directed=True,
                        num_vertices=None,        # for isolated vertices
                        ids="auto")               # int32 | int64 | str external IDs; canonical uint32 inside
G.num_vertices, G.num_edges, G.external_ids      # introspection

# Flagship: batched personalized PageRank with top-k.
# B queries packed CSR-style: seeds/weights + offsets delimiting each query.
ids, scores = mg.ppr_topk(G, seeds, seed_weights, seed_offsets,
                          k=64, alpha=0.85, tol=1e-6, max_iter=50)
# -> ids: int32 (B, k), scores: float32 (B, k); never materializes B×V at the Python boundary

pr  = mg.pagerank(G, alpha=0.85, tol=1e-6, max_iter=100, personalization=None)  # full vector
dist, parent = mg.bfs(G, sources=[42], direction="out")                          # multi-source
vs, es = mg.k_hop(G, seeds, k=2, direction="both",
                  max_vertices=None, max_edges=None)     # bounded; returns index arrays (views);
                                                         # as_graph=True to materialize a subgraph
comp = mg.experimental.wcc(G)                            # canonical-partition semantics

mg.set_execution(mode="auto")   # "auto" | "gpu" | "cpu" — per-OPERATION planner (see §6)
```

`edge_types=` and `time_range=` kwargs are **reserved** (raise `NotImplementedError` naming v0.2) so callers can write forward-compatible code. The C ABI (`include/mg.h`) mirrors this 1:1 (`mg_graph_create_from_coo`, `mg_ppr_topk`, `mg_pagerank`, `mg_bfs`, `mg_khop`, `mg_wcc`).

## 4. Repository layout

```
metal-graph/
├── CMakeLists.txt  LICENSE  README.md
├── include/mg.h                      # C ABI (stable surface)
├── src/
│   ├── runtime/                      # device, queues, pipeline cache, tracked-buffer allocator,
│   │   ...                          #  command-batch helper, per-operation GPU/CPU planner
│   ├── graph/                        # csr.cpp, build.cpp (CPU renumber+CSR+worklists),
│   │   ...                          #  worklists.cpp (per-orientation degree bins), extid_map.cpp
│   ├── kernels/                      # MSL: mg_common.h (Kahan, bitmap ops, simd helpers)
│   │   ├── spmv.metal               #  per-vertex gather: tg/sg/thread variants ×{1,B} batch
│   │   ├── scan.metal               #  two-pass device exclusive scan (M0 primitive)
│   │   ├── frontier.metal           #  degree-binned expand, bottom-up probe, compaction,
│   │   │                            #   GPU-written indirect dispatch args
│   │   ├── wcc.metal                #  hook (both endpoints) + pointer-jump + change flag
│   │   ├── select.metal             #  radix-select top-k per query (M3)
│   │   └── reduce.metal             #  dangling reduce, residual partials, fill/iota
│   ├── engines/                      # host drivers: spmv_engine, frontier_engine, reduce, select
│   ├── algos/                        # pagerank.cpp (incl. ppr_topk)  bfs.cpp  wcc.cpp  khop.cpp
│   │   └── cpu/                      # threaded CPU implementations: oracles + planner fallback
│   └── c_api/
├── python/metal_graph/               # nanobind module + sugar (experimental/, planner config)
├── tests/                            # pytest golden tests vs NetworkX/igraph; C++ unit tests
├── bench/                            # harness, dataset fetch, contention scenario, reports
└── .github/workflows/ci.yml          # hosted arm64 runner (GPU REQUIRED) + self-hosted perf job
```

## 5. Data structures

**CSR container** (per orientation; PageRank pulls along incoming edges, so directed graphs materialize the transposed CSR on demand and cache both — documented 2× edge memory when both orientations are used):

| Buffer | Type | Size | Notes |
|---|---|---|---|
| `row_offsets` | uint32 | V+1 | canonical vertex order (out-degree-descending for edge locality) |
| `col_indices` | uint32 | E | neighbor IDs (canonical) |
| `weights` | float32 | E or 0 | optional; BFS/WCC never touch it |
| `worklists[orient]` | uint32 ×3 | Σ=V_nonzero | **per-orientation** high/mid/low degree-bin vertex lists + counts |
| `zero_fill_list[orient]` | uint32 | — | vertices with zero degree *in that orientation* (must still be written each PR iteration) |
| `dangling_list` | uint32 | — | zero-out-degree vertices (dangling mass reduction) |
| `out_weight_sum` | float32 | V | weighted-graph normalization (Σ outgoing weights, **not** outdegree) |
| `extid_map` | host | V | canonical→external; hash map external→canonical (int64/str) |
| scratch | — | — | frontier ×2, visited bitmap (V/32, bound as `atomic_uint`), partials, indirect-args |

**Why worklists, not permutation ranges** (review item 3): a single degree-sorted ID permutation cannot make high/mid/low ranges contiguous for *both* orientations of a directed graph — in-degree and out-degree orders differ, and adversarial graphs make them disjoint. So segmentation is decoupled from identity: canonical IDs are fixed once, and each orientation carries three vertex worklists binned by *that orientation's* degree (initial thresholds 1024/32, retuned in M4). Kernels read `v = worklist[wi]` — one extra 4-byte streaming read per vertex, negligible against 8 B/edge. A hypersparse/DCSR fourth segment remains out of scope because it primarily serves multi-GPU partitioning.

**Build pipeline (CPU, threaded)**: external-ID map → degree histograms (both directions) → canonical order by out-degree (stable) → counting-sort edges into CSR → emit worklists, zero-fill lists, dangling list, `out_weight_sum`. All buffers allocated shared-mode up front — the build *is* the upload. Throughput is **measured, not asserted**: the harness times build separately (§10), and provisional estimates never become release claims without a physical run.

## 6. Kernels and host patterns

Kernel inventory (~18 entry points; several are segment/batch template expansions):

| Kernel | Mapping | Key Metal features |
|---|---|---|
| `mg_pr_gather_{tg,sg,thread}[_b]` | per-vertex pull reduce; worklist-driven; `_b` = B-query batch variant | `simd_sum`, threadgroup reduce; batch variant streams `col_indices` once for B accumulators |
| `mg_pr_prepare` | rank→contrib (÷ `out_weight_sum`), residual partials | fused |
| `mg_pr_zero_fill` | write teleport+dangling share to zero-in-degree vertices | review fix: these must be written every iteration |
| `mg_dangling_reduce` | Σ rank over `dangling_list` → device scalar | feeds the *next* iteration on-GPU |
| `mg_exscan_{partial,apply}` | two-pass device exclusive scan | M0 primitive (frontier offsets, compaction) |
| `mg_frontier_bin` | split frontier by degree into 3 bins; write per-bin **indirect dispatch args** | GPU-generated `MTLDispatchThreadgroupsIndirectArguments` |
| `mg_bfs_expand_{tg,sg,thread}` | degree-binned top-down expansion | `simd_prefix_exclusive_sum` compaction; atomic bitmap claim |
| `mg_bfs_bottomup_probe` | per-unvisited-vertex parent scan, early exit | shares spmv skeleton |
| `mg_khop_gather` | induced vertex/edge extraction with `max_*` caps | bitmap tests + scan compaction |
| `mg_wcc_hook` / `mg_wcc_jump` | per-edge min-hook **both endpoints** / pointer jumping | `atomic_fetch_min_explicit`; ping-pong label buffers |
| `mg_topk_select` | per-query radix-select over fp32-ordered bits | M3; CPU selection fallback behind same API |
| `mg_fill`, `mg_iota`, `mg_reduce_finalize` | utilities | — |

Corrected sketches (rev B — these follow MSL: no invented attributes, `simd_vote` avoided in favor of prefix sums, atomic bitmap accessed only through `atomic_uint`, SIMD width taken from the attribute):

```metal
// spmv.metal — mid bin: one SIMD-group per worklist vertex
kernel void mg_pr_gather_sg(
    device const uint*   worklist   [[buffer(0)]],   // mid-degree vertices, this orientation
    device const uint*   row_off    [[buffer(1)]],
    device const uint*   col_idx    [[buffer(2)]],
    device const float*  contrib    [[buffer(3)]],   // rank / out_weight_sum, prepared
    device float*        rank_next  [[buffer(4)]],
    device const float*  iter_scalars [[buffer(5)]], // [0] = dangling mass (GPU-reduced last iter)
    device const float*  pvec         [[buffer(6)]], // normalized teleport vector (uniform or personalization)
    constant MGPrParams& p            [[buffer(7)]],
    uint   tg_id      [[threadgroup_position_in_grid]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]])
{
    const uint wi = tg_id * sgs_per_tg + sg_in_tg;          // global SIMD-group index, derived
    if (wi >= p.mid_count) return;
    const uint v = worklist[wi];
    float acc = 0.0f;
    for (uint e = row_off[v] + lane; e < row_off[v + 1]; e += simd_sz)
        acc += contrib[col_idx[e]];                          // the bandwidth-bound gather
    acc = simd_sum(acc);
    if (lane == 0)   // rank = (1-α)·p[v] + α·(gather + D·p[v]); dangling mass D redistributed by pvec
        rank_next[v] = p.one_minus_alpha * pvec[v]
                     + p.alpha * (acc + iter_scalars[0] * pvec[v]);
}
```

```metal
// frontier.metal — top-down expansion core: exactly-once claim + SIMD compaction
device atomic_uint* w   = &visited[nbr >> 5];
const uint          bit = 1u << (nbr & 31u);
bool discovered = false;
if ((atomic_load_explicit(w, memory_order_relaxed) & bit) == 0u)          // cheap peek (atomic)
    discovered = (atomic_fetch_or_explicit(w, bit, memory_order_relaxed) & bit) == 0u;

const uint rank_in_sg = simd_prefix_exclusive_sum(uint(discovered));
const uint sg_count   = simd_sum(uint(discovered));
uint base = 0u;
if (lane == 0u && sg_count != 0u)
    base = atomic_fetch_add_explicit(next_size, sg_count, memory_order_relaxed);
base = simd_broadcast_first(base);
if (discovered) {
    next_frontier[base + rank_in_sg] = nbr;
    parent[nbr] = v;                                          // depth written by level constant
}
```

Host-side patterns (rev B):

**GPU-resident iteration, CPU at boundaries only.** PageRank encodes K iterations (default 8) into one command buffer; *every* per-iteration dependency — dangling-mass reduction, zero-in-degree fill, residual partials — is a GPU dispatch inside the batch, so iteration N+1 consumes iteration N's scalars from device buffers with no CPU involvement. Between batches the CPU (a) sums the residual partials in fp64 as the *convergence audit* and decides continue/stop, and (b) optionally re-audits dangling mass. BFS runs whole levels GPU-side: `mg_frontier_bin` writes per-bin indirect dispatch arguments and the next-level flag; the host encodes K levels of `dispatchThreadgroups(indirectBuffer:)` blindly and checks the done-flag after the batch completes. No indirect *command buffers* needed in v0.1 — indirect dispatch args cover it.

**Per-operation planner (not per-level).** `mode="auto"` chooses CPU or GPU **once per operation**, from graph size, batch size, and orientation residency (initial rule: `E < E_gpu_min` → CPU path; `E_gpu_min` measured in M4, seeded ~1M). The per-*level* BFS CPU/GPU flip from rev A conflicted with batched encoding (review item 4) and is demoted to an experiment (`MG_EXPERIMENTAL_LEVEL_HYBRID=1`) gated on measured sync cost; the direction-optimizing top-down/bottom-up switch remains GPU-side, driven by GPU-computed frontier-edge counts and standard α/β heuristics (β=24 as the seed constant).

**Synchronization protocol.** v0.1: tracked `MTLBuffer`s + one `MTLCommandQueue`; results surface to Python only after `waitUntilCompleted` at operation end; no CPU touches buffers mid-flight except designated readback buffers after completion. Heaps/fences and finer-grained overlap are v0.2 optimizations behind the allocator interface.

## 7. Per-algorithm design notes

**PageRank / batched PPR (rev B).** Pull iteration, all-GPU: `mg_pr_prepare` (contrib = rank ÷ `out_weight_sum`; residual partials) → `mg_pr_gather_*` over the three in-orientation worklists → `mg_pr_zero_fill` over zero-in-degree vertices (teleport + α·dangling share — they were previously, wrongly, excluded from dispatch) → `mg_dangling_reduce` into `iter_scalars` for the next iteration. Personalization: dense teleport vector per query; dangling mass redistributed by the personalization vector (standard choice, documented). Batched variant: B queries share the `col_indices` stream with B fp32 accumulators (B capped by a memory check; B≤32 typical) — rank state is `V × B`, top-k selection collapses it before Python. Convergence per query (converged queries masked out of further batches). Tests: dangling-heavy graphs, isolated vertices, weighted (normalization by weight-sum verified against NetworkX weighted PageRank), personalized with disjoint seed sets, B=1 ≡ unbatched equivalence.

**BFS / k-hop.** Direction-optimizing, degree-binned top-down (`mg_frontier_bin` + three expand variants — the review is right that flat "thread per frontier-edge chunk" was underspecified; binning *is* the load-balancing strategy, with the device scan available for exact edge partitioning if bin skew demands it), bottom-up probe when GPU-computed frontier-edge count crosses the α threshold. Multi-source native. `k_hop`: BFS truncated at depth k with `max_vertices`/`max_edges` caps enforced at compaction time (caps make agent-side latency predictable); returns index-array views, `as_graph=True` materializes. `direction="in"|"out"|"both"` selects orientation(s) — "both" unions per-level expansions over CSR and CSC.

**WCC (experimental).** Directed edges treated as undirected by hooking **both endpoints** from the single stored orientation (`atomic_min(label[u], ...)` *and* `atomic_min(label[v], ...)` per edge — no CSC needed). FastSV-style rounds: hook + pointer-jump on ping-pong label buffers + device change-flag, batched K rounds per command buffer. Tests compare **canonicalized partitions** (relabel by first-seen order) against igraph, never raw representative IDs. Ships under `experimental` unless release gates are comfortably green; it exists in M1–M2 regardless as the atomics/correctness canary.

## 8. Milestones — 12 weeks, staffed honestly

**Staffing statement (review item):** the 12-week schedule assumes **~2 GPU/systems engineers plus part-time Python/eval support**. A solo build is viable but should plan **~18–20 weeks** at the same scope, or drop the M3 GPU top-k and the M4 stretch items to hold 12.

**M0 — kill-risk spike (weeks 1–2).** Throwaway harness measuring: (a) achieved bandwidth of the raw gather loop on LiveJournal-shape data (gate below); (b) per-dispatch overhead + batched-command-buffer amortization curve; (c) atomic bitmap `fetch_or`/`fetch_min` throughput under contention; (d) **two-pass device scan + GPU-written indirect-args round-trip** (promoted to M0 per review — frontier machinery depends on both); (e) a first **contention probe**: the gather kernel running while an MLX decode loop generates tokens, both directions of interference recorded. **Gate: ≥40% of peak bandwidth on the gather kernel** (M4 Pro: ≥109 GB/s ⇒ ≤5.5 ms/LiveJournal-iteration). Miss → publish findings and reassess; hit → numbers seed the M3/M4 targets.

**M1 — engine + correct PageRank (weeks 3–5).** Runtime (tracked buffers, queue, pipeline cache, command-batch helper), build pipeline (external IDs, worklists, dangling/zero lists, measured build throughput), spmv engine, reduce/scan utilities; **single-query PageRank/PPR fully correct**: per-iteration GPU dangling handling, zero-in-degree fill, weighted normalization — validated against NetworkX/igraph on the §9 matrix including the adversarial in/out-degree-disjoint graph. Exit: LiveJournal PageRank within 1.5× of the M0 spike number, end-to-end from Python.

**M2 — batched PPR + frontier tier (weeks 6–8).** Batch-variant gather + per-query convergence masking; BFS (both directions, GPU-resident indirect dispatch, α/β switch), k-hop with caps and directions, WCC canary; per-operation planner with `E_gpu_min` placeholder. Exit: BFS parent/depth structural validation and canonicalized WCC equivalence vs igraph across the matrix; batched PPR B∈{1,4,16,64} bit-equivalent to sequential single queries (same iteration counts).

**M3 — top-k, packaging, demos (weeks 9–10).** `mg_topk_select` radix-select (CPU selection fallback stays behind the same API and is the correctness oracle); nanobind wheels; docs; the two demos that are the marketing: `examples/hipporag_ppr.py` (batched PPR vs igraph loop) and `examples/repo_map.py` (aider-style per-message PPR vs NetworkX). Exit: the parameterized PPR gate below, green.

**M4 — tune, contend, release (weeks 11–12).** Threshold sweeps (worklist bin boundaries, batch K, `E_gpu_min`, α/β); full bench suite incl. the contention scenario on M4 Pro + M4 Max (+ borrowed M3 Ultra); `metal-graph 0.1.0` on PyPI. Stretch: Katz; `nx-metal-graph` preview (`pagerank`/`bfs_tree`/`connected_components` only).

**Release gates for v0.1.0** (rev B — workload-level, per review):

1. **Primary ship gate:** on every *eligible* workload (resident graph above the planner threshold), metal-graph is **≥2× faster than the fastest maintained CPU baseline at identical semantics** (igraph or rustworkx, whichever wins per algorithm — NetworkX numbers are reported as context, never as the gate); the planner's automatic fallback shows **no material regression (<10%) on small graphs**; **at least one real agent workflow improves end-to-end** (the HippoRAG-style demo: retrieval latency at equal quality); and the contention run stays within a **declared unified-memory + GPU-time budget** alongside an active LLM decode, published in the README.
2. **Engineering targets** (expected, not ship-blocking, from the original roofline model): PageRank ≤6 ms/iteration on soc-LiveJournal1 on M4 Pro (≥36% BW; stretch ≤4 ms); 30-iteration run ≥15× igraph. BFS LiveJournal ≤50 ms (stretch ≤20 ms). WCC ≤5× BFS time.
3. **The PPR gate, fully parameterized** (review item): graph V=100k/E=2M (synthetic HippoRAG-shape KG, fp32 weights), **batch B=16, k=64, α=0.85, tol=1e-6, max 50 iterations, warm resident graph, top-k included, timed from Python call to NumPy result**: **≤10 ms for the batch** (≤0.7 ms/query amortized) and **≥5× the igraph per-query loop** at identical iteration counts. Cold-start (first call, pipeline compile) reported separately.

## 9. Testing and CI

Golden matrix vs NetworkX + igraph: {directed, undirected} × {weighted, unweighted} × {LiveJournal sample, RMAT-18, star/path/ring/clique, empty/singleton, disconnected forests, multi-edges/self-loops, **dangling-heavy**, **isolated-vertex**, **high-in/high-out-disjoint adversarial**}. Semantics tests for the §2 input policy. Property tests: external-ID round-trip, worklist invariants (partition + degree-class membership per orientation), bitmap idempotence, batched-vs-sequential PPR equivalence. Determinism documented per algorithm (PageRank deterministic at fixed iterations; BFS parents tie-nondeterministic, validated structurally; WCC canonicalized).

CI (review item — **no silent CPU fallback**): all GPU tests run with `MG_REQUIRE_GPU=1`, which makes Metal-device absence a hard failure, and assert the executed path via planner telemetry. Hosted `macos-15` arm64 runners (M1-class VMs — Metal present, performance meaningless) run build + correctness. The physical-runner job is manual and remains disabled until a suitably isolated self-hosted Apple Silicon runner is provisioned. Perf numbers only ever come from physical hardware.

## 10. Benchmarks shipped in-repo

`bench/run.py --suite v01` fetches soc-LiveJournal1 (69M) and com-orkut (117M), generates RMAT-22/24 and the HippoRAG-shape KG, and runs metal-graph vs igraph, rustworkx, SciPy (`sparse.csgraph`), and NetworkX (context only) per algorithm. Per review, reporting is decomposed and honest: **build, transpose, pipeline compile, warm kernel execution, convergence iteration/cadence metadata, top-k selection, and aggregate Python-boundary overhead are separate line items**; the fp64 audit is not separately timed. Every cell reports **median and p95 over ≥20 runs, cold and warm**, peak memory (RSS + `recommendedMaxWorkingSetSize` headroom), achieved GB/s (traffic model ÷ warm kernel time), and energy (`powermetrics` at 100 ms sampling, idle-subtracted, methodology in `bench/ENERGY.md`); chip model, core counts, macOS, compiler, and metal-graph versions recorded in the JSON. The **contention scenario** runs the PPR batch benchmark while `mlx_lm` decodes a fixed prompt stream, reporting graph-latency inflation *and* tokens/s impact — because an isolated GPU speedup that starves the LLM is a net loss for an agent, and nobody else publishes this number. The README table is regenerated only from physical-hardware runs.

## 11. Risks specific to this slice

Carried and mitigated: **gather efficiency** below 40% on power-law graphs (M0 gate exists to surface it before architecture; fallback: CSR tiling / degree-then-BFS vertex ordering); **bottom-up BFS underperforming** Apple's cache hierarchy (fallback: top-down + planner-level CPU path still meets gates on scale-free graphs); **GPU top-k slipping** (CPU selection behind the same API keeps the contract; gate is end-to-end latency, top-k line item shows where time went); **batched-PPR memory** (`V × B` fp32 state; B capped by a working-set check with a documented error); **`std::execution::par` gaps in Apple Clang** (thread-pool counting sort is the plan of record); **32-bit ID ceiling** (documented; v64 a v0.2+ template axis); **scope pressure from rev B additions** (pre-agreed drop order: nx preview → Katz → GPU top-k → WCC-from-experimental; the three core families + gates never drop).

## 12. After v0.1 (preview, not commitment)

v0.2 = label propagation, core number, Jaccard/overlap top-k (first intersection kernels), typed/temporal k-hop filters on the edge-property/mask subsystem, `MTLHeap` allocator + finer overlap, MLX device-resident/async results, full `nx-metal-graph`, CPU delta-overlay design for incremental graphs. v0.3 = the index path: GPU hash map + Louvain → Leiden with CPU-hybrid coarsening, targeting the GraphRAG `hierarchical_leiden` swap. v0.4 = sampling/walks + mlx-graphs integration. Discipline unchanged: ship only what beats the best maintained CPU option on the same machine at identical semantics by a margin worth the dependency.
