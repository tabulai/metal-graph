# Apple Metal HITS benchmark — 2026-09-04

## Technical summary

The new Metal HITS implementation completed the two measured graphs in
**1.518–2.337 ms per warm call** on an Apple M4 Max. Against the matching
threaded fp64 implementation, Metal was **6.75–8.06× faster**. Against
rustworkx—the fastest completed installed public CPU HITS baseline—it
was **32.99× faster** on the 2.0M-edge knowledge-graph shape and **217.09×
faster** on the 4.19M-edge RMAT graph.

The most conservative independent comparison is the prebuilt SciPy CSR
recurrence in the benchmark harness. It performs the same updates, uses the
same audit cadence, and stops at the same absolute L1 target. Metal was
**2.40× faster** on the duplicate-heavy knowledge graph and **14.24× faster**
on RMAT-18. Every result admitted to a ratio passed the recorded pointwise and
aggregate agreement checks against the fp64 reference.

These results establish an Apple Metal implementation, not the first GPU HITS
implementation generally. NVIDIA cuGraph already provides a CUDA HITS API and
was the design reference for `max_iter=100`, `tol=1e-5`, L1-normalized output,
and ignored edge weights. As of 2026-09-04, a targeted GitHub and general-web
search for `HITS Apple Metal`, `hub authority Metal GPU`, and `HITS MPS` did
not identify a public Metal-specific HITS implementation. This is a scoped
search result, not a first-of-kind claim.

## Metal wins across both measured graph shapes

Warm timings cover the complete algorithm call through materialized hub and
authority outputs. Values are medians of 20 timed calls; parentheses show the
CPU implementation's slowdown relative to Metal. Lower is better.

| Dataset | Metal | Matching fp64 CPU | Prebuilt SciPy recurrence | NetworkX `hits` | rustworkx `hits` | python-igraph pair |
|---|---:|---:|---:|---:|---:|---:|
| RMAT-18, 262k vertices / 4.19M edges | **2.337 ms** | 18.834 ms (8.06×) | 33.280 ms (14.24×) | capped | 507.428 ms (217.09×) | 640.071 ms (273.84×) |
| KG shape, 100k vertices / 2.0M edges | **1.518 ms** | 10.251 ms (6.75×) | 3.650 ms (2.40×) | 136.503 ms (89.93×) | 50.072 ms (32.99×) | 207.297 ms (136.58×) |

The public-library ratios are end-to-end API comparisons, not kernel-only
comparisons. NetworkX converts its graph to a SciPy matrix inside each call;
rustworkx constructs and coalesces sparse structures inside its call; and the
python-igraph API requires separate hub and authority calls. Those behaviors
are part of the measured public APIs but explain why their ratios are much
larger than the prebuilt SciPy comparison.

The p95 Metal times were 2.844 ms on RMAT-18 and 3.881 ms on the KG shape.
The latter has visible timing variance around a 1.518 ms median, so the
median is the appropriate steady-state headline while p95 should remain
visible in latency-sensitive use cases.

## Setup costs are excluded rather than hidden

The warm table excludes graph construction and reverse-CSR materialization.
They are separate rows in the source artifact:

| Dataset | metal-graph build | Reverse CSR | SciPy CSR | NetworkX graph | rustworkx graph | igraph graph |
|---|---:|---:|---:|---:|---:|---:|
| RMAT-18 | 61.581 ms | 31.217 ms | 108.901 ms | capped | 354.463 ms | 1,032.843 ms |
| KG shape | 41.939 ms | 59.274 ms | 33.587 ms | 220.284 ms | 261.213 ms | 211.564 ms |

Metal's first HITS call on an already prepared graph took 6.828 ms for RMAT-18
and 6.360 ms for the KG shape. These are algorithm-cold measurements after the
process runtime and reverse CSR were warmed; they are not process-startup or
graph-build measurements.

## Scope and metric definitions

- **RMAT-18:** deterministic Graph500-style RMAT, seed 1, 262,144 vertices and
  4,194,304 directed input edges. Its structural SciPy matrix contains
  3,940,023 unique nonzero positions after duplicate coalescing.
- **KG shape:** deterministic power-law synthetic graph, seed 7, 100,000
  vertices and 2,000,000 directed input edges. HITS ignores the generated edge
  weights. Duplicate multiplicity is retained by metal-graph; the equivalent
  coalesced SciPy matrix has only 305,667 unique nonzero positions. This
  substantially favors the prebuilt SciPy and NetworkX baselines because they
  process one weighted nonzero per repeated endpoint pair.
- **Iteration:** one `authority = L1(A.T @ hubs)` update followed by one
  `hubs = L1(A @ authority)` update.
- **Accuracy target:** absolute `L1(hubs_new - hubs_old) < 1e-5`. Because
  `metal_graph.hits` defines convergence using `V * tol`, the Metal, fp64, and
  matched SciPy rows use `tol = 1e-5 / V`. External solvers receive
  solver-appropriate tolerances and are admitted only after post-hoc agreement
  checks.
- **Speedup:** baseline warm median divided by Metal warm median for the same
  dataset. A ratio is emitted only when the candidate vectors satisfy both
  `L1 error <= 0.005` and `max absolute error <= 5e-5` against fp64.

All Metal and exact-recurrence rows completed five iterations. The largest
observed aggregate L1 difference between Metal and fp64 was `2.415e-7`; the
largest observed elementwise difference was `7.551e-8`.

## Methodology and environment

The benchmark used an optimized Release build (`-O3`) on an Apple M4 Max
MacBook Pro with 16 CPU cores and 128 GB unified memory, running macOS 26.6.2,
Xcode 26.4, and CPython 3.13.3 while connected to AC power. Recorded package
versions were NumPy 2.5.2, SciPy 1.18.1, NetworkX 3.4.2, rustworkx 0.18.0, and
python-igraph 1.0.0.

Each Metal and fp64 row received two warm-up calls; external implementations
received one. Every warm median and p95 uses 20 timed calls, whose individual
samples are retained in the JSON. Accuracy is checked on an actual timed
result, and the raw Metal/fp64 L1 norms are recorded before the scale-invariant
cross-solver comparison. Metal calls are synchronous at the Python boundary.
Original edge weights are ignored in all HITS paths, parallel edges contribute
their multiplicity, self-loops remain, all isolated vertices remain in the
output, and both vectors are L1-normalized.

The exact SciPy row is an independent benchmark-harness implementation over a
prebuilt CSR matrix. NetworkX uses its installed sparse-SVD implementation,
rustworkx uses its installed power iteration, and igraph uses two ARPACK score
calls with `tol=1e-5` and `maxiter=100`. Because equal numeric tolerances have
different meanings for those solvers, their returned scores are checked
post-hoc rather than assumed equivalent.

## Limitations and robustness

- This is one Apple M4 Max and two deterministic synthetic graphs. It does not
  establish performance on other Apple chips or every graph topology.
- NetworkX was capped at 2.5M input edges, so its RMAT-18 row is intentionally
  absent rather than represented as an infinite speedup.
- The SciPy recurrence is the strongest algorithm-matched baseline, but it is
  harness code rather than a named public HITS function. The public-library
  rows answer the practical API question and perform different internal work.
- The source artifact records `git_dirty=true` at commit `0fd0dc590313`.
  Its native-module SHA-256 is
  `c9f023f5be2db53cd29f42588511616d0772d37631cbceda95d92a450134b764`.
  The benchmark-harness SHA-256 is
  `141fe9cc40547a8e9ee5bd31c2220e704130cc2d6feab1c9a6537576f7b5a774`.
  The result is suitable for reviewing this working-tree feature, but a clean
  post-commit rerun is required before treating it as a release artifact.
- The warning emitted by igraph for many zero scores on these sparse graphs did
  not invalidate its result: both vectors still agreed with the fp64 reference
  well inside the recorded thresholds.

## Recommended next steps

Use the **6.75–8.06× fp64 CPU speedup** as the implementation-matched
CPU/GPU claim. Across all valid baselines, the lowest observed speedup was the
**2.40× prebuilt-SciPy comparison** on the duplicate-heavy KG shape.
Use the **32.99–217.09× rustworkx comparison** only with the end-to-end API
qualification above. Before release, rerun the focused command from a clean
commit and add at least one real directed web or citation graph. Measurements
on an M-series base chip and an M-series Pro chip would establish the useful
hardware range and the CPU/GPU crossover.

## Further questions

- At what edge count does Metal overtake the fp64 and prebuilt-SciPy paths?
- How do cadence 1 and cadence 5 trade convergence responsiveness for GPU
  synchronization cost on slow-converging graphs?
- Does unified-memory energy per completed HITS solve improve along with wall
  time? The existing powermetrics workflow can answer this in a follow-up run.

## Reproducible artifacts

- [Machine-readable benchmark JSON](../bench/results/bench-20260904T235705Z.json)
- [Harness-rendered benchmark table](../bench/results/bench-20260904T235705Z.md)
- [Benchmark methodology and focused command](../bench/README.md)
- [NVIDIA cuGraph HITS reference](https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.hits/)
- [NetworkX HITS API](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.hits_alg.hits.html)
- [NetworkX 3.4.2 HITS source](https://github.com/networkx/networkx/blob/networkx-3.4.2/networkx/algorithms/link_analysis/hits_alg.py#L8-L96)
- [rustworkx HITS API](https://www.rustworkx.org/dev/apiref/rustworkx.hits.html)
- [rustworkx 0.18.0 HITS source](https://github.com/Qiskit/rustworkx/blob/0.18.0/src/link_analysis.rs#L228-L382)
- [python-igraph hub and authority APIs](https://python.igraph.org/en/main/api/igraph.GraphBase.html)
