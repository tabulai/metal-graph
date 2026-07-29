# metal-graph Benchmark Report

**Run date:** 2026-07-28–29
**Hardware:** Apple M4 Max, 16 CPU cores, 40 GPU cores, 128 GB unified memory
**Source:** commit `a4a8bc325785b4b73f92ff896ec4e7e44ed5b4a6`
**Benchmark scope:** v0.1 core suite, six datasets, 20 warm runs per
metal-graph line item

## Executive summary

The rerun completed every metal-graph core benchmark across RMAT-18,
HippoRAG KG, RMAT-22, RMAT-24, LiveJournal, and Orkut. PageRank, PPR,
WCC, and all substantial BFS workloads ran on the GPU. The tiny
HippoRAG source-0 and 64-source BFS workloads correctly selected the
new bounded sparse CPU path.

The implementation passed 17 of 18 comparable PageRank, BFS, and WCC
`>=2x` external-baseline gates. The only miss was Orkut WCC, which was
1.21x faster than igraph. The strongest result was Orkut BFS at 8.29 ms,
approximately 4,965x faster than the equivalent dense-output rustworkx
comparison. On the HippoRAG-shaped KG, high-degree GPU BFS was 26.3x
faster than rustworkx, while the sparse source-0 query improved from
0.950 ms in the previous artifact to 0.012 ms in this run.

Two release concerns remain. HippoRAG PPR took 15.116 ms per batch
(0.945 ms/query), missing the 10 ms and 0.7 ms/query targets. RMAT-18
also ran 14–29% slower than the previous canonical smoke artifact on
the principal GPU algorithms and showed elevated p95 variability.

The monolithic run did not produce a final canonical JSON artifact.
LiveJournal's contextual igraph personalized-PageRank comparator ran
for more than six hours without completing. It was stopped, and Orkut
was completed in a bounded follow-up run that skipped only that
unbounded exact-solver comparator. This is a benchmark-harness
scalability problem, not a metal-graph kernel failure.

## Environment and provenance

| Item | Value |
|---|---|
| Chip | Apple M4 Max |
| CPU / GPU | 16 CPU cores (12 performance, 4 efficiency) / 40 GPU cores |
| Memory | 128 GB unified memory |
| macOS | 26.2 |
| Python | 3.13.3 |
| Xcode | 26.4 |
| metal-graph source | `a4a8bc325785b4b73f92ff896ec4e7e44ed5b4a6` |
| Native module SHA-256 | `90d359d7e7fffa0f896892fecdc22620432df71cb6ab43cbb60cc768dcb21b37` |
| Build | Release, `-O3`, arm64 |
| Required execution setting | `MG_REQUIRE_GPU=1` |
| NumPy | 2.4.6 |
| NetworkX | 3.4.2 |
| SciPy | 1.15.2 |
| rustworkx | 0.18.0 |
| igraph | 1.0.0 |

The machine remained on AC power, low-power mode was disabled, and
macOS reported no thermal or performance warning during the run.

Before benchmarking, the full Python test suite passed with 972 tests
and 54 skips. All three CTest targets passed. The source tree and native
module were clean and reproducibly identified before timing began.

## Methodology

The canonical command was:

```bash
MG_REQUIRE_GPU=1 PYTHONPATH=python \
python3 bench/run.py --suite v01 --runs 20 --fetch
```

The suite used the two synthetic graphs, the deterministic HippoRAG
shape, and both pinned SNAP datasets:

| Dataset | Vertices | Edges | Directed | Weighted |
|---|---:|---:|:---:|:---:|
| RMAT-18 | 262,144 | 4,194,304 | yes | no |
| HippoRAG KG | 100,000 | 2,000,000 | yes | yes |
| RMAT-22 | 4,194,304 | 67,108,864 | yes | no |
| RMAT-24 | 16,777,216 | 268,435,456 | yes | no |
| LiveJournal | 4,847,571 | 68,993,773 | yes | no |
| Orkut | 3,072,441 | 117,185,083 | no | no |

The LiveJournal and Orkut archives matched the sizes and SHA-256
digests pinned in `bench/run.py`. Orkut's raw vertex labels are sparse
despite the declared SNAP vertex count, so the benchmark loader
densely normalized them before graph construction. That normalization
fix is included in the source commit cited above.

All metal-graph warm rows used 20 timed repetitions. Expensive external
baseline rows generally used four timed samples plus one warm-up,
following the current harness policy. Sub-2 ms BFS baselines received
additional samples. Median and p95 below are wall-clock milliseconds
from the Python call to the returned result.

For BFS, the valid rustworkx comparator constructs dense `int32[V]`
distance and parent arrays, matching metal-graph's output shape.
Rustworkx's no-output visitor and sparse `bfs_layers` measurements are
retained only as non-equivalent context.

## Core results

Each cell is **median / p95 milliseconds**. PageRank is the full warm
run; the iteration-normalized value is reported separately.

| Dataset | PageRank | PR / iteration | PPR B=16, k=64 | BFS source 0 | BFS high-degree | BFS 64-source | WCC | k-hop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RMAT-18 | 1.759 / 6.059 | 0.352 / 1.212 | 10.202 / 11.543 | 1.538 / 2.215 | 1.446 / 2.135 | 1.679 / 3.331 | 3.819 / 5.679 | 75.940 / 77.128 |
| HippoRAG KG | 3.074 / 3.981 | 0.205 / 0.265 | 15.116 / 19.085 | 0.012 / 0.030 | 1.905 / 2.094 | 0.014 / 0.018 | 4.090 / 4.324 | 24.166 / 24.811 |
| RMAT-22 | 16.892 / 17.689 | 3.378 / 3.538 | 121.993 / 124.861 | 13.920 / 14.233 | 13.746 / 14.183 | 12.498 / 16.671 | 32.362 / 32.858 | 376.852 / 380.855 |
| RMAT-24 | 73.464 / 75.610 | 14.693 / 15.122 | 555.634 / 571.736 | 46.096 / 49.893 | 46.841 / 48.392 | 44.378 / 48.384 | 119.823 / 120.996 | 478.631 / 481.056 |
| LiveJournal | 19.773 / 20.483 | 3.955 / 4.097 | 173.635 / 175.465 | 13.398 / 14.013 | 14.454 / 15.261 | 15.342 / 20.337 | 24.932 / 25.435 | 5.551 / 6.101 |
| Orkut | 39.501 / 41.915 | 7.900 / 8.383 | 257.417 / 265.154 | 8.290 / 8.540 | 9.482 / 9.725 | 8.086 / 8.558 | 55.942 / 57.855 | 0.367 / 0.425 |

Graph construction and lazy-transpose measurements were:

| Dataset | Build | Transpose estimate |
|---|---:|---:|
| RMAT-18 | 66.479 | 33.574 |
| HippoRAG KG | 30.664 | 58.420 |
| RMAT-22 | 1,021.045 | 466.030 |
| RMAT-24 | 4,306.288 | 1,902.829 |
| LiveJournal | 527.018 | 341.979 |
| Orkut | 1,856.951 | 17.215 |

## External baseline comparison

The PageRank ratio uses the faster available rustworkx or igraph
baseline. BFS uses the equivalent dense-output rustworkx implementation.
WCC uses the faster rustworkx or igraph result. PPR comparisons against
igraph are contextual because igraph uses PRPACK without matching the
metal-graph iteration count.

| Dataset | PageRank speedup | BFS source-0 speedup | BFS high-degree speedup | WCC speedup | Contextual PPR speedup |
|---|---:|---:|---:|---:|---:|
| RMAT-18 | 174.2x | 670.2x | 711.3x | 32.9x | 585.4x |
| HippoRAG KG | 11.7x | 4.8x | 26.3x | 3.2x | 43.9x |
| RMAT-22 | 375.7x | 1,264.4x | 1,255.7x | 67.2x | 872.1x |
| RMAT-24 | 418.8x | 1,718.3x | 1,678.8x | 75.6x | 1,059.2x |
| LiveJournal | 344.1x | 702.9x | 651.1x | 66.3x | not completed |
| Orkut | 613.9x | 4,965.2x | 4,442.7x | 1.21x | skipped |

The small HippoRAG source-0 BFS row is a hybrid-planner win rather than
a GPU win. Source 0 has out-degree four, reaches only 52 vertices, and
scans 344 reachable edges. The new bounded sparse CPU path completes
that query in approximately 12 microseconds. The deterministic
high-degree source exercises the GPU and is the meaningful KG GPU
comparison: 1.905 ms versus rustworkx's 50.058 ms.

## Gate assessment

### Comparable algorithm gates

Seventeen of 18 comparable PageRank, BFS, and WCC cells met the `>=2x`
external-baseline threshold. The only failure was:

- **Orkut WCC:** 55.942 ms versus igraph's 67.772 ms, or 1.21x.

All six high-degree BFS workloads passed against the equivalent
dense-output rustworkx comparison.

### HippoRAG PPR gate

HippoRAG PPR remains a release blocker:

| Metric | Measured | Target | Result |
|---|---:|---:|---|
| Batch latency, B=16 | 15.116 ms | <=10 ms | fail; 51.2% over |
| Amortized latency | 0.945 ms/query | <=0.7 ms/query | fail; 35.0% over |
| Contextual igraph ratio | 43.9x | >=5x | pass, but solver is not iteration-matched |

The formal identical-iteration PPR comparison remains open.

### LiveJournal engineering targets

All documented LiveJournal targets passed:

| Metric | Measured | Target | Result |
|---|---:|---:|---|
| PageRank per iteration | 3.955 ms | <=6 ms | pass |
| PageRank stretch target | 3.955 ms | <=4 ms | pass |
| BFS | 13.398 ms | <=50 ms | pass |
| BFS stretch target | 13.398 ms | <=20 ms | pass |
| WCC / BFS | 1.861x | <=5x | pass |

The PageRank traffic model reported approximately 169 GB/s. This is a
model derived from documented byte traffic, not a hardware-counter
measurement.

## Comparison with the previous canonical smoke artifact

The previous canonical artifact covers only RMAT-18 and HippoRAG KG.
Lower latency is better.

| Dataset / operation | Previous | Current | Change |
|---|---:|---:|---:|
| RMAT-18 PageRank | 1.532 ms | 1.759 ms | 14.8% slower |
| RMAT-18 PPR B=16 | 8.934 ms | 10.202 ms | 14.2% slower |
| RMAT-18 BFS source 0 | 1.276 ms | 1.538 ms | 20.6% slower |
| RMAT-18 BFS 64-source | 1.304 ms | 1.679 ms | 28.8% slower |
| RMAT-18 WCC | 2.986 ms | 3.819 ms | 27.9% slower |
| HippoRAG PageRank | 3.175 ms | 3.074 ms | 3.2% faster |
| HippoRAG PPR B=16 | 15.277 ms | 15.116 ms | 1.1% faster |
| HippoRAG BFS source 0 | 0.950 ms | 0.012 ms | 79.2x faster |
| HippoRAG WCC | 4.166 ms | 4.090 ms | 1.8% faster |

The HippoRAG BFS change is an execution-path improvement: the old row
used the GPU for a tiny traversal, while the new planner chooses the
bounded sparse CPU path. It must not be presented as a 79x GPU-kernel
speedup. The high-degree KG row provides the current GPU evidence.

The RMAT-18 regression is not explained by source changes and should be
rechecked in a short, isolated smoke run. Its elevated PageRank p95
suggests system jitter may have affected the long collection, but the
medians are sufficiently different that the regression should not be
dismissed without another controlled run.

## Incomplete and invalid baseline rows

The following rows are not usable as completed gate evidence:

1. **LiveJournal igraph PPR:** stopped after more than six hours inside
   the 16-query exact-solver loop. The monolithic process had completed
   the other LiveJournal rows but had not serialized its JSON artifact.
2. **Orkut igraph PPR:** deliberately skipped in the bounded completion
   pass after the LiveJournal behavior.
3. **Orkut rustworkx PageRank:** failed because the harness passed an
   undirected `PyGraph` to an API requiring `PyDiGraph`.
4. **NetworkX on large datasets:** intentionally skipped above the
   harness's 2.5-million-edge cap.

The first five datasets therefore exist as completed timing evidence in
the preserved run log, not as a canonical machine-readable artifact.
The Orkut bounded artifact is strict JSON with 25 unique rows, clean
source provenance, and a rendered Markdown report that exactly
regenerates from the JSON.

## Outstanding issues and recommended next actions

1. **Add dataset-level checkpointing and baseline timeouts.** A
   contextual comparator must not prevent serialization of hours of
   completed core measurements. The igraph PPR loop should be capped,
   calibrated, or moved out of the default full suite.
2. **Fix the undirected rustworkx PageRank adapter.** Use an API that
   accepts `PyGraph`, or explicitly convert the graph while documenting
   the changed build and memory cost.
3. **Optimize HippoRAG PPR.** The current implementation misses both
   absolute latency targets despite a large contextual CPU-baseline
   advantage.
4. **Investigate Orkut WCC.** It is the only comparable `>=2x` gate
   failure.
5. **Repeat RMAT-18 in isolation.** Confirm whether the 14–29% median
   regressions are reproducible or were caused by long-run system
   variability.
6. **Add a direct metal-graph CPU-versus-GPU matrix.** The canonical
   harness compares automatic execution to external CPU libraries; it
   does not currently quantify forced metal-graph GPU acceleration over
   its own CPU path.
7. **Complete optional operational scenarios.** Energy capture requires
   a live sudo credential for `powermetrics`. MLX contention requires a
   configured `MG_BENCH_MLX_MODEL` and should also record solo model
   throughput so token-rate impact can be calculated.
8. **Collect the remaining release hardware matrix.** The implementation
   plan also calls for M4 Pro and M3 Ultra measurements and an end-to-end
   agent workflow.

## Artifacts

- [Preserved full-run log](../bench/results/bench-20260729-full-partial.log)
- [Bounded Orkut JSON](../bench/results/bench-20260729T131101Z-orkut-bounded.json)
- [Bounded Orkut rendered report](../bench/results/bench-20260729T131101Z-orkut-bounded.md)
- [Bounded Orkut console log](../bench/results/bench-20260729-orkut-bounded.log)
- [Previous canonical smoke JSON](../bench/results/bench-20260729T000207Z.json)
- [Benchmark harness documentation](../bench/README.md)

The `--energy` and `--contention` scenarios were not run. Energy capture
was blocked by the absence of a cached sudo credential, and contention
was not configured with an MLX model. Neither scenario is part of the
ordinary 153-row v0.1 core matrix.
