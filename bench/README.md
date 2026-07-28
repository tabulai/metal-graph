# metal-graph benchmark harness

`bench/run.py` is the only source of performance numbers for this project.

## Honesty rules

1. **No number appears in repo docs unless it came from a physical run of
   this harness on real Apple Silicon hardware.** CI VMs have Metal but
   meaningless performance; they never produce published numbers.
2. Reporting is **decomposed**: build, transpose, pipeline warm-up, warm
   kernel execution, convergence audit, top-k selection, and Python
   boundary overhead are separate line items. A headline number that hides
   cold-start or boundary cost is a lie by omission.
3. Warm kernel cells are **median + p95 over >= 20 runs** (`--runs`,
   default 20). Cold (first-call) numbers are reported separately, never
   averaged in.
4. Baselines are context: NetworkX (capped, because it is not the
   competition), SciPy sparse power iteration, a pure-python deque BFS.
   rustworkx / igraph are feature-detected line items for **BFS, WCC,
   and PageRank** (the plan-§10 gate shapes) and reported as
   `not installed` when absent — never silently skipped.
5. Every line item records `t_start_utc`/`t_end_utc` (ISO-8601, for
   powermetrics window alignment — see `ENERGY.md`) and peak-RSS
   bracketing: `peak_rss_mb` is the process high-water mark at the end of
   the item and `peak_rss_delta_mb` its growth during the item.
   `ru_maxrss` is monotonic, so a delta of 0 means "stayed under an
   earlier peak", not "allocated nothing".
6. `pagerank / warm_full_run` carries `achieved_gb_s` from a stated
   traffic model — `bytes ≈ iterations × (8·E + 24·V)`, divided by the
   warm median time. It is a model, not a counter; the formula is
   recorded in the JSON row (`traffic_model`) so it can be audited.

## Running

```bash
# quick smoke (RMAT-18 + HippoRAG-shape KG)
PYTHONPATH=python python3 bench/run.py --suite smoke

# full v0.1 suite (adds RMAT-22, RMAT-24 and the SNAP datasets if fetched)
PYTHONPATH=python python3 bench/run.py --suite v01

# SNAP datasets are downloaded ONLY with an explicit --fetch
PYTHONPATH=python python3 bench/run.py --suite v01 --fetch
```

Datasets cache under `bench/data/` (`.txt.gz` plus a parsed `.npz`).
Results land in `bench/results/` as JSON (with chip / macOS / python /
package versions and all `MG_*` env vars) plus a rendered markdown table.

## Line items

| item | meaning |
|---|---|
| `build / from_edges` | COO -> snapshot; the build **is** the upload (shared buffers) |
| `build / transpose_estimate` | first `direction="in"` touch minus the second call: lazy CSC materialization |
| `pipeline / first_op_vs_second` | first tiny op in the process vs second: pipeline-compile + runtime init |
| `pagerank / warm_full_run` | full converged run, median/p95; `iterations`, executed `path`, and modeled `achieved_gb_s` attached |
| `pagerank / per_iteration` | warm run divided by executed iterations |
| `ppr_topk / warm_batch16_k64` | the flagship gate shape: B=16, k=64, from Python call to NumPy result |
| `ppr_topk / python_boundary` | wall median minus `last_run_info()["ms"]` (engine time) |
| `ppr_topk / topk_selection_estimate` | median(k=1024) − median(k=1): selection cost estimate |
| `bfs / warm_single_source`, `warm_64_sources` | levels attached from telemetry |
| `wcc / warm` | hook+jump rounds attached |
| `k_hop / warm_k2_capped` | capped extraction (agent-latency shape) |
| `baseline_*` | context numbers (see honesty rules); `baseline_rustworkx` / `baseline_igraph` appear under `pagerank`, `bfs`, **and** `wcc` |

Every row additionally carries `t_start_utc` / `t_end_utc`,
`peak_rss_mb`, and `peak_rss_delta_mb` (honesty rule 5); the markdown
table shows the peak-RSS delta per line item.

The v0.1 suite includes `rmat24` (V≈16.8M, E≈268M — the ship-gate scale
point); smoke deliberately does not, so it stays fast.

The convergence-audit share is visible as the attached iteration counts:
iterations land on `MG_PR_AUDIT_INTERVAL` boundaries by design, so audit
overhead = (reported wall − engine ms) plus the overshoot iterations.

## Contention scenario (`--contention`)

Runs the B=16 PPR batch while `mlx_lm` decodes a fixed prompt stream on
the same GPU, reporting graph-latency inflation AND tokens/s impact. An
isolated GPU speedup that starves the LLM is a net loss for an agent.
`mlx_lm` is feature-detected, never a dependency:

```bash
pip install mlx-lm
MG_BENCH_MLX_MODEL=mlx-community/Llama-3.2-1B-Instruct-4bit \
  PYTHONPATH=python python3 bench/run.py --contention --suite smoke
```

## Energy (`--energy`)

Requires `sudo` (powermetrics). The harness refuses politely otherwise.
After starting powermetrics it captures an idle baseline window
(`--idle-seconds`, default 60) before any workload; the window and every
line item's timestamps land in the JSON so the streams can be aligned
and the idle power subtracted. Methodology: `bench/ENERGY.md`.
