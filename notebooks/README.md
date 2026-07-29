# Example notebooks

Executed, self-contained tours of metal-graph. Every notebook uses small
seeded synthetic data (no downloads) and runs end-to-end in well under a
minute on an Apple Silicon Mac; embedded outputs are from an M4 Max.

| Notebook | What it shows |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | The core API in five minutes: building graphs from arbitrary (string) IDs, PageRank with a NetworkX agreement check, BFS/k-hop/WCC, and the per-operation planner with honest execution telemetry. |
| [`02_batched_ppr_retrieval.ipynb`](02_batched_ppr_retrieval.ipynb) | The flagship: batched personalized PageRank with top-k on a 1.2M-edge knowledge-graph shape — one GPU call for 16 agent queries at ~0.3 ms/query, bit-identical to sequential execution, ~70× an igraph loop. |
| [`03_bfs_latency_planner.ipynb`](03_bfs_latency_planner.ipynb) | BFS at both extremes on a 2M-edge graph: GPU-resident full sweeps in milliseconds, tiny-component answers in ~15 µs via the bounded latency path (~150× faster than forcing the GPU), the sparse-output API, capped ego extraction, and WCC. |

Run them yourself after installing (`python3 -m pip install .` from the
repo root, or a dev build with `PYTHONPATH=<repo>/python`):

```bash
python3 -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
