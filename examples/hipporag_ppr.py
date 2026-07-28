#!/usr/bin/env python3
"""HippoRAG-style retrieval: batched personalized PageRank with top-k.

Builds the synthetic knowledge-graph shape from the benchmark harness
(V=100k, E=2M, power-law out-degree, fp32 weights), issues a batch of
B=16 entity-seeded queries through mg.ppr_topk(k=64), and times it
against a naive per-query NetworkX loop on a SUBSAMPLED graph (NetworkX
on the full KG would take minutes; the subsample keeps its loop under
~30 s and the table says so honestly).

Run from the repo root after building:
    PYTHONPATH=python python3 examples/hipporag_ppr.py
"""

import sys
import time

import numpy as np

try:
    import metal_graph as mg
except ImportError:
    mg = None

B = 16
K = 64
ALPHA, TOL, MAX_ITER = 0.85, 1e-6, 50
NX_SUB_V = 20_000  # subsample cap for the NetworkX baseline


def gen_kg(v=100_000, e=2_000_000, seed=7):
    """Synthetic HippoRAG-shape KG (same generator as bench/run.py)."""
    rng = np.random.default_rng(seed)
    raw = rng.zipf(1.8, size=v).astype(np.float64)
    deg = np.maximum(1, np.round(raw * (e / raw.sum()))).astype(np.int64)
    src = np.repeat(np.arange(v, dtype=np.uint32), deg)
    if src.size > e:
        src = src[:e]
    elif src.size < e:
        src = np.concatenate(
            [src, rng.integers(0, v, e - src.size).astype(np.uint32)])
    dst_rank = (rng.zipf(1.6, size=e) - 1) % v
    perm = rng.permutation(v).astype(np.uint32)
    dst = perm[dst_rank]
    w = rng.uniform(0.1, 2.0, e).astype(np.float32)
    return src, dst, w, v


def make_queries(v, batch, seed=3):
    """B entity-seeded queries, 1-5 seed entities each, packed CSR-style."""
    rng = np.random.default_rng(seed)
    seeds, weights, offsets = [], [], [0]
    for _ in range(batch):
        n = int(rng.integers(1, 6))
        seeds.append(rng.choice(v, size=n, replace=False).astype(np.uint32))
        weights.append(rng.uniform(0.5, 2.0, n).astype(np.float32))
        offsets.append(offsets[-1] + n)
    return (np.concatenate(seeds), np.concatenate(weights),
            np.asarray(offsets, dtype=np.uint64))


def nx_baseline(src, dst, w, queries, k):
    """Naive per-query loop on the subsampled graph (context only)."""
    import networkx as nx
    seeds, weights, offsets = queries
    keep = (src < NX_SUB_V) & (dst < NX_SUB_V)
    gx = nx.DiGraph()
    gx.add_nodes_from(range(NX_SUB_V))
    gx.add_weighted_edges_from(
        zip(src[keep].tolist(), dst[keep].tolist(),
            w[keep].astype(np.float64).tolist()))
    n_sub_edges = int(keep.sum())
    t0 = time.perf_counter()
    for q in range(len(offsets) - 1):
        lo, hi = int(offsets[q]), int(offsets[q + 1])
        pers = {}
        for s, sw in zip(seeds[lo:hi], weights[lo:hi]):
            pers[int(s) % NX_SUB_V] = pers.get(int(s) % NX_SUB_V, 0.0) \
                + float(sw)
        scores = nx.pagerank(gx, alpha=ALPHA, tol=TOL, max_iter=MAX_ITER,
                             personalization=pers, weight="weight")
        arr = np.zeros(NX_SUB_V)
        for node, sc in scores.items():
            arr[node] = sc
        order = np.lexsort((np.arange(NX_SUB_V), -arr))
        _ = order[:k]
    elapsed = (time.perf_counter() - t0) * 1e3
    return elapsed, n_sub_edges


def main():
    if mg is None:
        print("metal_graph not importable — build first, then run from the "
              "repo root with PYTHONPATH=python")
        return 1

    print("generating synthetic HippoRAG-shape KG (V=100k, E=2M)...")
    src, dst, w, v = gen_kg()
    t0 = time.perf_counter()
    g = mg.Graph.from_edges(src, dst, weights=w, directed=True,
                            num_vertices=v)
    build_ms = (time.perf_counter() - t0) * 1e3
    print(f"build: {build_ms:.1f} ms  (V={v:,}, E={len(src):,})")

    queries = make_queries(v, B)
    seeds, weights, offsets = queries

    def run_batch():
        return mg.ppr_topk(g, seeds, weights, offsets, k=K, alpha=ALPHA,
                           tol=TOL, max_iter=MAX_ITER)

    cold_ms, _ = _timed(run_batch)
    warm = [_timed(run_batch)[0] for _ in range(10)]
    warm_ms = float(np.median(warm))
    ids, scores = run_batch()
    info = mg.last_run_info()

    print(f"\nquery 0 top-5 entity ids: "
          f"{np.asarray(ids).reshape(B, K)[0, :5].tolist()}")

    nx_ms, n_sub_edges = nx_baseline(src, dst, w, queries, K)

    rows = [
        ("metal-graph ppr_topk batch (cold)", f"V=100k E=2.0M", B,
         cold_ms, cold_ms / B),
        ("metal-graph ppr_topk batch (warm)", f"V=100k E=2.0M", B,
         warm_ms, warm_ms / B),
        (f"networkx per-query loop (SUBSAMPLED)",
         f"V={NX_SUB_V // 1000}k E={n_sub_edges / 1e3:.0f}k", B,
         nx_ms, nx_ms / B),
    ]
    print(f"\n{'workload':<42s} {'graph':<18s} {'B':>3s} "
          f"{'total ms':>10s} {'ms/query':>9s}")
    for name, graph, batch, total, per in rows:
        print(f"{name:<42s} {graph:<18s} {batch:>3d} {total:>10.2f} "
              f"{per:>9.2f}")
    print(f"\nexecuted path: {info['path']} · iterations: "
          f"{info['iterations']} · engine: {info['ms']:.2f} ms")
    print("note: the NetworkX row runs on a subsampled graph so the loop "
          "stays under ~30 s — it is context, not a like-for-like ratio.")
    return 0


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1e3, out


if __name__ == "__main__":
    sys.exit(main())
