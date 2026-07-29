#!/usr/bin/env python3
"""metal-graph benchmark harness (v0.1, plan section 10).

Decomposed, honest reporting: build / transpose / pipeline warm / warm
kernels / convergence iteration metadata / top-k / Python boundary are
separate line items; every kernel cell is median + p95 over >= --runs runs.
Baselines
(NetworkX, SciPy, pure-python BFS, rustworkx/igraph BFS/WCC/PageRank when
installed) are context, never the gate. Every line item carries
t_start_utc/t_end_utc (powermetrics window alignment, bench/ENERGY.md) and
peak-RSS bracketing; pagerank warm runs carry a modeled achieved_gb_s.
Numbers belong in docs ONLY when they came from a physical run of this
script. See bench/README.md.

Usage:
  PYTHONPATH=python python3 bench/run.py --suite smoke
  PYTHONPATH=python python3 bench/run.py --suite v01 --fetch
"""

import argparse
import gzip
import hashlib
import importlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
RESULT_SCHEMA_VERSION = 1
SNAP_CACHE_SCHEMA_VERSION = 1
DOWNLOAD_TIMEOUT_SECONDS = 60

SNAP_DATASETS = {
    "soc-LiveJournal1": {
        "url": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
        "sha256":
            "d7bcd5a87b88c896c35fdb9611e804c3f4033c39b58c4c9ea3ba53c680d516d8",
        "bytes": 259_619_239,
        "vertices": 4_847_571,
        "edges": 68_993_773,
        "directed": True,
    },
    "com-orkut": {
        "url":
            "https://snap.stanford.edu/data/bigdata/communities/"
            "com-orkut.ungraph.txt.gz",
        "sha256":
            "f73e33fb685f411a10c952f2ba3ea788380b91a17bc636e38da1a23f6c6b2bc6",
        "bytes": 447_251_958,
        "vertices": 3_072_441,
        "edges": 117_185_083,
        "directed": False,
    },
}

# NetworkX baseline caps (context only; keeps the loop sane)
NX_MAX_EDGES = 2_500_000
PY_BFS_MAX_EDGES = 5_000_000


# ---------------------------------------------------------------------------
# generators (seeded, deterministic)
# ---------------------------------------------------------------------------

def gen_rmat(scale, edgefactor=16, seed=1):
    """Graph500-style RMAT (A=0.57, B=0.19, C=0.19), vectorized numpy."""
    rng = np.random.default_rng(seed)
    n_edges = edgefactor << scale
    a, b, c = 0.57, 0.19, 0.19
    ab = a + b
    c_norm = c / (1.0 - ab)
    a_norm = a / ab
    src = np.zeros(n_edges, dtype=np.int64)
    dst = np.zeros(n_edges, dtype=np.int64)
    for bit in range(scale):
        ii = rng.random(n_edges) > ab
        jj = rng.random(n_edges) > (c_norm * ii + a_norm * (~ii))
        src |= ii.astype(np.int64) << bit
        dst |= jj.astype(np.int64) << bit
    return (src.astype(np.uint32), dst.astype(np.uint32), None,
            1 << scale, True)


def gen_kg(v=100_000, e=2_000_000, seed=7):
    """HippoRAG-shape synthetic KG: power-law out-degree via zipf,
    weighted, directed."""
    rng = np.random.default_rng(seed)
    raw = rng.zipf(1.8, size=v).astype(np.float64)
    deg = np.maximum(1, np.round(raw * (e / raw.sum()))).astype(np.int64)
    src = np.repeat(np.arange(v, dtype=np.uint32), deg)
    if src.size > e:
        src = src[:e]
    elif src.size < e:
        pad = rng.integers(0, v, e - src.size).astype(np.uint32)
        src = np.concatenate([src, pad])
    dst_rank = (rng.zipf(1.6, size=e) - 1) % v
    perm = rng.permutation(v).astype(np.uint32)
    dst = perm[dst_rank]
    w = rng.uniform(0.1, 2.0, e).astype(np.float32)
    return src, dst, w, v, True


def stored_out_degrees(src, dst, v, directed):
    """USER-order degree of the traversal CSR used by direction='out'."""
    degree = np.bincount(src, minlength=v).astype(np.uint64, copy=False)
    if not directed:
        reverse = dst != src
        degree += np.bincount(
            dst[reverse], minlength=v
        ).astype(np.uint64, copy=False)
    return degree


def make_rustworkx_dense_bfs(rx, graph, v, source):
    """Return an API-equivalent dense dist+parent rustworkx BFS runner."""
    class OutputVisitor(rx.visit.BFSVisitor):
        def __init__(self):
            self.dist = np.full(v, -1, np.int32)
            self.parent = np.full(v, -1, np.int32)
            self.dist[source] = 0

        def tree_edge(self, edge):
            parent, child, _ = edge
            self.parent[child] = parent
            self.dist[child] = self.dist[parent] + 1

    def run():
        visitor = OutputVisitor()
        rx.bfs_search(graph, [source], visitor)
        return visitor.dist, visitor.parent

    return run


def make_igraph_dense_bfs(graph, v, source):
    """Return an API-equivalent dense dist+parent igraph BFS runner."""
    def run():
        order, layer_offsets, raw_parent = graph.bfs(source, mode="out")
        dist = np.full(v, -1, np.int32)
        parent = np.asarray(raw_parent, dtype=np.int32)
        parent[parent < 0] = -1

        order = np.asarray(order, dtype=np.intp)
        layer_offsets = np.asarray(layer_offsets, dtype=np.intp)
        depths = np.repeat(
            np.arange(layer_offsets.size - 1, dtype=np.int32),
            np.diff(layer_offsets),
        )
        dist[order] = depths
        return dist, parent

    return run


# ---------------------------------------------------------------------------
# SNAP datasets (download ONLY with --fetch; cached under bench/data/)
# ---------------------------------------------------------------------------

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise RuntimeError(
            f"{path}: expected {spec['bytes']} bytes, got "
            f"{path.stat().st_size}"
        )
    actual = file_sha256(path)
    if actual != spec["sha256"]:
        raise RuntimeError(
            f"{path}: SHA-256 mismatch; expected {spec['sha256']}, "
            f"got {actual}"
        )


def download_dataset(spec, destination):
    """Download one pinned dataset with a timeout, byte cap, and streaming hash."""
    digest = hashlib.sha256()
    total = 0
    request = urllib.request.Request(
        spec["url"],
        headers={"User-Agent": "metal-graph-benchmark/0.1"},
    )
    with urllib.request.urlopen(
        request, timeout=DOWNLOAD_TIMEOUT_SECONDS
    ) as response, destination.open("wb") as stream:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > spec["bytes"]:
                raise RuntimeError(
                    f"{spec['url']}: download exceeds pinned size "
                    f"{spec['bytes']}"
                )
            stream.write(chunk)
            digest.update(chunk)
    if total != spec["bytes"]:
        raise RuntimeError(
            f"{spec['url']}: expected {spec['bytes']} bytes, got {total}"
        )
    actual = digest.hexdigest()
    if actual != spec["sha256"]:
        raise RuntimeError(
            f"{spec['url']}: SHA-256 mismatch; expected {spec['sha256']}, "
            f"got {actual}"
        )


def fetch_datasets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in SNAP_DATASETS.items():
        path = DATA_DIR / (name + ".txt.gz")
        if path.exists():
            validate_dataset(path, spec)
            print(f"[fetch] cached: {path}")
            continue
        partial = path.with_suffix(path.suffix + ".part")
        print(f"[fetch] downloading {spec['url']} -> {path}")
        try:
            download_dataset(spec, partial)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    print("[fetch] done")


def load_snap_cache(path, spec):
    """Return a validated parsed SNAP cache, or None when it is unusable."""
    required = {
        "cache_schema_version",
        "source_sha256",
        "src",
        "dst",
        "v",
        "directed",
    }
    try:
        with np.load(path, allow_pickle=False) as cache:
            if not required.issubset(cache.files):
                return None
            schema = int(np.asarray(cache["cache_schema_version"]).item())
            digest = str(np.asarray(cache["source_sha256"]).item())
            src = np.asarray(cache["src"])
            dst = np.asarray(cache["dst"])
            v = int(np.asarray(cache["v"]).item())
            directed = bool(np.asarray(cache["directed"]).item())
    except (EOFError, KeyError, OSError, ValueError):
        return None

    if schema != SNAP_CACHE_SCHEMA_VERSION or digest != spec["sha256"]:
        return None
    if src.dtype != np.uint32 or dst.dtype != np.uint32:
        return None
    if src.ndim != 1 or dst.ndim != 1 or src.size != dst.size:
        return None
    if src.size != spec["edges"] or v != spec["vertices"]:
        return None
    if directed != spec["directed"]:
        return None
    if src.size and (int(src.max()) >= v or int(dst.max()) >= v):
        return None
    return src, dst, None, v, directed


def normalize_snap_ids(src, dst, expected_vertices, source):
    """Validate and densely renumber a pinned SNAP edge list.

    SNAP metadata reports the number of vertices, but some archives use a
    sparse external-ID range. Treating ``max_id + 1`` as the vertex count
    would add fake isolated vertices and can make benchmark source 0 fake as
    well. Preserve already-dense inputs and otherwise map sorted external IDs
    deterministically to ``[0, expected_vertices)``.
    """
    if src.size == 0:
        if expected_vertices != 0:
            raise RuntimeError(
                f"{source}: expected {expected_vertices} vertices, got 0"
            )
        return src, dst, 0

    max_id = int(max(src.max(), dst.max()))
    seen = np.zeros(max_id + 1, dtype=np.bool_)
    seen[src] = True
    seen[dst] = True
    ids = np.flatnonzero(seen)
    if ids.size != expected_vertices:
        raise RuntimeError(
            f"{source}: expected {expected_vertices} unique vertices, got "
            f"{ids.size}"
        )

    if ids[0] == 0 and ids[-1] + 1 == expected_vertices:
        return src, dst, expected_vertices

    dense_of_external = np.full(
        max_id + 1, np.iinfo(np.uint32).max, dtype=np.uint32
    )
    dense_of_external[ids] = np.arange(expected_vertices, dtype=np.uint32)
    return (
        dense_of_external[src],
        dense_of_external[dst],
        expected_vertices,
    )


def load_snap(name):
    """Parse an edge-list .txt.gz (cached as .npz after first parse).
    Returns None when the dataset was never fetched."""
    gz = DATA_DIR / (name + ".txt.gz")
    npz = DATA_DIR / (name + ".npz")
    spec = SNAP_DATASETS[name]
    if npz.exists():
        cached = load_snap_cache(npz, spec)
        if cached is not None:
            return cached
        if not gz.exists():
            raise RuntimeError(
                f"{npz}: invalid or legacy cache; rerun with --fetch"
            )
    if not gz.exists():
        return None
    validate_dataset(gz, spec)
    print(f"[load] parsing {gz} (first time; caching .npz)")
    src_l, dst_l = [], []
    with gzip.open(gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            a, b = line.split()
            src_l.append(int(a))
            dst_l.append(int(b))
    src = np.asarray(src_l, dtype=np.uint32)
    dst = np.asarray(dst_l, dtype=np.uint32)
    src, dst, v = normalize_snap_ids(
        src, dst, spec["vertices"], gz
    )
    directed = spec["directed"]
    if src.size != spec["edges"]:
        raise RuntimeError(
            f"{gz}: parsed shape does not match the pinned SNAP manifest"
        )
    partial = npz.with_suffix(npz.suffix + ".part")
    try:
        with partial.open("wb") as stream:
            np.savez_compressed(
                stream,
                cache_schema_version=SNAP_CACHE_SCHEMA_VERSION,
                source_sha256=spec["sha256"],
                src=src,
                dst=dst,
                v=v,
                directed=directed,
            )
        partial.replace(npz)
    finally:
        partial.unlink(missing_ok=True)
    return src, dst, None, v, directed


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------

# Tiny-component BFS reports an absolute-latency SLO assessment instead of
# treating a ratio as an enforceable gate:
# sub-20-microsecond cells are dominated by output-contract costs (dense
# int32[V] materialization), so ratios there compare formats, not engines.
TINY_BFS_SLO_MS = 0.050
TINY_BFS_SLO_GATE = (
    "absolute-latency SLO assessment; ratio excluded for sparse-path "
    "traversals"
)


def tiny_bfs_slo(median_ms, slo_ms=TINY_BFS_SLO_MS):
    """SLO metadata to attach to the measured tiny-component BFS row."""
    return {
        "slo_ms": slo_ms,
        "slo_pass": bool(median_ms <= slo_ms),
        "gate": TINY_BFS_SLO_GATE,
    }


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1e3, out


def stats_ms(fn, runs, warmup=2):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        ms, _ = timed(fn)
        samples.append(ms)
    samples.sort()
    p95 = samples[min(len(samples) - 1, int(round(0.95 * len(samples))) - 1)]
    return {"median_ms": statistics.median(samples), "p95_ms": p95,
            "min_ms": samples[0], "runs": runs}


def optional_import(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _maxrss_bytes():
    """Process high-water-mark RSS. ru_maxrss is bytes on macOS, KiB on
    Linux, and MONOTONIC: it only ever grows over the process lifetime."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v if sys.platform == "darwin" else v * 1024


class ItemMeter:
    """Brackets every line item with UTC wall-clock timestamps and
    peak-RSS samples.

    - t_start_utc / t_end_utc: ISO timestamps so each row's window can be
      aligned with the powermetrics sample stream (bench/ENERGY.md).
    - peak_rss_mb: the running process peak at the end of the item.
      Because ru_maxrss is monotonic, this is a high-water mark, NOT the
      item's own footprint.
    - peak_rss_delta_mb: growth of the running peak during the item.
      0.0 means the item stayed under a peak reached earlier, not that it
      allocated nothing.
    """

    def __init__(self):
        self.mark()

    def mark(self):
        self.t0 = datetime.now(timezone.utc)
        self.rss0 = _maxrss_bytes()

    def close(self):
        t1 = datetime.now(timezone.utc)
        rss1 = _maxrss_bytes()
        out = {"t_start_utc": self.t0.isoformat(),
               "t_end_utc": t1.isoformat(),
               "peak_rss_mb": rss1 / 1e6,
               "peak_rss_delta_mb": (rss1 - self.rss0) / 1e6}
        self.mark()
        return out


METER = ItemMeter()


# ---------------------------------------------------------------------------
# per-dataset benchmark
# ---------------------------------------------------------------------------

def make_queries(v, batch=16, max_seeds=5, seed=3):
    rng = np.random.default_rng(seed)
    seeds, weights, offsets = [], [], [0]
    for _ in range(batch):
        n = int(rng.integers(1, max_seeds + 1))
        seeds.append(rng.choice(v, size=n, replace=False).astype(np.uint32))
        weights.append(rng.uniform(0.5, 2.0, n).astype(np.float32))
        offsets.append(offsets[-1] + n)
    return (np.concatenate(seeds), np.concatenate(weights),
            np.asarray(offsets, dtype=np.uint64))


def bench_dataset(mg, name, data, runs, rows):
    src, dst, w, v, directed = data
    e = len(src)
    out_degree = stored_out_degrees(src, dst, v, directed)
    print(f"\n=== {name}: V={v:,} E={e:,} "
          f"{'weighted' if w is not None else 'unweighted'} ===")
    METER.mark()  # window starts here, not at generation/parse time

    def add(algo, item, st, **extra):
        row = {"dataset": name, "algo": algo, "item": item}
        row.update(st)
        row.update(extra)
        row.update(METER.close())  # t_start/t_end + peak-RSS bracketing
        rows.append(row)
        print(f"  {algo:>10s} {item:<26s} "
              f"median {st.get('median_ms', float('nan')):10.3f} ms   "
              f"p95 {st.get('p95_ms', float('nan')):10.3f} ms")

    # -- graph build (the build IS the upload) --------------------------
    ms, g = timed(lambda: mg.Graph.from_edges(
        src, dst, weights=w, directed=directed, num_vertices=v))
    add("build", "from_edges", {"median_ms": ms, "p95_ms": ms, "runs": 1})

    # -- transpose materialization: first orientation(in) touch ---------
    # A 1-source direction='in' BFS forces the transpose; the second
    # identical call reuses the cache. first - second ~= transpose cost.
    s0 = np.asarray([0], np.uint32)
    first, _ = timed(lambda: mg.bfs(g, s0, direction="in"))
    second, _ = timed(lambda: mg.bfs(g, s0, direction="in"))
    add("build", "transpose_estimate",
        {"median_ms": max(first - second, 0.0), "p95_ms": float("nan"),
         "runs": 1}, first_call_ms=first, second_call_ms=second)

    # -- pagerank --------------------------------------------------------
    pr_kw = dict(alpha=0.85, tol=1e-6, max_iter=100)
    cold, _ = timed(lambda: mg.pagerank(g, **pr_kw))
    st = stats_ms(lambda: mg.pagerank(g, **pr_kw), runs)
    info = mg.last_run_info()
    iters = max(info["iterations"], 1)
    # Achieved bandwidth from the documented traffic model: per pull
    # iteration the kernel streams the CSR structure once (~8 bytes/edge:
    # 4B column index + 4B gathered rank) and touches the rank / next-rank
    # / out-degree vectors (~24 bytes/vertex), so
    #     bytes ~= iterations * (8*E + 24*V)
    #     achieved_gb_s = bytes / warm_median_seconds / 1e9
    # This is a MODEL (no counters); the same formula is recorded in the
    # JSON row under "traffic_model" so numbers stay auditable.
    TRAFFIC_MODEL = ("bytes=iterations*(8*E+24*V); "
                     "achieved_gb_s=bytes/warm_median_s/1e9")
    model_bytes = iters * (8 * e + 24 * v)
    achieved = (model_bytes / (st["median_ms"] / 1e3) / 1e9
                if st["median_ms"] > 0 else float("nan"))
    add("pagerank", "warm_full_run", st, cold_ms=cold,
        iterations=info["iterations"], path=info["path"],
        engine_ms=info["ms"], achieved_gb_s=achieved,
        traffic_model=TRAFFIC_MODEL)
    add("pagerank", "per_iteration",
        {"median_ms": st["median_ms"] / iters,
         "p95_ms": st["p95_ms"] / iters, "runs": runs},
        iterations=iters)

    # -- ppr_topk (flagship: B=16, k=64) ---------------------------------
    seeds, weights, offsets = make_queries(v, batch=16)
    ppr_kw = dict(alpha=0.85, tol=1e-6, max_iter=50)

    def ppr(k=64):
        return mg.ppr_topk(g, seeds, weights, offsets, k=k, **ppr_kw)

    st = stats_ms(ppr, runs)
    info = mg.last_run_info()
    wall = st["median_ms"]
    engine = info["ms"]
    add("ppr_topk", "warm_batch16_k64", st, path=info["path"],
        iterations=info["iterations"], engine_ms=engine,
        amortized_per_query_ms=wall / 16.0)
    add("ppr_topk", "python_boundary",
        {"median_ms": max(wall - engine, 0.0), "p95_ms": float("nan"),
         "runs": runs})
    # top-k selection cost estimate: widen k, difference is selection.
    st_k1 = stats_ms(lambda: ppr(k=1), max(5, runs // 4))
    st_k1024 = stats_ms(lambda: ppr(k=min(1024, v)), max(5, runs // 4))
    add("ppr_topk", "topk_selection_estimate",
        {"median_ms": max(st_k1024["median_ms"] - st_k1["median_ms"], 0.0),
         "p95_ms": float("nan"), "runs": st_k1["runs"]},
        k_lo=1, k_hi=min(1024, v))

    # -- bfs ---------------------------------------------------------------
    def bfs_diagnostics(result):
        dist = np.asarray(result[0])
        reached = dist >= 0
        frontier_sizes = (
            np.bincount(dist[reached]).astype(int).tolist()
            if reached.any() else []
        )
        return {
            "reached_vertices": int(reached.sum()),
            "reachable_edges_scanned": int(out_degree[reached].sum()),
            "max_distance": int(dist[reached].max()) if reached.any() else -1,
            "frontier_sizes": frontier_sizes,
        }

    st = stats_ms(lambda: mg.bfs(g, s0, direction="out"), runs)
    info = mg.last_run_info()
    single_result = mg.bfs(g, s0, direction="out")
    slo = (
        tiny_bfs_slo(st["median_ms"])
        if info["op"] == "bfs_sparse" else {}
    )
    add("bfs", "warm_single_source", st, levels=info["iterations"],
        path=info["path"], variant=info["op"], source=0,
        source_out_degree=int(out_degree[0]),
        **bfs_diagnostics(single_result), **slo)

    if info["op"] == "bfs_sparse":
        # At microsecond scale a ratio measures output contracts, not
        # engines (the two dense int32[V] result arrays alone cost ~9 us at
        # V=100k). Record the absolute-latency assessment on the actual
        # measured row; same-contract ratios (igraph dense) remain in the
        # baseline rows, and rustworkx bfs_layers is excluded from gates as
        # a different output contract.
        st_sp = stats_ms(
            lambda: mg.bfs(g, s0, direction="out", output="sparse"), runs)
        info_sp = mg.last_run_info()
        add("bfs", "warm_single_source_sparse_output", st_sp,
            path=info_sp["path"], variant=info_sp["op"], source=0,
            semantics="opt-in sparse output: (vertices, dist, parent) of "
                      "length |reached|; context for the dense SLO row")

    high_source = int(np.argmax(out_degree)) if v else 0
    high = np.asarray([high_source], np.uint32)
    st = stats_ms(lambda: mg.bfs(g, high, direction="out"), runs)
    info = mg.last_run_info()
    high_result = mg.bfs(g, high, direction="out")
    add("bfs", "warm_high_degree_source", st, levels=info["iterations"],
        path=info["path"], variant=info["op"], source=high_source,
        source_out_degree=int(out_degree[high_source]),
        **bfs_diagnostics(high_result))

    src64 = np.arange(0, v, max(v // 64, 1), dtype=np.uint32)[:64]
    st = stats_ms(lambda: mg.bfs(g, src64, direction="out"), runs)
    info = mg.last_run_info()
    add("bfs", "warm_64_sources", st, path=info["path"],
        variant=info["op"])

    # -- wcc -----------------------------------------------------------------
    st = stats_ms(lambda: mg.experimental.wcc(g), runs)
    info = mg.last_run_info()
    add("wcc", "warm", st, rounds=info["iterations"], path=info["path"])

    # -- k_hop -----------------------------------------------------------
    st = stats_ms(lambda: mg.k_hop(g, s0, k=2, direction="both",
                                   max_vertices=100_000,
                                   max_edges=1_000_000), runs)
    add("k_hop", "warm_k2_capped", st)

    bench_baselines(name, data, runs, add)
    return g


def bench_baselines(name, data, runs, add):
    src, dst, w, v, directed = data
    e = len(src)

    # NetworkX: context only, capped.
    if e <= NX_MAX_EDGES:
        import networkx as nx
        gx = nx.DiGraph() if directed else nx.Graph()
        gx.add_nodes_from(range(v))
        if w is None:
            gx.add_edges_from(zip(src.tolist(), dst.tolist()))
        else:
            gx.add_weighted_edges_from(
                zip(src.tolist(), dst.tolist(),
                    w.astype(np.float64).tolist()))
        st = stats_ms(lambda: nx.pagerank(gx, alpha=0.85, tol=1e-6,
                                          max_iter=100),
                      max(3, runs // 5), warmup=1)
        add("pagerank", "baseline_networkx", st)
    else:
        add("pagerank", "baseline_networkx",
            {"median_ms": float("nan"), "p95_ms": float("nan"), "runs": 0},
            note=f"skipped: E={e:,} > cap {NX_MAX_EDGES:,}")

    # SciPy sparse power iteration (if importable).
    scipy_sparse = optional_import("scipy.sparse")
    if scipy_sparse is not None:
        data_w = np.ones(e, np.float32) if w is None else w
        adj = scipy_sparse.csr_matrix(
            (data_w, (src.astype(np.int64), dst.astype(np.int64))),
            shape=(v, v))
        ows = np.asarray(adj.sum(axis=1)).ravel()
        inv = np.where(ows > 0, 1.0 / np.where(ows > 0, ows, 1.0), 0.0)
        p_t = adj.T.multiply(inv).tocsr()
        alpha = 0.85

        def power_iter(n_iter=20):
            r = np.full(v, 1.0 / v)
            for _ in range(n_iter):
                dangling = r[ows == 0].sum()
                r = alpha * (p_t @ r + dangling / v) + (1 - alpha) / v
            return r

        st = stats_ms(power_iter, max(3, runs // 5), warmup=1)
        add("pagerank", "baseline_scipy_20iter", st)
    else:
        add("pagerank", "baseline_scipy_20iter",
            {"median_ms": float("nan"), "p95_ms": float("nan"), "runs": 0},
            note="scipy not installed")

    # Pure-python BFS (deque) context.
    if e <= PY_BFS_MAX_EDGES:
        adj = [[] for _ in range(v)]
        for a, b in zip(src.tolist(), dst.tolist()):
            adj[a].append(b)
            if not directed:
                adj[b].append(a)

        def py_bfs():
            dist = [-1] * v
            dist[0] = 0
            q = deque([0])
            while q:
                u = q.popleft()
                du = dist[u]
                for nb in adj[u]:
                    if dist[nb] < 0:
                        dist[nb] = du + 1
                        q.append(nb)
            return dist

        st = stats_ms(py_bfs, max(3, runs // 5), warmup=1)
        add("bfs", "baseline_python_deque", st)

    # rustworkx / igraph: feature-detected, never a dependency. BFS and
    # WCC are the plan-§10 ship-gate baseline shapes; pagerank is included
    # because both libraries expose it trivially. Absent libraries produce
    # explicit 'not installed' rows — never silently skipped.
    b_runs = max(3, runs // 5)

    def nan_stats():
        return {"median_ms": float("nan"), "p95_ms": float("nan"),
                "runs": 0}

    def baseline(algo, item, fn, sample_runs=None, **extra):
        try:
            st = stats_ms(
                fn, b_runs if sample_runs is None else sample_runs, warmup=1
            )
        except Exception as err:  # API drift across versions: report it
            add(algo, item, nan_stats(), note=f"failed: {err}")
            return
        add(algo, item, st, **extra)

    rx = optional_import("rustworkx")
    if rx is None:
        for algo in ("pagerank", "bfs", "wcc"):
            add(algo, "baseline_rustworkx", nan_stats(),
                note="not installed")
    else:
        gr, rx_weight = build_rustworkx_graph(
            rx, src, dst, w, v, directed
        )
        baseline("pagerank", "baseline_rustworkx",
                 lambda: rx.pagerank(gr, alpha=0.85, tol=1e-6,
                                     max_iter=100, weight_fn=rx_weight),
                 weighted=w is not None)
        degree = stored_out_degrees(src, dst, v, directed)
        # A handful of samples cannot support a microsecond-scale p95.
        # Calibrate from one traversal rather than source degree: a degree-1
        # source can still enter a giant component on SNAP/RMAT.
        def rx_bfs_noop():
            return rx.bfs_search(gr, [0], rx.visit.BFSVisitor())

        rx_bfs_dense = make_rustworkx_dense_bfs(rx, gr, v, 0)
        probe_ms, _ = timed(rx_bfs_dense)
        bfs_runs = max(200, runs) if probe_ms < 2.0 else b_runs
        baseline("bfs", "baseline_rustworkx", rx_bfs_dense, source=0,
                 semantics="dense int32 dist+parent",
                 sample_runs=bfs_runs)
        baseline("bfs", "baseline_rustworkx_noop", rx_bfs_noop,
                 source=0, semantics="traversal only; no returned result",
                 sample_runs=bfs_runs)
        baseline("bfs", "baseline_rustworkx_layers",
                 lambda: rx.bfs_layers(gr, [0]), source=0,
                 semantics="sparse layers; no parent array — different "
                           "output contract, context only, excluded from "
                           "gates",
                 sample_runs=bfs_runs)
        high_source = int(np.argmax(degree)) if v else 0
        rx_bfs_high = make_rustworkx_dense_bfs(
            rx, gr, v, high_source
        )
        baseline("bfs", "baseline_rustworkx_high_degree", rx_bfs_high,
                 source=high_source,
                 source_out_degree=int(degree[high_source]),
                 semantics="dense int32 dist+parent")
        rx_wcc = (rx.weakly_connected_components if directed
                  else rx.connected_components)
        baseline("wcc", "baseline_rustworkx", lambda: rx_wcc(gr))

    ig = optional_import("igraph")
    if ig is None:
        for algo in ("pagerank", "bfs", "wcc"):
            add(algo, "baseline_igraph", nan_stats(),
                note="not installed")
    else:
        gi, ig_weight = build_igraph_graph(ig, src, dst, w, v, directed)
        baseline("pagerank", "baseline_igraph",
                 lambda: gi.pagerank(damping=0.85, weights=ig_weight),
                 weighted=w is not None)

        ig_bfs_source_zero = make_igraph_dense_bfs(gi, v, 0)
        probe_ms, _ = timed(ig_bfs_source_zero)
        ig_bfs_runs = max(200, runs) if probe_ms < 2.0 else b_runs
        baseline("bfs", "baseline_igraph", ig_bfs_source_zero, source=0,
                 semantics="dense int32 dist+parent",
                 sample_runs=ig_bfs_runs)
        degree = stored_out_degrees(src, dst, v, directed)
        high_source = int(np.argmax(degree)) if v else 0

        ig_bfs_high_degree = make_igraph_dense_bfs(
            gi, v, high_source
        )
        probe_ms, _ = timed(ig_bfs_high_degree)
        ig_high_runs = max(200, runs) if probe_ms < 2.0 else b_runs
        baseline("bfs", "baseline_igraph_high_degree", ig_bfs_high_degree,
                 source=high_source,
                 source_out_degree=int(degree[high_source]),
                 semantics="dense int32 dist+parent",
                 sample_runs=ig_high_runs)
        ig_wcc = getattr(gi, "connected_components", None) or gi.clusters
        baseline("wcc", "baseline_igraph", lambda: ig_wcc(mode="weak"))

        # Plan-§8 PPR-gate comparator: per-query personalized-PageRank loop
        # with top-64 extraction over the SAME 16 queries as the
        # warm_batch16_k64 line item (make_queries is seeded). igraph solves
        # via PRPACK — an exact direct solver with no iteration-count
        # control — so "identical iteration counts" is not achievable; the
        # row notes the solver difference instead of pretending parity.
        if v >= 64:
            q_seeds, q_weights, q_offsets = make_queries(v, batch=16)
            k_gate = min(64, v)

            def ig_ppr_loop():
                out = []
                reset = np.zeros(v, dtype=np.float64)
                for qi in range(len(q_offsets) - 1):
                    lo, hi = int(q_offsets[qi]), int(q_offsets[qi + 1])
                    reset[q_seeds[lo:hi]] = q_weights[lo:hi]
                    scores = np.asarray(
                        gi.personalized_pagerank(
                            damping=0.85,
                            reset=reset,
                            weights=ig_weight,
                        )
                    )
                    top = np.argpartition(-scores, k_gate - 1)[:k_gate]
                    out.append(top[np.argsort(-scores[top], kind="stable")])
                    reset[q_seeds[lo:hi]] = 0.0
                return out

            baseline("ppr_topk", "baseline_igraph_query_loop", ig_ppr_loop,
                     note="B=16 sequential personalized_pagerank + top-64; "
                          "PRPACK exact solver (no iteration-count control)",
                     weighted=w is not None)


def build_rustworkx_graph(rx, src, dst, weights, v, directed):
    graph = rx.PyDiGraph() if directed else rx.PyGraph()
    graph.add_nodes_from(range(v))
    endpoints = zip(src.tolist(), dst.tolist())
    if weights is None:
        graph.add_edges_from_no_data(list(endpoints))
        weight_fn = None
    else:
        graph.add_edges_from([
            (source, target, float(weight))
            for (source, target), weight in zip(endpoints, weights)
        ])
        weight_fn = float
    return graph, weight_fn


def build_igraph_graph(ig, src, dst, weights, v, directed):
    graph = ig.Graph(
        n=v,
        edges=list(zip(src.tolist(), dst.tolist())),
        directed=directed,
    )
    if weights is None:
        weight_arg = None
    else:
        graph.es["weight"] = weights.astype(np.float64).tolist()
        weight_arg = "weight"
    return graph, weight_arg


# ---------------------------------------------------------------------------
# pipeline warm probe (process-level cold start)
# ---------------------------------------------------------------------------

def pipeline_warm_probe(mg, rows):
    METER.mark()
    src = np.arange(9, dtype=np.uint32)
    dst = (np.arange(9, dtype=np.uint32) + 1) % 10
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=10)
    first, _ = timed(lambda: mg.pagerank(g, alpha=0.85, tol=1e-6,
                                         max_iter=8))
    second, _ = timed(lambda: mg.pagerank(g, alpha=0.85, tol=1e-6,
                                          max_iter=8))
    row = {"dataset": "tiny10", "algo": "pipeline",
           "item": "first_op_vs_second",
           "median_ms": first, "p95_ms": float("nan"), "runs": 1,
           "second_ms": second,
           "pipeline_warm_estimate_ms": max(first - second, 0.0)}
    row.update(METER.close())
    rows.append(row)
    print(f"  pipeline warm: first tiny op {first:.3f} ms, "
          f"second {second:.3f} ms")


# ---------------------------------------------------------------------------
# environment metadata / reports
# ---------------------------------------------------------------------------

def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def collect_meta(mg, args):
    versions = {"metal_graph": getattr(mg, "__version__", "unknown"),
                "numpy": np.__version__}
    for pkg in ("networkx", "scipy", "rustworkx", "igraph"):
        m = optional_import(pkg)
        versions[pkg] = getattr(m, "__version__", None) if m else \
            "not installed"
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        timeout=10, cwd=BENCH_DIR.parent
    )
    git_dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
        timeout=10, cwd=BENCH_DIR.parent
    )
    xcode = subprocess.run(
        ["xcrun", "xcodebuild", "-version"], capture_output=True, text=True,
        timeout=10
    )
    native_module = Path(mg._core.__file__)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chip": sysctl("machdep.cpu.brand_string"),
        "macos": platform.mac_ver()[0],
        "python": sys.version.split()[0],
        "versions": versions,
        "suite": args.suite,
        "runs": args.runs,
        "argv": sys.argv[1:],
        "env": {k: v for k, v in os.environ.items()
                if k.startswith("MG_")},
        "git_sha": git_sha.stdout.strip() if git_sha.returncode == 0
        else "unknown",
        "git_dirty": bool(git_dirty.stdout.strip())
        if git_dirty.returncode == 0 else None,
        "xcode": xcode.stdout.strip() if xcode.returncode == 0 else "unknown",
        "native_module_sha256": file_sha256(native_module),
        "snap_datasets": SNAP_DATASETS,
    }


def render_markdown(meta, rows):
    dirty = meta.get("git_dirty")
    dirty_label = "dirty" if dirty is True else \
        "clean" if dirty is False else "unknown"
    git_sha = str(meta.get("git_sha", "unknown"))
    xcode = str(meta.get("xcode", "unknown"))
    lines = [
        f"# metal-graph bench — {meta['suite']} — {meta['timestamp_utc']}",
        "",
        f"- chip: `{meta['chip']}`  · macOS {meta['macos']} · "
        f"python {meta['python']} · metal_graph "
        f"{meta['versions']['metal_graph']}",
        f"- source: git `{git_sha[:12]}` ({dirty_label}) · "
        f"{xcode.splitlines()[0]}",
        "",
        "| dataset | algo | line item | median ms | p95 ms | "
        "peak RSS Δ MB | notes |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        notes = []
        note_keys = [
            "path", "iterations", "levels", "rounds", "note",
            "amortized_per_query_ms", "achieved_gb_s",
        ]
        # Preserve byte-for-byte rendering of legacy artifacts while showing
        # the richer fields on rows emitted by the corrected BFS harness.
        if "variant" in r:
            note_keys.extend([
                "variant", "source", "source_out_degree",
                "reached_vertices", "reachable_edges_scanned",
                "max_distance", "frontier_sizes",
            ])
        if "semantics" in r:
            note_keys.extend([
                "source", "source_out_degree", "semantics",
            ])
        if "slo_ms" in r:
            note_keys.extend(["slo_ms", "slo_pass", "gate"])
        for k in note_keys:
            if k in r and r[k] is not None:
                val = r[k]
                if isinstance(val, float):
                    val = f"{val:.4g}"
                notes.append(f"{k}={val}")
        med = r.get("median_ms")
        p95 = r.get("p95_ms")
        rssd = r.get("peak_rss_delta_mb")
        med = float("nan") if med is None else med
        p95 = float("nan") if p95 is None else p95
        rssd = float("nan") if rssd is None else rssd
        lines.append(f"| {r['dataset']} | {r['algo']} | {r['item']} | "
                     f"{med:.3f} | {p95:.3f} | {rssd:.1f} | "
                     f"{', '.join(notes)} |")
    return "\n".join(lines) + "\n"


def json_safe(value):
    """Replace non-finite floats so benchmark output is strict JSON."""
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# contention / energy (opt-in, feature-detected)
# ---------------------------------------------------------------------------

def run_contention(mg, args, rows):
    mlx_lm = optional_import("mlx_lm")
    if mlx_lm is None:
        print("--contention requires mlx_lm, which is not installed.\n"
              "  pip install mlx-lm\n"
              "then rerun with:\n"
              "  MG_BENCH_MLX_MODEL=<hf-model-id-or-path> "
              "python3 bench/run.py --contention ...")
        sys.exit(2)
    model_id = os.environ.get("MG_BENCH_MLX_MODEL")
    if not model_id:
        print("--contention needs MG_BENCH_MLX_MODEL "
              "(e.g. mlx-community/Llama-3.2-1B-Instruct-4bit).")
        sys.exit(2)
    import threading
    print(f"[contention] loading {model_id}")
    model, tokenizer = mlx_lm.load(model_id)
    stop = threading.Event()
    tokens = {"n": 0, "s": 0.0}

    def decode_loop():
        prompt = "Explain unified memory in one paragraph."
        while not stop.is_set():
            t0 = time.perf_counter()
            text = mlx_lm.generate(model, tokenizer, prompt=prompt,
                                   max_tokens=128, verbose=False)
            tokens["s"] += time.perf_counter() - t0
            tokens["n"] += 128 if text else 0

    src, dst, w, v, directed = gen_kg()
    g = mg.Graph.from_edges(src, dst, weights=w, directed=directed,
                            num_vertices=v)
    seeds, weights, offsets = make_queries(v, batch=16)
    kw = dict(k=64, alpha=0.85, tol=1e-6, max_iter=50)
    solo = stats_ms(lambda: mg.ppr_topk(g, seeds, weights, offsets, **kw),
                    args.runs)
    th = threading.Thread(target=decode_loop, daemon=True)
    th.start()
    time.sleep(3.0)  # let the decode loop reach steady state
    METER.mark()  # row timestamps bracket the CONTENDED window only
    contended = stats_ms(
        lambda: mg.ppr_topk(g, seeds, weights, offsets, **kw), args.runs)
    row = {"dataset": "kg-hipporag", "algo": "ppr_topk",
           "item": "contention_vs_solo",
           "median_ms": contended["median_ms"],
           "p95_ms": contended["p95_ms"], "runs": args.runs,
           "solo_median_ms": solo["median_ms"],
           "inflation": contended["median_ms"] / solo["median_ms"]}
    row.update(METER.close())
    stop.set()
    th.join(timeout=60)
    tps = tokens["n"] / tokens["s"] if tokens["s"] else float("nan")
    row["mlx_tokens_per_s_during"] = tps
    row["model"] = model_id
    rows.append(row)
    print(f"[contention] solo {solo['median_ms']:.2f} ms -> contended "
          f"{contended['median_ms']:.2f} ms; mlx {tps:.1f} tok/s")


@dataclass
class EnergyCaptureHandle:
    process: subprocess.Popen
    output_stream: object
    output_path: Path | None = None


def start_energy_capture(out_path):
    if os.geteuid() == 0:
        print("Refusing to run the benchmark harness as root. Run `sudo -v` "
              "first, then invoke the harness as your normal user.\n"
              "Methodology: bench/ENERGY.md")
        sys.exit(2)
    auth = subprocess.run(
        ["sudo", "-n", "true"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    if auth.returncode != 0:
        print("--energy needs a current sudo credential for powermetrics.\n"
              "Run `sudo -v`, then rerun this command without sudo.\n"
              "Methodology: bench/ENERGY.md")
        sys.exit(2)
    output_stream = out_path.open("xb")
    try:
        process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "/usr/bin/powermetrics",
                "-i",
                "100",
                "--samplers",
                "cpu_power,gpu_power",
            ],
            stdout=output_stream,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        output_stream.close()
        out_path.unlink(missing_ok=True)
        raise
    return EnergyCaptureHandle(
        process=process,
        output_stream=output_stream,
        output_path=out_path,
    )


def stop_energy_capture(handle):
    """Stop sudo, which relays the signal to its powermetrics command."""
    try:
        status = handle.process.poll()
        if status is None:
            handle.process.terminate()
            try:
                handle.process.wait(timeout=30)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "powermetrics did not stop after SIGTERM"
                ) from error
        elif status != 0:
            raise RuntimeError(
                f"powermetrics exited before cleanup with status {status}"
            )
    finally:
        handle.output_stream.close()
    if (handle.output_path is not None and
            handle.output_path.stat().st_size == 0):
        raise RuntimeError("powermetrics produced an empty capture")


@contextmanager
def energy_capture(enabled, out_path):
    proc = start_energy_capture(out_path) if enabled else None
    try:
        yield proc
    finally:
        if proc is not None:
            stop_energy_capture(proc)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_suite(suite):
    datasets = [("rmat18", lambda: gen_rmat(18)),
                ("kg-hipporag", lambda: gen_kg())]
    if suite == "v01":
        datasets.append(("rmat22", lambda: gen_rmat(22)))
        # RMAT-24 (V=16.8M, E=268M) is the plan-§10 ship-gate scale point;
        # deliberately NOT in smoke — smoke stays fast.
        datasets.append(("rmat24", lambda: gen_rmat(24)))
        for name in SNAP_DATASETS:
            datasets.append((name, lambda n=name: load_snap(n)))
    return datasets


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", choices=["smoke", "v01"], default="smoke")
    ap.add_argument("--runs", type=int, default=20,
                    help="warm-run repetitions per line item (>=20 for "
                         "publishable numbers)")
    ap.add_argument("--fetch", action="store_true",
                    help="download SNAP datasets (soc-LiveJournal1, "
                         "com-orkut) into bench/data/ — never implicit")
    ap.add_argument("--contention", action="store_true",
                    help="run the PPR batch under an mlx_lm decode loop")
    ap.add_argument("--energy", action="store_true",
                    help="capture powermetrics (requires sudo)")
    ap.add_argument("--idle-seconds", type=int, default=60,
                    help="with --energy: seconds of idle baseline captured "
                         "after powermetrics starts and before any "
                         "workload (bench/ENERGY.md); 0 disables; ignored "
                         "without --energy")
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args()

    print("SNAP datasets (downloaded only with --fetch):")
    for name, spec in SNAP_DATASETS.items():
        print(f"  {name}: {spec['url']}")
    if args.fetch:
        fetch_datasets()

    try:
        import metal_graph as mg
    except ImportError:
        print("\nerror: metal_graph not importable. Build first, then run "
              "from the repo root with PYTHONPATH=python.")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    energy_path = out_dir / f"powermetrics-{stamp}.txt"
    idle_window = None
    with energy_capture(args.energy, energy_path) as energy_proc:
        if energy_proc is not None and args.idle_seconds > 0:
            # Idle baseline INSIDE the powermetrics capture, before any
            # workload: its mean package power is what gets subtracted
            # from every line-item window (bench/ENERGY.md). The window
            # is recorded in the JSON so it aligns with the sample stream.
            print(f"[energy] capturing {args.idle_seconds}s idle baseline "
                  f"(do not touch the machine)")
            t0 = datetime.now(timezone.utc)
            time.sleep(args.idle_seconds)
            idle_window = {
                "t_start_utc": t0.isoformat(),
                "t_end_utc": datetime.now(timezone.utc).isoformat(),
                "seconds": args.idle_seconds,
            }

        meta = collect_meta(mg, args)
        if idle_window is not None:
            meta["energy_idle_window"] = idle_window
        print(f"\nchip: {meta['chip']} · macOS {meta['macos']} · "
              f"metal_graph {meta['versions']['metal_graph']}")

        rows = []
        METER.mark()  # exclude import/fetch/idle time from the first item
        pipeline_warm_probe(mg, rows)
        for name, load in build_suite(args.suite):
            data = load()
            if data is None:
                print(f"\n=== {name}: not fetched — rerun with --fetch ===")
                rows.append({
                    "dataset": name,
                    "algo": "-",
                    "item": "not_fetched",
                    "median_ms": float("nan"),
                    "p95_ms": float("nan"),
                    "runs": 0,
                    "note": "run bench/run.py --fetch",
                })
                continue
            bench_dataset(mg, name, data, args.runs, rows)

        if args.contention:
            run_contention(mg, args, rows)

    if energy_proc is not None:
        meta["powermetrics_file"] = str(energy_path)

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "meta": meta,
        "rows": rows,
    }
    json_path = out_dir / f"bench-{stamp}.json"
    json_path.write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False)
    )
    md = render_markdown(meta, rows)
    md_path = out_dir / f"bench-{stamp}.md"
    md_path.write_text(md)
    print(f"\nresults: {json_path}\n         {md_path}\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
