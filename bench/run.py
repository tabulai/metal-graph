#!/usr/bin/env python3
"""metal-graph benchmark harness (v0.1, plan section 10).

Decomposed, honest reporting: build / transpose / pipeline warm / warm
kernels / convergence audit / top-k / Python boundary are separate line
items; every kernel cell is median + p95 over >= --runs runs. Baselines
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"

SNAP_URLS = {
    "soc-LiveJournal1": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
    "com-orkut": "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
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


# ---------------------------------------------------------------------------
# SNAP datasets (download ONLY with --fetch; cached under bench/data/)
# ---------------------------------------------------------------------------

def fetch_datasets():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in SNAP_URLS.items():
        path = DATA_DIR / (name + ".txt.gz")
        if path.exists():
            print(f"[fetch] cached: {path}")
            continue
        print(f"[fetch] downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)  # noqa: S310 (explicit opt-in)
    print("[fetch] done")


def load_snap(name):
    """Parse an edge-list .txt.gz (cached as .npz after first parse).
    Returns None when the dataset was never fetched."""
    gz = DATA_DIR / (name + ".txt.gz")
    npz = DATA_DIR / (name + ".npz")
    if npz.exists():
        z = np.load(npz)
        return z["src"], z["dst"], None, int(z["v"]), bool(z["directed"])
    if not gz.exists():
        return None
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
    v = int(max(src.max(), dst.max())) + 1
    directed = name != "com-orkut"
    np.savez_compressed(npz, src=src, dst=dst, v=v, directed=directed)
    return src, dst, None, v, directed


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------

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
    st = stats_ms(lambda: mg.bfs(g, s0, direction="out"), runs)
    info = mg.last_run_info()
    add("bfs", "warm_single_source", st, levels=info["iterations"],
        path=info["path"])
    src64 = np.arange(0, v, max(v // 64, 1), dtype=np.uint32)[:64]
    st = stats_ms(lambda: mg.bfs(g, src64, direction="out"), runs)
    add("bfs", "warm_64_sources", st)

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

    def baseline(algo, item, fn, **extra):
        try:
            st = stats_ms(fn, b_runs, warmup=1)
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
        gr = rx.PyDiGraph() if directed else rx.PyGraph()
        gr.add_nodes_from(range(v))
        gr.add_edges_from_no_data(list(zip(src.tolist(), dst.tolist())))
        baseline("pagerank", "baseline_rustworkx",
                 lambda: rx.pagerank(gr, alpha=0.85, tol=1e-6,
                                     max_iter=100))
        baseline("bfs", "baseline_rustworkx",
                 lambda: rx.bfs_search(gr, [0], rx.visit.BFSVisitor()),
                 source=0)
        rx_wcc = (rx.weakly_connected_components if directed
                  else rx.connected_components)
        baseline("wcc", "baseline_rustworkx", lambda: rx_wcc(gr))

    ig = optional_import("igraph")
    if ig is None:
        for algo in ("pagerank", "bfs", "wcc"):
            add(algo, "baseline_igraph", nan_stats(),
                note="not installed")
    else:
        gi = ig.Graph(n=v, edges=list(zip(src.tolist(), dst.tolist())),
                      directed=directed)
        baseline("pagerank", "baseline_igraph",
                 lambda: gi.pagerank(damping=0.85))
        baseline("bfs", "baseline_igraph", lambda: gi.bfs(0), source=0)
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
                        gi.personalized_pagerank(damping=0.85, reset=reset))
                    top = np.argpartition(-scores, k_gate - 1)[:k_gate]
                    out.append(top[np.argsort(-scores[top], kind="stable")])
                    reset[q_seeds[lo:hi]] = 0.0
                return out

            baseline("ppr_topk", "baseline_igraph_query_loop", ig_ppr_loop,
                     note="B=16 sequential personalized_pagerank + top-64; "
                          "PRPACK exact solver (no iteration-count control)")


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
    }


def render_markdown(meta, rows):
    lines = [
        f"# metal-graph bench — {meta['suite']} — {meta['timestamp_utc']}",
        "",
        f"- chip: `{meta['chip']}`  · macOS {meta['macos']} · "
        f"python {meta['python']} · metal_graph "
        f"{meta['versions']['metal_graph']}",
        "",
        "| dataset | algo | line item | median ms | p95 ms | "
        "peak RSS Δ MB | notes |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        notes = []
        for k in ("path", "iterations", "levels", "rounds", "note",
                  "amortized_per_query_ms", "achieved_gb_s"):
            if k in r and r[k] is not None:
                val = r[k]
                if isinstance(val, float):
                    val = f"{val:.4g}"
                notes.append(f"{k}={val}")
        med = r.get("median_ms", float("nan"))
        p95 = r.get("p95_ms", float("nan"))
        rssd = r.get("peak_rss_delta_mb", float("nan"))
        lines.append(f"| {r['dataset']} | {r['algo']} | {r['item']} | "
                     f"{med:.3f} | {p95:.3f} | {rssd:.1f} | "
                     f"{', '.join(notes)} |")
    return "\n".join(lines) + "\n"


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


def start_energy_capture(out_path):
    if os.geteuid() != 0:
        print("--energy needs powermetrics, which requires sudo.\n"
              "Politely refusing: rerun as\n"
              "  sudo PYTHONPATH=python python3 bench/run.py --energy ...\n"
              "Methodology: bench/ENERGY.md")
        sys.exit(2)
    proc = subprocess.Popen(
        ["powermetrics", "-i", "100",
         "--samplers", "cpu_power,gpu_power",
         "-o", str(out_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


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
        for name in SNAP_URLS:
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
    for name, url in SNAP_URLS.items():
        print(f"  {name}: {url}")
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

    energy_proc = None
    idle_window = None
    energy_path = out_dir / f"powermetrics-{stamp}.txt"
    if args.energy:
        energy_proc = start_energy_capture(energy_path)
        if args.idle_seconds > 0:
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
            rows.append({"dataset": name, "algo": "-", "item": "not_fetched",
                         "median_ms": float("nan"), "p95_ms": float("nan"),
                         "runs": 0, "note": "run bench/run.py --fetch"})
            continue
        bench_dataset(mg, name, data, args.runs, rows)

    if args.contention:
        run_contention(mg, args, rows)

    if energy_proc is not None:
        energy_proc.terminate()
        energy_proc.wait(timeout=30)
        meta["powermetrics_file"] = str(energy_path)

    result = {"meta": meta, "rows": rows}
    json_path = out_dir / f"bench-{stamp}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))
    md = render_markdown(meta, rows)
    md_path = out_dir / f"bench-{stamp}.md"
    md_path.write_text(md)
    print(f"\nresults: {json_path}\n         {md_path}\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
