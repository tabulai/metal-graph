# conftest.py — shared fixtures and reference helpers for the metal-graph
# v0.1 golden test suite.
#
# Tests are written AGAINST the documented API (README.md and
# src/algos/algos.hpp) — they are the acceptance bar for the implementation.
# NetworkX is the only golden oracle. Everything here must COLLECT without
# metal_graph installed: the module is imported lazily via
# pytest.importorskip inside fixtures/helpers.
#
# All fixture graphs use identity-mode ids (integer ids 0..V-1 with
# num_vertices passed explicitly), so user index == node id == nx node.

import os

# Tests assume the documented default audit interval (5); pin before any
# metal_graph import (conftest is imported before every test module).
os.environ.setdefault("MG_PR_AUDIT_INTERVAL", "5")

import dataclasses
import subprocess
import sys
import zlib
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: test requires a Metal device (skipped without one)")


@pytest.fixture(scope="session", autouse=True)
def _audit_interval_guard():
    # Session guard: keep the documented default in place for every test.
    os.environ.setdefault("MG_PR_AUDIT_INTERVAL", "5")
    yield


# ---------------------------------------------------------------------------
# metal_graph access (lazy — collection must work without the module)
# ---------------------------------------------------------------------------

def mg_mod():
    return pytest.importorskip("metal_graph")


def has_gpu():
    core = pytest.importorskip("metal_graph._core")
    return bool(core.has_gpu())


@pytest.fixture(params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def exec_path(request):
    """Pin the execution path for the test. Under 'auto' the default
    MG_E_GPU_MIN (1M stored edges) would route every fixture graph to the
    CPU — pinning is the only way to exercise both paths."""
    mg = mg_mod()
    path = request.param
    if path == "gpu" and not has_gpu():
        pytest.skip("no Metal device")
    mg.set_execution(path)
    yield path
    mg.set_execution("auto")


def bfs_expected_path(directed, direction, exec_path):
    """BFS direction='both' on DIRECTED graphs is CPU-routed in v0.1 (a
    documented limitation); telemetry reports the truth, so tests expect it."""
    if directed and direction == "both" and exec_path == "gpu":
        return "cpu"
    return exec_path


@pytest.fixture
def assert_path():
    """assert_path('cpu'|'gpu'): the last op really ran on that path."""
    mg = mg_mod()

    def _check(path):
        info = mg.last_run_info()
        assert info["path"] == path, (
            f"planner telemetry: expected path={path!r}, got {info!r}")

    return _check


# ---------------------------------------------------------------------------
# Graph fixture matrix (deterministic, V <= 400 so NetworkX stays fast)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GraphCase:
    name: str
    src: np.ndarray  # uint32 user indices
    dst: np.ndarray
    directed: bool
    num_vertices: int

    @property
    def n_edges(self):
        return int(self.src.size)


def _case(name, src, dst, directed, v):
    src = np.asarray(src, dtype=np.uint32)
    dst = np.asarray(dst, dtype=np.uint32)
    assert src.size == dst.size
    if src.size:
        assert int(max(src.max(), dst.max())) < v
    return GraphCase(name, src, dst, directed, v)


def _dedup_pairs(src, dst, directed):
    if len(src) == 0:
        return np.array([], np.uint32), np.array([], np.uint32)
    s = np.asarray(src, np.uint64)
    d = np.asarray(dst, np.uint64)
    if not directed:
        lo, hi = np.minimum(s, d), np.maximum(s, d)
        s, d = lo, hi
    key = (s << np.uint64(32)) | d
    _, idx = np.unique(key, return_index=True)
    idx.sort()
    return np.asarray(src)[idx], np.asarray(dst)[idx]


def _gnp_case(name, v, p, directed, seed):
    rng = np.random.default_rng(seed)
    m = rng.random((v, v)) < p
    np.fill_diagonal(m, False)
    if not directed:
        m = np.triu(m, 1)
    s, d = np.nonzero(m)
    # Add a ring so every vertex 0..V-1 appears in the edge list.
    ring = np.arange(v)
    s = np.concatenate([s, ring])
    d = np.concatenate([d, (ring + 1) % v])
    s, d = _dedup_pairs(s, d, directed)
    return _case(name, s, d, directed, v)


def _forest_case():
    # Two-component undirected forest: binary tree on 0..14, star on 15..29.
    s, d = [], []
    for i in range(1, 15):
        s.append((i - 1) // 2)
        d.append(i)
    for j in range(16, 30):
        s.append(15)
        d.append(j)
    return _case("forest2", s, d, False, 30)


def _multi_self_case():
    v = 25
    ring = np.arange(v)
    s = list(ring) + [0, 0, 0, 2, 2, 5, 5, 7, 9, 9]
    d = list((ring + 1) % v) + [1, 1, 1, 3, 3, 5, 5, 7, 3, 3]
    return _case("multi_self", s, d, True, v)


def _dangling_case():
    # 30% sinks (zero out-degree), every vertex present in the edge list.
    v, n_sink, seed = 100, 30, 7
    rng = np.random.default_rng(seed)
    sinks = np.arange(v - n_sink, v)
    s, d = [], []
    for u in range(v - n_sink):
        deg = int(rng.integers(1, 6))
        targets = rng.choice(v - 1, size=deg, replace=False)
        targets[targets >= u] += 1  # no self-loops here
        s.extend([u] * deg)
        d.extend(int(t) for t in targets)
    hit = set(d)
    for t in sinks:
        if int(t) not in hit:
            s.append(int(rng.integers(0, v - n_sink)))
            d.append(int(t))
    return _case("dangling", s, d, True, v)


def _isolated_case():
    # Edges only among 0..39; num_vertices=60 adds isolated tail vertices.
    rng = np.random.default_rng(13)
    ring = np.arange(40)
    s = list(ring) + [int(x) for x in rng.integers(0, 40, 30)]
    d = list((ring + 1) % 40) + [int(x) for x in rng.integers(0, 40, 30)]
    s, d = _dedup_pairs(s, d, True)
    keep = np.asarray(s) != np.asarray(d)
    return _case("isolated", np.asarray(s)[keep], np.asarray(d)[keep], True, 60)


def _adversarial_case():
    # High-in / high-out disjoint. Out-degree spec vertices 0..7 and
    # in-degree spec vertices 8..15 carry degrees {0,1,31,32,33,1023,1024,
    # 1500} in their respective orientations, crossing both worklist-bin
    # thresholds (32 / 1024). Pool A (16..215) has large IN-degree and zero
    # out-degree; pool B (216..399) provides the OUT edges. Multi-edges by
    # construction (sampling with replacement). Vertices 0 and 8 are fully
    # isolated (degree-0 bins in both orientations).
    rng = np.random.default_rng(42)
    v = 400
    degs = [0, 1, 31, 32, 33, 1023, 1024, 1500]
    pool_a = np.arange(16, 216)
    pool_b = np.arange(216, 400)
    s, d = [], []
    for u, deg in zip(range(0, 8), degs):
        if deg:
            s.append(np.full(deg, u, dtype=np.int64))
            d.append(rng.choice(pool_a, size=deg, replace=True))
    for w, deg in zip(range(8, 16), degs):
        if deg:
            s.append(rng.choice(pool_b, size=deg, replace=True))
            d.append(np.full(deg, w, dtype=np.int64))
    return _case("adversarial", np.concatenate(s), np.concatenate(d), True, v)


def _build_cases():
    cases = [
        _case("star", np.zeros(40, np.uint32), np.arange(1, 41), True, 41),
        _case("path", np.arange(0, 49), np.arange(1, 50), True, 50),
        _case("ring", np.arange(48), (np.arange(48) + 1) % 48, True, 48),
        _case("clique",
              np.repeat(np.arange(18), 17),
              np.concatenate([np.delete(np.arange(18), u)
                              for u in range(18)]), True, 18),
        _case("empty", np.array([], np.uint32), np.array([], np.uint32),
              True, 0),
        _case("singleton", np.array([], np.uint32), np.array([], np.uint32),
              True, 1),
        _forest_case(),
        _gnp_case("gnp_dir", 120, 0.06, True, 11),
        _gnp_case("gnp_undir", 120, 0.06, False, 12),
        _multi_self_case(),
        _dangling_case(),
        _isolated_case(),
        _adversarial_case(),
    ]
    return {c.name: c for c in cases}


ALL_CASES = _build_cases()
CASE_NAMES = list(ALL_CASES)


@pytest.fixture(params=CASE_NAMES)
def graph_case(request):
    return ALL_CASES[request.param]


def make_weights(case):
    """Deterministic weighted twin: uniform(0.1, 2.0) per input edge."""
    rng = np.random.default_rng(zlib.crc32(case.name.encode()) & 0xFFFF)
    return rng.uniform(0.1, 2.0, case.n_edges).astype(np.float32)


def build_mg(case, weights=None):
    mg = mg_mod()
    return mg.Graph.from_edges(case.src, case.dst, weights=weights,
                               directed=case.directed,
                               num_vertices=case.num_vertices)


# ---------------------------------------------------------------------------
# NetworkX golden builders
# ---------------------------------------------------------------------------

def build_nx_multi(case, weights=None):
    g = nx.MultiDiGraph() if case.directed else nx.MultiGraph()
    g.add_nodes_from(range(case.num_vertices))
    if weights is None:
        g.add_edges_from(zip(case.src.tolist(), case.dst.tolist()))
    else:
        g.add_weighted_edges_from(
            zip(case.src.tolist(), case.dst.tolist(),
                np.asarray(weights, np.float64).tolist()))
    return g


def build_nx_agg(case, weights=None):
    """Simple (Di)Graph with parallel-edge weights summed — exactly the
    PageRank semantics for duplicate edges (unweighted edges count 1)."""
    g = nx.DiGraph() if case.directed else nx.Graph()
    g.add_nodes_from(range(case.num_vertices))
    w = np.ones(case.n_edges) if weights is None else np.asarray(
        weights, np.float64)
    for u, v, wi in zip(case.src.tolist(), case.dst.tolist(), w.tolist()):
        if g.has_edge(u, v):
            g[u][v]["weight"] += wi
        else:
            g.add_edge(u, v, weight=wi)
    return g


def nx_pagerank_array(case, weights=None, alpha=0.85, tol=1e-10,
                      max_iter=500, personalization=None):
    v = case.num_vertices
    if v == 0:
        return np.zeros(0, dtype=np.float64)
    g = build_nx_agg(case, weights)
    d = nx.pagerank(g, alpha=alpha, tol=tol, max_iter=max_iter,
                    weight="weight", personalization=personalization)
    out = np.zeros(v, dtype=np.float64)
    for node, score in d.items():
        out[node] = score
    return out


def pr_tolerance(path):
    """Per-path comparison tolerance vs the fp64 NetworkX oracle."""
    if path == "cpu":
        return dict(atol=1e-6, rtol=0.0)
    return dict(atol=5e-5, rtol=1e-3)  # fp32 GPU gather


# ---------------------------------------------------------------------------
# Reference BFS / k-hop (pure python, deterministic)
# ---------------------------------------------------------------------------

def adjacency_sets(case, direction):
    """Traversal adjacency over the INPUT arrays, respecting direction."""
    adj = [set() for _ in range(case.num_vertices)]
    for u, v in zip(case.src.tolist(), case.dst.tolist()):
        if case.directed:
            if direction == "out":
                adj[u].add(v)
            elif direction == "in":
                adj[v].add(u)
            else:  # both — undirected view
                adj[u].add(v)
                adj[v].add(u)
        else:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def bfs_dist_ref(case, sources, direction):
    from collections import deque
    adj = adjacency_sets(case, direction)
    dist = np.full(case.num_vertices, -1, dtype=np.int32)
    q = deque()
    for s in sources:
        s = int(s)
        if dist[s] < 0:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def khop_ref(case, seeds, k, direction="both", max_vertices=0, max_edges=0):
    """Reference k-hop per src/algos/algos.hpp: BFS truncated at depth k;
    max_vertices admits boundary-level candidates in ascending user index;
    induced edges = input edge ids with both endpoints reached
    (direction-agnostic membership), sorted ascending, then truncated."""
    adj = adjacency_sets(case, direction)
    reached = {int(s) for s in seeds}
    frontier = sorted(reached)
    depth = 0
    while frontier and depth < k:
        cand = set()
        for u in frontier:
            cand |= adj[u]
        cand -= reached
        cand = sorted(cand)
        if max_vertices:
            room = max_vertices - len(reached)
            cand = cand[:max(room, 0)]
        reached.update(cand)
        frontier = cand
        depth += 1
    vs = np.array(sorted(reached), dtype=np.uint32)
    es = [i for i, (u, v) in enumerate(zip(case.src.tolist(),
                                           case.dst.tolist()))
          if u in reached and v in reached]
    es.sort()
    if max_edges:
        es = es[:max_edges]
    return vs, np.array(es, dtype=np.uint32)


# ---------------------------------------------------------------------------
# top-k reference with the documented tie rule (score desc, user index asc)
# ---------------------------------------------------------------------------

def topk_from_scores(scores, k):
    v = len(scores)
    order = np.lexsort((np.arange(v), -np.asarray(scores, np.float64)))
    kk = min(k, v)
    return order[:kk], np.asarray(scores, np.float64)[order][:kk]


# ---------------------------------------------------------------------------
# Subprocess runner (planner env-var tests)
# ---------------------------------------------------------------------------

def run_python(code, extra_env=None, timeout=180):
    env = os.environ.copy()
    pp = str(PYTHON_DIR)
    if env.get("PYTHONPATH"):
        pp = pp + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pp
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, cwd=str(REPO_ROOT),
                          timeout=timeout)
