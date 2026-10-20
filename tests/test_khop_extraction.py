# test_khop_extraction.py — k_hop induced-edge extraction on a graph large
# enough to reach every host strategy: the threaded full edge-list scan
# (hub neighborhoods; several marking chunks), and the serial CSR walk that
# either sorts its few ids or dedupes them through the edge-id bitmap. The
# choice depends on neighborhood size and MG_THREADS, so each query is
# checked against a NumPy oracle at several thread counts and max_edges
# caps (including caps that cut across 65536-id enumeration blocks).
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")


def _skewed_graph(directed, seed=3, scale=14, n_edges=300_000):
    rng = np.random.default_rng(seed)
    src = np.zeros(n_edges, np.int64)
    dst = np.zeros(n_edges, np.int64)
    for bit in range(scale):  # R-MAT (0.57, 0.19, 0.19)
        ii = rng.random(n_edges) > 0.76
        jj = rng.random(n_edges) > np.where(ii, 0.19 / 0.24, 0.57 / 0.76)
        src |= ii.astype(np.int64) << bit
        dst |= jj.astype(np.int64) << bit
    src[:50] = dst[:50]  # self-loops (duplicates come from R-MAT itself)
    v = 1 << scale
    src = src.astype(np.uint32)
    dst = dst.astype(np.uint32)
    return mg.Graph.from_edges(src, dst, directed=directed,
                               num_vertices=v), src, dst, v


def _reference(src, dst, v, directed, seeds, k, direction):
    reached = np.zeros(v, bool)
    reached[seeds] = True
    frontier = reached.copy()
    fwd = direction in ("out", "both") or not directed
    rev = direction in ("in", "both") or not directed
    for _ in range(k):
        nxt = np.zeros(v, bool)
        if fwd:
            nxt[dst[frontier[src]]] = True
        if rev:
            nxt[src[frontier[dst]]] = True
        nxt &= ~reached
        if not nxt.any():
            break
        reached |= nxt
        frontier = nxt
    es = np.flatnonzero(reached[src] & reached[dst]).astype(np.uint32)
    return np.flatnonzero(reached).astype(np.uint32), es


@pytest.fixture(scope="module", params=[True, False],
                ids=["directed", "undirected"])
def big_graph(request):
    return _skewed_graph(request.param) + (request.param,)


@pytest.mark.parametrize("mode", ["cpu", pytest.param("gpu",
                                                      marks=pytest.mark.gpu)])
def test_khop_extraction_paths_match_oracle(big_graph, mode, monkeypatch):
    g, src, dst, v, directed = big_graph
    if mode == "gpu" and not mg.has_gpu():
        pytest.skip("no Metal device")
    deg = np.bincount(src, minlength=v) + np.bincount(dst, minlength=v)
    order = np.argsort(-deg, kind="stable")
    seeds_list = [[int(order[r])] for r in (0, 3, 30, 300, 1000, 3000, 9000)]
    seeds_list.append([int(x) for x in order[[5, 2000, 7000]]])
    mg.set_execution(mode)
    try:
        for seeds in seeds_list:
            for k in (1, 2):
                for direction in ("out", "in", "both"):
                    vs_ref, es_ref = _reference(src, dst, v, directed,
                                                seeds, k, direction)
                    for threads in ("1", "3", "16"):
                        monkeypatch.setenv("MG_THREADS", threads)
                        for cap in (0, 1, 70_000, 140_000):
                            vs, es = mg.k_hop(g, seeds, k=k,
                                              direction=direction,
                                              max_edges=cap or None)
                            ctx = (seeds, k, direction, threads, cap)
                            np.testing.assert_array_equal(vs, vs_ref,
                                                          err_msg=str(ctx))
                            want = es_ref[:cap] if cap else es_ref
                            np.testing.assert_array_equal(es, want,
                                                          err_msg=str(ctx))
    finally:
        mg.set_execution("auto")
