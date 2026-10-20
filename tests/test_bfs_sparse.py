# test_bfs_sparse.py — opt-in sparse-output BFS: same reachability as the
# dense contract, O(|reached|) output, ascending user order on EVERY
# execution path (bounded serial collect, threaded CPU fallback, forced GPU).
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import ALL_CASES, CASE_NAMES, adjacency_sets, build_mg

DIRECTIONS = ["out", "in", "both"]


def _check_against_dense(case, g, sources, direction):
    vs, dist, parent = mg.bfs(g, np.asarray(sources, np.uint32),
                              direction=direction, output="sparse")
    sparse_info = dict(mg.last_run_info())
    vs, dist, parent = np.asarray(vs), np.asarray(dist), np.asarray(parent)
    dense_dist, _ = mg.bfs(g, np.asarray(sources, np.uint32),
                           direction=direction)
    dense_dist = np.asarray(dense_dist)

    # Exact reached set, ascending, aligned dist values.
    np.testing.assert_array_equal(vs, np.where(dense_dist >= 0)[0])
    np.testing.assert_array_equal(dist, dense_dist[vs])
    assert vs.dtype == np.uint32 and dist.dtype == np.int32
    assert parent.dtype == np.int32 and parent.shape == vs.shape

    # Parents validated structurally (dense parents may differ per path).
    adj = adjacency_sets(case, direction)
    pos = {int(v): i for i, v in enumerate(vs)}
    src_set = set(int(s) for s in sources)
    for i, v in enumerate(vs):
        if dist[i] == 0:
            assert int(v) in src_set and parent[i] == -1
        else:
            p = int(parent[i])
            assert p in pos, f"parent {p} of {v} not in reached set"
            assert dist[pos[p]] == dist[i] - 1
            assert int(v) in adj[p], f"no edge {p} -> {v} ({direction})"
    return sparse_info


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("name", CASE_NAMES)
def test_sparse_matches_dense(name, direction, exec_path):
    case = ALL_CASES[name]
    if case.num_vertices == 0:
        pytest.skip("no valid sources in an empty graph")
    g = build_mg(case)
    _check_against_dense(case, g, [0], direction)


@pytest.mark.parametrize("overflow_cap", ["vertices", "edges"])
def test_sparse_fallback_on_cap_overflow(
    monkeypatch, exec_path, overflow_cap
):
    # Either cap can force the dense-machinery fallback; the output contract
    # (ascending, exact reached set) must hold identically, and telemetry
    # captured immediately after the sparse-output call must report it.
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "1024")
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", "8192")
    cap_env = {
        "vertices": "MG_BFS_SPARSE_MAX_VERTICES",
        "edges": "MG_BFS_SPARSE_MAX_EDGES",
    }[overflow_cap]
    monkeypatch.setenv(cap_env, "1")
    sparse_info = _check_against_dense(case, g, [0, 3], "out")
    assert sparse_info["op"] == "bfs"


def test_sparse_fast_path_telemetry_and_determinism():
    case = ALL_CASES["path"]
    g = build_mg(case)
    mg.set_execution("cpu")
    try:
        a = mg.bfs(g, 0, direction="out", output="sparse")
        assert mg.last_run_info()["op"] == "bfs_sparse"
        b = mg.bfs(g, 0, direction="out", output="sparse")
    finally:
        mg.set_execution("auto")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_sparse_unreachable_excluded():
    case = ALL_CASES["path"]  # directed chain: nothing before the source
    g = build_mg(case)
    vs, dist, parent = mg.bfs(g, 25, direction="out", output="sparse")
    vs = np.asarray(vs)
    assert vs.min() == 25 and vs.size == case.num_vertices - 25


def test_sparse_empty_graph():
    g = build_mg(ALL_CASES["empty"])
    vs, dist, parent = mg.bfs(g, np.asarray([], np.uint32),
                              direction="out", output="sparse")
    assert np.asarray(vs).size == 0
    assert np.asarray(dist).size == 0
    assert np.asarray(parent).size == 0


def test_bfs_output_kwarg_validated():
    g = build_mg(ALL_CASES["ring"])
    with pytest.raises(ValueError):
        mg.bfs(g, 0, direction="out", output="csr")


@pytest.mark.parametrize("mode", ["cpu", pytest.param("gpu",
                                                      marks=pytest.mark.gpu)])
@pytest.mark.parametrize("threads", ["1", "16"])
def test_sparse_fallback_compaction_large_v(mode, threads, monkeypatch):
    """Dense-fallback compaction runs chunked in parallel from V = 2**18:
    output must be ascending user ids equal to the dense result's reached
    set and dist, with structurally valid parents, at any thread count."""
    if mode == "gpu" and not mg.has_gpu():
        pytest.skip("no Metal device")
    rng = np.random.default_rng(8)
    v = (1 << 18) + 4321  # last chunk partial
    e = 600_000
    src = rng.integers(0, v, e).astype(np.uint32)
    dst = rng.integers(0, v, e).astype(np.uint32)
    perm = rng.permutation(v).astype(np.uint32)  # user order != canonical
    g = mg.Graph.from_edges(perm[src], perm[dst], directed=True,
                            num_vertices=v)
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "0")  # force fallback
    monkeypatch.setenv("MG_THREADS", threads)
    mg.set_execution(mode)
    try:
        sources = np.asarray(perm[[0, 1]], np.uint32)
        vs, dist, parent = mg.bfs(g, sources, output="sparse")
        assert mg.last_run_info()["op"] == "bfs"
        dense, _ = mg.bfs(g, sources)
    finally:
        mg.set_execution("auto")
    vs, dist, parent = (np.asarray(a) for a in (vs, dist, parent))
    dense = np.asarray(dense)
    np.testing.assert_array_equal(vs, np.flatnonzero(dense >= 0))
    np.testing.assert_array_equal(dist, dense[vs])
    assert vs.size > 100_000
    child = dist > 0
    assert (parent[~child] == -1).all()
    p, c = parent[child].astype(np.int64), vs[child].astype(np.int64)
    assert (dense[p] == dist[child] - 1).all()
    keys = perm[src].astype(np.int64) * v + perm[dst]
    assert np.isin(p * v + c, keys).all()
