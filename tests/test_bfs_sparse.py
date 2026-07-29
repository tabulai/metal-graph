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


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("name", CASE_NAMES)
def test_sparse_matches_dense(name, direction, exec_path):
    case = ALL_CASES[name]
    if case.num_vertices == 0:
        pytest.skip("no valid sources in an empty graph")
    g = build_mg(case)
    _check_against_dense(case, g, [0], direction)


def test_sparse_fallback_on_cap_overflow(monkeypatch, exec_path):
    # Caps of 1 force the dense-machinery fallback; the output contract
    # (ascending, exact reached set) must hold identically, and telemetry
    # must report the full path honestly.
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "1")
    _check_against_dense(case, g, [0, 3], "out")
    assert mg.last_run_info()["op"] == "bfs"


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
