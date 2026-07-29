# test_bfs.py — multi-source BFS: golden dist + structural parent checks.
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import (ALL_CASES, adjacency_sets, bfs_dist_ref,
                      bfs_expected_path, build_mg)

DIRECTIONS = ["out", "in", "both"]


def _source_sets(case):
    v = case.num_vertices
    single = [0]
    multi = sorted({0, v // 2, v - 1})
    return {"single": single, "multi": multi}


# The sparse preflight would otherwise absorb every V<=400 fixture on the
# cpu/auto routes, silently stripping the golden matrix's coverage of the
# threaded cpu::bfs oracle — so the matrix runs both with the preflight
# enabled (default) and disabled (full path forced).
SPARSE_MODES = ["default", "nosparse"]


def _apply_sparse_mode(monkeypatch, mode):
    if mode == "nosparse":
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "0")
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", "0")


def _assert_full_path_op(mode):
    if mode == "nosparse":
        assert mg.last_run_info()["op"] == "bfs", (
            "preflight disabled: the full (threaded/GPU) path must run")


@pytest.mark.parametrize("sparse_mode", SPARSE_MODES)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("srcset", ["single", "multi"])
def test_bfs_dist_matches_reference(graph_case, direction, srcset, exec_path,
                                    assert_path, sparse_mode, monkeypatch):
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("no valid sources in an empty graph")
    _apply_sparse_mode(monkeypatch, sparse_mode)
    sources = _source_sets(case)[srcset]
    g = build_mg(case)
    dist, parent = mg.bfs(g, np.asarray(sources, np.uint32),
                          direction=direction)
    assert_path(bfs_expected_path(case.directed, direction, exec_path))
    _assert_full_path_op(sparse_mode)
    dist = np.asarray(dist)
    parent = np.asarray(parent)
    assert dist.shape == (case.num_vertices,)
    assert parent.shape == (case.num_vertices,)
    expected = bfs_dist_ref(case, sources, direction)
    np.testing.assert_array_equal(dist, expected)
    # unreachable vertices carry parent == -1
    assert ((dist >= 0) | (parent == -1)).all()


@pytest.mark.parametrize("sparse_mode", SPARSE_MODES)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_bfs_parent_structural(graph_case, direction, exec_path, assert_path,
                               sparse_mode, monkeypatch):
    # Parent choice may be nondeterministic on the GPU path — validate
    # structurally: parent is visited, one level closer, and the traversal
    # edge parent -> v exists in the requested direction.
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("no valid sources in an empty graph")
    _apply_sparse_mode(monkeypatch, sparse_mode)
    sources = _source_sets(case)["multi"]
    g = build_mg(case)
    dist, parent = mg.bfs(g, np.asarray(sources, np.uint32),
                          direction=direction)
    assert_path(bfs_expected_path(case.directed, direction, exec_path))
    _assert_full_path_op(sparse_mode)
    dist = np.asarray(dist)
    parent = np.asarray(parent)
    adj = adjacency_sets(case, direction)
    src_set = set(sources)
    for v in range(case.num_vertices):
        if dist[v] < 0:
            assert parent[v] == -1, f"unreachable {v} must have parent -1"
        elif dist[v] == 0:
            assert v in src_set
            assert parent[v] == -1, f"source {v} must have parent -1"
        else:
            p = int(parent[v])
            assert 0 <= p < case.num_vertices
            assert dist[p] == dist[v] - 1, (
                f"parent[{v}]={p}: dist {dist[p]} != {dist[v]} - 1")
            assert v in adj[p], (
                f"no traversal edge {p} -> {v} (direction={direction})")


def test_bfs_both_equals_undirected_view(exec_path, assert_path):
    # direction='both' == BFS over the underlying undirected view.
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    dist, _ = mg.bfs(g, np.asarray([3], np.uint32), direction="both")
    assert_path(bfs_expected_path(case.directed, "both", exec_path))
    expected = bfs_dist_ref(case, [3], "both")  # undirected adjacency
    np.testing.assert_array_equal(np.asarray(dist), expected)


def test_bfs_duplicate_sources(exec_path, assert_path):
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    d1, _ = mg.bfs(g, np.asarray([5, 5, 5, 9], np.uint32), direction="out")
    assert_path(exec_path)
    d2, _ = mg.bfs(g, np.asarray([5, 9], np.uint32), direction="out")
    np.testing.assert_array_equal(np.asarray(d1), np.asarray(d2))


def test_bfs_scalar_source(exec_path, assert_path):
    case = ALL_CASES["path"]
    g = build_mg(case)
    d1, p1 = mg.bfs(g, 7, direction="out")
    assert_path(exec_path)
    d2, p2 = mg.bfs(g, np.asarray([7], np.uint32), direction="out")
    np.testing.assert_array_equal(np.asarray(d1), np.asarray(d2))
    np.testing.assert_array_equal(np.asarray(p1), np.asarray(p2))


def test_bfs_unreachable_minus_one(exec_path, assert_path):
    # path graph, direction='out' from the middle: everything before the
    # source is unreachable.
    case = ALL_CASES["path"]
    g = build_mg(case)
    dist, parent = mg.bfs(g, np.asarray([25], np.uint32), direction="out")
    assert_path(exec_path)
    dist = np.asarray(dist)
    parent = np.asarray(parent)
    assert (dist[:25] == -1).all()
    assert (parent[:25] == -1).all()
    np.testing.assert_array_equal(dist[25:], np.arange(25, dtype=np.int32))


def test_bfs_source_out_of_range_raises():
    case = ALL_CASES["path"]
    g = build_mg(case)
    with pytest.raises(ValueError):
        mg.bfs(g, np.asarray([case.num_vertices], np.uint32),
               direction="out")


def test_sparse_cpu_path_and_cap_fallback_agree(monkeypatch):
    """The latency path and regular CPU fallback preserve dense semantics."""
    src = np.arange(63, dtype=np.uint32)
    dst = src + 1
    g = mg.Graph.from_edges(
        src, dst, directed=True, num_vertices=64
    )
    source = np.asarray([0], np.uint32)
    mg.set_execution("cpu")
    try:
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "128")
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", "128")
        sparse_dist, sparse_parent = mg.bfs(g, source, direction="out")
        assert mg.last_run_info()["op"] == "bfs_sparse"

        monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "1")
        fallback_dist, fallback_parent = mg.bfs(
            g, source, direction="out"
        )
        assert mg.last_run_info()["op"] == "bfs"

        monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "128")
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", "1")
        edge_fallback_dist, edge_fallback_parent = mg.bfs(
            g, source, direction="out"
        )
        assert mg.last_run_info()["op"] == "bfs"
    finally:
        mg.set_execution("auto")

    np.testing.assert_array_equal(sparse_dist, fallback_dist)
    np.testing.assert_array_equal(sparse_dist, edge_fallback_dist)
    np.testing.assert_array_equal(sparse_dist, np.arange(64, dtype=np.int32))
    for parent in (
        sparse_parent, fallback_parent, edge_fallback_parent
    ):
        assert parent[0] == -1
        np.testing.assert_array_equal(
            parent[1:], np.arange(63, dtype=np.int32)
        )


def test_sparse_cpu_path_is_reentrant(monkeypatch):
    """Concurrent calls use call-local queues and output arrays."""
    src = np.arange(255, dtype=np.uint32)
    dst = src + 1
    g = mg.Graph.from_edges(
        src, dst, directed=True, num_vertices=256
    )
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "512")
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", "512")
    mg.set_execution("cpu")
    try:
        def run(source):
            dist, parent = mg.bfs(g, [source], direction="out")
            return np.asarray(dist), np.asarray(parent)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, range(16)))
    finally:
        mg.set_execution("auto")

    for source, (dist, parent) in enumerate(results):
        np.testing.assert_array_equal(dist[:source], -1)
        np.testing.assert_array_equal(
            dist[source:], np.arange(256 - source, dtype=np.int32)
        )
        assert parent[source] == -1
        if source + 1 < 256:
            np.testing.assert_array_equal(
                parent[source + 1:], np.arange(source, 255, dtype=np.int32)
            )


@pytest.mark.gpu
def test_bfs_deep_chain_crosses_many_batches(monkeypatch):
    # 4096-level path graph: the traversal spans dozens of command buffers
    # under the adaptive 8->64 batch schedule (and hundreds under a pinned
    # small batch). Guards level/parity bookkeeping across commit boundaries
    # while exercising the adaptive default; dist on a chain is determined.
    from conftest import has_gpu
    if not has_gpu():
        pytest.skip("no Metal device")
    monkeypatch.delenv("MG_BFS_LEVELS_PER_BATCH", raising=False)
    v = 4096
    src = np.arange(v - 1, dtype=np.uint32)
    dst = np.arange(1, v, dtype=np.uint32)
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=v)
    mg.set_execution("gpu")
    try:
        dist, parent = mg.bfs(g, np.asarray([0], np.uint32), direction="out")
        assert mg.last_run_info()["path"] == "gpu"
        assert mg.last_run_info()["iterations"] == v
    finally:
        mg.set_execution("auto")
    np.testing.assert_array_equal(np.asarray(dist),
                                  np.arange(v, dtype=np.int32))
    par = np.asarray(parent)
    assert par[0] == -1
    np.testing.assert_array_equal(par[1:], np.arange(v - 1, dtype=np.int32))
