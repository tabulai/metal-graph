# test_bfs.py — multi-source BFS: golden dist + structural parent checks.
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


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("srcset", ["single", "multi"])
def test_bfs_dist_matches_reference(graph_case, direction, srcset, exec_path,
                                    assert_path):
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("no valid sources in an empty graph")
    sources = _source_sets(case)[srcset]
    g = build_mg(case)
    dist, parent = mg.bfs(g, np.asarray(sources, np.uint32),
                          direction=direction)
    assert_path(bfs_expected_path(case.directed, direction, exec_path))
    dist = np.asarray(dist)
    parent = np.asarray(parent)
    assert dist.shape == (case.num_vertices,)
    assert parent.shape == (case.num_vertices,)
    expected = bfs_dist_ref(case, sources, direction)
    np.testing.assert_array_equal(dist, expected)
    # unreachable vertices carry parent == -1
    assert ((dist >= 0) | (parent == -1)).all()


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_bfs_parent_structural(graph_case, direction, exec_path, assert_path):
    # Parent choice may be nondeterministic on the GPU path — validate
    # structurally: parent is visited, one level closer, and the traversal
    # edge parent -> v exists in the requested direction.
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("no valid sources in an empty graph")
    sources = _source_sets(case)["multi"]
    g = build_mg(case)
    dist, parent = mg.bfs(g, np.asarray(sources, np.uint32),
                          direction=direction)
    assert_path(bfs_expected_path(case.directed, direction, exec_path))
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
