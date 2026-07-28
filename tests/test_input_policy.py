# test_input_policy.py — the documented build-time input policy.
import networkx as nx
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import ALL_CASES, build_mg, pr_tolerance

SRC = np.array([0, 1, 2], dtype=np.uint32)
DST = np.array([1, 2, 0], dtype=np.uint32)


def test_nan_weight_rejected():
    w = np.array([1.0, np.nan, 1.0], dtype=np.float32)
    with pytest.raises(ValueError):
        mg.Graph.from_edges(SRC, DST, weights=w, directed=True,
                            num_vertices=3)


def test_negative_weight_rejected():
    w = np.array([1.0, -0.25, 1.0], dtype=np.float32)
    with pytest.raises(ValueError):
        mg.Graph.from_edges(SRC, DST, weights=w, directed=True,
                            num_vertices=3)


def test_num_vertices_smaller_than_max_id_rejected():
    with pytest.raises(ValueError):
        mg.Graph.from_edges(SRC, DST, directed=True, num_vertices=2)


def test_weights_length_mismatch_rejected():
    w = np.array([1.0, 1.0], dtype=np.float32)  # 2 weights, 3 edges
    with pytest.raises(ValueError):
        mg.Graph.from_edges(SRC, DST, weights=w, directed=True,
                            num_vertices=3)


def test_self_loops_kept(exec_path, assert_path):
    # PageRank must match NetworkX, which keeps self-loops.
    src = np.array([0, 0, 1, 2], dtype=np.uint32)
    dst = np.array([0, 1, 2, 0], dtype=np.uint32)
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=3)
    assert g.num_edges == 4
    pr = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-10, max_iter=500))
    assert_path(exec_path)
    gx = nx.DiGraph()
    gx.add_nodes_from(range(3))
    gx.add_edges_from(zip(src.tolist(), dst.tolist()))
    d = nx.pagerank(gx, alpha=0.85, tol=1e-10, max_iter=500)
    expected = np.array([d[i] for i in range(3)])
    np.testing.assert_allclose(pr, expected, **pr_tolerance(exec_path))


def test_duplicate_edges_kept(exec_path, assert_path):
    # Parallel edges count: the multigraph result must differ from the
    # deduplicated one and match nx.MultiDiGraph pagerank.
    src = np.array([0, 0, 0, 1, 2], dtype=np.uint32)
    dst = np.array([1, 1, 2, 0, 0], dtype=np.uint32)
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=3)
    assert g.num_edges == 5  # num_edges reports the input count
    pr = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-10, max_iter=500))
    assert_path(exec_path)
    gm = nx.MultiDiGraph()
    gm.add_edges_from(zip(src.tolist(), dst.tolist()))
    d = nx.pagerank(gm, alpha=0.85, tol=1e-10, max_iter=500)
    expected = np.array([d[i] for i in range(3)])
    np.testing.assert_allclose(pr, expected, **pr_tolerance(exec_path))

    g_dedup = mg.Graph.from_edges(np.array([0, 0, 1, 2], np.uint32),
                                  np.array([1, 2, 0, 0], np.uint32),
                                  directed=True, num_vertices=3)
    pr_dedup = np.asarray(mg.pagerank(g_dedup, alpha=0.85, tol=1e-10,
                                      max_iter=500))
    assert not np.allclose(pr, pr_dedup, atol=1e-4), (
        "duplicate edges silently deduplicated")


def test_zero_weight_out_edges_treated_dangling(exec_path, assert_path):
    # A vertex whose outgoing weights sum to zero is dangling even though
    # its degree is nonzero (documented in algos.hpp) — NetworkX agrees.
    src = np.array([0, 1, 2], dtype=np.uint32)
    dst = np.array([1, 2, 0], dtype=np.uint32)
    w = np.array([1.0, 0.0, 1.0], dtype=np.float32)  # vertex 1 dangles
    g = mg.Graph.from_edges(src, dst, weights=w, directed=True,
                            num_vertices=3)
    pr = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-10, max_iter=500))
    assert_path(exec_path)
    gx = nx.DiGraph()
    gx.add_nodes_from(range(3))
    gx.add_weighted_edges_from(
        zip(src.tolist(), dst.tolist(), w.astype(np.float64).tolist()))
    d = nx.pagerank(gx, alpha=0.85, tol=1e-10, max_iter=500, weight="weight")
    expected = np.array([d[i] for i in range(3)])
    np.testing.assert_allclose(pr, expected, **pr_tolerance(exec_path))


def test_undirected_num_edges_reports_input_count():
    # Symmetrization is internal; num_edges is the caller's edge count.
    case = ALL_CASES["gnp_undir"]
    g = build_mg(case)
    assert g.num_edges == case.n_edges


def test_reserved_kwargs_raise_not_implemented():
    # edge_types= / time_range= are reserved for v0.2 and must raise
    # NotImplementedError naming v0.2 (forward-compatible callers).
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    with pytest.raises(NotImplementedError):
        mg.k_hop(g, np.asarray([0], np.uint32), k=1, direction="out",
                 edge_types=["cites"])
    with pytest.raises(NotImplementedError):
        mg.k_hop(g, np.asarray([0], np.uint32), k=1, direction="out",
                 time_range=(0, 10))
