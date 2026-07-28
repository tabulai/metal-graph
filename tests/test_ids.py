# test_ids.py — external-ID identity model: int64 sparse ids, string ids,
# identity mode with num_vertices tails, np.unique ordering contract.
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import ALL_CASES, build_mg


def test_int64_sparse_ids_roundtrip():
    src = np.array([10_000_000_007, 42, 999, 42], dtype=np.int64)
    dst = np.array([42, 999, 10_000_000_007, 7], dtype=np.int64)
    g = mg.Graph.from_edges(src, dst, directed=True)
    all_ids = np.unique(np.concatenate([src, dst]))
    assert g.num_vertices == len(all_ids)
    ext = np.asarray(g.external_ids)
    np.testing.assert_array_equal(ext, all_ids)  # np.unique order
    # index_of: external id -> user index, full round trip
    idx = np.asarray(g.index_of(ext))
    np.testing.assert_array_equal(idx, np.arange(len(all_ids)))
    src_idx = np.asarray(g.index_of(src))
    np.testing.assert_array_equal(ext[src_idx], src)
    dst_idx = np.asarray(g.index_of(dst))
    np.testing.assert_array_equal(ext[dst_idx], dst)


def test_int64_sparse_ids_algorithms_use_user_indices():
    # The same topology under sparse int64 ids and dense identity ids must
    # produce identical results after mapping through index_of.
    src_ids = np.array([500, 900, 1300, 500], dtype=np.int64)
    dst_ids = np.array([900, 1300, 500, 1300], dtype=np.int64)
    g = mg.Graph.from_edges(src_ids, dst_ids, directed=True)
    uniq = np.unique(np.concatenate([src_ids, dst_ids]))
    src_u = np.searchsorted(uniq, src_ids).astype(np.uint32)
    dst_u = np.searchsorted(uniq, dst_ids).astype(np.uint32)
    g_dense = mg.Graph.from_edges(src_u, dst_u, directed=True,
                                  num_vertices=len(uniq))
    pr_sparse = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-10,
                                       max_iter=500))
    pr_dense = np.asarray(mg.pagerank(g_dense, alpha=0.85, tol=1e-10,
                                      max_iter=500))
    np.testing.assert_allclose(pr_sparse, pr_dense, atol=1e-7)


def test_string_ids_roundtrip():
    src = np.array(["zebra", "apple", "mango", "apple"])
    dst = np.array(["apple", "mango", "zebra", "kiwi"])
    g = mg.Graph.from_edges(src, dst, directed=True)
    all_ids = np.unique(np.concatenate([src, dst]))  # lexicographic
    assert g.num_vertices == len(all_ids)
    ext = np.asarray(g.external_ids)
    assert list(ext) == list(all_ids)
    idx = np.asarray(g.index_of(np.array(["kiwi", "zebra"])))
    assert list(ext[idx]) == ["kiwi", "zebra"]
    # algorithms speak user indices regardless of external id type
    pr = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-8, max_iter=200))
    assert pr.shape == (g.num_vertices,)
    assert abs(float(pr.sum()) - 1.0) < 1e-4


def test_identity_mode_num_vertices_isolated_tail_pagerank():
    # ids already 0..max: identity mode; num_vertices adds isolated tails
    # that must receive teleport (and dangling-redistribution) mass.
    src = np.array([0, 1, 2], dtype=np.uint32)
    dst = np.array([1, 2, 0], dtype=np.uint32)
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=8)
    assert g.num_vertices == 8
    pr = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-10, max_iter=500))
    assert (pr[3:] > 0).all(), "isolated vertices must hold teleport mass"
    # golden: nx graph with the isolated tail nodes present
    import networkx as nx
    gx = nx.DiGraph()
    gx.add_nodes_from(range(8))
    gx.add_edges_from([(0, 1), (1, 2), (2, 0)])
    d = nx.pagerank(gx, alpha=0.85, tol=1e-10, max_iter=500)
    expected = np.array([d[i] for i in range(8)])
    np.testing.assert_allclose(pr, expected, atol=1e-6)


def test_identity_mode_num_vertices_isolated_tail_bfs():
    src = np.array([0, 1], dtype=np.uint32)
    dst = np.array([1, 2], dtype=np.uint32)
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=6)
    dist, parent = mg.bfs(g, np.asarray([0], np.uint32), direction="out")
    dist = np.asarray(dist)
    parent = np.asarray(parent)
    np.testing.assert_array_equal(dist[:3], [0, 1, 2])
    assert (dist[3:] == -1).all(), "isolated tail vertices are unreachable"
    assert (parent[3:] == -1).all()


def test_np_unique_ordering_contract():
    # User index order IS np.unique order of the external ids — documented.
    src = np.array([70, 10, 50], dtype=np.int64)
    dst = np.array([10, 50, 70], dtype=np.int64)
    g = mg.Graph.from_edges(src, dst, directed=True)
    ext = np.asarray(g.external_ids)
    np.testing.assert_array_equal(ext, np.array([10, 50, 70]))
    idx = np.asarray(g.index_of(np.array([50], dtype=np.int64)))
    assert idx[0] == 1


def test_identity_mode_ids_are_user_indices():
    # 0..V-1 int ids: external_ids is the identity, index_of a no-op.
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    ext = np.asarray(g.external_ids)
    np.testing.assert_array_equal(
        np.asarray(ext, dtype=np.int64), np.arange(case.num_vertices))
    idx = np.asarray(g.index_of(ext[:5]))
    np.testing.assert_array_equal(idx, np.arange(5))
