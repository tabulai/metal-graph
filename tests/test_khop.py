# test_khop.py — bounded k-hop extraction vs an in-test reference.
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import ALL_CASES, build_mg, khop_ref

DIRECTIONS = ["out", "in", "both"]


def _seeds(case):
    v = case.num_vertices
    return sorted({0, v // 3, (2 * v) // 3}) if v >= 3 else [0]


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("k", [0, 2])
def test_khop_matches_reference(graph_case, direction, k, exec_path):
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("k_hop needs at least one seed vertex")
    seeds = _seeds(case)
    g = build_mg(case)
    vs, es = mg.k_hop(g, np.asarray(seeds, np.uint32), k=k,
                      direction=direction)
    vs = np.asarray(vs)
    es = np.asarray(es)
    ref_vs, ref_es = khop_ref(case, seeds, k, direction)
    np.testing.assert_array_equal(np.sort(vs), vs)  # sorted ascending
    np.testing.assert_array_equal(np.sort(es), es)
    np.testing.assert_array_equal(vs, ref_vs)
    np.testing.assert_array_equal(es, ref_es)


@pytest.mark.parametrize("k", [1, 3])
def test_khop_depths_gnp(k, exec_path):
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    vs, es = mg.k_hop(g, np.asarray([0], np.uint32), k=k, direction="out")
    ref_vs, ref_es = khop_ref(case, [0], k, "out")
    np.testing.assert_array_equal(np.asarray(vs), ref_vs)
    np.testing.assert_array_equal(np.asarray(es), ref_es)


def test_khop_k0_seeds_only(exec_path):
    # k=0: exactly the seed set, plus edges among the seeds.
    case = ALL_CASES["multi_self"]
    seeds = [0, 1, 5]
    g = build_mg(case)
    vs, es = mg.k_hop(g, np.asarray(seeds, np.uint32), k=0, direction="both")
    np.testing.assert_array_equal(np.asarray(vs), np.asarray(seeds, np.uint32))
    ref_vs, ref_es = khop_ref(case, seeds, 0, "both")
    np.testing.assert_array_equal(np.asarray(es), ref_es)
    assert len(ref_es) > 0  # fixture guarantees edges among these seeds


@pytest.mark.parametrize("max_vertices", [5, 10, 25])
def test_khop_max_vertices_deterministic_admission(max_vertices):
    # Cap-bounded runs are documented to execute on the (deterministic) CPU
    # path in v0.1, so the cap tests do not pin an execution path.
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    seeds = [0]
    vs, es = mg.k_hop(g, np.asarray(seeds, np.uint32), k=4, direction="out",
                      max_vertices=max_vertices)
    vs = np.asarray(vs)
    assert len(vs) <= max_vertices
    ref_vs, ref_es = khop_ref(case, seeds, 4, "out",
                              max_vertices=max_vertices)
    np.testing.assert_array_equal(vs, ref_vs)
    np.testing.assert_array_equal(np.asarray(es), ref_es)


@pytest.mark.parametrize("max_edges", [1, 7, 50])
def test_khop_max_edges_truncates_ascending(max_edges):
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    vs, es = mg.k_hop(g, np.asarray([0], np.uint32), k=2, direction="both",
                      max_edges=max_edges)
    es = np.asarray(es)
    assert len(es) <= max_edges
    ref_vs, ref_es = khop_ref(case, [0], 2, "both", max_edges=max_edges)
    np.testing.assert_array_equal(np.asarray(vs), ref_vs)
    np.testing.assert_array_equal(es, ref_es)  # smallest input edge ids kept


def test_khop_undirected_edge_ids_are_input_ids(exec_path):
    # Undirected symmetrization must report each input edge id once.
    case = ALL_CASES["gnp_undir"]
    g = build_mg(case)
    vs, es = mg.k_hop(g, np.asarray([0], np.uint32), k=2, direction="both")
    es = np.asarray(es)
    assert len(np.unique(es)) == len(es)
    assert (es < case.n_edges).all()
    ref_vs, ref_es = khop_ref(case, [0], 2, "both")
    np.testing.assert_array_equal(es, ref_es)


def test_khop_as_graph_roundtrip(exec_path):
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    vs, es = mg.k_hop(g, np.asarray([0, 7], np.uint32), k=2,
                      direction="both")
    res = mg.k_hop(g, np.asarray([0, 7], np.uint32), k=2, direction="both",
                   as_graph=True)
    sub = res if hasattr(res, "num_vertices") else res[-1]
    assert sub.num_vertices == len(np.asarray(vs))
    assert sub.num_edges == len(np.asarray(es))
    pr = np.asarray(mg.pagerank(sub, alpha=0.85, tol=1e-8, max_iter=100))
    assert pr.shape == (sub.num_vertices,)
    assert np.isfinite(pr).all()
    assert abs(float(pr.sum()) - 1.0) < 1e-4


def test_khop_seed_out_of_range_raises():
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    with pytest.raises(ValueError):
        mg.k_hop(g, np.asarray([case.num_vertices + 3], np.uint32), k=1,
                 direction="out")
