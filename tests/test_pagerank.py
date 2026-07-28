# test_pagerank.py — full-vector PageRank vs NetworkX (golden oracle).
import os

import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import (ALL_CASES, build_mg, build_nx_multi, make_weights,
                      nx_pagerank_array, pr_tolerance)

PR_KW = dict(alpha=0.85, tol=1e-10, max_iter=500)


@pytest.mark.parametrize("weighted", [False, True], ids=["unw", "wgt"])
def test_pagerank_matches_networkx(graph_case, weighted, exec_path,
                                   assert_path):
    case = graph_case
    w = make_weights(case) if weighted else None
    g = build_mg(case, weights=w)
    pr = np.asarray(mg.pagerank(g, **PR_KW))
    assert pr.shape == (case.num_vertices,)
    if case.num_vertices == 0:
        return  # V==0 short-circuits before the planner — no path to assert
    assert_path(exec_path)
    expected = nx_pagerank_array(case, weights=w, **PR_KW)
    np.testing.assert_allclose(pr, expected, **pr_tolerance(exec_path))


def test_pagerank_multigraph_duplicate_edges(exec_path, assert_path):
    # Duplicate edges are counted — golden is nx.pagerank on a MultiDiGraph.
    import networkx as nx
    case = ALL_CASES["multi_self"]
    g = build_mg(case)
    pr = np.asarray(mg.pagerank(g, **PR_KW))
    assert_path(exec_path)
    gm = build_nx_multi(case)
    d = nx.pagerank(gm, alpha=0.85, tol=1e-10, max_iter=500)
    expected = np.zeros(case.num_vertices)
    for node, score in d.items():
        expected[node] = score
    np.testing.assert_allclose(pr, expected, **pr_tolerance(exec_path))


@pytest.mark.parametrize("case_name", ["dangling", "gnp_dir", "star"])
def test_pagerank_personalization_dense(case_name, exec_path, assert_path):
    # Dense personalization with zeros: dangling mass must be redistributed
    # by the personalization vector (NetworkX semantics); the dangling-heavy
    # fixture is the probe.
    case = ALL_CASES[case_name]
    rng = np.random.default_rng(3)
    pvec = rng.uniform(0.0, 1.0, case.num_vertices)
    pvec[pvec < 0.5] = 0.0  # sparse support, still sum > 0
    pvec[0] = 1.0
    pvec32 = pvec.astype(np.float32)
    g = build_mg(case)
    pr = np.asarray(mg.pagerank(g, personalization=pvec32, **PR_KW))
    assert_path(exec_path)
    pers = {i: float(pvec32[i]) for i in range(case.num_vertices)}
    expected = nx_pagerank_array(case, personalization=pers, **PR_KW)
    np.testing.assert_allclose(pr, expected, **pr_tolerance(exec_path))


def test_pagerank_personalization_sparse_dict(exec_path, assert_path):
    # Sparse dict form: keys are user indices (== external ids for
    # identity-mode graphs), missing vertices get 0.
    case = ALL_CASES["dangling"]
    g = build_mg(case)
    pers = {0: 2.0, 5: 1.0, 97: 3.0}
    pr = np.asarray(mg.pagerank(g, personalization=pers, **PR_KW))
    assert_path(exec_path)
    expected = nx_pagerank_array(case, personalization=pers, **PR_KW)
    np.testing.assert_allclose(pr, expected, **pr_tolerance(exec_path))


@pytest.mark.parametrize(
    "bad_weight",
    [np.nan, np.inf, -np.inf],
    ids=["nan", "positive_infinity", "negative_infinity"],
)
def test_pagerank_non_finite_personalization_rejected(bad_weight):
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    personalization = np.ones(case.num_vertices, dtype=np.float32)
    personalization[0] = bad_weight
    with pytest.raises(ValueError, match="finite"):
        mg.pagerank(g, personalization=personalization, **PR_KW)


@pytest.mark.parametrize("case_name", ["gnp_dir", "adversarial", "dangling"])
def test_pagerank_deterministic_per_path(case_name, exec_path, assert_path):
    case = ALL_CASES[case_name]
    g = build_mg(case, weights=make_weights(case))
    a = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-6, max_iter=100))
    assert_path(exec_path)
    b = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-6, max_iter=100))
    assert_path(exec_path)
    assert np.array_equal(a, b), "two runs must be bit-identical per path"


def test_pagerank_sums_to_one(graph_case, exec_path, assert_path):
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("V=0 covered by test_pagerank_empty")
    g = build_mg(case)
    pr = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-8, max_iter=200))
    assert_path(exec_path)
    assert np.isfinite(pr).all()
    assert (pr >= 0).all()
    assert abs(float(pr.sum()) - 1.0) < 1e-4


def test_pagerank_empty(exec_path):
    case = ALL_CASES["empty"]
    g = build_mg(case)
    pr = np.asarray(mg.pagerank(g, **PR_KW))
    assert pr.shape == (0,)


def test_pagerank_singleton(exec_path, assert_path):
    case = ALL_CASES["singleton"]
    g = build_mg(case)
    pr = np.asarray(mg.pagerank(g, **PR_KW))
    assert_path(exec_path)
    np.testing.assert_allclose(pr, [1.0], atol=1e-6)


def test_pagerank_iteration_count_on_audit_boundary(exec_path, assert_path):
    # Convergence is audited every MG_PR_AUDIT_INTERVAL (=5, the documented
    # default, pinned in conftest) iterations,
    # so the reported iteration count lands on an audit boundary (or on the
    # max_iter budget).
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    mg.pagerank(g, alpha=0.85, tol=1e-10, max_iter=500)
    assert_path(exec_path)
    info = mg.last_run_info()
    iters = info["iterations"]
    assert iters >= 1
    interval = int(os.environ["MG_PR_AUDIT_INTERVAL"])
    assert iters % interval == 0 or iters == 500, (
        f"iterations {iters} not on an audit boundary (interval {interval})")


def test_pagerank_max_iter_is_budget_not_error(exec_path, assert_path):
    # Documented divergence from NetworkX: hitting max_iter returns the
    # current iterate instead of raising.
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    pr = np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-15, max_iter=3))
    assert_path(exec_path)
    assert pr.shape == (case.num_vertices,)
    assert np.isfinite(pr).all()
    assert mg.last_run_info()["iterations"] <= 3
