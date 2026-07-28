# test_ppr_topk.py — batched personalized PageRank with top-k (flagship).
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import (ALL_CASES, build_mg, nx_pagerank_array, pr_tolerance,
                      topk_from_scores)

PPR_KW = dict(alpha=0.85, tol=1e-10, max_iter=500)


def _pack(queries):
    """queries: list of (seeds, weights) -> (seeds, weights, offsets)."""
    seeds = np.concatenate([np.asarray(q[0], np.uint32) for q in queries]) \
        if queries else np.array([], np.uint32)
    w = np.concatenate([np.asarray(q[1], np.float32) for q in queries]) \
        if queries else np.array([], np.float32)
    offs = np.zeros(len(queries) + 1, np.uint64)
    np.cumsum([len(q[0]) for q in queries], out=offs[1:])
    return seeds, w, offs


def _call(g, queries, k, **kw):
    seeds, w, offs = _pack(queries)
    ids, scores = mg.ppr_topk(g, seeds, w, offs, k=k, **kw)
    ids = np.asarray(ids).reshape(len(queries), k)
    scores = np.asarray(scores).reshape(len(queries), k)
    return ids, scores


def _nx_query_scores(case, seeds, weights, **kw):
    pers = {}
    for s, w in zip(seeds, weights):
        pers[int(s)] = pers.get(int(s), 0.0) + float(w)
    return nx_pagerank_array(case, personalization=pers, **kw)


def _check_row(case, nx_scores, mg_ids, mg_scores, k, atol):
    """Score-tolerance-aware top-k comparison: sorted score arrays must
    agree within atol; ids must match at every rank whose nx score is
    separated from BOTH neighbors by more than 10x the tolerance."""
    v = case.num_vertices
    kk = min(k, v)
    nx_ids_sorted, nx_sorted = topk_from_scores(nx_scores, v)
    # mg scores must be non-increasing (rank order)
    assert (np.diff(mg_scores[:kk]) <= 1e-12).all()
    np.testing.assert_allclose(mg_scores[:kk], nx_sorted[:kk], atol=atol)
    # padding when k > V
    assert (mg_ids[kk:] == -1).all()
    assert (mg_scores[kk:] == 0.0).all()
    # returned ids must be valid, unique user indices
    assert len(set(mg_ids[:kk].tolist())) == kk
    assert (mg_ids[:kk] >= 0).all() and (mg_ids[:kk] < v).all()
    for i in range(kk):
        gap_lo = nx_sorted[i - 1] - nx_sorted[i] if i > 0 else np.inf
        gap_hi = nx_sorted[i] - nx_sorted[i + 1] if i + 1 < v else np.inf
        if gap_lo > 10 * atol and gap_hi > 10 * atol:
            assert mg_ids[i] == nx_ids_sorted[i], (
                f"rank {i}: id {mg_ids[i]} != nx {nx_ids_sorted[i]} "
                f"(unambiguous at 10x tolerance)")


@pytest.mark.parametrize("case_name", ["gnp_dir", "dangling", "adversarial",
                                       "gnp_undir"])
def test_ppr_topk_matches_networkx(case_name, exec_path, assert_path):
    case = ALL_CASES[case_name]
    g = build_mg(case)
    rng = np.random.default_rng(21)
    queries = []
    for _ in range(4):
        n_seed = int(rng.integers(1, 4))
        seeds = rng.choice(case.num_vertices, size=n_seed, replace=False)
        weights = rng.uniform(0.5, 2.0, n_seed).astype(np.float32)
        queries.append((seeds, weights))
    k = 16
    ids, scores = _call(g, queries, k, **PPR_KW)
    assert_path(exec_path)
    atol = pr_tolerance(exec_path)["atol"]
    for q, (seeds, weights) in enumerate(queries):
        nx_scores = _nx_query_scores(case, seeds, weights, **PPR_KW)
        _check_row(case, nx_scores, ids[q], scores[q], k, atol)


@pytest.mark.parametrize("batch", [1, 4, 16, 64])
def test_ppr_topk_batched_equals_sequential(batch, exec_path, assert_path):
    # Batched results are bit-identical to per-query calls on the same path.
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    rng = np.random.default_rng(batch)
    queries = []
    for _ in range(batch):
        n_seed = int(rng.integers(1, 5))
        seeds = rng.choice(case.num_vertices, size=n_seed, replace=False)
        weights = rng.uniform(0.1, 1.0, n_seed).astype(np.float32)
        queries.append((seeds, weights))
    kw = dict(alpha=0.85, tol=1e-6, max_iter=50)
    k = 8
    ids_b, scores_b = _call(g, queries, k, **kw)
    assert_path(exec_path)
    for q, query in enumerate(queries):
        ids_1, scores_1 = _call(g, [query], k, **kw)
        assert_path(exec_path)
        assert np.array_equal(ids_b[q], ids_1[0])
        assert np.array_equal(scores_b[q], scores_1[0])


def test_ppr_topk_multi_tile_batch(exec_path, assert_path):
    # B=20 crosses the MG_PPR_TILE=8 tile width (3 tiles, last partial).
    case = ALL_CASES["dangling"]
    g = build_mg(case)
    rng = np.random.default_rng(99)
    queries = [(rng.choice(case.num_vertices, size=2, replace=False),
                np.ones(2, np.float32)) for _ in range(20)]
    kw = dict(alpha=0.85, tol=1e-6, max_iter=50)
    ids_b, scores_b = _call(g, queries, 8, **kw)
    assert_path(exec_path)
    for q, query in enumerate(queries):
        ids_1, scores_1 = _call(g, [query], 8, **kw)
        assert np.array_equal(ids_b[q], ids_1[0])
        assert np.array_equal(scores_b[q], scores_1[0])


def test_ppr_topk_k_greater_than_v_pads(exec_path, assert_path):
    case = ALL_CASES["forest2"]  # V=30
    g = build_mg(case)
    k = 45
    ids, scores = _call(g, [([0, 20], [1.0, 1.0])], k, **PPR_KW)
    assert_path(exec_path)
    atol = pr_tolerance(exec_path)["atol"]
    nx_scores = _nx_query_scores(case, [0, 20], [1.0, 1.0], **PPR_KW)
    _check_row(case, nx_scores, ids[0], scores[0], k, atol)
    assert (ids[0, 30:] == -1).all()
    assert (scores[0, 30:] == 0.0).all()


def test_ppr_topk_k1(exec_path, assert_path):
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    ids, scores = _call(g, [([7], [1.0])], 1, **PPR_KW)
    assert_path(exec_path)
    nx_scores = _nx_query_scores(case, [7], [1.0], **PPR_KW)
    nx_ids, nx_sorted = topk_from_scores(nx_scores, 1)
    assert ids.shape == (1, 1) and scores.shape == (1, 1)
    assert ids[0, 0] == nx_ids[0]
    np.testing.assert_allclose(scores[0, 0], nx_sorted[0],
                               atol=pr_tolerance(exec_path)["atol"])


def test_ppr_topk_duplicate_seeds_summed(exec_path, assert_path):
    # [a, a, b] with weights [1, 2, 3] == [a, b] with weights [3, 3].
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    kw = dict(alpha=0.85, tol=1e-6, max_iter=50)
    ids_dup, scores_dup = _call(g, [([4, 4, 9], [1.0, 2.0, 3.0])], 8, **kw)
    assert_path(exec_path)
    ids_pre, scores_pre = _call(g, [([4, 9], [3.0, 3.0])], 8, **kw)
    assert np.array_equal(ids_dup, ids_pre)
    assert np.array_equal(scores_dup, scores_pre)


@pytest.mark.parametrize(
    "bad_weight",
    [np.nan, np.inf, -np.inf],
    ids=["nan", "positive_infinity", "negative_infinity"],
)
def test_ppr_topk_non_finite_seed_weight_rejected(bad_weight):
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    with pytest.raises(ValueError, match="finite"):
        mg.ppr_topk(
            g,
            np.array([1, 2], np.uint32),
            np.array([1.0, bad_weight], np.float32),
            np.array([0, 2], np.uint64),
            k=4,
            alpha=0.85,
            tol=1e-6,
            max_iter=10,
        )


def test_ppr_topk_invalid_inputs_raise():
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    kw = dict(k=4, alpha=0.85, tol=1e-6, max_iter=10)
    ok_seeds = np.array([1, 2], np.uint32)
    ok_w = np.array([1.0, 1.0], np.float32)

    # empty query (offsets[q] == offsets[q+1] -> zero seed mass)
    with pytest.raises(ValueError):
        mg.ppr_topk(g, ok_seeds, ok_w,
                    np.array([0, 0, 2], np.uint64), **kw)
    # negative seed weight
    with pytest.raises(ValueError):
        mg.ppr_topk(g, ok_seeds, np.array([1.0, -0.5], np.float32),
                    np.array([0, 2], np.uint64), **kw)
    # non-monotonic offsets
    with pytest.raises(ValueError):
        mg.ppr_topk(g, ok_seeds, ok_w, np.array([2, 0], np.uint64), **kw)
    # offsets not covering the seed array
    with pytest.raises(ValueError):
        mg.ppr_topk(g, ok_seeds, ok_w, np.array([0, 1], np.uint64), **kw)
    # seed index out of range
    with pytest.raises(ValueError):
        mg.ppr_topk(g, np.array([1, 100000], np.uint32), ok_w,
                    np.array([0, 2], np.uint64), **kw)
