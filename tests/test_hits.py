# test_hits.py — HITS contract, NumPy oracle, and CPU/GPU parity.
import networkx as nx
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import ALL_CASES, GraphCase, build_mg, has_gpu, make_weights


EDGEFUL_CASES = [
    "star",
    "path",
    "ring",
    "clique",
    "forest2",
    "gnp_dir",
    "gnp_undir",
    "multi_self",
    "dangling",
    "isolated",
    "adversarial",
]


@pytest.fixture(autouse=True)
def _pin_hits_audit_interval(monkeypatch):
    """Keep convergence checks reproducible independently of the shell."""
    monkeypatch.setenv("MG_HITS_AUDIT_INTERVAL", "5")


def _adjacency(case):
    """Dense unweighted adjacency built directly from the input edge list.

    ``np.add.at`` is intentional: unlike ordinary advanced indexing it counts
    every parallel edge.  Undirected inputs are stored in both orientations,
    except that a self-loop is stored only once.
    """
    v = case.num_vertices
    adjacency = np.zeros((v, v), dtype=np.float64)
    if case.n_edges == 0:
        return adjacency

    src = np.asarray(case.src, dtype=np.intp)
    dst = np.asarray(case.dst, dtype=np.intp)
    np.add.at(adjacency, (src, dst), 1.0)
    if not case.directed:
        non_loop = src != dst
        np.add.at(adjacency, (dst[non_loop], src[non_loop]), 1.0)
    return adjacency


def _hits_reference(case, tol, max_iter, audit_interval=5):
    """Independent L1-normalized HITS power iteration.

    Passing ``tol=0`` deliberately disables early convergence, which lets a
    test reproduce exactly the number of iterations reported by a backend.
    The public API itself rejects that value.
    """
    adjacency = _adjacency(case)
    v = case.num_vertices
    if v == 0 or not np.any(adjacency):
        zeros = np.zeros(v, dtype=np.float64)
        return zeros, zeros.copy(), 0

    hubs = np.full(v, 1.0 / v, dtype=np.float64)
    authorities = np.zeros(v, dtype=np.float64)
    audit_interval = max(int(audit_interval), 1)
    done = 0

    while done < max_iter:
        authorities = adjacency.T @ hubs
        authorities /= authorities.sum(dtype=np.float64)
        next_hubs = adjacency @ authorities
        next_hubs /= next_hubs.sum(dtype=np.float64)
        done += 1

        audit = done % audit_interval == 0 or done == max_iter
        error = np.abs(next_hubs - hubs).sum(dtype=np.float64) if audit else 0.0
        hubs = next_hubs
        if audit and error < v * tol:
            break

    return hubs, authorities, done


def _path_tolerance(path):
    if path == "cpu":
        return dict(atol=1e-6, rtol=0.0)
    return dict(atol=5e-5, rtol=2e-3)


def _assert_score_pair(pair, v, *, normalized):
    assert isinstance(pair, tuple) and len(pair) == 2
    hubs, authorities = pair
    assert isinstance(hubs, np.ndarray)
    assert isinstance(authorities, np.ndarray)
    assert hubs.shape == authorities.shape == (v,)
    assert hubs.dtype == authorities.dtype == np.float32
    assert np.isfinite(hubs).all() and np.isfinite(authorities).all()
    assert (hubs >= 0).all() and (authorities >= 0).all()
    expected_sum = 1.0 if normalized else 0.0
    assert abs(float(hubs.sum(dtype=np.float64)) - expected_sum) < 1e-4
    assert abs(float(authorities.sum(dtype=np.float64)) - expected_sum) < 1e-4
    return hubs, authorities


@pytest.mark.parametrize("case_name", EDGEFUL_CASES)
def test_hits_matches_independent_numpy_oracle(case_name, exec_path,
                                               assert_path):
    case = ALL_CASES[case_name]
    pair = mg.hits(build_mg(case), tol=1e-7, max_iter=75)
    hubs, authorities = _assert_score_pair(
        pair, case.num_vertices, normalized=True)
    assert_path(exec_path)

    info = mg.last_run_info()
    assert info["op"] == "hits"
    assert 1 <= info["iterations"] <= 75
    assert info["iterations"] % 5 == 0 or info["iterations"] == 75
    assert info["ms"] >= 0.0

    # Backends can cross the tolerance threshold at different audit points.
    # Comparing with an oracle run for the backend's actual iteration count
    # isolates the sparse-matrix and normalization semantics from that benign
    # fp32/fp64 difference.
    expected_hubs, expected_authorities, _ = _hits_reference(
        case, tol=0.0, max_iter=info["iterations"])
    tolerance = _path_tolerance(exec_path)
    np.testing.assert_allclose(hubs, expected_hubs, **tolerance)
    np.testing.assert_allclose(authorities, expected_authorities, **tolerance)


def test_hits_parallel_edges_count_individually(exec_path, assert_path):
    duplicated = GraphCase(
        "hits_duplicates",
        np.asarray([0, 0, 0, 0, 1, 2, 2, 3], dtype=np.uint32),
        np.asarray([1, 1, 1, 2, 2, 0, 3, 2], dtype=np.uint32),
        True,
        4,
    )
    deduplicated = GraphCase(
        "hits_deduplicated",
        np.asarray([0, 0, 1, 2, 2, 3], dtype=np.uint32),
        np.asarray([1, 2, 2, 0, 3, 2], dtype=np.uint32),
        True,
        4,
    )

    hubs, authorities = mg.hits(
        build_mg(duplicated), tol=1e-12, max_iter=41)
    assert_path(exec_path)
    iterations = mg.last_run_info()["iterations"]
    expected_hubs, expected_authorities, _ = _hits_reference(
        duplicated, tol=0.0, max_iter=iterations)
    simple_hubs, simple_authorities, _ = _hits_reference(
        deduplicated, tol=0.0, max_iter=iterations)

    # Guard the fixture itself: this topology makes multiplicity observable.
    assert not np.allclose(expected_hubs, simple_hubs, atol=1e-3, rtol=0.0)
    assert not np.allclose(
        expected_authorities, simple_authorities, atol=1e-3, rtol=0.0)
    tolerance = _path_tolerance(exec_path)
    np.testing.assert_allclose(hubs, expected_hubs, **tolerance)
    np.testing.assert_allclose(authorities, expected_authorities, **tolerance)


def test_hits_ignores_edge_weights(exec_path, assert_path):
    case = ALL_CASES["multi_self"]
    graph = build_mg(case)
    weighted_graph = build_mg(case, weights=make_weights(case))

    plain = mg.hits(graph, tol=1e-9, max_iter=53)
    assert_path(exec_path)
    weighted = mg.hits(weighted_graph, tol=1e-9, max_iter=53)
    assert_path(exec_path)

    # Same structure takes the same deterministic code path; weights must not
    # even perturb a rounding bit.
    assert np.array_equal(plain[0], weighted[0])
    assert np.array_equal(plain[1], weighted[1])


def test_hits_matches_networkx_on_a_simple_graph(exec_path, assert_path):
    case = ALL_CASES["gnp_dir"]
    graph = build_mg(case)
    hubs, authorities = mg.hits(graph, tol=1e-12, max_iter=500)
    assert_path(exec_path)

    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(case.num_vertices))
    nx_graph.add_edges_from(zip(case.src.tolist(), case.dst.tolist()))
    expected_hubs, expected_authorities = nx.hits(nx_graph, normalized=True)
    expected_hubs = np.asarray(
        [expected_hubs[i] for i in range(case.num_vertices)])
    expected_authorities = np.asarray(
        [expected_authorities[i] for i in range(case.num_vertices)])

    tolerance = _path_tolerance(exec_path)
    np.testing.assert_allclose(hubs, expected_hubs, **tolerance)
    np.testing.assert_allclose(
        authorities, expected_authorities, **tolerance)


def test_hits_one_iteration_returns_the_completed_iteration(exec_path,
                                                            assert_path):
    case = ALL_CASES["gnp_dir"]
    hubs, authorities = mg.hits(build_mg(case), tol=1e-30, max_iter=1)
    assert_path(exec_path)
    expected_hubs, expected_authorities, expected_iterations = _hits_reference(
        case, tol=0.0, max_iter=1)
    assert mg.last_run_info()["iterations"] == expected_iterations == 1
    tolerance = _path_tolerance(exec_path)
    np.testing.assert_allclose(hubs, expected_hubs, **tolerance)
    np.testing.assert_allclose(authorities, expected_authorities, **tolerance)


def test_hits_empty_graph_short_circuits_to_cpu(exec_path):
    case = ALL_CASES["empty"]
    pair = mg.hits(build_mg(case), tol=1e-5, max_iter=10)
    _assert_score_pair(pair, 0, normalized=False)
    info = mg.last_run_info()
    assert info["op"] == "hits"
    assert info["path"] == "cpu"
    assert info["iterations"] == 0
    assert info["ms"] >= 0.0


def test_hits_edgeless_graph_returns_zeros(exec_path, assert_path):
    case = GraphCase(
        "hits_edgeless",
        np.asarray([], dtype=np.uint32),
        np.asarray([], dtype=np.uint32),
        True,
        7,
    )
    pair = mg.hits(build_mg(case), tol=1e-5, max_iter=10)
    hubs, authorities = _assert_score_pair(pair, 7, normalized=False)
    assert np.array_equal(hubs, np.zeros(7, dtype=np.float32))
    assert np.array_equal(authorities, np.zeros(7, dtype=np.float32))
    assert_path(exec_path)
    assert mg.last_run_info()["iterations"] == 0


@pytest.mark.parametrize("bad_tol", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_hits_rejects_invalid_tolerance(bad_tol):
    graph = build_mg(ALL_CASES["gnp_dir"])
    with pytest.raises(ValueError, match="tol"):
        mg.hits(graph, tol=bad_tol, max_iter=10)


@pytest.mark.parametrize("bad_max_iter", [0, -1, -100])
def test_hits_rejects_invalid_max_iter(bad_max_iter):
    graph = build_mg(ALL_CASES["gnp_dir"])
    with pytest.raises(ValueError, match="max_iter"):
        mg.hits(graph, tol=1e-5, max_iter=bad_max_iter)


def test_hits_rejects_non_graph_input():
    with pytest.raises(TypeError, match="metal_graph.Graph"):
        mg.hits(object())


def test_hits_external_ids_still_return_user_order(exec_path, assert_path):
    src = np.asarray(["hub", "hub", "alpha", "beta", "gamma"])
    dst = np.asarray(["alpha", "beta", "beta", "gamma", "alpha"])
    graph = mg.Graph.from_edges(src, dst, directed=True)
    dense_src = np.asarray(graph.index_of(src), dtype=np.uint32)
    dense_dst = np.asarray(graph.index_of(dst), dtype=np.uint32)
    dense = mg.Graph.from_edges(
        dense_src,
        dense_dst,
        directed=True,
        num_vertices=graph.num_vertices,
    )

    actual = mg.hits(graph, tol=1e-10, max_iter=101)
    assert_path(exec_path)
    expected = mg.hits(dense, tol=1e-10, max_iter=101)
    assert_path(exec_path)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_hits_custom_audit_interval_and_telemetry(exec_path, assert_path,
                                                  monkeypatch):
    monkeypatch.setenv("MG_HITS_AUDIT_INTERVAL", "3")
    case = ALL_CASES["star"]
    hubs, authorities = mg.hits(build_mg(case), tol=1e-12, max_iter=17)
    assert_path(exec_path)
    info = mg.last_run_info()

    assert info["op"] == "hits"
    assert info["iterations"] == 3
    assert info["ms"] >= 0.0
    expected_hubs, expected_authorities, expected_iterations = _hits_reference(
        case, tol=1e-12, max_iter=17, audit_interval=3)
    assert expected_iterations == info["iterations"]
    tolerance = _path_tolerance(exec_path)
    np.testing.assert_allclose(hubs, expected_hubs, **tolerance)
    np.testing.assert_allclose(authorities, expected_authorities, **tolerance)


@pytest.mark.parametrize("case_name", ["gnp_dir", "multi_self", "adversarial"])
def test_hits_is_bit_deterministic_per_path(case_name, exec_path, assert_path):
    case = ALL_CASES[case_name]
    graph = build_mg(case, weights=make_weights(case))
    first = mg.hits(graph, tol=1e-8, max_iter=61)
    assert_path(exec_path)
    first_iterations = mg.last_run_info()["iterations"]

    for _ in range(3):
        again = mg.hits(graph, tol=1e-8, max_iter=61)
        assert_path(exec_path)
        assert mg.last_run_info()["iterations"] == first_iterations
        assert np.array_equal(again[0], first[0])
        assert np.array_equal(again[1], first[1])


@pytest.mark.gpu
def test_hits_cpu_gpu_agreement():
    if not has_gpu():
        pytest.skip("no Metal device")
    case = ALL_CASES["gnp_dir"]
    graph = build_mg(case, weights=make_weights(case))

    mg.set_execution("cpu")
    try:
        cpu_hubs, cpu_authorities = mg.hits(
            graph, tol=1e-8, max_iter=200)
        assert mg.last_run_info()["path"] == "cpu"
        mg.set_execution("gpu")
        gpu_hubs, gpu_authorities = mg.hits(
            graph, tol=1e-8, max_iter=200)
        assert mg.last_run_info()["path"] == "gpu"
    finally:
        mg.set_execution("auto")

    np.testing.assert_allclose(
        gpu_hubs, cpu_hubs, atol=5e-5, rtol=2e-3)
    np.testing.assert_allclose(
        gpu_authorities, cpu_authorities, atol=5e-5, rtol=2e-3)
