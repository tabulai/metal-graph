"""GPU union-find must preserve exact partitions under concurrent updates."""

import numpy as np
import pytest

from conftest import has_gpu

mg = pytest.importorskip("metal_graph")


@pytest.fixture
def gpu_execution():
    if not has_gpu():
        pytest.skip("no Metal device")
    mg.set_execution("gpu")
    yield
    mg.set_execution("auto")


def _disconnected_stress_graph(with_hubs, directed):
    # Two disconnected components with interleaved user ids, plus isolated
    # vertices. Alternating edge directions prevent directed reachability
    # from serving as an accidental substitute for weak connectivity.
    n = 20_000
    rng = np.random.default_rng(20261020)
    order = rng.permutation(2 * n + 37).astype(np.uint32)
    groups = [order[:n], order[n:2 * n]]
    src, dst = [], []
    for group in groups:
        a, b = group[:-1].copy(), group[1:].copy()
        a[1::2], b[1::2] = b[1::2].copy(), a[1::2].copy()
        src.append(a)
        dst.append(b)
        if with_hubs:
            # Shared roots with multiple edge tiles, one threadgroup, and
            # one simdgroup respectively, alongside the low-degree path.
            # The hubs connect to the same component from different ids.
            for hub, targets in ((group[0], group[1:]),
                                 (group[1], group[2:1202]),
                                 (group[2], group[1202:1266])):
                src.append(np.full(targets.size, hub, dtype=np.uint32))
                dst.append(targets)
        src.append(group[::4096])  # self-loops
        dst.append(group[::4096])
        src.append(a[:100])  # parallel edges
        dst.append(b[:100])

    src, dst = np.concatenate(src), np.concatenate(dst)
    shuffled = rng.permutation(src.size)
    graph = mg.Graph.from_edges(src[shuffled], dst[shuffled],
                                directed=directed, num_vertices=order.size)

    # The construction provides an independent exact oracle, including
    # the public first-occurrence numbering of singleton components.
    components = groups + [order[i:i + 1] for i in range(2 * n, order.size)]
    expected = np.empty(order.size, dtype=np.int32)
    for label, vertices in enumerate(sorted(components, key=lambda c: int(c.min()))):
        expected[vertices] = label
    return graph, expected


@pytest.mark.gpu
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("with_hubs", [False, True], ids=["paths", "shared-hubs"])
def test_wcc_disconnected_adversarial_repeated(gpu_execution, directed, with_hubs):
    graph, expected = _disconnected_stress_graph(with_hubs, directed)
    for run in range(20):
        actual = np.asarray(mg.experimental.wcc(graph))
        assert mg.last_run_info()["path"] == "gpu"
        np.testing.assert_array_equal(actual, expected, err_msg=f"run {run}")


# ---------------------------------------------------------------------------
# Sample / snapshot / filtered-link passes and the threaded numbering.
# ---------------------------------------------------------------------------

def _oracle(src, dst, v):
    # Independent exact labels: scipy's weak components, renumbered by first
    # occurrence in user index order (the public numbering).
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    adj = coo_matrix((np.ones(src.size, dtype=np.int8),
                      (src.astype(np.int64), dst.astype(np.int64))),
                     shape=(v, v))
    n, lab = connected_components(adj, directed=True, connection="weak")
    _, first = np.unique(lab, return_index=True)
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(first)] = np.arange(n)
    return rank[lab].astype(np.int32)


def _bins_graph(seed=5):
    # Components whose vertices span every OUT degree bin: huge (>= 16384,
    # several tiles), high, mid and low rows, zero-out-degree sinks reached
    # only through in-edges, plus parallel edges, self-loops, isolated
    # vertices and tiny components. Vertex 0 in canonical order (the largest
    # out-degree) is the first component's 40000-edge hub. A further
    # component of vertex pairs links each vertex to itself and its partner
    # FIRST (the sampled edges), so its snapshot sets are pairs and every
    # later edge between pairs crosses sets outside the giant set.
    rng = np.random.default_rng(seed)
    n = 150_000
    order = rng.permutation(n).astype(np.uint32)
    first_src, first_dst, src, dst = [], [], [], []

    def component(members, degrees):
        for vtx, deg in zip(members, degrees):
            if deg:
                src.append(np.full(deg, vtx, dtype=np.uint32))
                dst.append(rng.choice(members, deg))

    c1 = order[:60_000]
    deg1 = rng.choice([0, 0, 1, 2, 3, 5], c1.size)
    deg1[:4] = [40_000, 3_000, 2_000, 1_100]
    deg1[4:40] = 100
    component(c1, deg1)
    c2 = order[60_000:90_000]
    deg2 = rng.choice([0, 1, 2, 4], c2.size)
    deg2[:3] = [20_000, 1_500, 1_024]
    deg2[3:30] = rng.integers(32, 1_000, 27)
    component(c2, deg2)
    pairs = order[90_000:130_000].reshape(-1, 2)
    for a, b in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
        first_src += [a, a]
        first_dst += [a, b]
    cross = rng.integers(0, pairs.shape[0], (2, 15_000))
    src.append(pairs[cross[0], 0])
    dst.append(pairs[cross[1], 1])
    small = order[130_000:140_000].reshape(-1, 5)  # 2000 five-vertex paths
    for k in range(4):
        src.append(small[:, k])
        dst.append(small[:, k + 1])
    # order[140_000:] stay isolated.
    src, dst = np.concatenate(src), np.concatenate(dst)
    shuffled = rng.permutation(src.size)
    src, dst = src[shuffled], dst[shuffled]
    loops = rng.choice(order[:140_000], 500)
    dup = rng.integers(0, src.size, 1_000)
    src = np.concatenate(first_src + [src, loops, src[dup]]).astype(np.uint32)
    dst = np.concatenate(first_dst + [dst, loops, dst[dup]]).astype(np.uint32)
    return src, dst, n


def _hub_outside_largest():
    # Canonical vertex 0 (a 5000-leaf star's hub) is NOT in the largest
    # component (a 100000-vertex directed cycle), so the snapshot giant set
    # is the smaller component and the cycle is scanned as non-giant.
    leaves = np.arange(1, 5_001, dtype=np.uint32)
    ring = np.arange(5_001, 105_001, dtype=np.uint32)
    src = np.concatenate([np.zeros(leaves.size, np.uint32), ring])
    dst = np.concatenate([leaves, np.roll(ring, -1)])
    return src, dst, 105_001


def _wcc_runs(g, runs):
    first = np.asarray(mg.experimental.wcc(g)).copy()
    info = mg.last_run_info()
    for run in range(1, runs):
        np.testing.assert_array_equal(np.asarray(mg.experimental.wcc(g)),
                                      first, err_msg=f"run {run}")
    return first, info


@pytest.mark.gpu
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("case", [_bins_graph, _hub_outside_largest],
                         ids=["bins", "hub-outside-largest"])
def test_wcc_gpu_filtered_passes_match_oracle(gpu_execution, directed, case):
    src, dst, v = case()
    graph = mg.Graph.from_edges(src, dst, directed=directed, num_vertices=v)
    expected = _oracle(src, dst, v)
    actual, info = _wcc_runs(graph, 10)
    assert info["path"] == "gpu"
    assert info["iterations"] == 1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("v", [1, 3, 200_000])
def test_wcc_gpu_edgeless(gpu_execution, directed, v):
    empty = np.array([], dtype=np.uint32)
    graph = mg.Graph.from_edges(empty, empty, directed=directed,
                                num_vertices=v)
    actual, info = _wcc_runs(graph, 3)
    assert info["path"] == "gpu"
    np.testing.assert_array_equal(actual, np.arange(v, dtype=np.int32))


@pytest.mark.parametrize("path", ["cpu", pytest.param("gpu",
                                                     marks=pytest.mark.gpu)])
def test_wcc_numbering_independent_of_threads(monkeypatch, path):
    # V >= 2**17 takes the threaded first-occurrence numbering whenever more
    # than one thread is allowed; every thread count must give the oracle's
    # ids (MG_THREADS=1 is the serial scan).
    if path == "gpu" and not has_gpu():
        pytest.skip("no Metal device")
    src, dst, v = _bins_graph(seed=9)
    graph = mg.Graph.from_edges(src, dst, directed=True, num_vertices=v)
    expected = _oracle(src, dst, v)
    mg.set_execution(path)
    try:
        for threads in ("1", "2", "3", "16"):
            monkeypatch.setenv("MG_THREADS", threads)
            actual = np.asarray(mg.experimental.wcc(graph))
            assert mg.last_run_info()["path"] == path
            np.testing.assert_array_equal(actual, expected,
                                          err_msg=f"MG_THREADS={threads}")
    finally:
        mg.set_execution("auto")
