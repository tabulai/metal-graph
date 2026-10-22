# test_wcc_cpu_parallel.py — threaded CPU union-find WCC (and its Afforest
# skip on undirected storage) must reproduce the serial oracle exactly.
import networkx as nx
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")


@pytest.fixture
def cpu_parallel(monkeypatch):
    """CPU path with the parallel union-find forced on for small graphs."""
    monkeypatch.setenv("MG_WCC_CPU_PAR_MIN_EDGES", "0")
    mg.set_execution("cpu")
    yield monkeypatch
    mg.set_execution("auto")


def _wcc(g, monkeypatch, threads):
    monkeypatch.setenv("MG_THREADS", str(threads))
    comp = np.asarray(mg.experimental.wcc(g)).copy()
    assert mg.last_run_info()["path"] == "cpu"
    return comp


def _star_chain(n_chain=2000, n_leaves=1000):
    # chain c0 -> c1 -> ... -> c_{n-1} -> hub, hub -> leaf_j. The hub has the
    # largest out-degree (canonical 0) and joins the chain's (giant)
    # component through its IN edge, so its out-edges to the leaves are the
    # only stored copies on directed storage.
    hub = n_chain
    leaves = np.arange(n_chain + 1, n_chain + 1 + n_leaves)
    src = np.concatenate([np.arange(n_chain), np.full(n_leaves, hub)])
    dst = np.concatenate([np.arange(1, n_chain + 1), leaves])
    return src.astype(np.uint32), dst.astype(np.uint32), n_chain + 1 + n_leaves


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("threads", [2, 4, 16])
def test_star_chain_single_component(cpu_parallel, directed, threads):
    src, dst, v = _star_chain()
    g = mg.Graph.from_edges(src, dst, directed=directed, num_vertices=v)
    comp = _wcc(g, cpu_parallel, threads)
    assert np.all(comp == 0)
    np.testing.assert_array_equal(comp, _wcc(g, cpu_parallel, 1))


def _random_case(seed, directed, v=3000, n_comp=40):
    # Many components of mixed size and degree (some above the Afforest
    # sample-link width), duplicate edges, self-loops, isolated vertices.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(v)
    cuts = np.sort(rng.choice(np.arange(1, v - 50), n_comp - 1, replace=False))
    groups = np.split(perm[: v - 50], cuts)  # last 50 stay isolated
    src, dst = [], []
    for grp in groups:
        if grp.size < 2:
            continue
        m = int(grp.size * rng.integers(1, 6))
        src.append(rng.choice(grp, m))
        dst.append(rng.choice(grp, m))
        src.append(grp[:-1])  # spanning path keeps the group connected
        dst.append(grp[1:])
    src = np.concatenate(src).astype(np.uint32)
    dst = np.concatenate(dst).astype(np.uint32)
    loops = rng.choice(v, 20).astype(np.uint32)
    src = np.concatenate([src, loops, src[:100]])
    dst = np.concatenate([dst, loops, dst[:100]])
    return src, dst, v


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_parallel_matches_serial_and_networkx(cpu_parallel, directed, seed):
    src, dst, v = _random_case(seed, directed)
    g = mg.Graph.from_edges(src, dst, directed=directed, num_vertices=v)
    serial = _wcc(g, cpu_parallel, 1)
    for threads in (2, 3, 8, 16):
        np.testing.assert_array_equal(_wcc(g, cpu_parallel, threads), serial)
    nxg = nx.Graph()
    nxg.add_nodes_from(range(v))
    nxg.add_edges_from(zip(src.tolist(), dst.tolist()))
    expected = frozenset(frozenset(c) for c in nx.connected_components(nxg))
    parts = {}
    for u, c in enumerate(serial.tolist()):
        parts.setdefault(c, set()).add(u)
    assert frozenset(frozenset(s) for s in parts.values()) == expected


def _permuted_path(n, seed):
    # One long path whose vertex ids are randomly permuted: union-find trees
    # end up deep and every chain crosses many threads' vertex ranges, which
    # is where a find that writes into other threads' slots during the final
    # labeling pass would clobber their freshly stored roots.
    perm = np.random.default_rng(seed).permutation(n).astype(np.uint32)
    return perm[:-1], perm[1:], n


@pytest.mark.parametrize("directed", [True, False])
def test_permuted_long_path_repeated(cpu_parallel, directed):
    src, dst, v = _permuted_path(200_000, seed=11)
    g = mg.Graph.from_edges(src, dst, directed=directed, num_vertices=v)
    for run in range(30):
        comp = _wcc(g, cpu_parallel, 16)
        assert comp.max() == 0, (
            f"run {run}: {int(comp.max()) + 1} components on a connected path")
