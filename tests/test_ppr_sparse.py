# test_ppr_sparse.py — ppr_topk's bounded-reach latency path
# (MG_PPR_SPARSE_MAX_VERTICES / MG_PPR_SPARSE_MAX_EDGES). A query whose
# seeds' OUT-reach fits the caps is solved exactly on that set; it must be
# bit-identical to the dense CPU path (ids, scores, iterations), and mixed
# batches must route each row independently.
import numpy as np
import pytest

from conftest import has_gpu

mg = pytest.importorskip("metal_graph")

PPR_KW = dict(alpha=0.85, tol=1e-10, max_iter=500)
CAPS_ENV = ("MG_PPR_SPARSE_MAX_VERTICES", "MG_PPR_SPARSE_MAX_EDGES")


@pytest.fixture
def ppr_env(monkeypatch):
    for name in CAPS_ENV + ("MG_E_GPU_MIN", "MG_E_GPU_MIN_PPR"):
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch
    mg.set_execution("auto")


def _set_caps(monkeypatch, vertices, edges):
    monkeypatch.setenv("MG_PPR_SPARSE_MAX_VERTICES", str(vertices))
    monkeypatch.setenv("MG_PPR_SPARSE_MAX_EDGES", str(edges))


def _pack(queries):
    seeds = np.concatenate([np.asarray(s, np.uint32) for s, _ in queries])
    w = np.concatenate([np.asarray(x, np.float32) for _, x in queries])
    offs = np.zeros(len(queries) + 1, np.uint64)
    np.cumsum([len(s) for s, _ in queries], out=offs[1:])
    return seeds, w, offs


def _run(g, queries, k):
    seeds, w, offs = _pack(queries)
    ids, scores = mg.ppr_topk(g, seeds, w, offs, k=k, **PPR_KW)
    info = mg.last_run_info()
    return (np.asarray(ids).reshape(len(queries), k).copy(),
            np.asarray(scores).reshape(len(queries), k).copy(), info)


def _mixed_graph(directed, weighted, seed=5):
    """Short out-chains and sinks (small reach) next to a dense random core
    (large reach), plus self-loops, duplicate and zero-weight edges."""
    rng = np.random.default_rng(seed)
    core_v = 120
    src = [rng.integers(0, core_v, 1500)]
    dst = [rng.integers(0, core_v, 1500)]
    nxt = core_v
    for length in (1, 2, 4, 7, 12):  # chains: c0 -> c1 -> ... (dangling end)
        chain = np.arange(nxt, nxt + length + 1)
        src.append(chain[:-1])
        dst.append(chain[1:])
        nxt += length + 1
    src.append(np.array([nxt, nxt, nxt + 1]))  # self-loop + duplicate edge
    dst.append(np.array([nxt, nxt + 1, nxt + 1]))
    nxt += 2
    src.append(rng.integers(nxt, nxt + 40, 60))  # small cyclic islands
    dst.append(rng.integers(nxt, nxt + 40, 60))
    nxt += 40
    src = np.concatenate(src).astype(np.uint32)
    dst = np.concatenate(dst).astype(np.uint32)
    w = None
    if weighted:
        w = rng.uniform(0.1, 2.0, src.size).astype(np.float32)
        w[::17] = 0.0  # zero-weight out-edges (ows == 0 with degree > 0)
    g = mg.Graph.from_edges(src, dst, weights=w, directed=directed,
                            num_vertices=nxt + 5)  # + isolated tail
    return g, nxt + 5, core_v


def _queries(v, core_v, seed=9):
    rng = np.random.default_rng(seed)
    qs = []
    for i in range(24):
        if i % 3 == 0:  # core seed: reach overflows any small cap
            seeds = rng.integers(0, core_v, int(rng.integers(1, 4)))
        else:  # outside the core: small reach
            seeds = rng.integers(core_v, v, int(rng.integers(1, 4)))
        seeds = np.concatenate([seeds, seeds[:1]])  # duplicate seed
        qs.append((seeds, rng.uniform(0.0, 1.5, seeds.size)))
    qs.append(([v - 1], [1.0]))  # isolated vertex
    return qs


GRAPHS = [(True, False), (True, True), (False, False), (False, True)]


@pytest.mark.parametrize("directed,weighted", GRAPHS)
@pytest.mark.parametrize("caps", [(1024, 8192), (9, 25), (40, 300)])
def test_sparse_cpu_bit_identical_to_dense(ppr_env, directed, weighted, caps):
    """CPU mode: caps off (0) vs on, including small caps that mix sparse
    and dense rows in one batch, must agree bit for bit."""
    g, v, core_v = _mixed_graph(directed, weighted)
    qs = _queries(v, core_v)
    mg.set_execution("cpu")
    for k in (5, 64, v + 7):  # k > V pads with -1 / 0
        _set_caps(ppr_env, 0, 0)
        ids0, sc0, info0 = _run(g, qs, k)
        _set_caps(ppr_env, *caps)
        ids1, sc1, info1 = _run(g, qs, k)
        assert info0["path"] == info1["path"] == "cpu"
        np.testing.assert_array_equal(ids1, ids0)
        assert sc1.tobytes() == sc0.tobytes()
        assert info1["iterations"] == info0["iterations"]


@pytest.mark.parametrize("directed,weighted", GRAPHS)
def test_sparse_rows_alone_bit_identical(ppr_env, directed, weighted):
    """Each query run alone (sparse path when it fits) equals its dense CPU
    row, including the iteration count."""
    g, v, core_v = _mixed_graph(directed, weighted)
    mg.set_execution("cpu")
    for q in _queries(v, core_v)[:10]:
        _set_caps(ppr_env, 0, 0)
        ids0, sc0, info0 = _run(g, [q], 16)
        _set_caps(ppr_env, 1024, 8192)
        ids1, sc1, info1 = _run(g, [q], 16)
        np.testing.assert_array_equal(ids1, ids0)
        assert sc1.tobytes() == sc0.tobytes()
        assert info1["iterations"] == info0["iterations"]


@pytest.mark.gpu
@pytest.mark.parametrize("directed,weighted", GRAPHS)
def test_mixed_batch_rows_match_cpu_or_gpu(ppr_env, directed, weighted):
    """Auto mode with the GPU planned: rows that fit the caps are the exact
    CPU rows, the rest are the forced-GPU rows, and batched == sequential."""
    if not has_gpu():
        pytest.skip("no Metal device")
    g, v, core_v = _mixed_graph(directed, weighted)
    qs = _queries(v, core_v)
    k = 16
    _set_caps(ppr_env, 9, 25)
    mg.set_execution("cpu")
    ids_cpu, sc_cpu, _ = _run(g, qs, k)
    mg.set_execution("gpu")
    ids_gpu, sc_gpu, _ = _run(g, qs, k)

    mg.set_execution("auto")
    ppr_env.setenv("MG_E_GPU_MIN", "0")
    ids, sc, info = _run(g, qs, k)
    assert info["path"] == "gpu"  # mixed batch reports the planned path
    n_cpu = n_gpu = 0
    for q in range(len(qs)):
        if (np.array_equal(ids[q], ids_cpu[q])
                and sc[q].tobytes() == sc_cpu[q].tobytes()):
            n_cpu += 1
        else:
            np.testing.assert_array_equal(ids[q], ids_gpu[q])
            assert sc[q].tobytes() == sc_gpu[q].tobytes()
            n_gpu += 1
        ids_q, sc_q, _ = _run(g, [qs[q]], k)  # sequential
        np.testing.assert_array_equal(ids_q[0], ids[q])
        assert sc_q[0].tobytes() == sc[q].tobytes()
    assert n_cpu > 0 and n_gpu > 0  # the batch really was mixed


@pytest.mark.gpu
@pytest.mark.parametrize("two_sided", [False, True])
def test_cap_boundaries_and_knob_off(ppr_env, two_sided):
    """A chain head reaches L vertices over L - 1 out-edges: caps equal to
    those counts take the latency path (telemetry path == 'cpu' with the GPU
    planned), one less overflows to the GPU, and 0 disables the path."""
    if not has_gpu():
        pytest.skip("no Metal device")
    L = 50
    src = np.arange(L - 1, dtype=np.uint32)
    dst = src + 1
    if two_sided:  # undirected storage: reach is still the whole chain
        g = mg.Graph.from_edges(src, dst, directed=False, num_vertices=L)
        n_edges = 2 * (L - 1)  # each u != v edge stored both ways
    else:
        g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=L)
        n_edges = L - 1
    mg.set_execution("auto")
    ppr_env.setenv("MG_E_GPU_MIN", "0")
    q = [([0], [1.0])]

    def path_with(vertices, edges):
        _set_caps(ppr_env, vertices, edges)
        return _run(g, q, 8)[2]["path"]

    assert path_with(L, n_edges) == "cpu"
    assert path_with(L - 1, n_edges) == "gpu"
    assert path_with(L, n_edges - 1) == "gpu"
    assert path_with(0, n_edges) == "gpu"
    assert path_with(L, 0) == "gpu"
    mg.set_execution("gpu")  # forced GPU skips the attempt entirely
    assert path_with(L, n_edges) == "gpu"
