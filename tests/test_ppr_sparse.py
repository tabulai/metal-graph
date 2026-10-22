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


def _run(g, queries, k, options=None):
    seeds, w, offs = _pack(queries)
    ids, scores = mg.ppr_topk(g, seeds, w, offs, k=k,
                             **(PPR_KW if options is None else options))
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


def _shifted_huge_rows(unreachable_edges, extra_hubs=0):
    """Seed 2 reaches {0, 1, 2} over 8,004 edges. Unreachable source 3
    shifts tiny terms across target 1's global IN-row tile boundaries.
    Serial fp64 sums would tie 0 and 1 exactly; the dense gather's fixed
    tiles favor 1 even though the returned float32 scores are identical."""
    n = 4000
    src = np.concatenate(([2], np.repeat(3, unreachable_edges),
                          np.repeat(2, n), [2], np.repeat(2, n), [0, 1]))
    dst = np.concatenate(([1], np.repeat(1, unreachable_edges),
                          np.repeat(1, n), [0], np.repeat(0, n), [0, 1]))
    w = np.concatenate(([1.], np.ones(unreachable_edges),
                        np.full(n, 2.**-55), [1.],
                        np.full(n, 2.**-55), [1., 1.]))
    if extra_hubs:
        # Enough plan items to exercise dense PPR's threaded gather.
        hubs = np.repeat(np.arange(4, 4 + extra_hubs), 16384)
        src = np.concatenate((src, hubs))
        dst = np.concatenate((dst, hubs))
        w = np.concatenate((w, np.ones(hubs.size)))
    return mg.Graph.from_edges(src.astype(np.uint32), dst.astype(np.uint32),
                                weights=w.astype(np.float32), directed=True,
                                num_vertices=4 + extra_hubs)


@pytest.mark.parametrize("unreachable_edges", [16382, 16383, 32767])
@pytest.mark.parametrize("options", [{}, dict(max_iter=1)])
def test_huge_row_near_tie_matches_dense_tiles(
        ppr_env, unreachable_edges, options):
    """Default convergence and one iteration must keep the dense path's
    fp64 order before top-k selection, including k=1, different tile
    boundaries, thread counts, and the latency path on or off."""
    mg.set_execution("cpu")
    g = _shifted_huge_rows(unreachable_edges, extra_hubs=8)
    expected = {}
    for threads in (1, 4):
        ppr_env.setenv("MG_THREADS", str(threads))
        for cap in (0, 8192):
            _set_caps(ppr_env, 1024, cap)
            for k in (1, 4):
                ids, scores, info = _run(g, [([2], [1.])], k,
                                         options=options)
                # Target 1's row is tiled (a serial sum would tie, pick 0).
                assert ids[0, 0] == 1
                if k == 4:
                    np.testing.assert_array_equal(ids[0], [1, 0, 2, 3])
                    assert scores[0, 0] == scores[0, 1]
                result = (ids.tobytes(), scores.tobytes(), info["iterations"])
                assert expected.setdefault(k, result) == result


def test_huge_rows_in_mixed_sparse_dense_batch(ppr_env):
    """The near-tie sparse row and two overflowing rows must match both
    dense-only batches and individual calls, independently of workers."""
    mg.set_execution("cpu")
    g = _shifted_huge_rows(16383, extra_hubs=8)
    qs = [([2], [1.]), ([3], [1.]), ([4], [1.])]
    baseline = None
    for threads in (1, 4):
        ppr_env.setenv("MG_THREADS", str(threads))
        for cap in (0, 8192):
            _set_caps(ppr_env, 1024, cap)
            ids, scores, info = _run(g, qs, 4)
            result = (ids.tobytes(), scores.tobytes(), info["iterations"])
            if baseline is None:
                baseline = result
            assert result == baseline
            iterations = []
            for row, query in enumerate(qs):
                ids_q, scores_q, info_q = _run(g, [query], 4)
                np.testing.assert_array_equal(ids_q[0], ids[row])
                assert scores_q[0].tobytes() == scores[row].tobytes()
                iterations.append(info_q["iterations"])
            assert info["iterations"] == max(iterations)


@pytest.mark.parametrize("directed,weighted", GRAPHS)
def test_huge_reachable_rows_with_raised_caps(ppr_env, directed, weighted):
    """With raised caps the sparse row can itself exceed a tile. Cover
    both orientations, zero weights, duplicate edges, and padded top-k."""
    n = 17000
    src = np.concatenate((np.full(n, 2), np.full(n, 3), [0, 1]))
    dst = np.concatenate((np.zeros(n), np.ones(n), [2, 3]))
    w = None
    if weighted:
        w = np.ones(src.size, np.float32)
        w[1:n] = 2.**-55
        w[n + 1:2 * n] = 2.**-55
        w[::19] = 0.0
    g = mg.Graph.from_edges(src.astype(np.uint32), dst.astype(np.uint32),
                            weights=w, directed=directed, num_vertices=5)
    mg.set_execution("cpu")
    baseline = None
    for threads in (1, 4):
        ppr_env.setenv("MG_THREADS", str(threads))
        for cap in (0, 100000):
            _set_caps(ppr_env, 1024, cap)
            ids, scores, info = _run(g, [([2, 3], [1., 1.])], 8)
            result = (ids.tobytes(), scores.tobytes(), info["iterations"])
            if baseline is None:
                baseline = result
            assert result == baseline


# MG_HUGE_TILE_EDGES == MG_BIN_HUGE_MIN (src/kernels/mg_params.h): rows with
# at least this many stored in-edges are gathered in tiles of this size.
TILE = 16384


def _interleaved_huge_rows(directed, weighted, seed, m=1500):
    """Seed 0 has m out-edges to each target 1..4 with one shared weight
    sequence, so serial fp64 row sums tie exactly. Source 5 (unreachable
    from 0 when directed) adds 0, TILE // 2 + 1 or several tiles of edges
    into a target, interleaved at random with the reachable ones, so each
    target's global tile boundaries cut its reachable edges differently."""
    rng = np.random.default_rng(seed)
    src, dst, w = [], [], []
    w_reach = rng.uniform(0.5, 2.0, m)
    pick = rng.random(m)
    w_reach[pick < 0.15] = 2.0 ** -rng.integers(40, 56, (pick < 0.15).sum())
    w_reach[(pick >= 0.15) & (pick < 0.22)] = 0.0
    w_reach[(pick >= 0.22) & (pick < 0.25)] = -0.0  # valid: -0.0 >= 0
    for t, extra in enumerate([0, TILE // 2 + 1,
                               int(rng.integers(TILE, 2 * TILE)),
                               int(rng.integers(2 * TILE, 3 * TILE))]):
        reach = np.zeros(m + extra, bool)
        reach[rng.choice(m + extra, m, replace=False)] = True
        src.append(np.where(reach, 0, 5))
        dst.append(np.full(m + extra, t + 1))
        wt = rng.uniform(0.1, 3.0, m + extra)
        wt[reach] = w_reach
        w.append(wt)
    src = np.concatenate(src).astype(np.uint32)
    dst = np.concatenate(dst).astype(np.uint32)
    w = np.concatenate(w).astype(np.float32) if weighted else None
    g = mg.Graph.from_edges(src, dst, weights=w, directed=directed,
                            num_vertices=7)  # + isolated vertex 6
    return g, src, dst, w


def _tiled_first_iteration(src, dst, w, directed, v, alpha=0.85):
    """fp64 scores after one PPR iteration from seed 0: the dense CPU
    gather's documented arithmetic. Stored IN row of x = edges into x in
    input order (undirected: every edge at x); rows of >= TILE edges sum
    each TILE-edge tile serially, then the partials in order; products are
    rounded before they are added. Only seed 0 has nonzero mass."""
    n = src.size
    key, other, eid = dst, src, np.arange(n)
    if not directed:
        loop = src == dst
        key = np.concatenate((dst, src[~loop]))
        other = np.concatenate((src, dst[~loop]))
        eid = np.concatenate((eid, eid[~loop]))
    order = np.lexsort((eid, key))  # row, then stored (= input) order
    key, other, eid = key[order], other[order], eid[order]
    first = np.searchsorted(key, np.arange(v))
    deg = np.bincount(key, minlength=v)
    pos = np.arange(key.size) - first[key]
    # ows[0]: float32 sum over the seed's OUT row (stored order).
    out0 = eid[key == 0] if not directed else np.flatnonzero(src == 0)
    ows = np.float32(0.0)
    for e in out0:
        ows = np.float32(ows + (w[e] if w is not None else np.float32(1)))
    c = 1.0 / float(ows)
    parts = {}
    for i in np.flatnonzero(other == 0):
        x = int(key[i])
        tile = int(pos[i]) // TILE if deg[x] >= TILE else 0
        term = c * float(w[eid[i]]) if w is not None else c
        parts[(x, tile)] = parts.get((x, tile), 0.0) + term
    acc = [0.0] * v
    for (x, _), p in sorted(parts.items()):
        acc[x] += p
    r = [alpha * a for a in acc]
    r[0] += (1.0 - alpha) + alpha * 0.0  # seed teleport (no dangling mass)
    return r


@pytest.mark.parametrize("seed", [1, 2, 3, 6])
@pytest.mark.parametrize("directed,weighted", GRAPHS)
def test_interleaved_huge_rows_match_tiled_sums(ppr_env, directed, weighted,
                                                seed):
    """One iteration, every vertex ranked: the dense path and the latency
    path both reproduce the tiled fp64 sums exactly (near-ties ordered by
    the exact fp64 value, float32 scores equal), for any thread count.
    Weighted seeds 1 and 6 also fail if either path fuses multiply-adds."""
    g, src, dst, w = _interleaved_huge_rows(directed, weighted, seed)
    r = _tiled_first_iteration(src, dst, w, directed, 7)
    want_ids = sorted(range(7), key=lambda x: (-r[x], x))
    want_scores = np.array([r[x] for x in want_ids], np.float32)
    mg.set_execution("cpu")
    opts = dict(alpha=0.85, tol=1e-12, max_iter=1)
    # Undirected: the seed reaches all of the high-degree vertices.
    caps = (1024, 8192) if directed else (1024, 1 << 22)
    for threads in (1, 4):
        ppr_env.setenv("MG_THREADS", str(threads))
        for vertices, edges in ((0, 0), caps):
            _set_caps(ppr_env, vertices, edges)
            ids, scores, info = _run(g, [([0], [1.])], 7, options=opts)
            assert info["iterations"] == 1
            np.testing.assert_array_equal(ids[0], want_ids)
            assert scores[0].tobytes() == want_scores.tobytes()


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("directed,weighted", GRAPHS)
def test_interleaved_huge_rows_sparse_matches_dense(ppr_env, directed,
                                                    weighted, seed):
    """Converged queries over huge tiled rows: latency path on or off,
    batched or one at a time, 1 or 4 threads, all bit-identical. Directed
    batches mix latency rows (seed 0, targets) with a dense one (source 5)."""
    g, *_ = _interleaved_huge_rows(directed, weighted, seed)
    qs = [([0], [1.]), ([5], [1.]), ([0, 3], [0.25, 1.]), ([4], [1.])]
    caps = (1024, 8192) if directed else (1024, 1 << 22)
    mg.set_execution("cpu")
    baseline = None
    for threads in (1, 4):
        ppr_env.setenv("MG_THREADS", str(threads))
        for vertices, edges in ((0, 0), caps):
            _set_caps(ppr_env, vertices, edges)
            ids, scores, info = _run(g, qs, 7)
            result = (ids.tobytes(), scores.tobytes(), info["iterations"])
            if baseline is None:
                baseline = result
            assert result == baseline
            for row, query in enumerate(qs):
                ids_q, scores_q, _ = _run(g, [query], 7)
                np.testing.assert_array_equal(ids_q[0], ids[row])
                assert scores_q[0].tobytes() == scores[row].tobytes()


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
