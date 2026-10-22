# test_bfs.py — multi-source BFS: golden dist + structural parent checks.
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")
mg_core = pytest.importorskip("metal_graph._core")

from conftest import (ALL_CASES, adjacency_sets, bfs_dist_ref,
                      bfs_expected_path, build_mg)

DIRECTIONS = ["out", "in", "both"]


def _source_sets(case):
    v = case.num_vertices
    single = [0]
    multi = sorted({0, v // 2, v - 1})
    return {"single": single, "multi": multi}


# The sparse preflight would otherwise absorb every V<=400 fixture on the
# cpu/auto routes, silently stripping the golden matrix's coverage of the
# threaded cpu::bfs oracle — so the matrix runs both with the preflight
# enabled (default) and disabled (full path forced).
SPARSE_MODES = ["default", "nosparse"]


def _apply_sparse_mode(monkeypatch, mode):
    if mode == "default":
        # Test the documented defaults, not tuning inherited from the shell.
        monkeypatch.delenv("MG_BFS_SPARSE_MAX_VERTICES", raising=False)
        monkeypatch.delenv("MG_BFS_SPARSE_MAX_EDGES", raising=False)
    else:
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "0")
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", "0")


def _assert_bfs_op(mode, exec_path, case, direction):
    op = mg.last_run_info()["op"]
    if mode == "nosparse" or exec_path == "gpu":
        assert op == "bfs", (
            f"{mode=} {exec_path=}: preflight must not run")
        return

    # The default sparse preflight is guaranteed to complete when the whole
    # relevant orientation fits its edge cap (8192 when, as for these
    # CPU-routed fixtures, the fallback is the threaded CPU path; every
    # fixture is also below the 1024-vertex cap). Larger graphs may still fit
    # when the reachable component is small, so leave that data-dependent
    # case to the explicit cap/fallback test below.
    orientation_copies = 1 if case.directed and direction != "both" else 2
    if case.n_edges * orientation_copies <= 8192:
        assert op == "bfs_sparse", (
            f"default CPU preflight should cover {case.name}/{direction}")
    else:
        assert op in {"bfs_sparse", "bfs"}


@pytest.mark.parametrize("sparse_mode", SPARSE_MODES)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("srcset", ["single", "multi"])
def test_bfs_dist_matches_reference(graph_case, direction, srcset, exec_path,
                                    assert_path, sparse_mode, monkeypatch):
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("no valid sources in an empty graph")
    _apply_sparse_mode(monkeypatch, sparse_mode)
    sources = _source_sets(case)[srcset]
    g = build_mg(case)
    dist, parent = mg.bfs(g, np.asarray(sources, np.uint32),
                          direction=direction)
    assert_path(bfs_expected_path(case.directed, direction, exec_path))
    _assert_bfs_op(sparse_mode, exec_path, case, direction)
    dist = np.asarray(dist)
    parent = np.asarray(parent)
    assert dist.shape == (case.num_vertices,)
    assert parent.shape == (case.num_vertices,)
    expected = bfs_dist_ref(case, sources, direction)
    np.testing.assert_array_equal(dist, expected)
    # unreachable vertices carry parent == -1
    assert ((dist >= 0) | (parent == -1)).all()


@pytest.mark.parametrize("sparse_mode", SPARSE_MODES)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_bfs_parent_structural(graph_case, direction, exec_path, assert_path,
                               sparse_mode, monkeypatch):
    # Parent choice may be nondeterministic on the GPU path — validate
    # structurally: parent is visited, one level closer, and the traversal
    # edge parent -> v exists in the requested direction.
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("no valid sources in an empty graph")
    _apply_sparse_mode(monkeypatch, sparse_mode)
    sources = _source_sets(case)["multi"]
    g = build_mg(case)
    dist, parent = mg.bfs(g, np.asarray(sources, np.uint32),
                          direction=direction)
    assert_path(bfs_expected_path(case.directed, direction, exec_path))
    _assert_bfs_op(sparse_mode, exec_path, case, direction)
    dist = np.asarray(dist)
    parent = np.asarray(parent)
    adj = adjacency_sets(case, direction)
    src_set = set(sources)
    for v in range(case.num_vertices):
        if dist[v] < 0:
            assert parent[v] == -1, f"unreachable {v} must have parent -1"
        elif dist[v] == 0:
            assert v in src_set
            assert parent[v] == -1, f"source {v} must have parent -1"
        else:
            p = int(parent[v])
            assert 0 <= p < case.num_vertices
            assert dist[p] == dist[v] - 1, (
                f"parent[{v}]={p}: dist {dist[p]} != {dist[v]} - 1")
            assert v in adj[p], (
                f"no traversal edge {p} -> {v} (direction={direction})")


def test_bfs_both_equals_undirected_view(exec_path, assert_path):
    # direction='both' == BFS over the underlying undirected view.
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    dist, _ = mg.bfs(g, np.asarray([3], np.uint32), direction="both")
    assert_path(bfs_expected_path(case.directed, "both", exec_path))
    expected = bfs_dist_ref(case, [3], "both")  # undirected adjacency
    np.testing.assert_array_equal(np.asarray(dist), expected)


def test_bfs_duplicate_sources(exec_path, assert_path):
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    d1, _ = mg.bfs(g, np.asarray([5, 5, 5, 9], np.uint32), direction="out")
    assert_path(exec_path)
    d2, _ = mg.bfs(g, np.asarray([5, 9], np.uint32), direction="out")
    np.testing.assert_array_equal(np.asarray(d1), np.asarray(d2))


def test_bfs_scalar_source(exec_path, assert_path):
    case = ALL_CASES["path"]
    g = build_mg(case)
    d1, p1 = mg.bfs(g, 7, direction="out")
    assert_path(exec_path)
    d2, p2 = mg.bfs(g, np.asarray([7], np.uint32), direction="out")
    np.testing.assert_array_equal(np.asarray(d1), np.asarray(d2))
    np.testing.assert_array_equal(np.asarray(p1), np.asarray(p2))


def test_bfs_unreachable_minus_one(exec_path, assert_path):
    # path graph, direction='out' from the middle: everything before the
    # source is unreachable.
    case = ALL_CASES["path"]
    g = build_mg(case)
    dist, parent = mg.bfs(g, np.asarray([25], np.uint32), direction="out")
    assert_path(exec_path)
    dist = np.asarray(dist)
    parent = np.asarray(parent)
    assert (dist[:25] == -1).all()
    assert (parent[:25] == -1).all()
    np.testing.assert_array_equal(dist[25:], np.arange(25, dtype=np.int32))


def test_bfs_source_out_of_range_raises():
    case = ALL_CASES["path"]
    g = build_mg(case)
    with pytest.raises(ValueError):
        mg.bfs(g, np.asarray([case.num_vertices], np.uint32),
               direction="out")


def test_sparse_cpu_path_and_cap_fallback_agree(monkeypatch):
    """The latency path and regular CPU fallback preserve dense semantics."""
    src = np.arange(63, dtype=np.uint32)
    dst = src + 1
    g = mg.Graph.from_edges(
        src, dst, directed=True, num_vertices=64
    )
    source = np.asarray([0], np.uint32)
    mg.set_execution("cpu")
    try:
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "128")
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", "128")
        sparse_dist, sparse_parent = mg.bfs(g, source, direction="out")
        assert mg.last_run_info()["op"] == "bfs_sparse"

        monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "1")
        fallback_dist, fallback_parent = mg.bfs(
            g, source, direction="out"
        )
        assert mg.last_run_info()["op"] == "bfs"

        monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "128")
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", "1")
        edge_fallback_dist, edge_fallback_parent = mg.bfs(
            g, source, direction="out"
        )
        assert mg.last_run_info()["op"] == "bfs"
    finally:
        mg.set_execution("auto")

    np.testing.assert_array_equal(sparse_dist, fallback_dist)
    np.testing.assert_array_equal(sparse_dist, edge_fallback_dist)
    np.testing.assert_array_equal(sparse_dist, np.arange(64, dtype=np.int32))
    for parent in (
        sparse_parent, fallback_parent, edge_fallback_parent
    ):
        assert parent[0] == -1
        np.testing.assert_array_equal(
            parent[1:], np.arange(63, dtype=np.int32)
        )


def test_sparse_cpu_path_is_reentrant(monkeypatch):
    """Concurrent calls use call-local queues and output arrays."""
    src = np.arange(255, dtype=np.uint32)
    dst = src + 1
    g = mg.Graph.from_edges(
        src, dst, directed=True, num_vertices=256
    )
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "512")
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", "512")
    mg.set_execution("cpu")
    try:
        def run(source):
            dist, parent = mg.bfs(g, [source], direction="out")
            return np.asarray(dist), np.asarray(parent)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, range(16)))
    finally:
        mg.set_execution("auto")

    for source, (dist, parent) in enumerate(results):
        np.testing.assert_array_equal(dist[:source], -1)
        np.testing.assert_array_equal(
            dist[source:], np.arange(256 - source, dtype=np.int32)
        )
        assert parent[source] == -1
        if source + 1 < 256:
            np.testing.assert_array_equal(
                parent[source + 1:], np.arange(source, 255, dtype=np.int32)
            )


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("threads", ["1", "4"])
def test_cpu_dense_output_fused_gather_permuted_ids(
        monkeypatch, direction, directed, threads):
    """A wide layered graph exercises parallel claims, competing parents,
    and USER/canonical permutations. Direct dense output must match the
    canonical sparse fallback and the serial preflight, including empty
    sources, duplicate sources, and isolated vertices."""
    from collections import deque

    width = 4096  # each full layer exceeds the frontier's threading grain
    a = np.arange(1, width + 1, dtype=np.uint32)
    b = a + width
    c = b + width
    sink = 3 * width + 1
    v = sink + 17
    chain = np.arange(sink + 1, sink + 9, dtype=np.uint32)
    src = np.concatenate((np.zeros(width, np.uint32), a, a, b, c,
                          chain[:-1], [a[5], 0])).astype(np.uint32)
    dst = np.concatenate((a, b, np.roll(b, 1), c,
                          np.full(width, sink, np.uint32),
                          chain[1:], [a[5], 0])).astype(np.uint32)
    perm = np.random.default_rng(29).permutation(v).astype(np.uint32)
    src, dst = perm[src], perm[dst]
    g = mg.Graph.from_edges(src, dst, directed=directed, num_vertices=v)
    if not directed or direction == "both":
        ts, td = np.concatenate((src, dst)), np.concatenate((dst, src))
    elif direction == "in":
        ts, td = dst, src
    else:
        ts, td = src, dst
    adjacency = [[] for _ in range(v)]
    for s, d in zip(ts, td):
        adjacency[int(s)].append(int(d))
    edge_keys = ts.astype(np.int64) * v + td
    root = sink if direction == "in" else 0
    source_sets = [[], [int(perm[-1])],
                   perm[[v - 1, root, root, sink + 4]].tolist()]

    def expected_dist(sources):
        distance = np.full(v, -1, np.int32)
        queue = deque()
        for source in sources:
            if distance[source] < 0:
                distance[source] = 0
                queue.append(source)
        while queue:
            source = queue.popleft()
            for target in adjacency[source]:
                if distance[target] < 0:
                    distance[target] = distance[source] + 1
                    queue.append(target)
        return distance

    def assert_parents(vertices, distances, parents, expected):
        vertices = np.asarray(vertices)
        distances = np.asarray(distances)
        parents = np.asarray(parents)
        child = distances > 0
        assert (parents[~child] == -1).all()
        p = parents[child].astype(np.int64)
        children = vertices[child].astype(np.int64)
        assert ((p >= 0) & (p < v)).all()
        np.testing.assert_array_equal(expected[p], distances[child] - 1)
        assert np.isin(p * v + children, edge_keys).all()

    monkeypatch.setenv("MG_THREADS", threads)
    mg.set_execution("cpu")
    try:
        for sources in source_sets:
            expected = expected_dist(sources)
            for vertex_cap, edge_cap in ((0, 0), (1, 1), (v, 2 * src.size)):
                monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", str(vertex_cap))
                monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", str(edge_cap))
                source_array = np.asarray(sources, np.uint32)
                distances, parents = mg.bfs(g, source_array, direction=direction)
                info = dict(mg.last_run_info())
                assert info["path"] == "cpu"
                if vertex_cap == 0:
                    assert info["op"] == "bfs"  # isolate the new fused gather
                np.testing.assert_array_equal(distances, expected)
                assert info["iterations"] == int(expected.max()) + 1
                assert_parents(np.arange(v), distances, parents, expected)
                vs, ds, ps = mg.bfs(g, source_array, direction=direction,
                                    output="sparse")
                np.testing.assert_array_equal(vs, np.flatnonzero(expected >= 0))
                np.testing.assert_array_equal(ds, expected[vs])
                assert_parents(vs, ds, ps, expected)
    finally:
        mg.set_execution("auto")


def _debug_batch_schedule(monkeypatch, value=None, *, vertices=4096,
                          max_levels=0, encoded=0, max_batches=6):
    if value is None:
        monkeypatch.delenv("MG_BFS_LEVELS_PER_BATCH", raising=False)
    else:
        monkeypatch.setenv("MG_BFS_LEVELS_PER_BATCH", value)
    return mg_core._debug_bfs_batch_schedule(
        vertices, max_levels, encoded, max_batches
    )


@pytest.mark.parametrize(
    "value", [None, "", "not-an-int", "12junk", "9" * 100]
)
def test_bfs_batch_schedule_invalid_values_use_adaptive(monkeypatch, value):
    info = _debug_batch_schedule(monkeypatch, value)
    assert info["fixed"] is False
    assert info["configured_levels"] == 8
    assert list(info["batches"]) == [8, 8, 16, 32, 64, 64]


@pytest.mark.parametrize(
    ("value", "configured"),
    [("1", 1), ("16", 16), ("0", 1), ("-17", 1),
     ("999999999", 256)],
)
def test_bfs_batch_schedule_fixed_values_are_bounded(
        monkeypatch, value, configured):
    info = _debug_batch_schedule(monkeypatch, value, max_batches=3)
    assert info["fixed"] is True
    assert info["configured_levels"] == configured
    assert info["command_buffer_level_cap"] == 256
    assert list(info["batches"]) == [configured] * 3


def test_bfs_batch_schedule_honors_level_bound_without_underflow(monkeypatch):
    info = _debug_batch_schedule(
        monkeypatch, None, max_levels=10, max_batches=8
    )
    assert list(info["batches"]) == [8, 2]
    assert info["encoded"] == 10
    assert info["stopped"] == "level_bound"

    uint32_max = (1 << 32) - 1
    near_limit = _debug_batch_schedule(
        monkeypatch, "256", vertices=uint32_max, max_levels=uint32_max,
        encoded=uint32_max - 3, max_batches=2
    )
    assert list(near_limit["batches"]) == [3]
    assert near_limit["encoded"] == uint32_max
    assert near_limit["stopped"] == "level_bound"


def test_bfs_batch_schedule_has_overflow_safe_termination_guard(monkeypatch):
    info = _debug_batch_schedule(
        monkeypatch, None, vertices=1, max_batches=100
    )
    assert info["prepare_limit"] == 1 + 2 + 256
    assert info["encoded"] == info["prepare_limit"] + 1
    assert info["stopped"] == "safety"
    assert max(info["batches"]) <= info["command_buffer_level_cap"]

    # The protocol failure wins over a coincident k-hop depth boundary.
    bounded = _debug_batch_schedule(
        monkeypatch, "256", vertices=1,
        max_levels=info["prepare_limit"] + 1, max_batches=2
    )
    assert bounded["encoded"] == bounded["prepare_limit"] + 1
    assert bounded["stopped"] == "safety"


@pytest.mark.gpu
def test_bfs_deep_chain_crosses_many_batches(monkeypatch):
    # 4096-level path graph: the traversal spans dozens of command buffers
    # under the adaptive 8,8,16,32,64... schedule (and hundreds under a pinned
    # small batch). Guards level/parity bookkeeping across commit boundaries
    # while exercising the adaptive default; dist on a chain is determined.
    from conftest import has_gpu
    if not has_gpu():
        pytest.skip("no Metal device")
    monkeypatch.delenv("MG_BFS_LEVELS_PER_BATCH", raising=False)
    v = 4096
    src = np.arange(v - 1, dtype=np.uint32)
    dst = np.arange(1, v, dtype=np.uint32)
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=v)
    mg.set_execution("gpu")
    try:
        dist, parent = mg.bfs(g, np.asarray([0], np.uint32), direction="out")
        assert mg.last_run_info()["path"] == "gpu"
        assert mg.last_run_info()["iterations"] == v
    finally:
        mg.set_execution("auto")
    np.testing.assert_array_equal(np.asarray(dist),
                                  np.arange(v, dtype=np.int32))
    par = np.asarray(parent)
    assert par[0] == -1
    np.testing.assert_array_equal(par[1:], np.arange(v - 1, dtype=np.int32))


def _multi_hub_graph():
    """Source 0 -> hubs 1..3 (out-degrees 70k, 40k, 2k, so one level's high
    bin mixes split and unsplit rows); hub edges overlap on shared leaves so
    slices of different hubs race to claim the same vertices, and a second
    level fans out of a 1.5k-degree leaf."""
    rng = np.random.default_rng(17)
    n_leaves = 90_000
    leaf0 = 4
    src = [np.zeros(3, np.int64)]
    dst = [np.array([1, 2, 3])]
    for hub, deg in ((1, 70_000), (2, 40_000), (3, 2_000)):
        src.append(np.full(deg, hub))
        dst.append(leaf0 + rng.choice(n_leaves, deg, replace=False))
    deep = leaf0 + n_leaves
    src.append(np.full(1_500, dst[1][0]))  # a leaf of hub 1
    dst.append(np.arange(deep, deep + 1_500))
    src = np.concatenate(src).astype(np.uint32)
    dst = np.concatenate(dst).astype(np.uint32)
    return src, dst, deep + 1_500


@pytest.mark.gpu
@pytest.mark.parametrize("split_max", ["1", "2", "64"])
@pytest.mark.parametrize("direction", ["out", "in"])
def test_bfs_gpu_high_degree_split_matches_cpu(monkeypatch, split_max,
                                               direction):
    """GPU BFS splits a high-degree frontier row across up to
    MG_BFS_HIGH_SPLIT_MAX threadgroups; dist must equal the CPU oracle and
    every parent must be a real edge one level closer, at every cap."""
    if not mg_core.has_gpu():
        pytest.skip("no Metal device")
    src, dst, v = _multi_hub_graph()
    # direction="in" walks the IN orientation of the reversed graph, which
    # has the same hub rows as the OUT orientation of the original.
    gs, gd = (src, dst) if direction == "out" else (dst, src)
    g = mg.Graph.from_edges(gs, gd, directed=True, num_vertices=v)
    source = [0]
    mg.set_execution("cpu")
    try:
        ref, _ = mg.bfs(g, source, direction=direction)
        monkeypatch.setenv("MG_BFS_HIGH_SPLIT_MAX", split_max)
        mg.set_execution("gpu")
        dist, parent = mg.bfs(g, source, direction=direction)
        assert mg.last_run_info()["path"] == "gpu"
    finally:
        mg.set_execution("auto")
    dist = np.asarray(dist)
    parent = np.asarray(parent)
    np.testing.assert_array_equal(dist, np.asarray(ref))
    assert dist.max() >= 2
    child = np.flatnonzero(dist > 0)
    par = parent[child]
    assert (par >= 0).all()
    np.testing.assert_array_equal(dist[par], dist[child] - 1)
    edges = set(zip(src.tolist(), dst.tolist()))  # traversal edges p -> c
    assert all((int(p), int(c)) in edges for p, c in zip(par, child))


@pytest.mark.parametrize("direction", ["out", "both"])
def test_sparse_preflight_cap_boundaries(monkeypatch, direction):
    """The preflight succeeds iff the reached vertices fit the vertex cap
    and their summed degrees (out + in when two-sided) fit the edge cap;
    each reached vertex is charged when it is discovered."""
    n = 40
    src = np.arange(n - 1, dtype=np.uint32)
    g = mg.Graph.from_edges(src, src + 1, directed=True, num_vertices=n)
    edges = (n - 1) * (2 if direction == "both" else 1)
    mg.set_execution("cpu")

    def op_with(vertices, edge_cap):
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", str(vertices))
        monkeypatch.setenv("MG_BFS_SPARSE_MAX_EDGES", str(edge_cap))
        dist, _ = mg.bfs(g, [0], direction=direction)
        assert np.asarray(dist).tolist() == list(range(n))
        return mg.last_run_info()["op"]

    try:
        assert op_with(n, edges) == "bfs_sparse"
        assert op_with(n - 1, edges) == "bfs"
        assert op_with(n, edges - 1) == "bfs"
    finally:
        mg.set_execution("auto")


@pytest.mark.gpu
def test_sparse_preflight_default_caps_follow_fallback(monkeypatch):
    """Default caps are 8192 vertices / 524288 edges when the overflow
    fallback is the GPU and 1024 / 8192 when it is the threaded CPU: a
    2000-vertex reach is answered serially only in the first case."""
    if not mg_core.has_gpu():
        pytest.skip("no Metal device")
    for name in ("MG_BFS_SPARSE_MAX_VERTICES", "MG_BFS_SPARSE_MAX_EDGES",
                 "MG_E_GPU_MIN", "MG_E_GPU_MIN_BFS"):
        monkeypatch.delenv(name, raising=False)
    n = 2000
    src = np.arange(n - 1, dtype=np.uint32)
    g = mg.Graph.from_edges(src, src + 1, directed=True, num_vertices=n)
    try:
        mg.set_execution("cpu")
        mg.bfs(g, [0])
        assert mg.last_run_info()["op"] == "bfs"  # CPU fallback: 1024 cap
        mg.set_execution("auto")
        monkeypatch.setenv("MG_E_GPU_MIN_BFS", "0")  # plan the GPU
        dist, _ = mg.bfs(g, [0])
        info = mg.last_run_info()
        assert (info["op"], info["path"]) == ("bfs_sparse", "cpu")
        assert np.asarray(dist).tolist() == list(range(n))
        vs, _, _ = mg.bfs(g, [0], output="sparse")
        assert mg.last_run_info()["op"] == "bfs_sparse"
        assert np.asarray(vs).size == n
    finally:
        mg.set_execution("auto")


@pytest.mark.parametrize("degree", [65535, 65536, 65537, 131073])
@pytest.mark.parametrize("direction", ["out", "in", "both"])
@pytest.mark.parametrize("threads", ["1", "4"])
def test_cpu_bfs_edge_slice_degree_boundaries(degree, direction, threads, monkeypatch):
    leaves = np.arange(1, degree + 1, dtype=np.uint32)
    zeros = np.zeros(degree, dtype=np.uint32)
    # A source row crosses chunk boundaries. The two extra leaf edges
    # offer competing valid parents to one depth-2 vertex. Initial sources
    # include isolates before, between, and after two nonzero-degree rows,
    # exercising repeated edge-prefix offsets on the optimized both path.
    deep = degree + 1
    v = degree + 11
    if direction == "out":
        src = np.concatenate((zeros, [1, 2]))
        dst = np.concatenate((leaves, [deep, deep]))
    elif direction == "in":
        src = np.concatenate((leaves, [deep, deep]))
        dst = np.concatenate((zeros, [1, 2]))
    else:
        mid = degree // 2
        src = np.concatenate((zeros[:mid], leaves[mid:], [1, 2]))
        dst = np.concatenate((leaves[:mid], zeros[mid:], [deep, deep]))
    second_source, second_child = degree + 4, degree + 5
    extra_src, extra_dst = ((second_child, second_source) if direction == "in"
                            else (second_source, second_child))
    src = np.concatenate((src, [extra_src]))
    dst = np.concatenate((dst, [extra_dst]))
    perm = np.random.default_rng(37).permutation(v).astype(np.uint32)
    src = perm[src.astype(np.uint32)]
    dst = perm[dst.astype(np.uint32)]
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=v)
    monkeypatch.setenv("MG_THREADS", threads)
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "0")
    sources = perm[[degree + 2, 0, degree + 3, second_source, degree + 6, 0]]
    mg.set_execution("cpu")
    try:
        distance, parent = mg.bfs(g, sources, direction=direction)
        assert mg.last_run_info()["iterations"] == 3
        vs, ds, ps = mg.bfs(g, sources, direction=direction, output="sparse")
    finally:
        mg.set_execution("auto")
    expected = np.full(v, -1, np.int32)
    expected[sources] = 0
    expected[perm[leaves]] = 1
    expected[perm[deep]] = 2
    expected[perm[second_child]] = 1
    np.testing.assert_array_equal(distance, expected)
    np.testing.assert_array_equal(vs, np.flatnonzero(expected >= 0))
    np.testing.assert_array_equal(ds, expected[vs])
    if direction == "in":
        src, dst = dst, src
    edges = src.astype(np.int64) * v + dst
    if direction == "both":
        edges = np.concatenate((edges, dst.astype(np.int64) * v + src))
    for vertices, distances, parents in ((np.arange(v), distance, parent),
                                         (vs, ds, ps)):
        child = distances > 0
        assert (parents[~child] == -1).all()
        p = parents[child].astype(np.int64)
        c = vertices[child].astype(np.int64)
        assert ((p >= 0) & (p < v)).all()
        np.testing.assert_array_equal(expected[p], distances[child] - 1)
        assert np.isin(p * v + c, edges).all()


@pytest.mark.parametrize("threads", ["1", "4"])
def test_cpu_bfs_edge_slice_all_zero_degree_frontier(monkeypatch, threads):
    """>1024 isolated sources enter the parallel-size frontier branch with
    an all-zero edge prefix. The level still counts once and ends cleanly."""
    v = 2048
    empty = np.asarray([], np.uint32)
    g = mg.Graph.from_edges(empty, empty, directed=True, num_vertices=v)
    sources = np.random.default_rng(43).permutation(v)[:1500].astype(np.uint32)
    sources = np.concatenate((sources, sources[:2]))  # deduplicate before slicing
    expected = np.full(v, -1, np.int32)
    expected[sources] = 0
    monkeypatch.setenv("MG_THREADS", threads)
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "0")
    mg.set_execution("cpu")
    try:
        distance, parent = mg.bfs(g, sources, direction="both")
        assert mg.last_run_info()["iterations"] == 1
        vs, ds, ps = mg.bfs(g, sources, direction="both", output="sparse")
        assert mg.last_run_info()["iterations"] == 1
    finally:
        mg.set_execution("auto")
    np.testing.assert_array_equal(distance, expected)
    assert (parent == -1).all()
    np.testing.assert_array_equal(vs, np.flatnonzero(expected >= 0))
    np.testing.assert_array_equal(ds, np.zeros(1500, np.int32))
    assert (ps == -1).all()


@pytest.mark.parametrize("frontier_size", [262143, 262144, 262145])
@pytest.mark.parametrize("threads", ["1", "4", "default"])
def test_cpu_bfs_parallel_prefix_boundary(frontier_size, threads, monkeypatch):
    """Large initial frontiers cross the parallel-prefix threshold, with
    repeated offsets spanning scan blocks and hubs split across edge slices."""
    source_rows = [1, frontier_size // 4 - 1,
                   frontier_size // 2 + 1, frontier_size - 2]
    degrees = [65537, 3, 65539, 2]
    n_leaves = sum(degrees)
    leaves = np.arange(frontier_size, frontier_size + n_leaves, dtype=np.uint32)
    deep = frontier_size + n_leaves
    v = deep + 5
    src, dst = [], []
    cursor = 0
    for row, degree in zip(source_rows, degrees):
        neighbors = leaves[cursor:cursor + degree]
        cursor += degree
        split = degree // 2
        src.extend((np.full(split, row, np.uint32), neighbors[split:]))
        dst.extend((neighbors[:split], np.full(degree - split, row, np.uint32)))
    # A later level has competing parents and both orientation transitions.
    src.append(np.asarray([leaves[0], deep, leaves[0]], np.uint32))
    dst.append(np.asarray([deep, leaves[-1], deep], np.uint32))
    perm = np.random.default_rng(83).permutation(v).astype(np.uint32)
    src, dst = perm[np.concatenate(src)], perm[np.concatenate(dst)]
    sources = perm[np.arange(frontier_size, dtype=np.uint32)]
    sources = np.concatenate((sources, sources[[0, 1, -1]]))
    expected = np.full(v, -1, np.int32)
    expected[sources] = 0
    expected[perm[leaves]] = 1
    expected[perm[deep]] = 2
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=v)
    if threads == "default":
        monkeypatch.delenv("MG_THREADS", raising=False)
    else:
        monkeypatch.setenv("MG_THREADS", threads)
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "0")
    mg.set_execution("cpu")
    try:
        distance, parent = mg.bfs(g, sources, direction="both")
        assert mg.last_run_info()["iterations"] == 3
        vs, ds, ps = mg.bfs(g, sources, direction="both", output="sparse")
        assert mg.last_run_info()["iterations"] == 3
    finally:
        mg.set_execution("auto")
    np.testing.assert_array_equal(distance, expected)
    np.testing.assert_array_equal(vs, np.flatnonzero(expected >= 0))
    np.testing.assert_array_equal(ds, expected[vs])
    edges = np.concatenate((src.astype(np.int64) * v + dst,
                            dst.astype(np.int64) * v + src))
    for vertices, distances, parents in ((np.arange(v), distance, parent),
                                         (vs, ds, ps)):
        child = distances > 0
        assert (parents[~child] == -1).all()
        p, c = parents[child].astype(np.int64), vertices[child].astype(np.int64)
        assert ((p >= 0) & (p < v)).all()
        np.testing.assert_array_equal(expected[p], distances[child] - 1)
        assert np.isin(p * v + c, edges).all()


@pytest.mark.parametrize("frontier_size", [262144, 262145])
@pytest.mark.parametrize("threads", ["4", "default"])
def test_cpu_bfs_parallel_prefix_all_zero(frontier_size, threads, monkeypatch):
    """Every block total is zero; a large isolated-source level still runs
    exactly once and leaves disconnected nonsources untouched."""
    v = frontier_size + 7
    empty = np.asarray([], np.uint32)
    g = mg.Graph.from_edges(empty, empty, directed=True, num_vertices=v)
    sources = np.random.default_rng(89).permutation(v)[:frontier_size].astype(np.uint32)
    sources = np.concatenate((sources, sources[:3]))
    expected = np.full(v, -1, np.int32)
    expected[sources] = 0
    if threads == "default":
        monkeypatch.delenv("MG_THREADS", raising=False)
    else:
        monkeypatch.setenv("MG_THREADS", threads)
    monkeypatch.setenv("MG_BFS_SPARSE_MAX_VERTICES", "0")
    mg.set_execution("cpu")
    try:
        distance, parent = mg.bfs(g, sources, direction="both")
        assert mg.last_run_info()["iterations"] == 1
        vs, ds, ps = mg.bfs(g, sources, direction="both", output="sparse")
        assert mg.last_run_info()["iterations"] == 1
    finally:
        mg.set_execution("auto")
    np.testing.assert_array_equal(distance, expected)
    assert (parent == -1).all()
    np.testing.assert_array_equal(vs, np.flatnonzero(expected >= 0))
    np.testing.assert_array_equal(ds, np.zeros(frontier_size, np.int32))
    assert (ps == -1).all()
